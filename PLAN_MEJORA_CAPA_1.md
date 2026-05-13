# Plan de mejora — Capa 1: Datos y persistencia

> **Fuente:** Auditoría asistida 2026-05-12, complementaria a `PLAN_AUDITORIA.md`. Este plan **no se solapa** con los gaps de rúbrica T1–T11 del plan de auditoría; aborda calidad técnica, pedagógica y operacional dentro de la primera capa de la arquitectura (datos y persistencia).
> **Alcance:** 7 tareas, ~10 h de trabajo, todas tamaño S o M.
> **Estado inicial:** 37/37 tests verde; 12 gaps detectados (3 críticos, 6 medios, 3 bajos).

## Overview

La Capa 1 implementa ingesta de precios (yfinance) y macro (FRED), persistencia SQLite vía SQLAlchemy 2.0 y un cache transparente con TTL. La implementación actual pasa los tests y cubre el criterio funcional, pero tiene gaps en tres dimensiones: **robustez** (APIs externas sin retry/async, `datetime.utcnow` deprecated, falta validación de ticker), **calidad pedagógica** (docstrings sin anclaje al syllabus, tests sin semántica explícita de cache, falta script de bootstrap manual) y **performance/despliegue** (SQLite efímera en Render, FRED sin cache, queries no vectorizadas).

## Decisiones arquitectónicas (fijadas tras consulta)

1. **Persistencia en Render:** SQLite efímera + bootstrap automático en startup. Tras cada cold-start, una tarea en background descarga 2 años de histórico para los 5 activos + SPY. *Razón:* mantiene free-tier puro; aceptamos ~30–60 s adicionales en el arranque.
2. **APIs externas:** hardening completo con `tenacity` (backoff exponencial, 3 reintentos) + `asyncio.to_thread` para no bloquear el event loop + circuit breaker en memoria (3 fallos consecutivos → cool-down 5 min).
3. **Sin cambio de schema:** las migraciones se limitan a renombrar usos de `datetime.utcnow()` → `datetime.now(timezone.utc)`. No se altera tipo de columna ni se introduce Alembic.
4. **Tests sin red:** el bootstrap automático se desactiva con `Settings.bootstrap_on_startup: bool` (default `True` en prod, forzado `False` en `conftest.py`).

## Mapa de gaps verificados

| ID | Descripción | Dimensión | Severidad | Tarea |
|----|-------------|-----------|-----------|-------|
| C1-1 | `datetime.utcnow()` deprecated en Py 3.12+ (5 ocurrencias) | Robustez | 🟡 Media | T1.1 |
| C1-2 | `yf.download` y `requests.get` sync bloquean event loop async | Performance | 🔴 Alta | T1.2 |
| C1-3 | SQLite efímera en Render free-tier (cache se pierde tras redeploy) | Despliegue | 🔴 Alta | T1.6 |
| C1-4 | `_refresh_from_yfinance` no maneja 429/backoff | Robustez | 🔴 Alta | T1.2 |
| C1-5 | `fetch_fred_latest` no cachea — cada `/macro` pega a FRED | Performance | 🟡 Media | T1.7 |
| C1-6 | `get_prices` no valida ticker contra `assets` (cachea basura) | Robustez | 🟡 Media | T1.3 |
| C1-7 | Cache hit/miss/stale no observable (sin métrica/log diferenciado) | Pedagogía | 🟡 Media | T1.3 |
| C1-8 | Falta índice compuesto `(ticker, date)` para queries por rango | Performance | 🟢 Baja | T1.7 |
| C1-9 | Tests no verifican semántica de cache (solo el endpoint) | Pedagogía | 🟡 Media | T1.4 |
| C1-10 | Docstrings no anclan al syllabus (M5/M6/M9) | Pedagogía | 🟢 Baja | T1.4 |
| C1-11 | Sin script CLI de bootstrap manual del histórico | Pedagogía | 🟡 Media | T1.5 |
| C1-12 | `_read_prices_df` itera fila a fila (ineficiente >1k filas) | Performance | 🟢 Baja | T1.7 |

## Dependency Graph

```
T1.1 (datetime UTC)         ── independiente
T1.2 (API hardening) ──┐
                       ├──► T1.6 (bootstrap automático)
T1.5 (seed CLI) ───────┘
T1.3 (cache observability) ── independiente
T1.4 (docstrings + tests) ── después de T1.3
T1.7 (FRED cache + índice + vectorización) ── independiente
```

---

## Phase 1.A — Robustez técnica

### Task 1.1 — Migrar `datetime.utcnow()` → `datetime.now(timezone.utc)`

**Descripción:** `datetime.utcnow()` está deprecated en Python 3.12+. Reemplazar las 5 ocurrencias en la capa de datos por `datetime.now(timezone.utc)` y asegurar que las comparaciones de `_is_cache_stale` sigan funcionando (la fecha del último precio es `date`, no `datetime`; homogeneizar a tz-aware).

**Acceptance criteria:**
- [ ] Cero usos de `datetime.utcnow` en `backend/app/` (verificable con `grep -r "utcnow" backend/app/`)
- [ ] `db_models.py` usa `default=lambda: datetime.now(timezone.utc)` como default factory
- [ ] `_is_cache_stale` compara `datetime` tz-aware contra `datetime` tz-aware
- [ ] Todos los tests existentes siguen verdes (37/37)

**Verification:**
- `pytest backend/tests/ -v` → 37/37 verde
- `grep -r "utcnow" backend/app/` → vacío

**Dependencies:** Ninguna
**Files:**
- `backend/app/services/data.py`
- `backend/app/models/db_models.py`
- `backend/app/services/macro.py`

**Estimated scope:** XS

---

### Task 1.2 — Hardening de APIs externas (tenacity + async + circuit breaker)

**Descripción:** `yf.download` y `requests.get` son síncronos dentro de endpoints `async def`, bloquean el event loop. Además no hay reintentos ni protección contra rate-limiting (yfinance ya bloqueó al sandbox con 429 durante el primer entrenamiento ML). Introducir tres mecanismos coordinados:
- `tenacity.retry` con backoff exponencial (3 intentos: 1 s → 2 s → 4 s) para `_refresh_from_yfinance` y `fetch_fred_latest`.
- Wrapper async desde el router: `await asyncio.to_thread(_refresh_from_yfinance, ...)`. La signature del servicio no cambia.
- Circuit breaker por ticker: módulo-level `_circuit_state: dict[str, tuple[int, datetime]]`. Tras 3 fallos consecutivos, devolver `None` durante 5 min sin pegarle al API.

**Acceptance criteria:**
- [ ] `requirements.txt` añade `tenacity>=8.2`
- [ ] `_refresh_from_yfinance` decorado con `@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))`
- [ ] `precios.py` invoca `get_prices` vía `await asyncio.to_thread(...)`
- [ ] `services/data.py` expone `_circuit_state` y respeta cool-down de 5 min
- [ ] Nuevo test `test_data_resilience.py` simula 429 (mock de `yf.download` con `side_effect`) y verifica: reintenta 3 veces, abre circuit, cierra tras cool-down (manipulando el timestamp interno).

**Verification:**
- `pytest backend/tests/test_data_resilience.py -v`
- Manual: `time curl -s http://localhost:8000/precios/AAPL >/dev/null` antes y después; latencia comparable en happy path.

**Dependencies:** Ninguna
**Files:**
- `backend/app/services/data.py`
- `backend/app/services/macro.py`
- `backend/app/routers/precios.py`
- `backend/requirements.txt`
- `backend/tests/test_data_resilience.py` (nuevo)

**Estimated scope:** M (5 archivos — al límite)

---

### Task 1.3 — Validación de ticker + observabilidad de cache

**Descripción:** Hoy `get_prices(db, "FAKE123")` cachearía un ticker inexistente. Añadir validación: el ticker debe existir en `assets` o levantar `HTTPException(404)`. Adicionalmente, instrumentar el cache para distinguir tres estados — HIT, MISS, STALE — con log estructurado y un contador en memoria expuesto vía `GET /health/cache`. Esta es la base pedagógica para que el frontend o el docente explique el patrón cache transparente.

**Acceptance criteria:**
- [ ] `get_prices` levanta `ValueError("ticker not in assets table")` si el ticker no existe (el router lo traduce a 404)
- [ ] Cache emite log: `cache state=HIT|MISS|STALE ticker=AAPL ttl_remaining_min=42`
- [ ] Módulo `services/data.py` expone `CACHE_STATS: dict[str, dict[str, int]]` (por ticker → `{hit, miss, stale}`)
- [ ] `GET /health/cache` retorna `{tickers: {AAPL: {hit: 12, miss: 1, stale: 3}}, total_rows: 2580}`
- [ ] Test verifica los 3 estados explícitamente

**Verification:**
- `pytest -k 'cache or resilience'`
- `curl /health/cache` después de 5 requests retorna conteos coherentes

**Dependencies:** Ninguna
**Files:**
- `backend/app/services/data.py`
- `backend/app/routers/precios.py`
- `backend/app/routers/health.py`
- `backend/tests/test_cache_semantics.py` (nuevo)

**Estimated scope:** S

---

### Checkpoint Phase 1.A

- [ ] `pytest backend/tests/ -v` — todos verdes (≥40 tests, vs 37 iniciales)
- [ ] `grep -r "utcnow" backend/app/` — vacío
- [ ] `curl /health/cache` retorna JSON estructurado con contadores
- [ ] **Revisión humana antes de Fase 1.B**

---

## Phase 1.B — Calidad pedagógica

### Task 1.4 — Docstrings anclados al syllabus + tests didácticos de cache

**Descripción:** El estudiante debe poder leer `data.py` y reconocer qué módulo del curso ejerce. Añadir breadcrumbs en docstrings (estilo del que ya tiene `decorators.py` con `"(Semana 1 del curso)"`). Adicionalmente, los tests actuales de cache prueban el endpoint pero no la semántica: añadir 3 tests didácticos que el docente pueda mostrar en clase para explicar HIT/MISS/STALE.

**Acceptance criteria:**
- [ ] Cada función pública en `data.py`, `macro.py` y `database.py` tiene docstring que cita el módulo del syllabus (M5 Pydantic, M6 FastAPI, M9 SQLAlchemy, M13 ML Prod)
- [ ] `test_cache_semantics.py` extendido con 3 tests con docstring pedagógico explicativo:
  - `test_cache_cold_then_warm` (MISS seguido de HIT)
  - `test_cache_stale_triggers_refresh` (manipula `cache_ttl_minutes=0`)
  - `test_unknown_ticker_returns_404` (sin auto-cache)
- [ ] Cada test tiene comentario `# Pedagogía: ...` arriba del assert clave

**Verification:**
- `pytest backend/tests/test_cache_semantics.py -v --tb=short` → 3 verdes
- Revisión visual de docstrings

**Dependencies:** T1.3 (CACHE_STATS necesario para algunos asserts)
**Files:**
- `backend/app/services/data.py`
- `backend/app/services/macro.py`
- `backend/app/database.py`
- `backend/tests/test_cache_semantics.py` (extender)

**Estimated scope:** S

---

### Task 1.5 — Script CLI `seed_history` para bootstrap manual

**Descripción:** Comando `python -m app.scripts.seed_history` que descarga 2 años de histórico para los 5 tickers + SPY en un solo run. Lo usan: (a) el docente para preparar la DB de la máquina docente con datos reales, (b) el bootstrap automático del Task 1.6 internamente. Idempotente: si una fecha ya existe en la tabla, no la re-inserta.

**Acceptance criteria:**
- [ ] `app/scripts/__init__.py` + `app/scripts/seed_history.py`
- [ ] CLI acepta `--tickers AAPL,JPM,...` (default: los 5 seed + SPY) y `--years 2`
- [ ] Imprime progreso por ticker: `seeding AAPL [504/504 rows]`
- [ ] Idempotente (correr 2 veces no duplica filas)
- [ ] Termina con exit code 0 incluso si algún ticker falla (registra warning); retorna 1 solo si todos fallan
- [ ] Test verifica: `python -m app.scripts.seed_history --tickers FAKE` exit code 1 con log warning

**Verification:**
- `python -m app.scripts.seed_history --years 1 --tickers AAPL` → llena DB sin error
- `sqlite3 risk.db "SELECT COUNT(*) FROM prices WHERE ticker='AAPL'"` ≥ 250

**Dependencies:** T1.2 (usa `_refresh_from_yfinance` con tenacity)
**Files:**
- `backend/app/scripts/__init__.py` (nuevo)
- `backend/app/scripts/seed_history.py` (nuevo)
- `backend/tests/test_seed_history.py` (nuevo)

**Estimated scope:** S

---

## Phase 1.C — Performance y despliegue

### Task 1.6 — Bootstrap automático en startup (no bloqueante)

**Descripción:** En el `lifespan` event, tras `seed_assets_if_empty`, lanzar `asyncio.create_task` que ejecute `seed_history.run()` en background. La app sirve `/health` inmediatamente; el primer `/precios/AAPL` puede caer en MISS si el bootstrap aún no termina (eso es aceptable y observable en `/health/cache`). Controlado por `Settings.bootstrap_on_startup: bool`.

**Acceptance criteria:**
- [ ] `Settings.bootstrap_on_startup: bool = Field(default=True)`
- [ ] `Settings.bootstrap_years: int = Field(default=2)`
- [ ] En `main.py:lifespan`, si `settings.bootstrap_on_startup`, lanzar `asyncio.create_task(_bootstrap_in_background())` que llama a `seed_history.run()`
- [ ] `/health` retorna `{status: ok, bootstrap_state: running|complete|failed}` (estado en módulo-level `BOOTSTRAP_STATE`)
- [ ] `conftest.py` setea `settings.bootstrap_on_startup = False` para evitar background tasks en tests
- [ ] Test verifica que `bootstrap_on_startup=True` programa la tarea (mock de `seed_history.run`)

**Verification:**
- `uvicorn app.main:app` → log `bootstrap started years=2 tickers=6`; tras ~30–60 s log `bootstrap complete rows=3024`
- `curl /health` retorna `bootstrap_state` correcto
- `pytest backend/tests/` sigue verde y rápido (sin descargas reales)

**Dependencies:** T1.2, T1.5
**Files:**
- `backend/app/main.py`
- `backend/app/config.py`
- `backend/app/routers/health.py`
- `backend/tests/conftest.py`
- `backend/tests/test_bootstrap.py` (nuevo)

**Estimated scope:** M

---

### Task 1.7 — FRED cache + índice compuesto + vectorización de lectura

**Descripción:** Tres mejoras de performance agrupadas (cada una <30 min):
- **FRED cache:** wrapper de TTL sobre `fetch_fred_latest` (10 min). FRED cambia diariamente; cachear in-memory es suficiente.
- **Índice compuesto:** añadir `Index("ix_prices_ticker_date", "ticker", "date")` en `Price.__table_args__`. Los individuales se mantienen.
- **Vectorización:** `_read_prices_df` usa `pd.read_sql_query(stmt, db.bind)` en vez de iterar filas; reducción ~3× para datasets >500 filas.

**Acceptance criteria:**
- [ ] `fetch_fred_latest` con TTL 600 s; 2ª llamada en <10 min devuelve sin HTTP request (test con mock)
- [ ] `Price` define índice compuesto en `__table_args__`
- [ ] `_read_prices_df` usa `pd.read_sql_query`; resultado idéntico al actual
- [ ] Test `test_read_prices_df_matches_orm` compara ambos métodos con el mismo dataset

**Verification:**
- `pytest backend/tests/test_activos.py backend/tests/test_cache_semantics.py -v`
- Manual: `EXPLAIN QUERY PLAN SELECT * FROM prices WHERE ticker='AAPL' AND date>='2024-01-01'` debe mencionar el índice compuesto

**Dependencies:** Ninguna
**Files:**
- `backend/app/services/data.py`
- `backend/app/services/macro.py`
- `backend/app/models/db_models.py`

**Estimated scope:** S

---

## Checkpoint Phase 1 — Completa

- [ ] `pytest backend/tests/ -v` — ≥45 tests verde (vs 37 iniciales)
- [ ] `uvicorn app.main:app` arranca en <2 s, bootstrap completa en background en <60 s
- [ ] `curl /health/cache` y `curl /health` exponen estado coherente
- [ ] `python -m app.scripts.seed_history --tickers KO --years 1` funciona idempotente
- [ ] `docker build` sigue verde (tenacity añade ~1 MB)
- [ ] CI workflow no rompe (tests sin internet siguen pasando)
- [ ] **Revisión humana antes de pasar a Capa 2**

---

## Risks and Mitigations

| Riesgo | Impacto | Probabilidad | Mitigación |
|---|---|---|---|
| Bootstrap en background falla silenciosamente en Render | Alto | Media | `BOOTSTRAP_STATE` expuesto en `/health`; alerta visible en el frontend |
| yfinance sigue bloqueando 429 incluso con backoff | Alto | Media | Circuit breaker (T1.2) + fallback a serie sintética en bootstrap (warning en log) |
| `asyncio.to_thread` introduce concurrencia mal probada | Medio | Baja | Tests existentes corren contra el thread pool real (no mockean `to_thread`) |
| Wrapper TTL custom sobre FRED es frágil | Bajo | Baja | Mantener bajo 30 LOC y testear; usar `time.monotonic()` para evitar clock-skew |
| Bootstrap añade 30–60 s al cold-start de Render | Medio | Alta | Documentar en README; recomendar warm-up curl 2 min antes de demo |

## Open Questions

1. **`Asset.currency`:** ¿Vale la pena exponerlo como filtro futuro o lo dejamos hardcoded `USD`? *Default: dejar como está; no es prioridad de Capa 1.*
2. **Años de histórico para bootstrap:** ¿2 años es suficiente para todas las capas que dependen (GARCH, frontera)? *Default: sí; GARCH suele requerir ≥500 datos, 2 años hábiles ≈ 504.*
3. **Logging estructurado JSON:** ¿Migrar de texto a JSON para que Render Logs lo parsee? *Default: posponer a Capa 5 — Infraestructura.*

## Parallelization (si se quisiera ejecutar con varios agentes)

- **Lote independiente A:** T1.1, T1.3, T1.7 — no comparten archivos.
- **Lote secuencial B:** T1.2 → T1.5 → T1.6 (cadena de dependencias del bootstrap).
- **Final:** T1.4 (necesita T1.3 mergeado).

## Definition of Done

- [ ] Los 12 gaps verificados (C1-1 a C1-12) están cerrados.
- [ ] `pytest backend/tests/ -v` pasa con ≥45 tests (vs 37 actuales).
- [ ] Backend en Render arranca, bootstrap completa, `/health/cache` reporta estado coherente.
- [ ] README de `Proyecto_I/` documenta: política de cache, bootstrap, script `seed_history`, comportamiento ante 429.
- [ ] Commit final atomic en el submódulo + gitlink actualizado en el repo padre.
