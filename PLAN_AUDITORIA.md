# Plan de auditoría y cierre de gaps — Proyecto_I vs spec CIII

> **Fuente autoritativa:** `Proyecto_Integrador_Riesgo_Python_CIII.html` (raíz del repo padre).
> **Fecha:** 2026-05-12
> **Autor del plan:** Auditoría asistida (Claude Opus 4.7), revisada por Javier Sierra antes de ejecución.
> **Estado inicial:** 9/16 criterios completos, 5/16 con divergencias, 2/16 con falta significativa.

## Overview

La implementación de referencia `Proyecto_I` cumple la rúbrica con rigor matemático sólido, pero tiene **6 gaps con impacto directo en criterios de rúbrica** y **~7 gaps menores de pulido**. Este plan los cierra en 3 fases, cada una con checkpoint verificable. Total estimado: 4–6 horas de trabajo, todas tareas tamaño S o M.

## Architecture Decisions (fijadas tras consulta — 2026-05-12)

1. **Mantener `cvxpy` para Markowitz QP** *(decisión usuario)*
   *Razón:* `cp.quad_form` tiene valor pedagógico alto (formulación QP explícita y legible). La divergencia con la regla "<200 MB" se **documenta en README** como decisión consciente: se prioriza claridad matemática del código sobre el peso de la imagen.
   *Mitigación:* optimizar el Dockerfile al máximo (purgar `.pyc`, `tests/`, archivos de docs de paquetes) para minimizar peso aun con cvxpy.
2. **Renombrar columnas de `PredictionLog`** a `input_features` / `timestamp` + añadir `actual` (nullable). *(decisión usuario)*
   *Migración:* borrar `backend/risk.db`; el lifespan re-crea schema y `seed_assets_if_empty` regenera los 5 activos. Documentar en T2.
3. **Markowitz 10k portafolios = Monte Carlo Dirichlet** *(decisión usuario)*
   `Dirichlet(α=1)` si `non_negative=True`; `Normal(0, 0.3)` normalizado si no. Refleja literalmente "portafolios simulados" de la spec.
4. **CORS:** lista de orígenes vía `Settings.cors_origins` (CSV en `.env`); default `["*"]` solo si `env=dev`. En prod requiere lista explícita.
5. **Stress sobre tasas:** `rate_shock` mueve `rf_daily` en la valoración del bono y del CAPM (no shockea retornos de equity directamente — eso es `market_crash`).
6. **No introducir Alembic** (sigue siendo bonificación opcional). Las migraciones se hacen con `Base.metadata.create_all()` + recreación de DB en dev.

## Mapa de gaps verificados

| ID | Descripción | Criterio | Severidad | Tarea |
|----|-------------|----------|-----------|-------|
| G1 | Escenarios de stress no coinciden con spec (rate, market, combined) | 9 ★ 3% | 🔴 Alta | T1 |
| G2 | `PredictionLog` sin campo `actual`; nombres divergentes | 11 ★★ 7% | 🔴 Alta | T2 |
| G3 | Tabla `signals_log` ausente; `/alertas` no persiste | 1 ★ 7% | 🔴 Alta | T3 |
| G4 | Imagen Docker ~700 MB (spec exige <200 MB) | 13 ★★ 6% | 🔴 Alta | T4 |
| G5 | Sin CORSMiddleware | 14 funcional | 🔴 Alta | T5 |
| G6 | Frontend tab Stress sin bar/heatmap/comparación VaR | 9 ★ + 14 | 🟡 Media | T9 |
| G7 | Markowitz sin 10k portafolios simulados | 6 ★ 7% | 🟡 Media | T6 |
| G8 | Nelson-Siegel usa `Nelder-Mead` (spec: `least_squares`) | 7 ★ 6% | 🟡 Media | T7 |
| G9 | Familia GARCH sin ARCH(1) puro | 3 ★ 6% | 🟡 Media | T7 |
| G10 | `/alertas` sin `threshold_params` configurables | 1 ★ 7% | 🟡 Media | T3 |
| G11 | `/precios/{ticker}` sin `start`/`end` | 10 ★ 10% | 🟢 Baja | T8 |
| G12 | Frontend sin textos interpretativos prosa | 14 3% | 🟢 Baja | T10 |
| G13 | README sin justificación de activos seleccionados | 15 5% | 🟢 Baja | T11 |

## Dependency Graph

```
T1 (stress scenarios) ─┬─► T9 (frontend stress viz)
T2 (PredictionLog)    ─┘
T3 (signals_log)       ──► T10 (frontend interpretativo, opcional)
T4 (Docker / cvxpy)    ──► T6 (Markowitz 10k uses scipy too)
T5 (CORS)              ── independiente
T7 (NS + ARCH(1))      ── independiente
T8 (/precios fechas)   ── independiente
T11 (README)           ── después de T4 y T7 (para documentar decisiones)
```

---

## Phase 1 — Cierre de gaps críticos de rúbrica

### Task 1: Corregir escenarios de stress testing al magnitudes de spec

**Descripción:** Re-alinear `SCENARIO_SHOCKS` y la lógica de `apply_scenario` para reflejar las magnitudes exactas de la spec: `rate_shock` = Δr ±200 pb sobre tasa (aplicado a valuación de bono y a CAPM expected return), `market_crash` = -20 % y -30 % sobre benchmark (dos sub-escenarios), `vol_spike` = σ→σ·2 (ya OK), `combined` = -20 % + σ×2 + Δr +200 pb. Añadir sensibilidad por activo para el heatmap del frontend (T9).

**Acceptance criteria:**
- [ ] `SCENARIO_SHOCKS` reemplazado por dataclass `Scenario` con campos `rate_shock_bp`, `market_drop_pct`, `vol_multiplier`
- [ ] `apply_scenario` retorna `{var_base, var_stressed, portfolio_loss, sensitivity_by_asset: dict[str, float]}`
- [ ] `market_crash` expone dos variantes: `market_crash_20` y `market_crash_30`
- [ ] `StressOut` Pydantic actualizado para incluir `sensitivity_by_asset`
- [ ] Test `tests/test_stress.py` (nuevo) verifica las 5 magnitudes y respuesta del endpoint

**Verification:** `pytest backend/tests/test_stress.py -v` (debe pasar; al menos 5 assertions)
**Dependencies:** Ninguna
**Files:**
- `backend/app/services/stress.py`
- `backend/app/routers/stress.py`
- `backend/app/models/schemas.py`
- `backend/tests/test_stress.py` (nuevo)
**Estimated scope:** M (4 archivos)

---

### Task 2: Migrar `PredictionLog` a esquema spec-compliant

**Descripción:** Renombrar `features→input_features`, `ts→timestamp`; añadir columna `actual: float | None`; añadir endpoint `POST /predict/{id}/actual` para back-fill del valor real (habilita futura bonificación de drift). Actualizar `predict.py` y test correspondiente. Recrear `risk.db` (borrar archivo; lifespan rehace schema).

**Acceptance criteria:**
- [ ] `db_models.PredictionLog` con columnas: `id, ticker, input_features (JSON), prediction (Integer), probability (Float), actual (Float, nullable), model_version, timestamp`
- [ ] `routers/predict.py` insert usa nombres nuevos
- [ ] Endpoint `POST /predict/{log_id}/actual` con body `{actual: float}` actualiza el registro
- [ ] `tests/test_ml_singleton.py` verifica nombres nuevos y endpoint de actual

**Verification:** `pytest backend/tests/test_ml_singleton.py -v`
**Dependencies:** Ninguna
**Files:**
- `backend/app/models/db_models.py`
- `backend/app/routers/predict.py`
- `backend/app/models/schemas.py`
- `backend/tests/test_ml_singleton.py`
**Estimated scope:** S

---

### Task 3: Añadir `SignalLog` + `/alertas` con `threshold_params` y persistencia

**Descripción:** Modelar tabla `signals_log` (id, timestamp, ticker, rule, value), persistir cada detección en `GET /alertas`, y aceptar `threshold_params` opcionales por query string (`rsi_overbought`, `rsi_oversold`, `bb_k`). Añadir `@field_validator` con rangos válidos (RSI ∈ [50, 100], etc.). Test verifica persistencia y validación.

**Acceptance criteria:**
- [ ] `SignalLog` en `db_models.py` con índices en `ticker` y `timestamp`
- [ ] `/alertas` acepta `rsi_overbought: float = 70`, `rsi_oversold: float = 30`, `bb_k: float = 2.0` validados Pydantic
- [ ] Cada señal detectada se inserta en `signals_log` antes de retornar
- [ ] `services/signals.py:detect_signals` acepta thresholds parametrizados
- [ ] Test verifica persistencia (count > 0 después del endpoint) y rechazo 422 si `rsi_overbought < 50`

**Verification:** `pytest backend/tests/test_signals.py -v` (nuevo)
**Dependencies:** Ninguna
**Files:**
- `backend/app/models/db_models.py`
- `backend/app/models/schemas.py`
- `backend/app/services/signals.py`
- `backend/app/routers/alertas.py`
- `backend/tests/test_signals.py` (nuevo)
**Estimated scope:** M (5 archivos — en el límite, pero coherente)

---

### Task 4: Documentar trade-off de `cvxpy` + optimizar Dockerfile al máximo

**Descripción:** Por decisión consciente (valor pedagógico de `cp.quad_form`), se mantiene `cvxpy`. La spec exige <200 MB; con cvxpy es inalcanzable. Mitigación: (a) optimizar Dockerfile (purgar tests/, docs de paquetes, `__pycache__`, `.dist-info` opcional, etc.), (b) documentar la decisión en README con justificación pedagógica, (c) ajustar comentario del Dockerfile para evitar mentir sobre el objetivo.

**Acceptance criteria:**
- [ ] Dockerfile elimina archivos no necesarios del stage runtime (find /usr/local -name "tests" -type d -exec rm -rf, find -name "*.pyc", etc.)
- [ ] Comentario del Dockerfile actualizado: `# Multi-stage build. Imagen ~500 MB por cvxpy (decisión pedagógica documentada en README).`
- [ ] Imagen final reportada por `docker images` registrada en README (objetivo: <600 MB tras purga)
- [ ] Sección "Decisiones de implementación" del README explica por qué se mantuvo cvxpy

**Verification:**
- `docker build -t riesgo:slim ./backend && docker images riesgo:slim --format "{{.Size}}"` reporta tamaño
- README contiene la sección justificativa

**Dependencies:** Ninguna
**Files:**
- `backend/Dockerfile`
- `README.md` (parte de T11, pero el bloque cvxpy va aquí)
**Estimated scope:** XS
**Nota:** T6 ya no depende de T4 (ambas pueden ejecutarse en paralelo).

---

### Task 5: Añadir `CORSMiddleware` parametrizado

**Descripción:** Añadir `CORSMiddleware` a `main.py`, con `allow_origins` configurable vía `Settings.cors_origins` (CSV en `.env`). Por defecto en `env=dev` permite `["*"]`; en `env=prod` requiere lista explícita.

**Acceptance criteria:**
- [ ] `Settings.cors_origins: list[str]` con default `["*"]` en dev
- [ ] `CORSMiddleware` montado antes del primer router
- [ ] `.env.example` documenta `CORS_ORIGINS=https://app.streamlit.io,https://miapp.com`
- [ ] Test verifica header `Access-Control-Allow-Origin` en respuesta a `OPTIONS /health`

**Verification:** `pytest backend/tests/test_health.py -k cors`
**Dependencies:** Ninguna
**Files:**
- `backend/app/main.py`
- `backend/app/config.py`
- `backend/.env.example`
- `backend/tests/test_health.py`
**Estimated scope:** S

---

### Checkpoint Phase 1

- [ ] `pytest backend/tests/ -v` — todos los tests pasan (incluyendo los nuevos)
- [ ] `docker build -t riesgo:slim ./backend` — termina sin error, imagen <200 MB
- [ ] `rm backend/risk.db && uvicorn app.main:app` — arranca, crea schema nuevo, seedea activos, carga modelo
- [ ] `curl http://localhost:8000/alertas?rsi_overbought=75` — responde JSON con señales y persiste en `signals_log`
- [ ] `curl -X POST http://localhost:8000/stress -d '{"weights": {"AAPL": 0.5, "JPM": 0.5}}'` — responde con 5 escenarios (rate_shock, market_crash_20, market_crash_30, vol_spike, combined)
- [ ] **Revisión humana antes de Fase 2**

---

## Phase 2 — Cierre de gaps medios y mejoras pedagógicas

### Task 6: Markowitz con 10,000 portafolios Monte Carlo

**Descripción:** Añadir `simulate_random_portfolios(mu, cov, n=10000, non_negative=True)` que muestrea pesos: Dirichlet(α=1) si `non_negative=True`, Normal(0, 0.3) normalizado si no. Retornar lista de puntos `{ret, vol, sharpe, weights}`. Incluirlos en `FrontierOut`. Frontend dibuja nube + frontera encima.

**Acceptance criteria:**
- [ ] `services/portfolio.py:simulate_random_portfolios` retorna `np.ndarray (n, 3)` con [ret, vol, sharpe]
- [ ] `FrontierOut.simulated: list[PortfolioPoint]` (sin weights para no inflar respuesta — solo {ret, vol, sharpe})
- [ ] Endpoint `/frontera-eficiente` acepta `n_random: int = 10000` opcional
- [ ] Frontend renderiza scatter de simulados (alpha=0.2) + frontera (línea) + min_var/max_sharpe (markers)
- [ ] Test con `n=200` (rapidez) verifica que `min(vols_simulados) ≥ min_var.vol`

**Verification:** `pytest -k 'frontera or markowitz'`; manual en `streamlit run frontend/app.py` tab 6
**Dependencies:** T4 (debe usar scipy)
**Files:**
- `backend/app/services/portfolio.py`
- `backend/app/models/schemas.py`
- `backend/app/routers/frontera.py`
- `frontend/app.py`
- `backend/tests/test_portfolios.py`
**Estimated scope:** M

---

### Task 7: Nelson-Siegel con `least_squares` + ARCH(1) puro en familia GARCH

**Descripción:** (a) Migrar `fit_ns` a `scipy.optimize.least_squares` (residuos vectoriales: `f(x) = y_observado - y_NS(t)`). (b) Añadir modelo `arch_model(returns, mean='Constant', vol='ARCH', p=1)` a `fit_garch_family` para tener los 4 modelos que la spec menciona (ARCH(1), GARCH(1,1), EGARCH, GJR).

**Acceptance criteria:**
- [ ] `fit_ns` usa `least_squares` con `x0 = [yields.mean(), -0.02, 0.02, 2.0]` y `bounds` para `tau > 0`
- [ ] RMSE post-cambio ≤ RMSE actual (test compara contra valor de referencia)
- [ ] `fit_garch_family` retorna 4 entradas si `arch` disponible
- [ ] Test `test_options_bond.py::test_ns_fits_flat_curve_approx_constant` sigue pasando

**Verification:** `pytest -k 'ns or volatilidad'`
**Dependencies:** Ninguna
**Files:**
- `backend/app/services/fixed_income.py`
- `backend/app/services/volatility.py`
- `backend/tests/test_options_bond.py` (ajustar tolerancias si necesario)
**Estimated scope:** S

---

### Task 8: `/precios/{ticker}` con `?start=&end=`

**Descripción:** Añadir filtrado por rango de fechas. Si `start` o `end` se omiten, se interpreta como abierto en ese extremo. Filtrar en SQL (no en Python).

**Acceptance criteria:**
- [ ] `precios.py` acepta `start: date | None = None`, `end: date | None = None`
- [ ] `services/data.py:get_prices` propaga el filtro al SELECT
- [ ] Si `start > end` → 422
- [ ] Test verifica rango parcial (ej. último año)

**Verification:** `pytest -k precios`
**Dependencies:** Ninguna
**Files:**
- `backend/app/routers/precios.py`
- `backend/app/services/data.py`
- `backend/tests/test_activos.py`
**Estimated scope:** XS

---

### Task 9: Frontend tab Stress con 3 visualizaciones

**Descripción:** Refactor del tab 11 (Stress) para mostrar (a) bar chart de `portfolio_loss` por escenario (matplotlib), (b) comparación VaR base/estresado (st.bar_chart paired), (c) heatmap de `sensitivity_by_asset` por escenario (seaborn + st.pyplot). Backend ya expone `sensitivity_by_asset` por T1.

**Acceptance criteria:**
- [ ] Bar chart de pérdida por escenario (5 barras)
- [ ] Comparación VaR base vs estresado (paired bar)
- [ ] Heatmap activos×escenarios (seaborn `heatmap` o `st.dataframe.style.background_gradient`)
- [ ] Caption interpretativo de 2-3 frases bajo cada gráfico

**Verification:** Manual — `streamlit run frontend/app.py`, tab 11, observar 3 visualizaciones sin error
**Dependencies:** T1 (sensitivity_by_asset en el response)
**Files:**
- `frontend/app.py`
- `frontend/requirements.txt` (añadir `matplotlib`, `seaborn`)
**Estimated scope:** S

---

### Checkpoint Phase 2

- [ ] Tab Markowitz: nube + frontera + markers visibles
- [ ] Tab Stress: 3 visualizaciones renderizan
- [ ] `pytest backend/tests/ -v` — sigue verde
- [ ] `GET /precios/AAPL?start=2024-01-01` filtra correctamente
- [ ] **Revisión humana antes de Fase 3**

---

## Phase 3 — Pulido narrativo y documentación

### Task 10: Textos interpretativos por tab del frontend

**Descripción:** Añadir un bloque `st.markdown` con interpretación prosa (3-5 frases pedagógicas) al final de cada uno de los 12 tabs. Lenguaje accesible, explicar el "qué significa este resultado" sin jerga gratuita.

**Acceptance criteria:**
- [ ] 12 tabs con sección "Interpretación" o "Lectura del resultado"
- [ ] Cada texto cita al menos un valor numérico del resultado (no genérico)
- [ ] Sin errores ortográficos o de codificación

**Verification:** Revisión visual manual del frontend deployado
**Dependencies:** T9 (para que el tab Stress ya tenga las viz)
**Files:**
- `frontend/app.py`
**Estimated scope:** S

---

### Task 11: README — justificación de activos + sección de divergencias

**Descripción:** Añadir al README: (a) justificación pedagógica de los 5 activos (AAPL=tech, JPM=financiero, XOM=energía, JNJ=salud, KO=consumo defensivo) — diversificación sectorial intencional, (b) sección "Decisiones de implementación" listando: SLSQP en vez de cvxpy (T4), Monte Carlo Dirichlet (T6), nombres de `PredictionLog` (T2), CORS configurable (T5).

**Acceptance criteria:**
- [ ] Sección "Activos seleccionados" expandida con párrafo por activo
- [ ] Sección "Decisiones de implementación" con 4-5 ítems
- [ ] Tabla "Mapa rúbrica → código" actualizada si los archivos cambiaron de ruta

**Verification:** Revisión manual; `markdownlint` opcional
**Dependencies:** T4, T7
**Files:**
- `README.md`
**Estimated scope:** XS

---

### Checkpoint Phase 3 — Listo para sustentación

- [ ] Auditoría re-ejecutada: 16/16 criterios marcados ✅
- [ ] Demo en vivo end-to-end del frontend deployado en Render funciona
- [ ] README cubre arquitectura, deploy, ML, política IA, activos, decisiones de implementación
- [ ] Imagen Docker <200 MB confirmada en CI (workflow imprime tamaño)
- [ ] PR/commit final con mensaje "Auditoría CIII cumplida"

---

## Risks and Mitigations

| Riesgo | Impacto | Probabilidad | Mitigación |
|---|---|---|---|
| SLSQP no converge en QP mal condicionados | Medio | Baja | Regularización Tikhonov `Σ + 1e-8·I` + `x0 = pesos iguales` + fallback a `trust-constr` |
| Reemplazo de `cvxpy` rompe demo/tests existentes con tolerancias estrictas | Medio | Media | Ajustar tolerancias en tests a 1e-4 (suficiente para uso financiero) |
| Migración de schema requiere borrar `risk.db` y rompe entornos con datos | Bajo | Alta | Documentar en T2; el seed regenera 5 activos automáticamente |
| `seaborn` aumenta peso del frontend Streamlit | Bajo | Alta | Solo afecta frontend, no la imagen Docker del backend |
| `arch` library tarda mucho en compilar en Docker build | Bajo | Media | Ya está documentado; build cache en GitHub Actions |
| CORS `allow_origins=["*"]` en prod es inseguro | Medio | Baja | Default depende de `env`; en prod requiere lista explícita |

## Open Questions

> Estas 3 preguntas tienen defaults asumidos arriba. Si discrepas, revisar antes de empezar Fase 1.

1. **T4 (Docker)**: ¿Eliminar `cvxpy` (cumple spec) o documentar la divergencia y dejarlo (conserva valor pedagógico de `cp.quad_form`)? *Default: eliminar.*
2. **T2 (PredictionLog)**: ¿Renombrar `features→input_features` y `ts→timestamp` (rompe DB existente — hay que borrar `risk.db`)? *Default: sí, renombrar.*
3. **T6 (Markowitz nube)**: ¿Monte Carlo Dirichlet (10k aleatorios) o muestreo denso de frontera (10k puntos)? *Default: Monte Carlo, más educativo y matchea la frase "portafolios simulados".*

## Parallelization (si se quisiera ejecutar con varios agentes)

- **Lote independiente A**: T1, T2, T3, T5 — no comparten archivos.
- **Lote independiente B**: T7, T8 — no comparten archivos.
- **Secuencial**: T4 → T6 (T6 usa scipy de T4).
- **Después de T1+T9**: T10.
- **Al final**: T11.

## Definition of Done

- [ ] Los 13 gaps verificados (G1–G13) están cerrados o documentados como decisión explícita.
- [ ] `pytest backend/tests/ -v` pasa con 40+ tests (vs 37 actuales).
- [ ] `docker build` produce imagen <200 MB.
- [ ] CORS funciona end-to-end (frontend local → backend local; frontend Render → backend Render).
- [ ] Frontend tab Stress y tab Markowitz tienen visualizaciones exigidas por spec.
- [ ] README contiene secciones de "Activos seleccionados" y "Decisiones de implementación".
- [ ] Commit final atomic con mensaje descriptivo, push a GitHub, CI verde.
