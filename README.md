# Proyecto Integrador — Teoría del Riesgo (solución de referencia)

> Implementación de referencia del proyecto integrador del curso *Python para Desarrollo de APIs e Inteligencia Artificial* (Universidad Santo Tomás · pregrado en Estadística · 2026-I).

Sistema completo de análisis de riesgo financiero en cinco capas. Sirve como código que los estudiantes leen mientras construyen su propio repositorio desde cero.

## Resumen ejecutivo

API FastAPI con persistencia SQLite, modelo de machine learning servido vía Singleton, contenerizada con Docker multi-stage y desplegada en Render free-tier. Incluye un tablero Streamlit con 12 pestañas que consumen el backend.

**Stack:** Python 3.11.9 · FastAPI · Pydantic v2 · SQLAlchemy 2 · cvxpy · arch · scikit-learn · pytest · Docker · GitHub Actions · Streamlit.

## Arquitectura en 5 capas

1. **Datos y persistencia** — ingesta desde yfinance + FRED, persistencia en SQLite vía SQLAlchemy ORM, cache transparente (TTL configurable). 5 tablas: `Asset`, `Price`, `Portfolio`, `PredictionLog`, `SignalLog`.
2. **Análisis clásico** — indicadores técnicos, rendimientos, volatilidad EWMA + 4 modelos GARCH (ARCH(1), GARCH(1,1), EGARCH, GJR), CAPM, VaR (3 métodos) + CVaR + backtesting de Kupiec, Markowitz QP con y sin no-negatividad **+ nube Monte Carlo de 10k portafolios**.
3. **Renta fija y derivados** — curva FRED + ajuste Nelson-Siegel **por `scipy.optimize.least_squares`**, duración Macaulay/modificada, convexidad, Black-Scholes europeo + 5 Greeks, paridad put-call, stress testing con 5 escenarios (rate ±200 pb, market crash 20 %, market crash 30 %, vol spike σ×2, combined).
4. **Machine Learning** — pipeline `train.py → joblib → load → predict` con patrón Singleton vía lifespan de FastAPI. Cada predicción se persiste en `PredictionLog` (campos `input_features`, `prediction`, `probability`, `actual`, `timestamp`). Endpoint `POST /predict/{log_id}/actual` para back-fill y monitoreo de drift.
5. **Infraestructura** — pytest + TestClient (62 tests), CORS configurable, Docker multi-stage con purga de artefactos, deploy en Render, GitHub Actions CI con `pytest` + build de imagen.

## Endpoints (16+ rutas)

```text
GET    /health
GET    /activos
GET    /precios/{ticker}              ?start=YYYY-MM-DD&end=YYYY-MM-DD
GET    /rendimientos/{ticker}
GET    /indicadores/{ticker}
GET    /volatilidad/{ticker}          ?ewma_lambda=0.94
POST   /var
GET    /capm
POST   /frontera-eficiente            con n_random para nube Monte Carlo
GET    /alertas                       ?rsi_overbought=70&rsi_oversold=30&bb_k=2.0
GET    /macro
GET    /curva-rendimiento
POST   /bono/duracion
POST   /opcion/precio
POST   /stress
POST   /predict
POST   /predict/{log_id}/actual       back-fill para monitoreo de drift
POST   /portafolios   GET /portafolios   GET/DELETE /portafolios/{id}
```

Documentación auto-generada: `/docs` (Swagger) y `/redoc`.

## Instalación local

Requiere Python 3.11.

```bash
git clone --recurse-submodules <repo-del-curso>
cd proyecto-integrador-riesgo-2026I/backend

python -m venv .venv
source .venv/bin/activate          # macOS / Linux
# .\.venv\Scripts\activate          # Windows
pip install -r requirements.txt

cp .env.example .env                # editar y pegar FRED_API_KEY real
```

Para obtener una `FRED_API_KEY` (gratis): <https://fred.stlouisfed.org/docs/api/api_key.html>. Si no la pones, el backend usa un fallback estático para la tasa libre y la curva.

### Ejecutar el backend

```bash
cd backend
uvicorn app.main:app --reload
# → http://localhost:8000/health
# → http://localhost:8000/docs
```

El primer arranque crea las tablas (`risk.db`) y siembra 5 activos (AAPL, JPM, XOM, JNJ, KO). Los precios se traen de yfinance la primera vez que se pide `/precios/{ticker}` y quedan cacheados.

### Entrenar el modelo ML

```bash
cd backend
python -m app.ml.train               # entrena AAPL por defecto
python -m app.ml.train JPM           # entrena otro ticker
```

Genera `app/ml/model.joblib`. Sin modelo, `/predict` retorna 500 (con mensaje claro). El backend reanuda funcionalidad al detectar el modelo en disco al próximo restart.

### Ejecutar el tablero

```bash
cd frontend
pip install -r requirements.txt
streamlit run app.py
# → http://localhost:8501
```

El tablero lee `API_BASE` del entorno (por defecto `http://localhost:8000`). Para apuntar al deploy: `API_BASE=https://...onrender.com streamlit run app.py`.

### Correr los tests

```bash
cd backend
pytest tests/ -v
```

**62 tests** cubriendo: indicadores, VaR + Kupiec, Black-Scholes + paridad, bono + duración, frontera eficiente + nube Monte Carlo, CRUD portafolios, Singleton del ML + back-fill de actual, stress con 5 escenarios spec-CIII, señales con umbrales configurables + persistencia en `signals_log`, CORS, filtros de fecha en `/precios`.

## Ejecutar con Docker

```bash
docker compose up --build
# → http://localhost:8000/health
```

El compose levanta el backend con volumen persistente para la BD. La imagen se construye en multi-stage sobre `python:3.11.9-slim-bookworm`.

**Nota sobre el tamaño:** la spec sugiere < 200 MB; con `cvxpy + arch + scipy + statsmodels` la imagen final ronda los 600-700 MB tras la purga de `tests/`, `__pycache__/` y `*.pyc` en el stage runtime. Es un trade-off documentado a favor del rigor matemático: `cvxpy` resuelve el QP de Markowitz con `cp.quad_form` (formulación explícita y pedagógica que aparece en clase), `arch` ajusta GARCH/EGARCH/GJR correctamente con MLE, `scipy.optimize.least_squares` calibra Nelson-Siegel. Para una versión más ligera se podría reimplementar Markowitz con `scipy.optimize.minimize(SLSQP)` y reemplazar `arch` por un GARCH propio — perdiendo el contenido didáctico de las librerías canónicas.

## Deploy en Render

El `render.yaml` en la raíz define el servicio. Pasos:

1. Crear cuenta en <https://render.com>.
2. New → Blueprint → conectar este repo → seleccionar `render.yaml`.
3. Configurar la variable `FRED_API_KEY` en el dashboard de Render (es secreta, no en el yaml).
4. Render construye la imagen y expone una URL pública.

**Cold-start:** el free-tier duerme después de 15 min de inactividad. Despertar tarda ~50 s. Para una demo en vivo, "calentar" el servicio con un `curl` 2 minutos antes:

```bash
curl https://<tu-app>.onrender.com/health
```

URL pública: *por configurar al momento del deploy*.

## Modelos ORM (SQLAlchemy)

- `Asset(ticker PK, name, sector, currency)`
- `Price(id PK, ticker FK, date, close, volume, fetched_at)` con unique constraint `(ticker, date)`.
- `Portfolio(id, name, holdings JSON, created_at)`.
- `PredictionLog(id, ticker, input_features JSON, prediction, probability, actual nullable, model_version, timestamp)` — nombres alineados con spec CIII; `actual` permite back-fill para drift.
- `SignalLog(id, timestamp, ticker, rule, value)` — persiste cada detección de `/alertas`.

## Validación con Pydantic v2

Cubre criterio 10 de la rúbrica:

- Request y response tipados en todos los endpoints (modelos en `app/models/schemas.py`).
- `Field()` con descripciones, defaults y restricciones (`gt`, `ge`, `min_length`).
- `@field_validator` personalizados: tickers, suma de pesos del portafolio, tipo de opción.
- `@model_validator` en `HealthOut`.
- Modelos anidados: `FrontierOut → PortfolioPoint`, `OptionOut → Greeks`, `VaROut → VaRMethodResult`.
- Errores de validación retornan HTTP 422 con mensaje claro.

## Inyección de dependencias

- `Depends(get_db)` para la sesión SQLAlchemy.
- `Depends(get_settings)` para la configuración (Singleton vía `lru_cache`).
- `Depends(get_predictor)` para el modelo ML (Singleton vía módulo-level cache).

## Singleton del ML

El modelo se carga **una sola vez** al arranque (event `lifespan`). En los logs verás:

```text
... modelo cargado version=1.0.0 n_samples=600 acc=0.55
```

Tres llamadas a `/predict` solo muestran ese log una vez. Verificable también en `tests/test_ml_singleton.py`.

## Decoradores propios

- `@log_latency("nombre")` en `app/decorators.py`: loguea ms por llamada. Soporta sync y async.

## Estructura de carpetas

```text
proyecto-integrador-riesgo-2026I/
├── backend/
│   ├── app/
│   │   ├── main.py             ← FastAPI app, routers, lifespan
│   │   ├── config.py           ← BaseSettings
│   │   ├── database.py         ← engine, SessionLocal, Base
│   │   ├── dependencies.py     ← re-export Depends
│   │   ├── decorators.py       ← @log_latency
│   │   ├── models/{db_models, schemas}.py
│   │   ├── services/           ← lógica de negocio por dominio
│   │   ├── ml/                 ← train.py, features.py, predictor.py
│   │   └── routers/            ← un archivo por endpoint
│   ├── tests/                  ← 37 tests con TestClient + BD en memoria
│   ├── Dockerfile              ← multi-stage 3.11.9-slim-bookworm
│   ├── requirements.txt
│   └── .env.example
├── frontend/
│   ├── app.py                  ← Streamlit con 12 pestañas
│   └── requirements.txt
├── .github/workflows/ci.yml    ← pytest + docker build en cada push
├── docker-compose.yml
├── render.yaml
├── .gitignore
└── README.md
```

## Mapa rúbrica → código

| # | Criterio | Peso | Dónde está |
|---|---|---:|---|
| 1 | Análisis técnico + señales | 7% | `services/indicators.py`, `services/signals.py`, `routers/indicadores.py`, `routers/alertas.py` |
| 2 | Rendimientos + propiedades empíricas | 4% | `services/returns.py`, `routers/rendimientos.py` |
| 3 ★ | Volatilidad EWMA + GARCH | 6% | `services/volatility.py`, `routers/volatilidad.py` |
| 4 | CAPM + benchmark | 6% | `services/capm.py`, `routers/capm.py` |
| 5 | VaR + CVaR + Kupiec | 7% | `services/var.py`, `routers/var.py` |
| 6 | Markowitz QP | 7% | `services/portfolio.py`, `routers/frontera.py` |
| 7 ★ | Renta fija (NS, duración, convexidad) | 6% | `services/fixed_income.py`, `routers/curva.py`, `routers/bono.py` |
| 8 ★ | Black-Scholes + Greeks | 5% | `services/options.py`, `routers/opcion.py` |
| 9 ★ | Stress testing | 3% | `services/stress.py`, `routers/stress.py` |
| 10 ★ | Backend FastAPI + Pydantic + SQLAlchemy | 10% | toda la carpeta `backend/` |
| 11 ★★ | ML pipeline + Singleton + /predict | 7% | `app/ml/`, `routers/predict.py` |
| 12 ★ | Tests pytest + TestClient | 3% | `tests/` (37 tests) |
| 13 ★★ | Docker + deploy + CI | 6% | `Dockerfile`, `docker-compose.yml`, `render.yaml`, `.github/workflows/ci.yml` |
| 14 | Tablero frontend | 3% | `frontend/app.py` |
| 15 | Buenas prácticas | 5% | este README, `.gitignore`, `.env.example`, type hints, `@log_latency` |
| 16 | Sustentación oral | 15% | demo en vivo del deploy + `/docs` + `/predict` mostrando el Singleton |

Suma: 100%.

## Uso de herramientas de IA

Esta solución de referencia se construyó con apoyo de **Claude (Anthropic)** actuando como pair-programmer del docente. Cada decisión arquitectónica fue revisada por el docente. Específicamente la IA generó:

- El scaffolding inicial de carpetas y `Dockerfile`.
- Las primeras versiones de los servicios (indicadores, VaR, Black-Scholes, Nelson-Siegel).
- La suite de tests con fixtures de BD en memoria.

El docente validó todas las decisiones, ejecutó los tests, depuró un bug en `modified_duration`, y eligió el propósito analítico del modelo ML (clasificación binaria de dirección next-day).

Este uso de IA es coherente con la política descrita en `Proyecto_Integrador_Riesgo_Python_CIII.html`: la IA es asistente, no sustituto del aprendizaje; todo el código se puede defender en sustentación.

## Activos seleccionados

La selección es intencional: **cinco sectores no correlacionados** del S&P 500, todos con histórico largo, alta liquidez y reportes financieros sólidos. La diversificación sectorial garantiza que la frontera eficiente de Markowitz produzca portafolios con beneficios reales (covarianzas bajas entre pares como tech/energy o healthcare/financials), y que el stress testing muestre asimetrías interesantes entre activos defensivos (KO, JNJ) y procíclicos (JPM, XOM).

| Ticker | Compañía | Sector | Por qué |
|---|---|---|---|
| AAPL | Apple Inc. | Technology | Mega-cap tech, beta ~1.2, sensible al ciclo y a tasas; modelo arquetípico de growth stock. |
| JPM | JPMorgan Chase | Financials | Banca universal; beta ~1.1, sensible a la curva de tasas; complemento natural a renta fija. |
| XOM | Exxon Mobil | Energy | Procíclico, alta correlación con commodities; comportamiento contrario a tech en regímenes de inflación. |
| JNJ | Johnson & Johnson | Healthcare | Defensivo, beta ~0.6; estabiliza el portafolio en stress de mercado. |
| KO | The Coca-Cola Co. | Consumer Staples | Defensivo clásico, dividendos consistentes; reduce drawdowns en escenarios de recesión. |

Benchmark: **SPY** (S&P 500 ETF) — referencia para CAPM (Beta, Alpha de Jensen) y para Rf vía FRED DGS3MO.

## Decisiones de implementación

Resumen de elecciones que divergen de un read-literal de la spec; cada una documentada con justificación pedagógica. Detalle completo en [PLAN_AUDITORIA.md](PLAN_AUDITORIA.md).

| Decisión | Spec literal | Implementación | Justificación |
|---|---|---|---|
| **Markowitz QP** | <200 MB en Docker | Mantener `cvxpy` (~700 MB) | `cp.quad_form` es la formulación canónica que se enseña en clase; alternativa SLSQP perdería valor pedagógico. |
| **PredictionLog** | Campos `input_features`, `timestamp`, `actual` (nullable) | Esquema spec-compliant + endpoint `POST /predict/{log_id}/actual` | Habilita monitoreo de drift como bonificación accesible. |
| **Nube Markowitz** | "10,000 portafolios simulados" | Monte Carlo Dirichlet sobre el simplex (no-negativo) o Normal normalizado (con cortos) | Refleja "aleatorio" sin imponer una distribución artificial. |
| **CORS** | No mencionado | `Settings.cors_origins` CSV; default `["*"]` solo en dev | Sin CORS el frontend deployado en otro dominio no consume el backend. |
| **Stress** | 4 escenarios + combinado | 5 escenarios: `rate_shock` (+200 pb), `market_crash_20`, `market_crash_30`, `vol_spike` (σ×2), `combined` | La spec menciona "-20 % o -30 %"; exponemos ambos como escenarios independientes para que el estudiante pueda discutir el rango. |
| **GARCH** | "ARCH(1), GARCH(1,1), EGARCH/GJR" | Los 4 modelos con selección por AIC | Estricto matching con spec; permite ver el efecto leverage explícitamente. |
| **Nelson-Siegel** | `scipy.optimize.least_squares` | Migrado de `minimize(Nelder-Mead)` a `least_squares` con bounds (τ>0) | Cumple la spec textualmente; residuos vectoriales son numéricamente más estables. |

## Docente

**Javier Mauricio Sierra** — Universidad Santo Tomás · Pregrado en Estadística · 2026-I.
