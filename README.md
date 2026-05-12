# Proyecto Integrador — Teoría del Riesgo (solución de referencia)

> Implementación de referencia del proyecto integrador del curso *Python para Desarrollo de APIs e Inteligencia Artificial* (Universidad Santo Tomás · pregrado en Estadística · 2026-I).

Sistema completo de análisis de riesgo financiero en cinco capas. Sirve como código que los estudiantes leen mientras construyen su propio repositorio desde cero.

## Resumen ejecutivo

API FastAPI con persistencia SQLite, modelo de machine learning servido vía Singleton, contenerizada con Docker multi-stage y desplegada en Render free-tier. Incluye un tablero Streamlit con 12 pestañas que consumen el backend.

**Stack:** Python 3.11.9 · FastAPI · Pydantic v2 · SQLAlchemy 2 · cvxpy · arch · scikit-learn · pytest · Docker · GitHub Actions · Streamlit.

## Arquitectura en 5 capas

1. **Datos y persistencia** — ingesta desde yfinance + FRED, persistencia en SQLite vía SQLAlchemy ORM, cache transparente (TTL configurable).
2. **Análisis clásico** — indicadores técnicos, rendimientos, volatilidad EWMA + 3 modelos GARCH (incluido EGARCH/GJR), CAPM, VaR (3 métodos) + CVaR + backtesting de Kupiec, Markowitz QP con y sin restricción de no-negatividad.
3. **Renta fija y derivados** — curva FRED + ajuste Nelson-Siegel, duración Macaulay/modificada, convexidad, Black-Scholes europeo + 5 Greeks, paridad put-call, stress testing con 4 escenarios.
4. **Machine Learning** — pipeline `train.py → joblib → load → predict` con patrón Singleton vía lifespan de FastAPI. Cada predicción se persiste en `PredictionLog`. Propósito: clasificación binaria de dirección next-day.
5. **Infraestructura** — pytest + TestClient (37 tests), Docker multi-stage, deploy en Render, GitHub Actions CI con `pytest` + build de imagen.

## Endpoints (16+ rutas)

```text
GET    /health
GET    /activos
GET    /precios/{ticker}
GET    /rendimientos/{ticker}
GET    /indicadores/{ticker}
GET    /volatilidad/{ticker}     ?ewma_lambda=0.94
POST   /var
GET    /capm
POST   /frontera-eficiente
GET    /alertas
GET    /macro
GET    /curva-rendimiento
POST   /bono/duracion
POST   /opcion/precio
POST   /stress
POST   /predict
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

**37 tests** cubriendo: indicadores, VaR + Kupiec, Black-Scholes + paridad, bono + duración, frontera eficiente, CRUD portafolios, Singleton del ML, endpoints con TestClient.

## Ejecutar con Docker

```bash
docker compose up --build
# → http://localhost:8000/health
```

El compose levanta el backend con volumen persistente para la BD. La imagen se construye en multi-stage sobre `python:3.11.9-slim-bookworm`.

**Nota sobre el tamaño:** la spec sugiere < 200 MB; con `cvxpy + arch + scipy + statsmodels` la imagen final ronda los 700 MB. Es un tradeoff documentado entre rigor matemático (cvxpy resuelve QP exacto, arch ajusta GARCH propiamente) y peso de la imagen. Para una versión más ligera se puede sustituir cvxpy por una implementación manual con scipy.

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
- `PredictionLog(id, ticker, features JSON, prediction, probability, model_version, ts)`.

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

Los 5 activos seed cubren sectores diversos:

| Ticker | Compañía | Sector |
|---|---|---|
| AAPL | Apple Inc. | Technology |
| JPM | JPMorgan Chase & Co. | Financials |
| XOM | Exxon Mobil Corp. | Energy |
| JNJ | Johnson & Johnson | Healthcare |
| KO | The Coca-Cola Company | Consumer Staples |

Benchmark: **SPY** (S&P 500 ETF).

## Docente

**Javier Mauricio Sierra** — Universidad Santo Tomás · Pregrado en Estadística · 2026-I.
