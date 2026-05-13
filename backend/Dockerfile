# Multi-stage build.
# Decision documentada: se mantiene `cvxpy` por su valor pedagogico (cp.quad_form),
# aceptando que la imagen final excede el objetivo "<200 MB" de la spec.
# Mitigacion: purgamos tests/, docs y artefactos de cache para reducir lo posible.
# Ver seccion "Decisiones de implementacion" en README.md para justificacion.

FROM python:3.11.9-slim-bookworm AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /build

# Dependencias del sistema para compilar arch/cvxpy si hace falta.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --prefix=/install --no-cache-dir -r requirements.txt

# Purgar artefactos no necesarios en runtime (reduce ~30-50 MB).
RUN find /install -type d -name "tests" -exec rm -rf {} + 2>/dev/null || true && \
    find /install -type d -name "test" -exec rm -rf {} + 2>/dev/null || true && \
    find /install -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true && \
    find /install -type f -name "*.pyc" -delete 2>/dev/null || true && \
    find /install -type f -name "*.pyo" -delete 2>/dev/null || true && \
    find /install -type d -name "*.dist-info" -path "*/locale/*" -exec rm -rf {} + 2>/dev/null || true


# Stage 2: runtime - copia solo lo necesario.
FROM python:3.11.9-slim-bookworm AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000

WORKDIR /app

# Usuario no-root.
RUN useradd --create-home --shell /bin/bash appuser

COPY --from=builder /install /usr/local
COPY --chown=appuser:appuser app /app/app

USER appuser

EXPOSE 8000

# Render expone $PORT; uvicorn lo respeta.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
