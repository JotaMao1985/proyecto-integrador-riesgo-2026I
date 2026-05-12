# Multi-stage build para mantener la imagen final < 200 MB.
# Stage 1: builder — instala dependencias en /install.
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


# Stage 2: runtime — copia solo lo necesario.
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
