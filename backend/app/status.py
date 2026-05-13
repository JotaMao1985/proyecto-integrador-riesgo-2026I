"""Estado modulo-level del backend (observabilidad cross-modulo).

Centralizado aqui para evitar ciclos de import entre `main.py` y `routers/health.py`.
"""
from __future__ import annotations

# Estados validos del bootstrap del historico de precios (T1.6):
# - "pending":  bootstrap aun no se lanzo (o esta deshabilitado)
# - "running":  background task en ejecucion
# - "complete": termino con al menos un ticker exitoso
# - "failed":   todos los tickers fallaron o hubo excepcion no recuperable
BOOTSTRAP_STATE: dict[str, object] = {"state": "pending"}
