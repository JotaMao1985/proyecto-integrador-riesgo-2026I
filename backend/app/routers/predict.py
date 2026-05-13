"""Endpoint /predict: modelo ML (Singleton) + back-fill de actual."""
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.database import get_db
from app.ml.features import build_features
from app.ml.predictor import Predictor, get_predictor
from app.models.db_models import PredictionLog
from app.models.schemas import ActualUpdate, PredictOut, PredictRequest
from app.services.data import get_prices

router = APIRouter(tags=["ml"])


@router.post("/predict", response_model=PredictOut, status_code=status.HTTP_200_OK)
async def predict(
    req: PredictRequest,
    db: Session = Depends(get_db),
    predictor: Predictor = Depends(get_predictor),
) -> PredictOut:
    df = get_prices(db, req.ticker)
    if df.empty:
        raise HTTPException(status_code=404, detail=f"Sin datos para {req.ticker}")
    close = df["close"].tail(req.lookback_days)
    if len(close) < 60:
        raise HTTPException(status_code=400, detail="Lookback insuficiente")

    X = build_features(close).dropna()
    if X.empty:
        raise HTTPException(status_code=400, detail="Features no calculables")

    pred, proba = predictor.predict(X.tail(1)[predictor.features])

    ts = datetime.now(timezone.utc)
    log = PredictionLog(
        ticker=req.ticker.upper(),
        input_features=X.tail(1).iloc[0].to_dict(),
        prediction=int(pred),
        probability=float(proba),
        model_version=predictor.model_version,
        timestamp=ts,
    )
    db.add(log)
    db.commit()
    db.refresh(log)

    return PredictOut(
        log_id=log.id,
        ticker=req.ticker.upper(),
        prediction=int(pred),
        probability=float(proba),
        model_version=predictor.model_version,
        features_used=predictor.features,
        timestamp=ts,
    )


@router.post("/predict/{log_id}/actual", status_code=status.HTTP_200_OK)
async def set_actual(
    log_id: int, body: ActualUpdate, db: Session = Depends(get_db)
) -> dict:
    """Back-fill del valor real observado. Habilita monitoreo de drift."""
    log = db.get(PredictionLog, log_id)
    if log is None:
        raise HTTPException(status_code=404, detail=f"PredictionLog id={log_id} no existe")
    log.actual = body.actual
    db.commit()
    return {"log_id": log_id, "actual": body.actual}
