"""Endpoint /predict: modelo ML (Singleton)."""
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.database import get_db
from app.ml.features import build_features
from app.ml.predictor import Predictor, get_predictor
from app.models.db_models import PredictionLog
from app.models.schemas import PredictOut, PredictRequest
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

    ts = datetime.utcnow()
    log = PredictionLog(
        ticker=req.ticker.upper(),
        features=X.tail(1).iloc[0].to_dict(),
        prediction=int(pred),
        probability=float(proba),
        model_version=predictor.model_version,
        ts=ts,
    )
    db.add(log)
    db.commit()

    return PredictOut(
        ticker=req.ticker.upper(),
        prediction=int(pred),
        probability=float(proba),
        model_version=predictor.model_version,
        features_used=predictor.features,
        ts=ts,
    )
