"""Endpoint /activos."""
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.schemas import AssetOut
from app.services.data import list_assets

router = APIRouter(tags=["datos"])


@router.get("/activos", response_model=list[AssetOut], summary="Lista de activos seed")
async def get_activos(db: Session = Depends(get_db)) -> list[AssetOut]:
    return [AssetOut.model_validate(a) for a in list_assets(db)]
