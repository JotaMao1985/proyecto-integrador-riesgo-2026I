"""CRUD basico de portafolios (criterio 10)."""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.db_models import Portfolio
from app.models.schemas import PortfolioCreate, PortfolioOut

router = APIRouter(prefix="/portafolios", tags=["portafolios"])


@router.post("", response_model=PortfolioOut, status_code=201)
async def crear(payload: PortfolioCreate, db: Session = Depends(get_db)) -> PortfolioOut:
    p = Portfolio(name=payload.name, holdings=payload.holdings)
    db.add(p)
    db.commit()
    db.refresh(p)
    return PortfolioOut.model_validate(p)


@router.get("", response_model=list[PortfolioOut])
async def listar(db: Session = Depends(get_db)) -> list[PortfolioOut]:
    rows = list(db.scalars(select(Portfolio).order_by(Portfolio.id.desc())))
    return [PortfolioOut.model_validate(r) for r in rows]


@router.get("/{pid}", response_model=PortfolioOut)
async def obtener(pid: int, db: Session = Depends(get_db)) -> PortfolioOut:
    p = db.get(Portfolio, pid)
    if not p:
        raise HTTPException(status_code=404, detail="Portfolio no encontrado")
    return PortfolioOut.model_validate(p)


@router.delete("/{pid}", status_code=204)
async def borrar(pid: int, db: Session = Depends(get_db)) -> None:
    p = db.get(Portfolio, pid)
    if not p:
        raise HTTPException(status_code=404, detail="Portfolio no encontrado")
    db.delete(p)
    db.commit()
