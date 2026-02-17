from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
from db import get_db
from schemas.comic_schema import ComicCreate, ComicOut
import models

router = APIRouter(
    prefix="/comics",
    tags=["Comics"]
)

@router.post("/", response_model=ComicOut)
def create_comic(comic: ComicCreate, db: Session = Depends(get_db)):
    new_comic = models.Comic(**comic.dict())
    db.add(new_comic)
    db.commit()
    db.refresh(new_comic)
    return new_comic

@router.get("/", response_model=List[ComicOut])
def list_comics(db: Session = Depends(get_db)):
    return db.query(models.Comic).all()
