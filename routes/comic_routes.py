from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
from db import get_db
from schemas.comic_schema import ComicCreate, ComicOut
from repositories import comic_repository

router = APIRouter(
    prefix="/comics",
    tags=["Comics"]
)

@router.post("/", response_model=ComicOut)
def create_comic(comic: ComicCreate, db: Session = Depends(get_db)):
    return comic_repository.create_comic(db=db, comic=comic)

@router.get("/", response_model=List[ComicOut])
def list_comics(db: Session = Depends(get_db)):
    return comic_repository.get_comics(db)
