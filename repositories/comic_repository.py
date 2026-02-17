from sqlalchemy.orm import Session
import models
from schemas.comic_schema import ComicCreate

def create_comic(db: Session, comic: ComicCreate):
    db_comic = models.Comic(**comic.dict())
    db.add(db_comic)
    db.commit()
    db.refresh(db_comic)
    return db_comic

def get_comics(db: Session, skip: int = 0, limit: int = 100):
    return db.query(models.Comic).offset(skip).limit(limit).all()
