from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
from db import get_db
from schemas.video_schema import VideoCreate, VideoOut
import models

router = APIRouter(
    prefix="/videos",
    tags=["Videos"]
)

@router.post("/", response_model=VideoOut)
def create_video(video: VideoCreate, db: Session = Depends(get_db)):
    new_video = models.Video(**video.dict())
    db.add(new_video)
    db.commit()
    db.refresh(new_video)
    return new_video

@router.get("/", response_model=List[VideoOut])
def list_videos(db: Session = Depends(get_db)):
    return db.query(models.Video).all()
