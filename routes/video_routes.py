from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
from db import get_db
from schemas.video_schema import VideoCreate, VideoOut
from repositories import video_repository

router = APIRouter(
    prefix="/videos",
    tags=["Videos"]
)

@router.post("/", response_model=VideoOut)
def create_video(video: VideoCreate, db: Session = Depends(get_db)):
    return video_repository.create_video(db=db, video=video)

@router.get("/", response_model=List[VideoOut])
def list_videos(db: Session = Depends(get_db)):
    return video_repository.get_videos(db)
