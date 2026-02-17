from sqlalchemy.orm import Session
import models
from schemas.video_schema import VideoCreate

def create_video(db: Session, video: VideoCreate):
    db_video = models.Video(**video.dict())
    db.add(db_video)
    db.commit()
    db.refresh(db_video)
    return db_video

def get_videos(db: Session, skip: int = 0, limit: int = 100):
    return db.query(models.Video).offset(skip).limit(limit).all()
