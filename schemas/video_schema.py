from pydantic import BaseModel
from datetime import datetime
from typing import Optional

class VideoBase(BaseModel):
    title: str
    story_id: int

class VideoCreate(VideoBase):
    video_path: str

class VideoOut(VideoBase):
    id: int
    video_path: str
    created_at: datetime

    class Config:
        from_attributes = True
