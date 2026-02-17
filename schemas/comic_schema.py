from pydantic import BaseModel
from datetime import datetime
from typing import Optional, List

class ComicBase(BaseModel):
    title: str
    story_id: int

class ComicCreate(ComicBase):
    image_paths: str # Stored as a comma-separated string for now

class ComicOut(ComicBase):
    id: int
    image_paths: str
    created_at: datetime

    class Config:
        from_attributes = True
