from pydantic import BaseModel
from datetime import datetime
from typing import Optional

class StoryBase(BaseModel):
    title: str
    content: str

class StoryCreate(StoryBase):
    pass

class StoryGenerate(BaseModel):
    title: str
    prompt: str

class StoryOut(StoryBase):
    id: int
    user_id: int
    created_at: datetime

    class Config:
        from_attributes = True

