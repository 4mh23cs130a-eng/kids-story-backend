from pydantic import BaseModel
from datetime import datetime
from typing import List

class ComicBase(BaseModel):
    title: str
    story_id: int

class ComicCreate(ComicBase):
    # Stored in DB as a JSON string, e.g. '["path1.png","path2.png",...]'
    image_paths: str

class ComicOut(ComicBase):
    id: int
    image_paths: str        # raw JSON string; client should json.loads() it to get List[str]
    created_at: datetime

    class Config:
        from_attributes = True
