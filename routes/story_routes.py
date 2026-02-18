from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from typing import List

from db import get_db
from schemas.story_schema import StoryCreate, StoryOut, StoryGenerate
from repositories import story_repository
from utils.ai_utils import generate_story

router = APIRouter(
    prefix="/stories",
    tags=["Stories"]
)

# CREATE STORY USING AI + SAVE IN DATABASE
@router.post("/generate", response_model=StoryOut)
def create_story(story_data: StoryGenerate, db: Session = Depends(get_db)):
    
    # 1️⃣ Generate story using AI
    ai_story_text = generate_story(story_data.prompt)

    
    # 2️⃣ Save generated story into database
    saved_story = story_repository.create_story(
        db=db,
        title=story_data.title,
        content=ai_story_text,
        user_id=1   # temporary user
    )

    return saved_story


# GET ALL STORIES FROM DATABASE
@router.get("/", response_model=List[StoryOut])
def list_stories(db: Session = Depends(get_db)):
    return story_repository.get_stories(db)

