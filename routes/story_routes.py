from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
from db import get_db
from schemas.story_schema import StoryCreate, StoryOut
from repositories import story_repository

router = APIRouter(
    prefix="/stories",
    tags=["Stories"]
)

@router.post("/", response_model=StoryOut)
def create_story(story: StoryCreate, db: Session = Depends(get_db)):
    # Placeholder: Assuming user_id=1 for now
    return story_repository.create_story(db=db, story=story, user_id=1)

@router.get("/", response_model=List[StoryOut])
def list_stories(db: Session = Depends(get_db)):
    return story_repository.get_stories(db)
