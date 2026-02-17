from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
from db import get_db
from schemas.story_schema import StoryCreate, StoryOut
import models

router = APIRouter(
    prefix="/stories",
    tags=["Stories"]
)

@router.post("/", response_model=StoryOut)
def create_story(story: StoryCreate, db: Session = Depends(get_db)):
    # Placeholder: Assuming user_id=1 for now
    new_story = models.Story(**story.dict(), user_id=1)
    db.add(new_story)
    db.commit()
    db.refresh(new_story)
    return new_story

@router.get("/", response_model=List[StoryOut])
def list_stories(db: Session = Depends(get_db)):
    return db.query(models.Story).all()
