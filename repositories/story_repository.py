from sqlalchemy.orm import Session
import models
from schemas.story_schema import StoryCreate

def create_story(db: Session, title: str, content: str, user_id: int):
    db_story = models.Story(title=title, content=content, user_id=user_id)
    db.add(db_story)
    db.commit()
    db.refresh(db_story)
    return db_story

def get_stories(db: Session, skip: int = 0, limit: int = 100):
    return db.query(models.Story).offset(skip).limit(limit).all()

def get_story(db: Session, story_id: int):
    return db.query(models.Story).filter(models.Story.id == story_id).first()
