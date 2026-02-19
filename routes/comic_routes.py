from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
from db import get_db
from schemas.comic_schema import ComicCreate, ComicOut
from repositories import comic_repository, story_repository
from utils.image_gen import generate_full_comic_page
import json

router = APIRouter(
    prefix="/comics",
    tags=["Comics"]
)


@router.post("/generate/{story_id}", response_model=ComicOut)
def generate_comic(story_id: int, db: Session = Depends(get_db)):
    """
    Generate a full comic page (4-panel 2x2 grid) from a story.
    Uses AI images per scene, composed into one single PNG.
    """
    # 1. Fetch the story
    story = story_repository.get_story(db=db, story_id=story_id)
    if not story:
        raise HTTPException(status_code=404, detail=f"Story with id {story_id} not found.")

    # 2. Generate full comic page (4 panels)
    try:
        page_path = generate_full_comic_page(
            story_id=story_id,
            story_content=story.content,
            story_title=story.title or "",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Comic generation failed: {str(e)}")

    # 3. Save comic record
    comic_data = ComicCreate(
        title=f"Comic: {story.title}",
        story_id=story_id,
        image_paths=page_path,
    )
    comic = comic_repository.create_comic(db=db, comic=comic_data)
    return comic



@router.post("/", response_model=ComicOut)
def create_comic(comic: ComicCreate, db: Session = Depends(get_db)):
    return comic_repository.create_comic(db=db, comic=comic)


@router.get("/", response_model=List[ComicOut])
def list_comics(db: Session = Depends(get_db)):
    return comic_repository.get_comics(db)


@router.get("/{story_id}", response_model=ComicOut)
def get_comic_by_story(story_id: int, db: Session = Depends(get_db)):
    comic = comic_repository.get_comic_by_story_id(db=db, story_id=story_id)
    if not comic:
        raise HTTPException(status_code=404, detail="No comic found for this story.")
    return comic
