import json
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
from db import get_db
from schemas.comic_schema import ComicCreate, ComicOut
from repositories import comic_repository, story_repository
from utils.image_gen import generate_comic_panels

router = APIRouter(
    prefix="/comics",
    tags=["Comics"]
)


@router.post("/generate/{story_id}", response_model=ComicOut)
def generate_comic(story_id: int, db: Session = Depends(get_db)):
    """
    Generate 4 individual comic images from a story — one image per scene.
    Each image contains:
      - AI-generated illustration for that scene (640x512 px)
      - Full scene text as a caption below the illustration
    The 4 file paths are stored as a JSON array in image_paths.
    """
    # 1. Fetch the story
    story = story_repository.get_story(db=db, story_id=story_id)
    if not story:
        raise HTTPException(status_code=404, detail=f"Story with id {story_id} not found.")

    # 2. Generate 4 separate comic images
    try:
        paths: list = generate_comic_panels(
            story_id=story_id,
            story_content=story.content,
            story_title=story.title or "",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Comic generation failed: {str(e)}")

    # 3. Save comic record  (paths stored as JSON string)
    comic_data = ComicCreate(
        title=f"Comic: {story.title}",
        story_id=story_id,
        image_paths=json.dumps(paths),   # e.g. '["generated_comics/story_1_scene_1.png", ...]'
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
