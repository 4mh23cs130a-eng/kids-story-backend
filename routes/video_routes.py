from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
from db import get_db
from schemas.video_schema import VideoCreate, VideoOut
from repositories import video_repository, comic_repository, story_repository
from utils.video_gen import generate_video

router = APIRouter(
    prefix="/videos",
    tags=["Videos"]
)


@router.post("/generate/{story_id}", response_model=VideoOut)
def generate_video_for_story(story_id: int, db: Session = Depends(get_db)):
    """
    Generate a slideshow MP4 video from the comic images linked to a story.
    Requires the comic to have been generated first via POST /comics/generate/{story_id}.
    """
    # 1. Fetch the story (for title)
    story = story_repository.get_story(db=db, story_id=story_id)
    if not story:
        raise HTTPException(status_code=404, detail=f"Story with id {story_id} not found.")

    # 2. Fetch the comic linked to this story
    comic = comic_repository.get_comic_by_story_id(db=db, story_id=story_id)
    if not comic:
        raise HTTPException(
            status_code=404,
            detail=f"No comic found for story {story_id}. Generate the comic first via POST /comics/generate/{story_id}."
        )

    # 3. Parse comma-separated image paths
    image_paths = [p.strip() for p in comic.image_paths.split(",") if p.strip()]
    if not image_paths:
        raise HTTPException(status_code=400, detail="Comic has no image paths.")

    # 4. Generate the video
    try:
        video_path = generate_video(
            story_id=story_id,
            image_paths=image_paths,
            story_title=story.title,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Video generation failed: {str(e)}")

    # 5. Save video record to DB
    video_data = VideoCreate(
        title=f"Video: {story.title}",
        story_id=story_id,
        video_path=video_path,
    )
    video = video_repository.create_video(db=db, video=video_data)
    return video


@router.post("/", response_model=VideoOut)
def create_video(video: VideoCreate, db: Session = Depends(get_db)):
    return video_repository.create_video(db=db, video=video)


@router.get("/", response_model=List[VideoOut])
def list_videos(db: Session = Depends(get_db)):
    return video_repository.get_videos(db)


@router.get("/{story_id}", response_model=VideoOut)
def get_video_by_story(story_id: int, db: Session = Depends(get_db)):
    video = db.query(__import__('models').Video).filter(
        __import__('models').Video.story_id == story_id
    ).first()
    if not video:
        raise HTTPException(status_code=404, detail="No video found for this story.")
    return video
