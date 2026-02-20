from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import models
from db import engine
from routes import auth_routes, story_routes, comic_routes, video_routes
from pathlib import Path

# Ensure output directories exist
Path("generated_comics").mkdir(exist_ok=True)
Path("generated_videos").mkdir(exist_ok=True)

# Create database tables
models.Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Kids Story Backend",
    description="Backend API for AI-powered kids stories, comics, and videos.",
    version="1.0.0"
)

# ── CORS — allow the Vite frontend during development ──────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],     # Allow all origins for development
    allow_credentials=False, # Must be False when allow_origins=["*"]
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth_routes.router)
app.include_router(story_routes.router)
app.include_router(comic_routes.router)
app.include_router(video_routes.router)

# Serve generated files as static URLs
app.mount("/static/comics", StaticFiles(directory="generated_comics"), name="comics")
app.mount("/static/videos", StaticFiles(directory="generated_videos"), name="videos")

@app.get("/")
def read_root():
    return {"message": "Welcome to Kids Story Backend! Visit /docs for API documentation."}

