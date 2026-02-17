from fastapi import FastAPI
import models
from db import engine
from routes import auth_routes, story_routes, comic_routes, video_routes

# Create database tables
models.Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Kids Story Backend",
    description="Backend API for AI-powered kids stories, comics, and videos.",
    version="1.0.0"
)

# Include routers
app.include_router(auth_routes.router)
app.include_router(story_routes.router)
app.include_router(comic_routes.router)
app.include_router(video_routes.router)

@app.get("/")
def read_root():
    return {"message": "Welcome to Kids Story Backend! Visit /docs for API documentation."}
