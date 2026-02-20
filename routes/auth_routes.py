from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from db import get_db
from schemas.user_schema import UserCreate, UserOut, UserLogin
from repositories import user_repository
from utils import auth_utils

router = APIRouter(
    prefix="/auth",
    tags=["Authentication"]
)

@router.post("/register", response_model=UserOut)
def register(user: UserCreate, db: Session = Depends(get_db)):
    db_user = user_repository.get_user_by_email(db, email=user.email)
    if db_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    hashed_password = auth_utils.get_password_hash(user.password)
    return user_repository.create_user(db=db, user=user, hashed_password=hashed_password)

@router.post("/login")
def login(credentials: UserLogin, db: Session = Depends(get_db)):
    """Login with email + password only (no username required)."""
    db_user = user_repository.get_user_by_email(db, email=credentials.email)
    if not db_user or not auth_utils.verify_password(credentials.password, db_user.hashed_password):
        raise HTTPException(status_code=400, detail="Invalid email or password")
    return {
        "message": "Login successful",
        "user_id":  db_user.id,
        "username": db_user.username,
        "email":    db_user.email,
    }
