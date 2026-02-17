from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from db import get_db
from schemas.user_schema import UserCreate, UserOut
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
def login(user: UserCreate, db: Session = Depends(get_db)):
    db_user = user_repository.get_user_by_email(db, email=user.email)
    if not db_user or not auth_utils.verify_password(user.password, db_user.hashed_password):
        raise HTTPException(status_code=400, detail="Invalid credentials")
    return {"message": "Login successful", "user_id": db_user.id}
