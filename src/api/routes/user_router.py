from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from datetime import timedelta
from src.api.database.database import get_db
from src.api.models.user_models import User
from src.api.schemas.user_schemas import UserCreate, UserLogin
from src.api.utils.auth import hash_password, verify_password, create_access_token,get_current_user
from fastapi.security import OAuth2PasswordRequestForm 

router = APIRouter()



@router.post("/register")
def register(user: UserCreate, db: Session = Depends(get_db)):
    existing_user = db.query(User).filter(User.email == user.email).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="Email already registered")

    # bcrypt limit: 72 characters max
    if len(user.password) > 72:
        raise HTTPException(status_code=400, detail="Password must be less than 72 characters")

    # Trim the password just in case (optional)
    safe_password = user.password[:72]

    hashed_pw = hash_password(safe_password)
    new_user = User(name=user.name, email=user.email, password=hashed_pw)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return {"message": "User registered successfully"}


@router.post("/login")
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.email == form_data.username).first()
    if not db_user or not verify_password(form_data.password, db_user.password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    access_token_expires = timedelta(minutes=30)
    access_token = create_access_token(
        data={"sub": db_user.email}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}


@router.get("/me")
def get_me(current_user = Depends(get_current_user)):
    return {
        "name": current_user.name,
        "email": current_user.email,
        "id": current_user.id
    }