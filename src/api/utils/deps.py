from fastapi import Depends,HTTPException 

from fastapi.security import OAuth2PasswordBearer 
from jose import jwt,JWTError 
from src.api.utils.auth import SECRET_KEY, ALGORITHM 
from sqlalchemy.orm import Session 
from src.api.database.database import SessionLocal 
from src.api.models.user_models import User 


oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/users/login")


def get_db():
    db = SessionLocal()
    try: 
        yield db 
    finally:
        db.close()
        

def get_current_user(token:str = Depends(oauth2_scheme),db: Session = Depends(get_db)):
    
    credentials_exception = HTTPException(
        status_code = 401, 
        detail = "Could not validate credentials",
        headers = {"WWW-Authenticate": "Bearer"},
    )
    
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms = [ALGORITHM])
        user_id = payload.get("user_id")
        if user_id is None:
            raise credentials_exception 
    
    except JWTError:
        raise credentials_exception 
    
    
    user = db.query(User).filter(User.id == user_id).first()
    
    if user is None:
        raise credentials_exception 
    return user 