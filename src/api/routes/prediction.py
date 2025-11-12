from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
import pandas as pd
from src.api.schemas.dataschema import HouseData
from src.api.database.database import SessionLocal
from src.api.models.prediction_models import HousePrediction
from src.api.models.user_models import User
from src.pipeline.predict_pipeline import PredictPipeline
from src.api.utils.helper import convert_numpy_types
from src.api.utils.auth import get_current_user

router = APIRouter()
pipeline = PredictPipeline()

# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.get("/")
def root():
    return {"message": "Welcome to House Prediction API"}

@router.post("/predict")
def predict_house(
    house: HouseData, 
    db: Session = Depends(get_db), 
    current_user: User = Depends(get_current_user)
):
    try:
        # Convert input to DataFrame
        data = house.dict()
        df = pd.DataFrame([data])
        df = df.rename(columns={'FirstFlrSF': '1stFlrSF'})
        
        # Make prediction
        preds = pipeline.predict(df)
        predicted_price = float(preds[0])
        
        # Prepare record for DB
        df = df.rename(columns={'1stFlrSF': 'FirstFlrSF'})
        row_data = convert_numpy_types(df.iloc[0].to_dict())
        
        # Create prediction record with user relation
        record = HousePrediction(
            **row_data, 
            predicted_price=predicted_price, 
            
        )
        db.add(record)
        db.commit()
        db.refresh(record)
        
        current_user.last_prediction_id = record.id 
        db.commit()
        return {"predicted_price": predicted_price, "user_id": current_user.id,"last_prediction_id":record.id}
        
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
