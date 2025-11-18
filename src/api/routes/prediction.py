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

# -------------------------------
# Router & Pipeline
# -------------------------------
router = APIRouter()
pipeline = PredictPipeline()  # ML model pipeline instance

# -------------------------------
# DB Dependency for FastAPI
# -------------------------------
def get_db():
    """
    Provide a database session per request and close it after.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# -------------------------------
# Root Route
# -------------------------------
@router.get("/")
def root():
    return {"message": "Welcome to House Prediction API"}

# -------------------------------
# Prediction Route
# -------------------------------
@router.post("/predict")
def predict_house(
    house: HouseData,                # Validated input data via Pydantic
    db: Session = Depends(get_db),   # Database session injected
    current_user: User = Depends(get_current_user)  # Authenticated user
):
    try:
        # -------------------------------
        # Convert input to DataFrame for ML model
        # -------------------------------
        data = house.dict()
        df = pd.DataFrame([data])
        df = df.rename(columns={'FirstFlrSF': '1stFlrSF'})  # match model features

        # -------------------------------
        # Make prediction
        # -------------------------------
        preds = pipeline.predict(df)
        predicted_price = float(preds[0])  # Convert numpy to native float

        # -------------------------------
        # Prepare data for DB
        # -------------------------------
        df = df.rename(columns={'1stFlrSF': 'FirstFlrSF'})  # match DB column names
        row_data = convert_numpy_types(df.iloc[0].to_dict())

        # Create prediction record
        record = HousePrediction(
            **row_data,
            predicted_price=predicted_price,
        )

        # -------------------------------
        # Save record in DB
        # -------------------------------
        db.add(record)
        db.commit()
        db.refresh(record)

        # Link prediction to current user
        current_user.last_prediction_id = record.id
        db.commit()

        # -------------------------------
        # Return response
        # -------------------------------
        return {
            "predicted_price": predicted_price,
            "user_id": current_user.id,
            "last_prediction_id": record.id
        }

    except Exception as e:
        # Catch any error and return as HTTP 500
        raise HTTPException(status_code=500, detail=str(e))
