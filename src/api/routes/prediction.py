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
    house: HouseData,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    try:
        # Convert input to DataFrame
        data = house.dict()
        df = pd.DataFrame([data])
        df = df.rename(columns={'FirstFlrSF': '1stFlrSF'})  # match model features

        # -------------------------------
        # Debug print: Check input before prediction
        # -------------------------------
        print("Input DataFrame for prediction:", df.to_dict(orient='records'))

        # Make prediction
        preds = pipeline.predict(df)

        # If pipeline returned an error dict, raise HTTPException
        if isinstance(preds, dict) and "error" in preds:
            raise HTTPException(status_code=500, detail=f"Prediction pipeline error: {preds['error']}")

        # Convert prediction to float
        predicted_price = float(preds[0])

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

        # Save record in DB
        db.add(record)
        db.commit()
        db.refresh(record)

        # Link prediction to current user
        current_user.last_prediction_id = record.id
        db.commit()

        # Return response
        return {
            "predicted_price": predicted_price,
            "user_id": current_user.id,
            "last_prediction_id": record.id
        }

    except HTTPException:
        # Re-raise known HTTPExceptions
        raise
    except Exception as e:
        # Catch-all with debug info
        import traceback
        tb = traceback.format_exc()
        print("Prediction endpoint error:", tb)
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
