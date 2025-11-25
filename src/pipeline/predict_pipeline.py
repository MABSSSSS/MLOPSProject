import os, sys, dill
import pandas as pd
import numpy as np
from sqlalchemy.orm import Session
from datetime import datetime

from src.exception import CustomException
from src.api.models.prediction_models import HousePrediction
from src.api.database.database import get_db
from src.api.utils.helper import convert_numpy_types


class PredictPipeline:
    def __init__(self):
        self.model_path = os.path.join("artifacts", "catboost_model.pkl")
        self.preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")

    def load_object(self, path):
        try:
            with open(path, "rb") as f:
                return dill.load(f)
        except Exception as e:
            raise CustomException(f"Error loading object from {path}: {e}", sys)

    def prepare_df(self, data: dict):
        return pd.DataFrame([data])

    def predict(self, features: pd.DataFrame, db: Session = None):
        try:
            print("📦 Loading model and preprocessor...")
            model = self.load_object(self.model_path)
            preprocessor = self.load_object(self.preprocessor_path)

            expected_cols = list(preprocessor.feature_names_in_)
            print("Expected columns:",expected_cols)
            print("Inout Features before aligning",features.dtypes)
            
            for col in expected_cols:
                if col not in features.columns:
                    features[col] = 0
            features = features[expected_cols]
            print("Features after aligning:", features.dtypes)
            

            transformed = preprocessor.transform(features)
            preds = model.predict(transformed)

            if db:
                for i, row in features.iterrows():
                    clean_row = convert_numpy_types(row.to_dict())
                    record = HousePrediction(
                        **clean_row,
                        predicted_price=float(preds[i])
                    )
                    db.add(record)
                db.commit()
                

            return preds
        except Exception as e:
            # raise CustomException(e, sys)
            return {'error': str(e)}
