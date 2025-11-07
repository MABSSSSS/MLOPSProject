import os, sys, mlflow, pandas as pd
from src.exception import CustomException

class PredictPipeline:
    def __init__(self):
        # DagsHub repo info
        self.repo_owner = "happinesswhat31"
        self.repo_name = "MLOPSProject"
        self.run_id = "f496b708fe1f436b87bc7f281ec54fe5"
        
    def load_model_from_dagshub(self):
        try:
            print("📦 Loading model from DagsHub MLflow...")
            mlflow.set_tracking_uri(f"https://dagshub.com/{self.repo_owner}/{self.repo_name}.mlflow")
            
            # Enable DagsHub authentication (optional if public)
            os.environ["MLFLOW_TRACKING_USERNAME"] = "happinesswhat31"
            os.environ["MLFLOW_TRACKING_PASSWORD"] = "46a2b7c3aa44a97964edfe6ee84c0e0c9cb094f3"

            model_uri = f"runs:/{self.run_id}/model"
            model = mlflow.pyfunc.load_model(model_uri)
            print("✅ Model loaded successfully from DagsHub MLflow!")
            return model
        except Exception as e:
            raise CustomException(f"Error loading from DagsHub: {e}", sys)
    
    def predict(self, features: pd.DataFrame):
        try:
            model = self.load_model_from_dagshub()
            preds = model.predict(features)
            return preds
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    try:
        print("🚀 Starting prediction pipeline...")
        df = pd.read_csv("C:/MLOPS/notebook/data/test.csv")
        pipeline = PredictPipeline()
        preds = pipeline.predict(df)
        print("✅ Predictions generated successfully!")
        pd.DataFrame(preds, columns=["Predicted_SalePrice"]).to_csv("artifacts/predictions.csv", index=False)
        print("💾 Predictions saved successfully!")
    except Exception as e:
        print(f"❌ Error in prediction pipeline {e}")
