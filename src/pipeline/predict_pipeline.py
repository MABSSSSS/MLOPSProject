import os, sys, dill
import pandas as pd 

from src.exception import CustomException 

class PredictPipeline:
    
    def __init__(self):
        self.model_path = os.path.join("artifacts","catboost_model.pkl")
        self.preprocessor_path = os.path.join("artifacts","preprocessor.pkl")
        
    def load_object(self,path):
        with open(path, "rb") as f:
            return dill.load(f)
        
    def predict(self, features: pd.DataFrame):
        try:
            model = self.load_object(self.model_path)
            preprocessor = self.load_object(self.preprocessor_path)
            
            transformed = preprocessor.transform(features)
            preds = model.predict(transformed)
            
            
            return preds 
        
        except Exception as e:
            raise CustomException(e,sys)
        

if __name__ == "__main__":
    
    df = pd.read_csv("C:/MLOPS/notebook/data/test.csv")
    
    pipeline = PredictPipeline()
    preds = pipeline.predict(df) 
    
    print("Predictions")
    print(preds)
    
    pd.DataFrame(preds, columns=["Predicted_SalePrice"]).to_csv("artifacts/predictions.csv",index=False)
    print("Saved successfully")