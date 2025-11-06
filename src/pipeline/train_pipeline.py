from src.components.data_ingestion import DataIngestion 
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer 

from src.logger import logging 
from src.exception import CustomException 
import sys 


def train_pipeline():
    try:
        logging.info("Starting training pipeline")
        
        # data ingestion 
        ingestion = DataIngestion()
        train_data_path , test_data_path = ingestion.initiate_data_ingestion()
        
        # Data transformation 
        
        transformation = DataTransformation()
        train_arr, test_arr, _ = transformation.initiate_data_transformation(train_data_path,test_data_path)
        
        # Model training
        
        trainer = ModelTrainer()
        r2 = trainer.initiate_model_trainer(train_arr, test_arr)
        
        print(f"Trainer complete ✅ Best model: {r2['best_model_name']} | R2 Score: {r2['r2_score']:.4f}")

        # print(f"Trainer complete Best model R2 score: {r2:.4f}")
        logging.info(f"Training pipeline completed successfully with R2 = {r2}")
        
    except Exception as e:
        raise CustomException(e,sys) 
    
if __name__ == "__main__":
    train_pipeline()
            
            