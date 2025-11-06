import sys 
from dataclasses import dataclass 

import numpy as np 

import pandas as pd

from sklearn.compose import ColumnTransformer 
from sklearn.impute import SimpleImputer 
from sklearn.pipeline import Pipeline 
from sklearn.preprocessing import OneHotEncoder, StandardScaler 

from src.exception import CustomException 
from src.logger import logging 
import os 

from src.utils import save_object 

@dataclass 
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts',"preprocessor.pkl")
    

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        
    def get_transformer_object(self):
        '''
        This function is responsible for data transformation 
        '''
        
        try:
            numerical_columns = ['GrLivArea', 'OverallQual', 'GarageCars', 'YearBuilt','TotalBsmtSF','FullBath','1stFlrSF']
            categorical_columns = [
                'MSZoning',
                'Exterior2nd',
                'Exterior1st',
                'BsmtQual',
                'Foundation',
                'ExterQual',
                'HouseStyle'
                ]
            
            num_pipeline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="median")),
                    ("scaler",StandardScaler())
                    
                ]
            )
                
            cat_pipeline = Pipeline(
                
                steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder",OneHotEncoder(handle_unknown='ignore',drop='first',sparse_output=False)),
                    ("scaler", StandardScaler(with_mean=False))
                ]
                
                
            )
            
            logging.info(f"Numerical Columns standard scaling completed: {numerical_columns}")
             
            logging.info(f"Categorical Columns encoding completed: {categorical_columns}")
            
            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline,numerical_columns),
                    ("cat_pipeline", cat_pipeline,categorical_columns)
                ],
                remainder="passthrough"
                
            )
            
            return preprocessor 
        
            
        except Exception as e:
            raise CustomException(e,sys) 
        
        
    def initiate_data_transformation(self,train_path,test_path):
        
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            print(train_df.columns)
            
            logging.info("Read train and test data completed")
            
            logging.info("Obtaing preprocessing object")
            
            preprocessor_obj  = self.get_transformer_object()
            
            target_column_name = "SalePrice"
                        
            input_feature_train_df = train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df = train_df[target_column_name]
            print(f"Input feature train df:\n {input_feature_train_df.head()}")
            
            input_feature_test_df = test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df = test_df[target_column_name]
            print(f"Input feature test df:\n {input_feature_test_df.head()}")
            logging.info(
                f"Applying preprocessing object on training and testing datasets."
            )
            
            print("Applying preprocessing object on training and testing datasets.")
            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)
            
            print(input_feature_train_arr.shape)
            print(input_feature_test_arr.shape)
            print(np.array(target_feature_test_df).shape)
            print(np.array(target_feature_train_df).shape)
            
            train_arr = np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
            
            
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            
            print("concatination done")
            logging.info(f"Saved preprocessing object.")
            
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj 
            )
            print("save done")
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
            
        
        except Exception as e:
            raise CustomException(e,sys)
        
            
     