import os 
import sys 
from dataclasses import dataclass 
import dill 
import optuna 
import numpy as np 
import mlflow 
import mlflow.sklearn 
from catboost import CatBoostRegressor 
from urllib.parse import urlparse

from sklearn.ensemble import (
    
    AdaBoostRegressor, 
    GradientBoostingRegressor, 
    RandomForestRegressor,
)

from sklearn.linear_model import LinearRegression 
from sklearn.metrics import r2_score , mean_squared_error,mean_absolute_error 
from sklearn.neighbors import KNeighborsRegressor 
from sklearn.tree import DecisionTreeRegressor 
from xgboost import XGBRegressor 

from src.exception import CustomException 
from src.logger import logging 

from src.utils import save_object ,evaluate_model 
# import dagshub 

# dagshub.init(repo_owner='happinesswhat31', repo_name='MLOPSProject', mlflow=True)
# print("Tracking URI:", mlflow.get_tracking_uri())

@dataclass 
class ModelTrainerConfig:
    # path where final model will be saved
    trained_model_file_path = os.path.join("artifacts","catboost_model.pkl")
    # path to saved preprocessor (from data transformation appliance)
    preprocessor_object_path: str = os.path.join('artifacts','preprocessor.pkl')
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()
    
    def eval_metrics(self,actual,pred):
        rmse = np.sqrt(mean_squared_error(actual,pred))
        mae = mean_absolute_error(actual,pred)
        r2 = r2_score(actual,pred)
        return rmse, mae, r2 
        
    def run_catboost_optuna(self,X_train,y_train,X_test,y_test,n_trials=25):
        """
        Run Optuna specifically for CatBoost to return best params and best score.
        X_train etc are already transformed arrays (as per your current workflow).
        """
        
        def objective(trial):
            params = {
                "depth": trial.suggest_int("depth",4,10),
                "learning_rate": trial.suggest_float("learning_rate",0.01,0.2,log=False),
                "iterations":trial.suggest_int("iterations",50,400),
                # optional regulation
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg",1e-2,10.0),
            }
            
            model = CatBoostRegressor(verbose=False,**params)
            model.fit(X_train,y_train)
            preds = model.predict(X_test)
            return r2_score(y_test,preds)
        
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials, show_progress_bar=False)
        
        return study.best_params, study.best_value 
    
    
        
        
    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and test input data")
            print("Trsinin dtsrt")
            X_train,y_train,X_test,y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
                
            )
            # cleaning data:removing NAN Values in target
            train_nan_mask = ~np.isnan(y_train)
            test_nan_mask = ~np.isnan(y_test)
            
            X_train = X_train[train_nan_mask]
            y_train = y_train[train_nan_mask]
            X_test = X_test[test_nan_mask]
            y_test = y_test[test_nan_mask]
            print(f"Cleaned NaNs - Remaining samples:Train ={len(y_train)}, Test={len(y_test)}")
            
            
            print(X_train)
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "K-Neighbors Classifier": KNeighborsRegressor(),
                "XGBClassifier": XGBRegressor(),
                "CatBoosting Classifier": CatBoostRegressor(verbose=False),
                "AdaBoost Classifier": AdaBoostRegressor(),
            }
            
            best_params = {
                "Decision Tree": {
                    'criterion':['squared_error','friedman_mse','absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features': ['sqrt','log2']
                },
                "Random Forest":{
                    # 'criterion':['squared_error', 'firedman_mse','absolute_error','poisson'],
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boosting":{
                    # 'loss':['squared_error', 'huber','absolute_error',''quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error','firedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Linear Regression":{},
                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators':[8,16,32,64,128,256]
                },
                "CatBoostingRegressor":{
                    'depth':[6,8,10],
                    'learning_rate':[0.01,0.05,0.1],
                    'iterations':[30,50,100]
                },
                "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators':[8,16,32,64,128,256]
                }
            }
            
            
            model_report:dict =evaluate_model(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                              models=models,best_params=best_params,n_trials=20)
            
        #    To get best model score from dict 
            best_model_score = max(sorted(model_report.values()))
            
            # To get best model name from dict
            
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]
            
            print("This is best model")
            print(best_model_name)
            
            model_names = list(best_params.keys())
            
            actual_model = "" 
            
            for model in model_names:
                if best_model_name == model:
                    actual_model = actual_model + model 
                    
            best_params = best_params[actual_model]
            
            mlflow.set_registry_uri("https://dagshub.com/happinesswhat31/MLOPSProject.mlflow")
            tracking_url_type_score = urlparse(mlflow.get_tracking_uri()).scheme 
                #  password  9@SNfYYzZPezthw 
            # best model params
            
            with mlflow.start_run(run_name=best_model_name):
                
                predicted_qualities = best_model.predict(X_test)
                rmse,mae,r2 = self.eval_metrics(y_test, predicted_qualities)
                
                # Log params and metrics to ML Flow 
                
                mlflow.log_params(best_params)
                mlflow.log_metric("RMSE", rmse)
                mlflow.log_metric("MAE",mae)
                mlflow.log_metric("R2",r2)
                
                if tracking_url_type_score != "file":
                    mlflow.sklearn.log_model(
                        best_model, artifact_path="model", registered_model_name =  best_model_name 
                        
                    )
                    
                else:
                    mlflow.sklearn.log_model(best_model,"model")
                    
            
            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model")
            
            
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )
            
            predicted = best_model.predict(X_test)
            
            r2_square = r2_score(y_test, predicted)
            return r2_square 
        
            
            
            
                
            
        except Exception as e:
            raise CustomException(e,sys)
        
if __name__ == "__main__":
            # usage for local development only
            from src.components.data_ingestion import DataIngestion 
            
            di = DataIngestion()
            train_path, test_path = di.initiate_data_ingestion()
            
            # data trasnformation returns transformed arrays and path - call it to get train_arr/test_arr
            
            from src.components.data_transformation import DataTransformation 
            dt = DataTransformation()
            
            train_arr, test_arr,_ = dt.initiate_data_transformation(train_path,test_path)
            mt = ModelTrainer()
            print(mt.initiate_model_trainer(train_arr, test_arr))
            
            
            