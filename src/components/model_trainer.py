import os 
import sys 
from dataclasses import dataclass 
import dill 
import optuna 


from catboost import CatBoostRegressor 

from sklearn.ensemble import (
    
    AdaBoostRegressor, 
    GradientBoostingRegressor, 
    RandomForestRegressor,
)

from sklearn.linear_model import LinearRegression 
from sklearn.metrics import r2_score 
from sklearn.neighbors import KNeighborsRegressor 
from sklearn.tree import DecisionTreeRegressor 
from xgboost import XGBRegressor 

from src.exception import CustomException 
from src.logger import logging 

from src.utils import save_object ,evaluate_model 

@dataclass 
class ModelTrainerConfig:
    # path where final model will be saved
    trained_model_file_path = os.path.join("artifacts","catboost_model.pkl")
    # path to saved preprocessor (from data transformation appliance)
    preprocessor_object_path: str = os.path.join('artifacts','preprocessor.pkl')
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()
        
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
            
            params = {
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
                                              models=models,param=params,n_trials=20)
            
            logging.info(f"Model report: {model_report}")
            # choose best model by highest score 
            
            best_model_name = max(model_report, key=model_report.get)
            best_model_score = model_report[best_model_name]
            logging.info(f"Best model name:{best_model_name} with score {best_model_score:.4f}")
            
            
            if best_model_name == "Catboosting Classifer":
                logging.info("CatBoost selected as best model -")
                best_params, best_score = self.run_catboost_optuna(X_train,y_train,X_test,y_test, n_trials=50)
                
                final_model = CatBoostRegressor(verbose=False, **best_params)
                final_model.fit(X_train,y_train)
                
                # saving final model 
                
                save_object(file_path = self.model_trainer_config.trained_model_file_path, obj=final_model)
                logging.info(f"Sved final catboost model to {self.model_trainer_config.trained_model_file_path}")
                
                # return metrics 
                preds = final_model.predict(X_test)
                final_r2 = r2_score(y_test,preds)
                
                return {
                    "best_model_name": "CatBoost",
                    "best_params":best_params,
                    "r2_score":float(final_r2)
                }
            
            else:
                logging.info(f"{best_model_name} selected as best model - saving without Optuna tuning.")
                best_model = models[best_model_name]
                best_model.fit(X_train, y_train)
                save_object(file_path=self.model_trainer_config.trained_model_file_path, obj=best_model)

                preds = best_model.predict(X_test)
                final_r2 = r2_score(y_test, preds)

                return {
                    "best_model_name": best_model_name,
                    "best_params": "Default/GridSearch",
                    "r2_score": float(final_r2)
                }
            
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
            
            
            