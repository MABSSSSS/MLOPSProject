import os 
import sys 
import optuna 
import numpy as np 
import pandas as pd 
import dill
from src.logger import logging 
from src.exception import CustomException 
from sklearn.metrics import r2_score 

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        
        os.makedirs(dir_path, exist_ok=True)
        
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
            
    except Exception as e:
        raise CustomException(e,sys)
    

def evaluate_model(X_train,y_train,X_test,y_test, models,best_params,n_trials=30):
    try:
       
        report = {}
        print("Shapes:")
        print("X_train:", X_train.shape)
        print("y_train:", np.array(y_train).shape)
        print("X_test:", X_test.shape)
        print("y_test:", np.array(y_test).shape)

        
        for model_name, model in models.items():
            logging.info(f"Starting Optuna tuning for: {model_name}")
            print(f"\n Tuning {model_name}...")
            
            # If models has no parameters then tain directly 
            
            if model_name not in best_params or len(best_params[model_name]) == 0:
                model.fit(X_train,y_train)
                y_pred = model.predict(X_test)
                score = r2_score(y_test,y_pred)
                report[model_name] = score 
                print(f" { model_name}: No params, R2 = {score:.4f}")
                continue 
            
            search_space = best_params[model_name]
            
            # Define objective for Optuna 
            
            def objective(trial):
                trial_params = {}
                
                for key,values in search_space.items():
                    if isinstance(values[0], float):
                        trial_params[key] = trial.suggest_float(key, min(values), max(values))
                    elif isinstance(values[0],int):
                        trial_params[key] = trial.suggest_int(key, min(values),max(values))
                    else:
                        trial_params[key] = trial.suggest_categorical(key,values)
                
                
                model.set_params(**trial_params)
                model.fit(X_train,y_train)
                preds = model.predict(X_test)
                return r2_score(y_test,preds)
            
            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
            
            best_params = study.best_params 
            best_score = study.best_value 
            
            
            logging.info(f"Best params for {model_name}: {best_params}")
            logging.info(f"Best R2 Score: {best_score:.4f}")
            
            print(f"Best params for {model_name}: {best_params}")
            print(f"Best R score: {best_score: .4f}")
            
            report[model_name] = best_score 
        
        return report  
    
    except  Exception as e:
        raise CustomException(e,sys)
    
            
            