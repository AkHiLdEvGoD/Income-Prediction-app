import os
import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score,recall_score,f1_score,precision_score
import joblib
import dagshub
import pandas as pd
from src.utils.logger import logger
import json

def load_model(model_path):
    try:
        model = joblib.load(model_path)
        logger.info('Model Loaded for evaluation')
        return model
    
    except FileNotFoundError:
        logger.error(f'Model not found at {model_path}')
        raise

    except Exception as e:
        logger.error(f'Unexpected error occured while loading the model {e}')
        raise

def load_data(data_path:str):
    try:
        df = pd.read_csv(data_path)
        logger.infO(f'Data loaded from path {data_path}')
        return df
    
    except pd.errors.ParserError as e:
        logger.error(f'Failed to parse the CSV file: {e}')
        raise
    
    except Exception as e:
        logger.error(f'Unexpected error occurred while loading the data: {e}')
        raise

def evaluate_model(model,df:pd.DataFrame):
    try:
        X = df.drop(columns = ['target'])
        y = df['target']

        y_pred = model.predict(X)

        accuracy = accuracy_score(y,y_pred)
        precision = precision_score(y,y_pred)
        recall = recall_score(y,y_pred)
        f1 = f1_score(y,y_pred)

        metric_dict = {
            'accuracy' : accuracy,
            'precision' : precision,
            'recall' : recall,
            'f1_score' : f1
        }

        logger.info('All metrics of model evaluated')
        return metric_dict
    
    except Exception as e:
        logger.error(f'Unexpected error occured while evaluating model {e}')
        raise

def save_metrics(metric_dict,save_path):
    try:
        metric_save_path = os.path.join(save_path,'metrics')
        os.makedirs(metric_save_path,exist_ok=True)
        with open(os.path.join(metric_save_path,'metrics.json'),'w') as f:
            json.dump(metric_dict,f,indent=4)
        logger.info(f'Metrics saved at {os.path.join(metric_save_path,'metrics.json')}')
    
    except Exception as e:
        logger.error(f'Unexpected error occured while saving metrics : {e}')
        raise
