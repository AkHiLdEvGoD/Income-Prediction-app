import os
import pandas as pd
import joblib
import mlflow.sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from src.utils.logger import logger
import yaml

def load_params(params_path:str):
    try:
        with open(params_path,'r') as f:
            params = yaml.safe_load(f)
        logger.info(f'Parameter retrieved from {params_path}')
        return params
    
    except FileNotFoundError:
        logger.error('File not found: %s', params_path)
        raise
    except yaml.YAMLError as e:
        logger.error('YAML error: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error: %s', e)
        raise

def load_data(data_path:str):
    try:
        df = pd.read_csv(data_path)
        X = df.drop('target',axis=1)
        y = df.loc[:,'target']

        logger.info(f"Train data loaded with shape: {df.shape}")
        return X, y
    except Exception as e:
        logger.error(f"Error loading training data: {e}")
        raise

def train_model(X,y,params):
    try:
        # model = KNeighborsClassifier(metric='manhattan', n_neighbors=6,weights='distance')
        model_type = params['model_training']['model_type']
        if model_type == 'logistic_regression':
            model_params = params['model_training']['logistic_regression']
            model = LogisticRegression(**model_params)

        elif model_type == 'knn':
            model_params = params['model_training']['knn']
            model = KNeighborsClassifier(**model_params)
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")
        
        model.fit(X,y)
        logger.success("Model training completed")
        return model
    except Exception as e:
        logger.error(f"Error during model training: {e}")
        raise

def save_model(model,destination_path):
    try:
        model_path = os.path.join(destination_path,'model')
        os.makedirs(model_path,exist_ok=True)
        # joblib.dump(model,os.path.join(model_path,'model.pkl'))
        mlflow.sklearn.save_model(sk_model=model, path=model_path)
        logger.info(f'Model saved to path : {os.path.join(model_path,"model.pkl")}')
    except Exception as e:
        logger.error(f'Error occurred while saving the model: {e}')
        raise
