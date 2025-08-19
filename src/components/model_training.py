import os
import pandas as pd
import joblib
from sklearn.neighbors import KNeighborsClassifier
from src.utils.logger import logger
from src.utils.config import ARTIFACTS_DIR,PROCESSED_DATA_PATH

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

def train_model(X,y):
    try:
        model = KNeighborsClassifier(metric='manhattan', n_neighbors=6,weights='distance')
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
        joblib.dump(model,os.path.join(model_path,'model.pkl'))
        logger.info(f'Model saved to path : {os.path.join(model_path,"model.pkl")}')
    except Exception as e:
        logger.error(f'Error occurred while saving the model: {e}')
        raise

def main():
    try:
        X,y = load_data(PROCESSED_DATA_PATH)
        model = train_model(X,y)
        save_model(model,ARTIFACTS_DIR)
    except Exception as e:
        logger.error(f"Model training failed: {e}")

if __name__ == '__main__':
    main()