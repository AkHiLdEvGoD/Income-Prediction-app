import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.components.model_training import load_data,save_model,train_model,load_params
from src.utils.logger import logger
from src.utils.config import ARTIFACTS_DIR,PROCESSED_DATA_PATH

def main():
    try:
        params = load_params('./params.yaml')
        X,y = load_data(PROCESSED_DATA_PATH)
        model = train_model(X,y,params)
        save_model(model,ARTIFACTS_DIR)
    except Exception as e:
        logger.error(f"Model training failed: {e}")

if __name__ == '__main__':
    main()