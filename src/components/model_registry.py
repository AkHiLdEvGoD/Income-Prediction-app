from src.utils.logger import logger
import mlflow
import json

def load_model_info(file_path:str):
    try:
        with open(file_path,'r') as f:
            model_info = json.load(f)
        logger.success(f'Model info loaded from {file_path}')
        return model_info
    except FileNotFoundError:
        logger.error(f'File not found: {file_path}')
        raise
    except Exception as e:
        logger.error(f'Unexpected error occurred while loading the model info: {e}')
        raise

def register_model(model_name:str,model_info:str):
    try:
        run_id = model_info.get("run_id")
        if not run_id:
            raise ValueError("run_id missing from model_info.")
        model_uri = f'runs:/{run_id}/model'
        model_version = mlflow.register_model(model_uri,model_name)

        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name=model_name,
            version = model_version.version,
            stage='Staging'
        )
        logger.success(f"Model '{model_name}' (version {model_version.version}) registered and moved to 'Staging'.")

    except Exception as e:
        logger.error(f"Model registration failed: {e}")
        raise