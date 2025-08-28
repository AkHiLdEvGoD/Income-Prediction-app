import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.utils.logger import logger
from src.components.model_registry import load_model_info,register_model
from src.utils.config import MODEL_INFO_PATH
import mlflow
import dagshub
# from dotenv import load_dotenv

def main():
    # load_dotenv()
    dagshub_username = os.getenv('DAGSHUB_USERNAME')
    dagshub_token = os.getenv('DAGSHUB_PASSWORD')
    if not dagshub_token:
        raise EnvironmentError('DAGSHUB_PASSWORD variable not set')
    os.environ['MLFLOW_TRACKING_USERNAME']= dagshub_username
    os.environ['MLFLOW_TRACKING_PASSWORD'] = dagshub_token
    dagshub_url = "https://dagshub.com"
    repo_owner = "AkHiLdEvGoD"
    repo_name = "Income-Prediction-app"

    mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')
    # mlflow.set_tracking_uri(tracking_uri)
    # dagshub.init(repo_name=repo_name,repo_owner=repo_owner,mlflow=True)
    try:
        model_info = load_model_info(MODEL_INFO_PATH)
        model_name = 'income_pred_model'
        register_model(model_name,model_info)
    except Exception as e:
        logger.error(f'Failed to complete the model registration process: {e}')

if __name__ == '__main__':
    main()