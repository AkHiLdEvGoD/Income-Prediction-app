import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.components.model_evaluation import load_model,load_data,evaluate_model,save_metrics,save_model_info,load_params
from src.utils.logger import logger
from src.utils.config import MODEL_PATH,TEST_DATA_PATH,ARTIFACTS_DIR
import mlflow
import dagshub
# from dotenv import load_dotenv

def main():
    # load_dotenv()
    # dagshub_username = os.getenv('DAGSHUB_USERNAME')
    dagshub_token = os.getenv('DAGSHUB_PASSWORD')
    if not dagshub_token:
        raise EnvironmentError('DAGSHUB_PASSWORD variable not set')
    # # tracking_uri = os.getenv('MLFLOW_TRACKING_URI')
    # # repo_name = os.getenv('DAGSHUB_REPO_NAME')
    # # repo_owner = os.getenv('DAGSHUB_REPO_OWNER')
    os.environ['MLFLOW_TRACKING_USERNAME']= dagshub_token
    os.environ['MLFLOW_TRACKING_PASSWORD'] = dagshub_token
    dagshub_url = "https://dagshub.com"
    repo_owner = "AkHiLdEvGoD"
    repo_name = "Income-Prediction-app"

    mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')
    # mlflow.set_tracking_uri(tracking_uri)
    # dagshub.init(repo_name=repo_name,repo_owner=repo_owner,mlflow=True)
    mlflow.set_experiment('project_pipeline')
    with mlflow.start_run() as run:
        try:
            params = load_params('./params.yaml')           
            model = load_model(MODEL_PATH)
            test_data = load_data(TEST_DATA_PATH)
            metrics,cm,clf_report = evaluate_model(model,test_data)
            save_metrics(metrics,cm,clf_report,ARTIFACTS_DIR)
            for key, value in metrics.items():
                mlflow.log_metric(key, value)
            model_type = params['model_training']['model_type']
            mlflow.log_param('Model_type',model_type)

            if hasattr(model,'get_params'):
                params = model.get_params()
                for param_name, param_value in params.items():
                    mlflow.log_param(param_name, param_value)

            mlflow.sklearn.log_model(model,'model')
            mlflow.log_artifacts(ARTIFACTS_DIR)
            save_model_info(run.info.run_id,MODEL_PATH,ARTIFACTS_DIR)
            logger.success('Model Evaluation logged and Completed')

        except Exception as e:
            logger.error(f'Unexpected error occure during Model Evaluation : {e}')

if __name__ == '__main__':
    main()