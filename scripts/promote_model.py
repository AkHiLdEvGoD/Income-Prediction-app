import os
import mlflow
from mlflow.tracking import MlflowClient
from dotenv import load_dotenv
import dagshub

def promote_best_model():
    load_dotenv()
    dagshub_url = "https://dagshub.com"
    repo_owner = "AkHiLdEvGoD"
    repo_name = "Income-Prediction-app"

    if os.getenv('CI')=='true':
        dagshub_username = os.getenv('DAGSHUB_USERNAME')
        dagshub_token = os.getenv('DAGSHUB_PASSWORD')
        if not dagshub_token:
            raise EnvironmentError('DAGSHUB_PASSWORD variable not set')
        os.environ['MLFLOW_TRACKING_USERNAME']= dagshub_username
        os.environ['MLFLOW_TRACKING_PASSWORD'] = dagshub_token
        mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')

    else: 
        tracking_uri = os.getenv('MLFLOW_TRACKING_URI')
        repo_name = os.getenv('DAGSHUB_REPO_NAME')
        repo_owner = os.getenv('DAGSHUB_REPO_OWNER')
        mlflow.set_tracking_uri(tracking_uri)
        dagshub.init(repo_name=repo_name, repo_owner=repo_owner, mlflow=True)

    client = MlflowClient()
    MODEL_NAME = 'income_pred_model'
    latest_staging_versions = client.get_latest_versions(MODEL_NAME,stages=['Staging'])
    if not latest_staging_versions:
        print("No model in Staging to promote.")
        return
    
    METRIC_KEY = 'accuracy'
    staging_model = latest_staging_versions[0]
    staging_run_id = staging_model.run_id
    staging_run = client.get_run(staging_run_id)
    staging_metric = staging_run.data.metrics.get(METRIC_KEY)

    if staging_metric is None:
        print("No metric found for staging model.")
        return

    print(f"Latest Staging model v{staging_model.version} with {METRIC_KEY}: {staging_metric}")
    prod_versions = client.get_latest_versions(MODEL_NAME, stages=["Production"])
    current_prod_metric = None

    if prod_versions:
        prod_model = prod_versions[0]
        prod_run = client.get_run(prod_model.run_id)
        current_prod_metric = prod_run.data.metrics.get(METRIC_KEY)
        print(f"Current Production model v{prod_model.version} with {METRIC_KEY}: {current_prod_metric}")

    if current_prod_metric is not None and staging_metric <= current_prod_metric:
        print("Staging model is not better than Production. Skipping promotion.")
        return
    
    for version in prod_versions:
        client.transition_model_version_stage(
            name=MODEL_NAME,
            version=version.version,
            stage="Archived"
        )
    
    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=staging_model.version,
        stage="Production"
    )
    print(f"Model version {latest_staging_versions} promoted to Production")

if __name__ == "__main__":
    promote_best_model()