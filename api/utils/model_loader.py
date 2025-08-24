import os
import joblib
import mlflow
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient
from fastapi import FastAPI
from contextlib import asynccontextmanager
import dagshub

from .config import PREPROCESSOR_PATH

load_dotenv()

tracking_uri = os.getenv('MLFLOW_TRACKING_URI')
repo_name = os.getenv('DAGSHUB_REPO_NAME')
repo_owner = os.getenv('DAGSHUB_REPO_OWNER')

mlflow.set_tracking_uri(tracking_uri)
dagshub.init(repo_name=repo_name, repo_owner=repo_owner, mlflow=True)

def get_latest_model_version(model_name: str):
    client = MlflowClient()
    latest = client.get_latest_versions(model_name, stages=["Staging"])
    if not latest:
        latest = client.get_latest_versions(model_name, stages=["None"])
    return latest[0].version if latest else None

@asynccontextmanager
async def lifespan(app: FastAPI):
    model_name = "income_prediction_model"
    model_version = get_latest_model_version(model_name)
    if not model_version:
        raise RuntimeError(f"No versions found for model: {model_name}")

    model_uri = f"models:/{model_name}/{model_version}"
    print(f"Loading model from {model_uri}")
    
    app.state.model = mlflow.pyfunc.load_model(model_uri)
    app.state.preprocessor = joblib.load(PREPROCESSOR_PATH)

    print("Model and pipeline loaded.")
    yield
