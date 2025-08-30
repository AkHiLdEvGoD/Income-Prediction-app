import pytest
import pandas as pd
import mlflow,dagshub
from mlflow.tracking import MlflowClient 
import os
import joblib
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score

@pytest.fixture(scope='module')
def model():
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

    def get_latest_model_version(model_name: str):
        client = MlflowClient()
        latest = client.get_latest_versions(model_name, stages=["Staging"])
        return latest[0].version if latest else None
    
    model_name = "income_pred_model"
    model_version = get_latest_model_version(model_name)
    model_uri = f"models:/{model_name}/{model_version}"
    model = mlflow.pyfunc.load_model(model_uri)
    PREPROCESSOR_PATH = os.getenv('PREPROCESSOR_PATH')
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    TEST_DATA_PATH = os.getenv('TEST_DATA')
    test_data = pd.read_csv(TEST_DATA_PATH)

    return model,preprocessor,test_data

def test_model_loaded_properly(model):
    model,preprocessor,_ = model
    assert model is not None, "Model failed to load from MLflow"
    assert preprocessor is not None, "Preprocessor failed to load from path"
    assert hasattr(model, "predict"), "Loaded model does not have a predict method"

def test_model_signature(model):
    model,preprocessor,_=model
    input_data = pd.DataFrame({
        'age':[25],
        'educational-num':[7],
        'occupation':['Machine-op-inspct'],
        'relationship':['Own-child'],
        'capital-gain':[0],
        'capital-loss':[0],
        'hours-per-week':[40],
        'work_type':['Private'],
        'is_married':[0],
        'region':['North America'],
        'race_gender':['Black_Male'],
        'has_capital_gain':[0],
        'has_capital_loss':[0]
    })
    processed_data = preprocessor.transform(input_data)
    preds = model.predict(processed_data)
    assert preds.shape[0] == 1, "Model should return 1 prediction for 1 input"
    assert preds[0] in [0, 1], "Model should return binary classification output"

def test_model_performance(model):
    model,_,test_data = model
    X = test_data.drop(columns=['target'])
    y = test_data['target']
        
    y_pred = model.predict(X)

    accuracy_new = accuracy_score(y, y_pred)
    precision_new = precision_score(y, y_pred)
    recall_new = recall_score(y, y_pred)
    f1_new = f1_score(y, y_pred)

    expected_accuracy = 0.80
    expected_precision = 0.60
    expected_recall = 0.60
    expected_f1 = 0.60

    assert accuracy_new >= expected_accuracy,f'Accuracy should be at least {expected_accuracy}'
    assert precision_new >= expected_precision,f'Precision should be at least {expected_precision}'
    assert recall_new >= expected_recall,f'Recall should be at least {expected_recall}'
    assert f1_new >= expected_f1,f'F1 should be at least {expected_f1}'


    