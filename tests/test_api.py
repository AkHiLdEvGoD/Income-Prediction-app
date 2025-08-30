from api.main import app
from fastapi.testclient import TestClient
import pytest

@pytest.fixture(scope='module')
def client():
    with TestClient(app) as c:
        yield c

# def test_root_endpoint(client):
#     response = client.get('/')
#     assert response.status_code==200

def test_predict_endpoint(client):
    sample_input = {
        "age": 25,
        "workclass": "Private",
        "education": "11th",
        "educational-num": 7,
        "marital-status": "Never-married",
        "occupation": "Machine-op-inspct",
        "relationship": "Own-child",    
        "race": "Black",
        "gender": "Male",
        "capital-gain":0,
        "capital-loss": 0,
        "hours-per-week": 40, 
        "native-country": "United-States"
    }
    response = client.post('/predict',json=sample_input)
    print(response.json())
    assert response.status_code==200
    assert response.json()['Predicted Income'] in ['<=50k','>50k']