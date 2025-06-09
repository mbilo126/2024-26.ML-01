import pytest
from bilotta.app import app as flask_app

@pytest.fixture()
def client():
    flask_app.config.update({"TESTING": True})
    with flask_app.test_client() as client:
        yield client
def test_hello(client):
    input_data = {
        'danceability': 0.5,
        'energy': 0.08,          
        'loudness': 0.3,         
        'speechiness': 0.6,      
        'acousticness': 0.1,     
        'instrumentalness': 0.15,
        'liveness': 0.04,        
        'valence': 0.4,          
        'tempo': 120.0,          
        'duration_ms': 180000,
        'key': -4,            
        'time_signature': 4,  
        'macro_genre': "Pop",
        'mode': 1,
        'year': 2023
    }
    response = client.post("/infer", json={"input_data": input_data})
    assert response.status_code == 200
    data = response.get_json()
    assert "prediction" in data
