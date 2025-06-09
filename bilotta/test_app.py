import pytest
from bilotta.app import app as flask_app

@pytest.fixture()
def client():
    flask_app.config.update({"TESTING": True})
    with flask_app.test_client() as client:
        yield client
def test_hello(client):
    input_data = {
        'danceability': 0.5,               # float
        'energy': 0.08,                    # float
        'loudness': 0.3,                   # float
        'speechiness': 0.6,                # float
        'acousticness': 0.1,               # float
        'instrumentalness': 0.15,          # float
        'liveness': 0.04,                  # float
        'valence': 0.4,                    # float
        'tempo': 120.0,                    # float
        'duration_ms': 180000,             # int
        'key': -4,                         # int
        'time_signature': 4,               # int
        'macro_genre': "Pop",              # object (string)
        'mode': 1,                         # int (rappresenta "Major")
        'year': 2023                       # int
    }
    response = client.post("/infer", json={"input_data": input_data})
    assert response.status_code == 200
    data = response.get_json()
    assert "prediction" in data
