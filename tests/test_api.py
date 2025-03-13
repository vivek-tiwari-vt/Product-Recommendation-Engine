import requests

def test_recommendation_endpoint():
    response = requests.get(
        "http://localhost:5000/recommend",
        params={"user_id": "A123456789"}
    )
    assert response.status_code == 200
    assert len(response.json()['recommendations']) == 10