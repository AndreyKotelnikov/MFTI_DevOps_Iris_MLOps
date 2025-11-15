from fastapi.testclient import TestClient


def test_predict_endpoint_returns_valid_response(tmp_path, monkeypatch):
    """Проверяем, что /predict возвращает корректный ответ."""
    monkeypatch.setenv("MODEL_DIR", str(tmp_path))

    from src.models.train import train_and_save_model
    train_and_save_model()

    from src.api.main import app
    client = TestClient(app)

    payload = {
        "samples": [
            {
                "sepal_length": 5.1,
                "sepal_width": 3.5,
                "petal_length": 1.4,
                "petal_width": 0.2,
            }
        ]
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert "predictions" in data
    assert len(data["predictions"]) == 1

    pred = data["predictions"][0]
    assert pred["class_id"] in [0, 1, 2]
    assert pred["class_name"] in ["setosa", "versicolor", "virginica"]
    assert 0.0 <= pred["probability"] <= 1.0
