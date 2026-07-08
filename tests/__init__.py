import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fastapi.testclient import TestClient
import pytest
from unittest.mock import Mock, patch

# از آنجایی که api ما هنگام import، مدل را بارگذاری می‌کند،
# باید با patch جلوی بارگذاری واقعی را بگیریم.
@pytest.fixture(autouse=True)
def mock_model_load():
    with patch("mlflow.sklearn.load_model") as mock_load:
        mock_model = Mock()
        mock_model.predict.return_value = [0]
        mock_model.predict_proba.return_value = [[0.8, 0.2]]
        mock_load.return_value = mock_model
        yield

def test_health():
    from api import app
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["model_loaded"] == True

def test_predict():
    from api import app
    client = TestClient(app)
    response = client.post("/predict", json={"features": {"age": 25, "current_streak_days": 5}})
    assert response.status_code == 200
    assert "prediction" in response.json()
    assert "probability" in response.json()
    assert response.json()["prediction"] == 0