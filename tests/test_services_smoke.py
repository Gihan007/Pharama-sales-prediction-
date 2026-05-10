import sys
from pathlib import Path

from fastapi.testclient import TestClient

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from apps.api_gateway.main import app as api_gateway_app
from services.forecast_service.app.forecasting import forecast_sales
from services.forecast_service.app.main import app as forecast_service_app
from services.frontend_service.app.main import app as frontend_service_app


def test_api_gateway_health():
    client = TestClient(api_gateway_app)
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["service"] == "api-gateway"


def test_frontend_service_renders_forecast_page():
    client = TestClient(frontend_service_app)
    response = client.get("/forecast")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_forecast_service_historical_forecast():
    client = TestClient(forecast_service_app)
    response = client.post(
        "/forecast",
        json={"category": "C1", "date": "2015-01-01", "model_type": "ensemble"},
    )

    assert response.status_code == 200
    assert response.json()["success"] is True


def test_forecast_uses_restructured_data_path():
    value, _, _, model_used = forecast_sales("C1", "2015-01-01", "ensemble")

    assert isinstance(float(value), float)
    assert model_used == "Historical Data"
