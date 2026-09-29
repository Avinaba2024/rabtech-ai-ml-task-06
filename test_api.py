from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

VALID_PAYLOAD = {
    "age": 35,
    "annual_income": 65000,
    "loan_amount": 30000,
    "credit_score": 700,
    "employment_years": 8,
    "debt_to_income": 0.25,
    "late_payments": 1,
    "home_ownership": "MORTGAGE",
    "loan_purpose": "DEBT_CONSOLIDATION",
    "employment_type": "SALARIED",
}

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True

def test_predict_schema():
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {
        "prediction",
        "probability_default",
        "probability_non_default",
        "model_version",
    }
    assert body["prediction"] in [0, 1]
    assert 0 <= body["probability_default"] <= 1
    assert 0 <= body["probability_non_default"] <= 1
    assert abs(
        body["probability_default"] + body["probability_non_default"] - 1
    ) < 1e-5

def test_invalid_payload_returns_422():
    bad = dict(VALID_PAYLOAD)
    bad["credit_score"] = 1200
    response = client.post("/predict", json=bad)
    assert response.status_code == 422

def test_openapi_available():
    response = client.get("/openapi.json")
    assert response.status_code == 200
    assert "/predict" in response.json()["paths"]
