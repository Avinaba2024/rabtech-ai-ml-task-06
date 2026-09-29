# RabTech Academy — Task 06: Real-Time ML Inference REST API & Capstone

## Overview

This repository packages the **Task 04 champion credit-default classification model** behind a FastAPI REST API.

### Architecture

```text
Client
  |
  | POST /predict (JSON)
  v
FastAPI + Pydantic validation
  |
  v
Pandas feature frame
  |
  v
Serialized champion Scikit-Learn Pipeline
  |-- preprocessing
  |-- trained classifier
  |
  v
Prediction + probability response
```

The serialized Pipeline contains the preprocessing and classifier from Task 04, so inference applies the same transformations used during training.

## Repository contents

- `app.py` — FastAPI service
- `champion_credit_default_model.joblib` — serialized Task 04 champion pipeline
- `test_api.py` — API unit/integration tests
- `Dockerfile` — container image definition
- `requirements.txt` — pinned runtime dependencies
- `.dockerignore` — Docker build exclusions
- `example_request.json` — sample `/predict` payload
- `credit_default_risk.csv` — example dataset

## API endpoints

### GET `/health`

Returns service/model health.

Example response:

```json
{
  "status": "ok",
  "model_loaded": true
}
```

### POST `/predict`

Accepts JSON customer features and returns a binary prediction plus probabilities.

Request:

```json
{
  "age": 35,
  "annual_income": 65000,
  "loan_amount": 30000,
  "credit_score": 700,
  "employment_years": 8,
  "debt_to_income": 0.25,
  "late_payments": 1,
  "home_ownership": "MORTGAGE",
  "loan_purpose": "DEBT_CONSOLIDATION",
  "employment_type": "SALARIED"
}
```

Response shape:

```json
{
  "prediction": 0,
  "probability_default": 0.123456,
  "probability_non_default": 0.876544,
  "model_version": "task04-champion-v1"
}
```

The exact probabilities depend on the serialized champion model.

## Interactive API documentation

When running locally:

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`
- OpenAPI JSON: `http://localhost:8000/openapi.json`

## Run locally

```bash
python -m venv .venv
```

Activate the environment, then:

```bash
pip install -r requirements.txt
uvicorn app:app --reload
```

Open:

```text
http://localhost:8000/docs
```

## Run tests

```bash
pip install pytest httpx
pytest -q
```

Tests verify:

- `/health` response and model loading
- `/predict` response schema
- probability bounds and probability sum
- validation failure returns HTTP 422
- OpenAPI documentation exposes `/predict`

## Docker

Build:

```bash
docker build -t rabtech-ml-api .
```

Run:

```bash
docker run --rm -p 8000:8000 rabtech-ml-api
```

Then open:

```text
http://localhost:8000/docs
```

## Production considerations

This is an internship capstone demonstration. Before real production use, add authentication/authorization, HTTPS termination, structured logging, rate limiting, request tracing, monitoring, model/data drift detection, model registry/versioning, secrets management, and a documented rollback process.

The API should not be treated as an automatic adverse-decision system without appropriate human oversight and governance.
