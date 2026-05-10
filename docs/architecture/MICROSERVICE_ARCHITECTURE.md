# Microservice Architecture

This repository is now arranged as a microservice-style monorepo. The original
root imports are kept as thin compatibility shims, while new development should
target the service folders below.

## Top-Level Layout

```text
apps/
  api_gateway/              API-only FastAPI gateway
  web_ui/                   Compatibility shim for frontend imports

services/
  frontend_service/         Standalone frontend service, Jinja templates, CSS, JS
  forecast_service/         Forecast inference API and forecasting logic
  training_service/         Model training API
  advanced_ai_service/      NAS, federated learning, causal inference
  explainability_service/   SHAP and explanation APIs

src/                        Shared ML/model code used by the services
data/raw/                   C1-C8 CSV files and source datasets
artifacts/models/           Trained model artifacts
artifacts/results/          Generated experiment/result outputs
infra/docker/               Dockerfile, Compose, Nginx
infra/monitoring/           Prometheus config
```

## Local Entrypoints

```bash
python app.py
```

Runs the API gateway on port `5000`.

Individual services:

```bash
uvicorn services.frontend_service.app.main:app --port 3001
uvicorn services.forecast_service.app.main:app --port 5001
uvicorn services.training_service.app.main:app --port 5002
uvicorn services.advanced_ai_service.app.main:app --port 5003
uvicorn services.explainability_service.app.main:app --port 5004
```

The web UI service owns all frontend pages and static files. The API gateway
does not render frontend templates. The web UI service proxies `/api/*`
requests to the API gateway, so the existing JavaScript can keep calling
relative `/api/...` URLs.

Docker Compose:

```bash
docker compose -f infra/docker/docker-compose.yml up --build
```

## Compatibility Shims

These files remain at the repo root so existing scripts keep working:

- `app.py`
- `forecast_utils.py`
- `shap_explainer.py`
- `apps/web_ui/main.py`

New code should import from the service packages directly.
