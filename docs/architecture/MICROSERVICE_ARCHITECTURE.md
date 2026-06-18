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

## Dockerized Services

The Docker Compose stack runs each application boundary as its own container on
the shared `drug_prediction_network` network:

| Service | Container DNS | Host Port | Purpose |
| --- | --- | ---: | --- |
| Web UI | `web-ui` | `3001` | Frontend pages and static assets |
| API Gateway | `api-gateway` | `5000` | Public API entrypoint used by the frontend |
| Forecast Service | `forecast-service` | `5001` | Forecast inference API |
| Training Service | `training-service` | `5002` | Model training API |
| Advanced AI Service | `advanced-ai-service` | `5003` | NAS, federated learning, causal analysis |
| Explainability Service | `explainability-service` | `5004` | Explainability API |
| Redis | `redis` | `6379` | Cache/message infrastructure |
| PostgreSQL | `postgres` | `5432` | Relational database |
| Redis Exporter | `redis-exporter` | `9121` | Redis metrics for Prometheus |
| Postgres Exporter | `postgres-exporter` | `9187` | PostgreSQL metrics for Prometheus |
| Nginx | `nginx` | `80`, `443` | Reverse proxy for frontend and `/api/*` |
| Prometheus | `prometheus` | `9090` | Metrics scraping |
| Grafana | `grafana` | `3000` | Dashboards |
| Node Exporter | `node-exporter` | `9100` | Host/container metrics |

Run only the application services:

```bash
docker compose -f infra/docker/docker-compose.yml up --build web-ui api-gateway forecast-service training-service advanced-ai-service explainability-service redis
```

Run the full platform, including reverse proxy and monitoring:

```bash
docker compose -f infra/docker/docker-compose.yml up --build
```

Useful URLs after startup:

```text
Frontend direct:  http://localhost:3001
Frontend via Nginx: http://localhost
API gateway:      http://localhost:5000/health
Forecast API:     http://localhost:5001/health
Training API:     http://localhost:5002/health
Advanced AI API:  http://localhost:5003/health
Explainability:   http://localhost:5004/health
Service metrics:  http://localhost:5000/metrics
Prometheus:       http://localhost:9090
Grafana:          http://localhost:3000
```

Stop and remove containers:

```bash
docker compose -f infra/docker/docker-compose.yml down
```

## Compatibility Shims

These files remain at the repo root so existing scripts keep working:

- `app.py`
- `forecast_utils.py`
- `shap_explainer.py`
- `apps/web_ui/main.py`

New code should import from the service packages directly.
