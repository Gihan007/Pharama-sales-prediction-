# GKE Kubernetes Manifests

This folder contains Kubernetes YAML for deploying the microservice version of
the drug sales platform to Google Kubernetes Engine.

## Folder Layout

```text
infra/kubernetes/gke/
  shared/                  Namespace, ConfigMap, Secret, Ingress
  frontend-service/        Frontend UI deployment/service
  api-gateway/             API gateway deployment/service
  forecast-service/        Forecast API deployment/service
  training-service/        Training API deployment/service
  advanced-ai-service/     NAS, federated, causal API deployment/service
  explainability-service/  SHAP/explanation API deployment/service
  redis/                   Redis deployment/service/PVC
  postgres/                Postgres StatefulSet/service
```

## Image Placeholder

All app services currently use this placeholder image:

```text
REGION-docker.pkg.dev/PROJECT_ID/REPOSITORY/drug-sales-platform:latest
```

Replace it with your Artifact Registry image before applying.

Example:

```powershell
$env:PROJECT_ID = "your-gcp-project"
$env:REGION = "asia-south1"
$env:REPOSITORY = "drug-sales"
$env:IMAGE = "$env:REGION-docker.pkg.dev/$env:PROJECT_ID/$env:REPOSITORY/drug-sales-platform:latest"

docker build -f infra/docker/Dockerfile -t $env:IMAGE .
docker push $env:IMAGE
```

Then update the manifests:

```powershell
Get-ChildItem infra/kubernetes/gke -Recurse -Include *.yml |
  ForEach-Object {
    (Get-Content $_.FullName) -replace "REGION-docker.pkg.dev/PROJECT_ID/REPOSITORY/drug-sales-platform:latest", $env:IMAGE |
      Set-Content $_.FullName
  }
```

## Deploy

```bash
kubectl apply -k infra/kubernetes/gke
```

Check rollout:

```bash
kubectl -n drug-sales get pods
kubectl -n drug-sales get svc
kubectl -n drug-sales get ingress
```

## Notes

- `shared/secret.yml` contains a development placeholder password. Replace it
  before production.
- For production GKE, prefer Cloud SQL over the included in-cluster Postgres.
- Generated forecast/SHAP images are container-local unless you add shared
  storage such as GCS Fuse or Filestore.
- Large model artifacts are currently expected to be included in the container
  image or mounted separately.
