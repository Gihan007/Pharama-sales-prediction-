from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from libs.common.metrics import install_metrics
from libs.common.serialization import make_json_serializable

app = FastAPI(title="Advanced AI Service")
install_metrics(app, "advanced-ai-service")


class NASRequest(BaseModel):
    category: str = "C1"
    generations: int = 3


class FederatedTrainRequest(BaseModel):
    category: str = "C1"
    num_clients: int = 5
    num_rounds: int = 8
    distribution_type: str = "iid"


class CausalDiscoveryRequest(BaseModel):
    category: str = "C1"
    max_lags: int = 5


@app.get("/health")
async def health():
    return {"status": "healthy", "service": "advanced-ai-service"}


@app.post("/nas/search")
async def nas_search(payload: NASRequest):
    try:
        from src.models.advanced.nas_drug_prediction import DrugPredictionNAS

        nas = DrugPredictionNAS(save_dir="./artifacts/results/nas_results")
        result = nas.search_optimal_architecture(payload.category, payload.generations)
        return {"success": True, "result": make_json_serializable(result)}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/federated/train")
async def federated_train(payload: FederatedTrainRequest):
    try:
        from src.models.advanced.federated_learning import run_federated_drug_prediction

        result = run_federated_drug_prediction(
            category=payload.category,
            num_clients=payload.num_clients,
            num_rounds=payload.num_rounds,
            distribution_type=payload.distribution_type,
        )
        return {"success": True, "result": make_json_serializable(result)}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/causal/discovery")
async def causal_discovery(payload: CausalDiscoveryRequest):
    try:
        from src.models.advanced.causal_inference import CausalInferenceEngine

        engine = CausalInferenceEngine(save_dir="./artifacts/results/causal_results")
        result = engine.discover_causal_relationships(payload.category, max_lags=payload.max_lags)
        return {"success": True, "result": make_json_serializable(result)}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
