from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict

from libs.common.metrics import install_metrics
from services.explainability_service.app.shap_explainer import get_model_explainability


app = FastAPI(title="Explainability Service")
install_metrics(app, "explainability-service")


class ExplainabilityRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    category: str = "C1"
    model_type: str = "xgboost"


@app.get("/health")
async def health():
    return {"status": "healthy", "service": "explainability-service"}


@app.post("/explainability")
async def explainability(payload: ExplainabilityRequest):
    results = get_model_explainability(payload.category, payload.model_type)
    if results is None:
        raise HTTPException(
            status_code=500,
            detail=f"Could not generate explainability for {payload.category} using {payload.model_type}",
        )

    return {"success": True, "results": results}
