from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict
from typing import List, Optional


app = FastAPI(title="Training Service")


class TrainingRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    categories: Optional[List[str]] = None
    model_type: str = "xgboost"


@app.get("/health")
async def health():
    return {"status": "healthy", "service": "training-service"}


@app.post("/training/run")
async def run_training(payload: TrainingRequest):
    try:
        from src import pipeline

        trainers = {
            "xgboost": pipeline.train_xgboost_models,
            "transformer": pipeline.train_transformer_models,
            "gru": pipeline.train_gru_models,
            "lstm": pipeline.train_lstm_models,
            "lightgbm": pipeline.train_lightgbm_models,
            "prophet": pipeline.train_prophet_models,
            "tft": pipeline.train_tft_models,
            "nbeats": pipeline.train_nbeats_models,
            "informer": pipeline.train_informer_models,
        }

        trainer = trainers.get(payload.model_type)
        if trainer is None:
            raise HTTPException(status_code=400, detail=f"Unsupported model_type: {payload.model_type}")

        model_dir_names = {
            "xgboost": "models_xgb",
            "transformer": "models_transformer",
            "gru": "models_gru",
            "lstm": "models_lstm",
            "lightgbm": "models_lightgbm",
            "prophet": "models_prophet",
            "tft": "models_tft",
            "nbeats": "models_nbeats",
            "informer": "models_informer",
        }

        trainer(
            categories=payload.categories,
            base_path="./data/raw/",
            model_dir=f"./artifacts/models/{model_dir_names[payload.model_type]}/",
        )
        return {"success": True, "model_type": payload.model_type, "categories": payload.categories}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
