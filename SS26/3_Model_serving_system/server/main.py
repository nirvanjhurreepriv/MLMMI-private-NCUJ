""" from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import asyncio
from .registry import ModelRegistry
from .predictor import Predictor
from typing import Optional

app = FastAPI(title="Model Serving System", version="1.0.0")

class PredictRequest(BaseModel):
    text: str
    model_id: str

class PredictResponse(BaseModel):
    prediction: int
    model_id: str
    text: str

registry = ModelRegistry()
predictor: Optional[Predictor] = None

@app.on_event("startup")
async def startup_event():
    global predictor
    registry.load_all()
    predictor = Predictor(registry)
    asyncio.create_task(predictor.process_queue())
    print("Server started. Available models:", registry.list_models())

@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    if predictor is None:
        raise HTTPException(status_code=503, detail="Server not ready")
    
    if request.model_id not in registry.list_models():
        raise HTTPException(status_code=400, detail=f"Unknown model_id: {request.model_id}")
    
    try:
        result = await predictor.predict(request.text, request.model_id)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "ok", "models": registry.list_models()}

@app.get("/models")
async def list_models():
    return {"models": registry.list_models()} """

# --------------------------------------------------------------------------------

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import asyncio
from typing import Optional
from .registry import ModelRegistry
from .predictor import Predictor

app = FastAPI(title="Model Serving System", version="1.0.0")

class PredictRequest(BaseModel):
    text: str
    model_id: str

class PredictResponse(BaseModel):
    prediction: int
    model_id: str
    text: str

registry = ModelRegistry()
predictor: Optional[Predictor] = None

@app.on_event("startup")
async def startup_event():
    global predictor
    registry.load_all()
    predictor = Predictor(
        registry,
        max_batch_size=8,
        max_wait_ms=10,
        cache_capacity=1000,
    )
    asyncio.create_task(predictor.process_queue())
    print("Server started. Available models:", registry.list_models())

@app.on_event("shutdown")
async def shutdown_event():
    if predictor:
        predictor.stop()

@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    if predictor is None:
        raise HTTPException(status_code=503, detail="Server not ready")
    if request.model_id not in registry.list_models():
        raise HTTPException(status_code=400, detail=f"Unknown model_id: {request.model_id}")
    try:
        result = await predictor.predict(request.text, request.model_id)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "ok", "models": registry.list_models()}

@app.get("/models")
async def list_models():
    return {"models": registry.list_models()}

""" @app.get("/metrics")
async def get_metrics():
    if predictor is None:
        return {"error": "Server not ready"}
    cache_metrics = predictor.get_cache_metrics()
    queue_size = predictor.queue.qsize() if predictor else 0
    return {
        "cache": cache_metrics,
        "queue_size": queue_size,
    } """

@app.get("/metrics")
async def get_metrics():
    if predictor is None:
        return {"error": "Server not ready"}
    metrics = predictor.get_metrics()
    return {
        "prediction_cache": metrics["prediction_cache"],
        "embedding_cache": metrics["embedding_cache"],
        "queue_size": metrics["queue_size"],
    }