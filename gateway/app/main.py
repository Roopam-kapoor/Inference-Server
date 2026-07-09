from fastapi import FastAPI, Header, HTTPException, Depends, Response
import httpx
import os
from pydantic import BaseModel
import time
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingRequest(BaseModel):
    input: str
    model: str


app = FastAPI()

from prometheus_fastapi_instrumentator import Instrumentator

Instrumentator().instrument(app).expose(app)

API_KEY = os.getenv("API_KEY", "dev-secret-key")
MODEL_SERVER_URL = os.getenv("MODEL_SERVER_URL", "http://localhost:8001")


async def verify_api_key(x_api_key: str = Header(default=None)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")


@app.middleware("http")
async def log_requests(request, call_next):
    start = time.time()
    response = await call_next(request)
    duration = (time.time() - start) * 1000

    log_data = {
        "method": request.method,
        "path": request.url.path,
        "status_code": response.status_code,
        "duration_ms": round(duration, 2),
    }

    logger.info(json.dumps(log_data))
    return response


@app.get("/ready")
async def readiness():
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{MODEL_SERVER_URL}/health")

            if response.status_code == 200:
                return {"status": "ready"}

            return Response(
                content='{"status":"unavailable"}',
                status_code=503,
                media_type="application/json",
            )

    except Exception:
        return Response(
            content='{"status":"unavailable"}',
            status_code=503,
            media_type="application/json",
        )


@app.get("/health")
async def health_check():
    return {"status": "ok"}


@app.post("/v1/embeddings")
async def get_embeddings(
    request: EmbeddingRequest,
    _: str = Depends(verify_api_key),
):
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{MODEL_SERVER_URL}/v1/embeddings",
            json=request.model_dump(),
        )

    return response.json()