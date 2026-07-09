import os

os.environ["TRANSFORMERS_OFFLINE"] = "1"

from transformers import AutoTokenizer, AutoModel
import torch
from fastapi import FastAPI
from pydantic import BaseModel


class EmbeddingRequest(BaseModel):
    input: str
    model: str


app = FastAPI()

tokenizer = None
model = None

MODEL_PATH = os.getenv("MODEL_PATH", "./models")


@app.on_event("startup")
async def startup_event():
    global tokenizer, model

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
    )

    model = AutoModel.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
    )

    model.eval()

    print(f"Model loaded successfully from {MODEL_PATH}.")


@app.get("/health")
async def health_check():
    return {"status": "ok"}


@app.post("/v1/embeddings")
async def get_embeddings(request: EmbeddingRequest):
    inputs = tokenizer(
        request.input,
        return_tensors="pt",
    )

    with torch.no_grad():
        outputs = model(**inputs)

    return {
        "object": "list",
        "data": [
            {
                "object": "embedding",
                "embedding": outputs.last_hidden_state[:, 0, :]
                .squeeze()
                .tolist(),
                "index": 0,
            }
        ],
        "model": request.model,
    }