"""
FastAPI inference server for the custom BERT fine-tuned on Amazon book reviews.

Called from prompt-lens/app/api/analyze/route.ts via BERT_API_URL.

Endpoints:
  GET  /health   → { "status": "ok" }
  POST /predict  → { "rating": 1-5 }

Request body for /predict:
  {
    "review":   "...",
    "mode":     "zero-shot" | "few-shot",
    "examples": [{ "review": "...", "rating": 1-5 }, ...]   // ignored for encoder
  }

Start with:
  pip install -r requirements_bert.txt
  HF_TOKEN=<your_token> python bert_server.py
"""

import os
from contextlib import asynccontextmanager
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

load_dotenv()

import bert_inference as bi


@asynccontextmanager
async def lifespan(app: FastAPI):
    bi.load_model(hf_token=os.getenv("HF_TOKEN"))
    yield


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class Example(BaseModel):
    review: str
    rating: int = Field(ge=1, le=5)


class PredictRequest(BaseModel):
    review: str
    mode: str = "zero-shot"
    examples: Optional[list[Example]] = []


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(req: PredictRequest):
    if not req.review.strip():
        raise HTTPException(status_code=400, detail="review cannot be empty")

    examples = [e.model_dump() for e in (req.examples or [])]

    if req.mode == "few-shot" and examples:
        result = bi.predict_few_shot(req.review, examples)
    else:
        result = bi.predict_zero_shot(req.review)

    return {"rating": result["rating"]}


if __name__ == "__main__":
    port = int(os.getenv("BERT_PORT", 8001))
    uvicorn.run(app, host="0.0.0.0", port=port)
