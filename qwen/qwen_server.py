"""
FastAPI server wrapping qwen_inference.py.

Arin calls this from prompt-lens/app/api/analyze/route.ts.

Endpoints:
  GET  /health          → { status: "ok" }
  POST /predict         → { rating: 1-5 }

Request body for /predict:
  {
    "review":   "...",
    "mode":     "zero-shot" | "few-shot",
    "examples": [{ "review": "...", "rating": 1-5 }, ...]   // optional
  }

Start with:
  pip install fastapi uvicorn
  HF_TOKEN=<your_token> python qwen_server.py
"""

import os
from contextlib import asynccontextmanager
from dotenv import load_dotenv

load_dotenv()
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

import qwen_inference as qi


@asynccontextmanager
async def lifespan(app: FastAPI):
    qi.load_model(hf_token=os.getenv("HF_TOKEN"))
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
        result = qi.predict_few_shot(req.review, examples)
    else:
        result = qi.predict_zero_shot(req.review)

    return {"rating": result["rating"]}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
