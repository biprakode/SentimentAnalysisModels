# PromptLens: Comparative Study of SLMs and LLMs for Amazon Book Review Rating

## Overview

PromptLens is a comparative AI system that analyzes Amazon book reviews and predicts a star rating on a scale of 1 to 5. The project explores the behavior and performance differences between Small Language Models (SLMs) and Large Language Models (LLMs), with a focus on **zero-shot and few-shot prompting techniques**.

Users input a book review, choose a prompting mode, and observe how different models interpret the same input. Predictions are then compared against a user-provided ground truth rating and evaluated using accuracy and mean absolute error.

---

## Objectives

- Implement and fine-tune Small Language Models (SLMs) for sentiment classification
- Experiment with zero-shot and few-shot prompting strategies
- Analyze how prompting style influences model output
- Compare SLM outputs against a strong LLM baseline
- Evaluate model performance using standard metrics

---

## Models

### 1. Custom Encoder SLM — BERT (Fine-tuned)
- BERT-base architecture built from scratch and fully fine-tuned
- Trained on 1.5M Amazon book reviews (`cogsci13/Amazon-Reviews-2023-Books-Review`)
- Served locally via `bert-from-scratch/bert_server.py` (FastAPI, port 8001)
- Checkpoint: `AgentPhoenix7/SLM-project` on HuggingFace Hub

### 2. Optimized Small LLM — Qwen2.5-0.5B (LoRA Fine-tuned)
- Qwen2.5-0.5B decoder fine-tuned with LoRA (PEFT)
- Trained on the same Amazon book review dataset
- Served locally via `qwen/qwen_server.py` (FastAPI, port 8000)
- Checkpoint: `AgentPhoenix7/SLM-project-qwen` on HuggingFace Hub

### 3. Standard LLM — Llama 3.3 70B (Groq)
- `llama-3.3-70b-versatile` accessed via Groq API
- No fine-tuning — relies entirely on prompt engineering
- Serves as the high-capability reference baseline

---

## Fine-tuning Comparison

| | Custom Encoder (BERT) | Small LLM (Qwen 0.5B) |
|---|---|---|
| **Base architecture** | BERT-base (encoder, built from scratch) | Qwen2.5-0.5B (decoder, pretrained) |
| **Training method** | Full fine-tune | LoRA (PEFT) — adapters only |
| **Trainable parameters** | ~110M (all) | ~10M (LoRA r=8 + classifier head) |
| **Dataset** | Amazon Reviews 2023 — Books (1.5M samples) | Same |
| **HF Hub repo** | `AgentPhoenix7/SLM-project` | `AgentPhoenix7/SLM-project-qwen` |
| **Kaggle notebook** | `bert-finetune-kaggle.ipynb` | `qwen/qwen-finetune-kaggle.ipynb` |
| **Inference server** | `bert-from-scratch/bert_server.py` (port 8001) | `qwen/qwen_server.py` (port 8000) |

---

## Prompting Techniques

### Zero-shot
- No examples provided to the model
- Model relies solely on the system instruction and its weights

### Few-shot
- Up to 3 labeled examples are provided before the target review
- Demonstrates how in-context examples shift model behavior
- Applied to all three models; encoder models use examples for UI consistency only

---

## System Workflow

1. User enters an Amazon book review
2. Selects prompting mode (Zero-shot / Few-shot)
3. Optionally provides up to 3 labeled few-shot examples
4. All three models run in parallel and return a 1–5 rating
5. User submits their own rating as ground truth
6. System evaluates predictions and displays accuracy / MAE per model

---

## Project Structure

```
SentimentAnalysisModels/
├── prompt-lens/              # Next.js frontend + API routes
│   ├── app/
│   │   ├── page.tsx          # Main input page (review + mode toggle)
│   │   ├── output/page.tsx   # Results page (model cards + metrics)
│   │   └── api/analyze/route.ts  # Calls all three model servers
│   └── components/
├── bert-from-scratch/        # Custom BERT model + inference server
│   ├── bert-finetune-kaggle.ipynb
│   ├── bert_inference.py     # Model definition + checkpoint loading
│   ├── bert_server.py        # FastAPI server (port 8001)
│   └── requirements_bert.txt
├── qwen/                     # Qwen2.5-0.5B LoRA model + inference server
│   ├── qwen-finetune-kaggle.ipynb
│   ├── qwen_inference.py
│   ├── qwen_server.py        # FastAPI server (port 8000)
│   └── requirements_qwen.txt
├── examples/                 # Sample book review inputs
│   ├── zero-shot/
│   └── few-shot/
└── package.json              # Root scripts: setup + dev
```

---

## Getting Started

### Prerequisites
- Node.js 18+
- Python 3.10+
- Groq API key

### Setup

```bash
npm run setup
```

This installs all Node and Python dependencies and downloads `best.pt` for the BERT model from HuggingFace Hub.

### Environment Variables

Copy the example files and fill in your keys:

```bash
cp prompt-lens/.env.example prompt-lens/.env.local
cp bert-from-scratch/.env.example bert-from-scratch/.env
cp qwen/.env.example qwen/.env
```

**`prompt-lens/.env.local`**
```
GROQ_API_KEY=your_groq_api_key
QWEN_API_URL=http://localhost:8000   # optional
BERT_API_URL=http://localhost:8001   # optional
```

**`bert-from-scratch/.env`**
```
HF_TOKEN=your_hf_token
BERT_PORT=8001
```

**`qwen/.env`**
```
HF_TOKEN=your_hf_token
```

### Run

```bash
npm run dev
```

Starts three processes concurrently:
| Process | Color | URL |
|---|---|---|
| Next.js frontend | cyan | http://localhost:3000 |
| Qwen inference server | yellow | http://localhost:8000 |
| BERT inference server | magenta | http://localhost:8001 |

---

## Evaluation Metrics

- **Accuracy** — exact match between predicted and actual rating
- **Mean Absolute Error (MAE)** — average deviation from user's rating

---

## Tech Stack

- **Frontend:** Next.js, React, Tailwind CSS v4, TypeScript
- **API Routes:** Next.js server functions (`@ai-sdk/groq`)
- **Inference Servers:** FastAPI + Uvicorn (Python)
- **ML:** PyTorch, HuggingFace Transformers, PEFT
- **LLM API:** Groq (`llama-3.3-70b-versatile`)

---

## Team Contributions

- **Biprarshi** — Custom BERT encoder: architecture, training, dataset pipeline
- **Anis & David** — Qwen 0.5B: LoRA fine-tuning, prompt engineering
- **Arin** — Frontend, backend API integration, Groq prompt engineering, system orchestration
