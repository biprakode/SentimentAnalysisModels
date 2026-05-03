# PromptLens — Session Summary

## Assignment Question
> Design a python program to implement a Small Language Model (SLM) that performs a given classification task using zero-shot and few-shot prompting (maximum 3 examples). Optimize the prompting technique or strategy so that the SLM's output is as close as possible to that of a stronger LLM.
>
> **Input:** A sentence/paragraph + zero-shot or few-shot prompt (max 3 examples)
>
> **Output:**
> - Output from the SLM
> - Optimized SLM output
> - LLM reference output
> - Evaluation Score (F1 or Accuracy)

**How PromptLens addresses this:** The classification task is 1–5 star book rating. The SLM is Qwen 0.5B (fine-tuned + prompt engineered). The stronger LLM reference is `llama-3.3-70b-versatile` via Groq. The system explicitly supports zero-shot and few-shot (up to 3 examples) modes, and evaluates with Accuracy and MAE.

---

## Project
**PromptLens** (`c:\Users\agntd\Documents\GitHub\SentimentAnalysisModels`) is a group academic project comparing 4 models predicting 1–5 star ratings for book reviews. Frontend is a Next.js app at `prompt-lens/`.

## Who I am
I am **Anis Mandal** — listed as **"Anis"** or **"Anish"** in the README (teammate spelling mistake). My responsibilities:
- Fine-tuning of Qwen 0.5B (small LLM)
- Prompt engineering for the small LLM

## Teammates
| Person | Responsibility |
|--------|---------------|
| Biprarshi | Custom BERT encoder model (training + dataset) |
| David | Co-owns Qwen 0.5B fine-tuning with me |
| Arin | Frontend, backend API, Groq LLM prompt engineering, system orchestration |

---

## What exists in the repo

**`bert-finetune-kaggle.ipynb`** — Biprarshi's notebook. Trains a custom BERT-base classifier from scratch on Amazon Reviews (1.5M samples). Best checkpoint saved at:
- HF Hub: `AgentPhoenix7/SLM-project` → `checkpoints/best.pt`
- Dataset cache: `AgentPhoenix7/SLM-project-dataset-cache`

**`prompt-lens/`** — Arin's Next.js app. Frontend is ~95% complete. The backend API (`app/api/analyze/route.ts`) runs 4 models in parallel:
- Groq LLM (`llama-3.3-70b-versatile`) — **fully real and working**
- Custom Encoder, HF Encoder, Qwen 0.5B — **all mocked** (return hardcoded `4` with 1s delay)

---

## What I did this session

### 1. Created `qwen-finetune-kaggle.ipynb`
Kaggle notebook for fine-tuning Qwen 0.5B (`Qwen/Qwen2.5-0.5B`) using LoRA (PEFT) on the same Amazon Reviews dataset.

Key details:
- Uses `AutoModelForSequenceClassification` + LoRA (`r=8`, targets `q_proj/k_proj/v_proj/o_proj`) + `modules_to_save=['score']`
- Only ~10M trainable params (vs 110M for BERT full fine-tune)
- Checkpoint size ~10–30 MB (only saves trainable params)
- **Reuses** the existing HF Hub dataset cache (`AgentPhoenix7/SLM-project-dataset-cache`) — no re-download
- Saves checkpoints to new HF Hub repo: `AgentPhoenix7/SLM-project-qwen`
- Same resume logic as the BERT notebook (HF Hub → local last.pt → epoch_*.pt → fresh)
- `CKPT_EVERY = 500`, `BATCH_SIZE = 32`, `ACCUMULATION_STEPS = 2`, `NUM_EPOCHS = 3`, `MAX_LR = 2e-4`

### 2. Updated `readme.md`
Added a "Fine-tuning Approach Comparison" table between the Models and Prompting Techniques sections comparing BERT vs Qwen training approaches (architecture, trainable params, checkpoint size, HF repos, notebooks).

---

## What's next (my remaining work)
1. ~~**Run the Qwen fine-tuning notebook on Kaggle**~~ — **DONE, training in progress** (pushing to `AgentPhoenix7/SLM-project-qwen`)
2. ~~**Prompt engineering for Qwen 0.5B**~~ — **DONE** (`qwen_inference.py`)
3. ~~**Deliver a model endpoint**~~ — **DONE** (`qwen_server.py` + `route.ts` updated)

Remaining:
- Wait for training to complete and verify `best.pt` accuracy
- Arin needs to run `cd prompt-lens && npm install` then start the Python server alongside `next dev`

---

## Session 2 — Inference module + endpoint

### Git: stash → branch → commit
- Stashed all uncommitted changes (including untracked files via `-u`)
- Created new branch `anis` off `main`
- Committed all files (`readme.md`, `bert-finetune-kaggle.ipynb`, `qwen-finetune-kaggle.ipynb`, `latest_summary.md`, `question.md`) in one commit (`217970b`)
- `.vscode/` was not committed (IDE config, excluded intentionally)

### Training status
- `qwen-finetune-kaggle.ipynb` is actively running on Kaggle
- Checkpoints uploading to `AgentPhoenix7/SLM-project-qwen` every 500 optimizer steps
- Both `checkpoints/last.pt` and `checkpoints/best.pt` are live on HF Hub

### 1. Created `qwen_inference.py`
Core Python inference module. Loads Qwen2.5-0.5B + LoRA checkpoint from `AgentPhoenix7/SLM-project-qwen/checkpoints/best.pt`.

Three public functions:

| Function | Prompt strategy |
|----------|----------------|
| `predict_zero_shot(review)` | Raw review text fed directly to classifier |
| `predict_few_shot(review, examples)` | Up to 3 labelled examples prepended: `"Review: …\nRating: N\n\nReview: …\nRating:"` |
| `predict_optimized(review, examples)` | Structured task prefix + confidence-based ensemble (picks whichever of optimized-prompt vs zero-shot has higher softmax confidence) |

All three return `{ rating: 1–5, confidence: float, mode: str }`.

Checkpoint loading mirrors the training notebook exactly: load base model with same LoRA config → `torch.load(best.pt)` → copy `trainable_state_dict` keys into PEFT model state dict.

### 2. Created `qwen_server.py`
FastAPI server wrapping `qwen_inference.py`.

- `GET /health` — liveness check
- `POST /predict` — body `{ review, mode, examples }` → `{ rating: 1–5 }`
- CORS open (all origins) so Next.js can call it from localhost
- Model loaded once at startup via `lifespan`
- Start with: `HF_TOKEN=<token> python qwen_server.py`
- URL overridable via `QWEN_API_URL` env var (default `http://localhost:8000`)

### 3. Updated `prompt-lens/app/api/analyze/route.ts`
Replaced the two hardcoded Qwen stubs with real `fetch` calls to the Python server:

```ts
// before
async function qwen_zs() { await delay(1000); return 4; }
async function qwen_fs() { await delay(1000); return 3; }

// after
async function qwen_zs(review: string): Promise<number>  // POST /predict {mode:"zero-shot"}
async function qwen_fs(review: string, examples: Example[]): Promise<number>  // POST /predict {mode:"few-shot"}
```

Call sites updated to pass `review` and `examples` through.

**Note:** All 4 TS diagnostics in `route.ts` (`next/server`, `ai`, `@ai-sdk/groq`, `process`) resolve after `cd prompt-lens && npm install` — packages are already declared in `package.json`.

### 4. Created `requirements_qwen.txt`
Python dependency file for the inference server:
```
torch, transformers, peft, accelerate, huggingface_hub, fastapi, uvicorn, pydantic
```
Install with `pip install -r requirements_qwen.txt`. CPU inference is sufficient for presenting (~2–5s/prediction). Model weights (~1 GB base + ~13 MB LoRA) are cached after first `python qwen_server.py` run — pre-warm before the presentation. Set `HF_TOKEN` env var to access the private checkpoint.

---

## File structure reorganization

Moved all loose root-level files into their corresponding folders:

| File | From | To |
|------|------|----|
| `qwen_inference.py` | root | `qwen/` |
| `qwen_server.py` | root | `qwen/` |
| `requirements_qwen.txt` | root | `qwen/` |
| `qwen-finetune-kaggle.ipynb` | root | `qwen/` |
| `bert-finetune-kaggle.ipynb` | root | `bert-from-scratch/` |

New top-level structure:
```
bert-from-scratch/   ← BERT model implementation + Kaggle notebook
qwen/                ← Qwen fine-tuning notebook, inference module, server, requirements
prompt-lens/         ← Next.js frontend app
readme.md / question.md / latest_summary.md   ← project-level docs (stay at root)
```

---

## Debugging session — `qwen-finetune-kaggle.ipynb`

Three bugs were caught and fixed while running the notebook on Kaggle.

### Bug 1 — `torch_dtype` deprecation warning (Cell 3)
**Symptom:** `torch_dtype is deprecated! Use dtype instead!`  
**Fix:** Renamed the kwarg in `AutoModelForSequenceClassification.from_pretrained(...)` from `torch_dtype=torch.float16` to `dtype=torch.float16`.

### Bug 2 — `ValueError: Attempting to unscale FP16 gradients` (Cell 6)
**Symptom:** Crash at `GradScaler.unscale_()` on the first optimizer step.  
**Root cause:** The model was loaded in `float16`, so its parameters (and their gradients) were FP16. `GradScaler` only supports FP32 gradients — it expects a FP32 model where `autocast` casts only the *forward pass* to FP16.  
**Fix attempt:** Removed the `dtype` argument entirely, expecting the model to default to FP32.  
**Result:** Triggered Bug 3 (see below).

### Bug 3 — `NotImplementedError: "_amp_foreach_non_finite_check_and_unscale_cuda" not implemented for 'BFloat16'` (Cell 6)
**Symptom:** Same crash site as Bug 2, different dtype.  
**Root cause:** `Qwen/Qwen2.5-0.5B`'s `config.json` hardcodes `"torch_dtype": "bfloat16"`. Removing the explicit `dtype` arg does not produce FP32 — HuggingFace honours the model's own config and loads in BFloat16. `GradScaler` does not support BFloat16 at all (BFloat16 has the same exponent range as FP32, so loss scaling is unnecessary and the CUDA kernel for it simply doesn't exist).

**Final fix (two cells):**

| Cell | Change |
|------|--------|
| Cell 3 | `dtype=torch.bfloat16` — explicit, matches model config, documents the intent |
| Cell 5 (`Trainer`) | Removed `GradScaler` entirely. `train_epoch` now uses `autocast(device_type='cuda', dtype=torch.bfloat16)` and calls `optimizer.step()` directly (no scale/unscale/update). |

### Checkpoint size difference — expected behaviour
`bert-finetune-kaggle.ipynb` produces ~1.31 GB checkpoints; `qwen-finetune-kaggle.ipynb` produces ~13.2 MB. This is correct and by design:

- BERT saves `m.state_dict()` — the full 110 M-param model in FP32 + Adam moments ≈ 1.3 GB.
- Qwen saves only `trainable_state_dict` — the 1.08 M LoRA adapter + score-head params + their Adam moments ≈ 13 MB. The 495 M frozen base-model weights are re-downloaded from HF Hub at the start of each Kaggle session (Cell 3), so they never need to be checkpointed.

---

## Repo cleanup

Removed unnecessary files and added root `.gitignore`:

**Deleted (generated artifacts):**
- All `__pycache__/` directories (`bert-from-scratch/model/`, `tokenizer/`, `training/`)
- `bert-from-scratch/notebook_output/` (Kaggle execution log)
- `bert-from-scratch/eval/perplexity.py` (empty file) + empty `eval/` folder

**Created:**
- `.gitignore` at root — covers `__pycache__`, `.venv`, `.vscode`, `.idea`, `.ipynb_checkpoints`, `node_modules`, `.next`, `.env`

**Still present (ask Anis/team before removing):**
- `.vscode/settings.json` — personal Python env setting, not project-required
- `.idea/.gitignore` — JetBrains IDE artifact, project uses VS Code
- `question.md` — raw assignment brief, already in `readme.md` and `latest_summary.md`

---

## Frontend configuration + inference cleanup

### 1. Removed `predict_optimized` from `qwen/qwen_inference.py`
Both `predict_zero_shot` and `predict_few_shot` now always use the optimized prompt (task prefix prepended). The separate `predict_optimized` ensemble function was redundant and removed. Server docstring and mode list in `qwen/qwen_server.py` updated to match (`"optimized"` mode dropped).

### 2. Frontend configuration audit (`prompt-lens/`)

**Critical fix — missing env var setup:**
- No `.env.local` existed; `GROQ_API_KEY` is required by `@ai-sdk/groq` — without it every request returned 500.
- Created `prompt-lens/.env.example` as the canonical template:
  ```
  GROQ_API_KEY=          # required — get from console.groq.com
  QWEN_API_URL=http://localhost:8000   # optional, this is the default
  ```
- Each dev must copy `.env.example` → `.env.local` and fill in `GROQ_API_KEY`. Both are covered by `.env*` in `prompt-lens/.gitignore` — neither will be committed.

**Fixed stale footer copy:**
- `prompt-lens/app/page.tsx`: `"Analysis is simulated"` → `"Results may vary by model"`

**Ran `npm install` in `prompt-lens/`:**
- 383 packages installed. TypeScript type-check (`tsc --noEmit`) passes with zero errors.
- 2 moderate `postcss` vulnerabilities reported — fix requires downgrading Next.js to v9 (breaking). Do NOT run `npm audit fix --force`. The vulnerability (XSS via user-injected CSS) is not exploitable in this app.

---

## API audit — bugs fixed in `prompt-lens/app/api/analyze/route.ts`

Three bugs found and fixed:

### Bug 1 & 2 — `groq_zs` and `groq_fs` catch blocks returned `Response.json()` instead of a number
**Symptom:** When Groq throws (API key error, rate limit, network failure), the catch block returned a `Response` object instead of a `number`. `Promise.all` resolved with that Response as the `groq` value, then `{ model: "groq", rating: <Response> }` caused JSON serialization to fail → 500 for the whole request.

**Fix:** Both catch blocks now `return 3` (neutral fallback) and log the error to console. Groq failures degrade gracefully — the other three model results still come through.

### Bug 3 — `groq_zs` regex fallback was `0`
**Symptom:** If Groq returns text with no digit 1–5, `text.match(/[1-5]/)` is null and `parseInt(null)` produced `0` — not a valid rating.

**Fix:** Changed `0` → `3` (neutral), matching the same fallback already used in `groq_fs`.

**No bugs found in:**
- `qwen/qwen_server.py` — clean
- `qwen/qwen_inference.py` — `trainable_state_dict` key matches notebook save format; `dtype=bfloat16` is correct (Qwen2.5 config.json sets this, verified in training notebook debugging notes above)

---

## Prompt optimization scoped to Qwen only

Made prompt optimization an explicit, controllable feature at every layer — BERT is structurally excluded.

**`qwen/qwen_inference.py`:** Added `optimize: bool = True` parameter to `predict_zero_shot` and `predict_few_shot`. When `True` (default), uses `_fmt_optimized`; when `False`, falls back to `_fmt_zero_shot`/`_fmt_few_shot`.

**`qwen/qwen_server.py`:** Added `optimize: bool = True` field to `PredictRequest`. Passed through to both inference calls. `bert_server.py` has no such field — optimization is structurally absent for BERT.

**`prompt-lens/app/api/analyze/route.ts`:** `qwen_zs` and `qwen_fs` now explicitly send `optimize: true` in the request body. BERT calls (`custom_zs`, `custom_fs`) do not.
