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
1. **Run the Qwen fine-tuning notebook on Kaggle** (needs `HF_TOKEN` Kaggle Secret — already set from BERT notebook)
2. **Prompt engineering for Qwen 0.5B** — design zero-shot and few-shot prompt templates for the inference path
3. **Deliver a model endpoint** that Arin can plug into the mocked Qwen slot in `prompt-lens/app/api/analyze/route.ts`

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
