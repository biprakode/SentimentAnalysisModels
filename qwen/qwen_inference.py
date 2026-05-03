"""
Qwen2.5-0.5B inference module for PromptLens.

Loads the LoRA fine-tuned checkpoint from HF Hub and exposes two prediction
modes:
  - zero_shot  : optimized task prefix + raw review text
  - few_shot   : optimized task prefix + up to 3 labelled examples
"""

import os
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import LoraConfig, get_peft_model, TaskType
from huggingface_hub import hf_hub_download, login

MODEL_NAME = "Qwen/Qwen2.5-0.5B"
HF_REPO_ID = "AgentPhoenix7/SLM-project-qwen"
NUM_LABELS  = 5
MAX_LENGTH  = 256

_LORA_CONFIG = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    modules_to_save=["score"],
    bias="none",
)

_model     = None
_tokenizer = None
_device    = None


def load_model(hf_token: str | None = None, device: str | None = None) -> None:
    global _model, _tokenizer, _device

    if _model is not None:
        return

    if hf_token:
        login(token=hf_token)

    _device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if _tokenizer.pad_token is None:
        _tokenizer.pad_token = _tokenizer.eos_token
    _tokenizer.padding_side = "right"

    base = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        dtype=torch.bfloat16,
    )
    base.config.pad_token_id = _tokenizer.pad_token_id

    model = get_peft_model(base, _LORA_CONFIG)

    ckpt_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename="checkpoints/best.pt",
        repo_type="model",
    )
    ckpt = torch.load(ckpt_path, map_location=_device, weights_only=False)
    sd = model.state_dict()
    for k, v in ckpt["trainable_state_dict"].items():
        if k in sd:
            sd[k].copy_(v.to(_device))
    model.load_state_dict(sd)

    model.to(_device).eval()
    _model = model
    print(f"Model loaded on {_device}")


# ── prompt formatters ─────────────────────────────────────────────────────────

def _fmt_zero_shot(review: str) -> str:
    return review.strip()


def _fmt_few_shot(review: str, examples: list[dict]) -> str:
    parts = [
        f"Review: {ex['review'].strip()}\nRating: {ex['rating']}"
        for ex in examples[:3]
    ]
    parts.append(f"Review: {review.strip()}\nRating:")
    return "\n\n".join(parts)


def _fmt_optimized(review: str, examples: list[dict]) -> str:
    prefix = (
        "Task: Rate this book review from 1 (very negative) "
        "to 5 (very positive) stars.\n\n"
    )
    body = _fmt_few_shot(review, examples) if examples else _fmt_zero_shot(review)
    return prefix + body


# ── core inference ────────────────────────────────────────────────────────────

def _infer(text: str) -> dict:
    enc = _tokenizer(
        text,
        max_length=MAX_LENGTH,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        logits = _model(
            enc["input_ids"].to(_device),
            attention_mask=enc["attention_mask"].to(_device),
        ).logits
    probs      = F.softmax(logits.float(), dim=-1)[0]
    label      = probs.argmax().item()
    return {"rating": label + 1, "confidence": probs[label].item()}


# ── public API ────────────────────────────────────────────────────────────────

def predict_zero_shot(review: str, optimize: bool = True) -> dict:
    text = _fmt_optimized(review, []) if optimize else _fmt_zero_shot(review)
    return {**_infer(text), "mode": "zero-shot"}


def predict_few_shot(review: str, examples: list[dict], optimize: bool = True) -> dict:
    text = _fmt_optimized(review, examples) if optimize else _fmt_few_shot(review, examples)
    return {**_infer(text), "mode": "few-shot"}


