"""
Inference module for the custom BERT fine-tuned on Amazon book reviews.

Checkpoint: AgentPhoenix7/SLM-project  (checkpoints/best.pt)
Tokenizer : bert-base-uncased (max_length=256)
Labels    : logit index 0-4  →  star rating 1-5
"""

import math
import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers import BertTokenizer
from huggingface_hub import hf_hub_download

HF_REPO_ID = "AgentPhoenix7/SLM-project"
CKPT_FILENAME = "checkpoints/best.pt"
LOCAL_CKPT = os.path.join(os.path.dirname(__file__), "checkpoints", "best.pt")
MAX_LENGTH = 256

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── model definition (must match bert-finetune-kaggle.ipynb exactly) ──────────

class ModelConfig:
    n_embd     = 768
    n_layer    = 12
    n_head     = 12
    block_size = 512
    vocab_size = 30522
    eps        = 1e-12
    embd_pdrop = 0.1
    num_labels = 5

class TrainingConfig:
    resid_pdrop = 0.1
    attn_pdrop  = 0.1


class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x**3)))


class LayerNorm(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(config.n_embd))
        self.beta  = nn.Parameter(torch.zeros(config.n_embd))
        self.eps   = config.eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var  = x.var(-1, keepdim=True, unbiased=False)
        return self.gamma * (x - mean) / (var + self.eps).sqrt() + self.beta


class BertEmbeddings(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.word_embeddings       = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embeddings   = nn.Embedding(config.block_size, config.n_embd)
        self.token_type_embeddings = nn.Embedding(2, config.n_embd)
        self.LayerNorm             = nn.LayerNorm(config.n_embd, eps=config.eps)
        self.dropout               = nn.Dropout(config.embd_pdrop)

    def forward(self, input_ids: Tensor,
                token_type_ids: Optional[Tensor] = None,
                position_ids:   Optional[Tensor] = None) -> Tensor:
        T = input_ids.shape[1]
        if position_ids is None:
            position_ids = torch.arange(T, device=input_ids.device).unsqueeze(0)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        hidden = (self.word_embeddings(input_ids)
                  + self.position_embeddings(position_ids)
                  + self.token_type_embeddings(token_type_ids))
        return self.dropout(self.LayerNorm(hidden))


class MultiHeadAttention(nn.Module):
    def __init__(self, config: ModelConfig, train_config: TrainingConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_embd    = config.n_embd
        self.num_heads = config.n_head
        self.head_dim  = config.n_embd // config.n_head
        self.W_Q           = nn.Linear(config.n_embd, config.n_embd)
        self.W_K           = nn.Linear(config.n_embd, config.n_embd)
        self.W_V           = nn.Linear(config.n_embd, config.n_embd)
        self.final_linear  = nn.Linear(config.n_embd, config.n_embd)
        self.attn_dropout  = nn.Dropout(train_config.attn_pdrop)
        self.resid_dropout = nn.Dropout(train_config.resid_pdrop)

    def _split(self, x: Tensor) -> Tensor:
        B, T, _ = x.shape
        return x.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

    def _merge(self, x: Tensor) -> Tensor:
        B, H, T, D = x.shape
        return x.transpose(1, 2).contiguous().view(B, T, self.n_embd)

    def forward(self, x: Tensor, attention_mask: Optional[Tensor] = None) -> Tensor:
        q = self._split(self.W_Q(x))
        k = self._split(self.W_K(x))
        v = self._split(self.W_V(x))
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            scores = scores.masked_fill(
                attention_mask[:, None, None, :] == 0, float('-inf')
            )
        probs = self.attn_dropout(F.softmax(scores, dim=-1))
        return self.resid_dropout(self.final_linear(self._merge(torch.matmul(probs, v))))


class FeedForward(nn.Module):
    def __init__(self, config: ModelConfig, train_config: TrainingConfig):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu    = GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(train_config.resid_pdrop)

    def forward(self, x):
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))


class TransformerBlock(nn.Module):
    def __init__(self, modelconfig: ModelConfig, trainconfig: TrainingConfig):
        super().__init__()
        self.ln1  = LayerNorm(modelconfig)
        self.attn = MultiHeadAttention(modelconfig, trainconfig)
        self.ln2  = LayerNorm(modelconfig)
        self.mlp  = FeedForward(modelconfig, trainconfig)

    def forward(self, x, attention_mask=None):
        x = self.ln1(x + self.attn(x, attention_mask))
        x = self.ln2(x + self.mlp(x))
        return x


class BertPooler(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.dense      = nn.Linear(config.n_embd, config.n_embd)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        return self.activation(self.dense(hidden_states[:, 0, :]))


class Bert(nn.Module):
    def __init__(self, modelconfig: ModelConfig, trainingconfig: TrainingConfig):
        super().__init__()
        self.embeddings = BertEmbeddings(modelconfig)
        self.h          = nn.ModuleList([
            TransformerBlock(modelconfig, trainingconfig)
            for _ in range(modelconfig.n_layer)
        ])
        self.pooler     = BertPooler(modelconfig)
        self.classifier = nn.Linear(modelconfig.n_embd, modelconfig.num_labels)
        self.config     = modelconfig
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        x = self.embeddings(input_ids, token_type_ids)
        for block in self.h:
            x = block(x, attention_mask)
        return self.classifier(self.pooler(x))


# ── globals set by load_model() ───────────────────────────────────────────────
_model: Optional[Bert] = None
_tokenizer: Optional[BertTokenizer] = None


def load_model(hf_token: Optional[str] = None) -> None:
    global _model, _tokenizer

    if os.path.exists(LOCAL_CKPT):
        print(f"Loading checkpoint from local path: {LOCAL_CKPT}")
        ckpt_path = LOCAL_CKPT
    else:
        print(f"Local checkpoint not found — downloading from {HF_REPO_ID} …")
        ckpt_path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=CKPT_FILENAME,
            repo_type="model",
            token=hf_token,
        )

    cfg       = ModelConfig()
    train_cfg = TrainingConfig()
    model     = Bert(cfg, train_cfg)

    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    _model     = model
    _tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    print(f"Model loaded on {DEVICE}  (best_val_loss={ckpt.get('best_val_loss', 'n/a')})")


def _predict(review: str) -> int:
    assert _model is not None and _tokenizer is not None, "call load_model() first"
    enc = _tokenizer(
        review,
        max_length=MAX_LENGTH,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        logits = _model(
            enc["input_ids"].to(DEVICE),
            attention_mask=enc["attention_mask"].to(DEVICE),
        )
    return int(logits.argmax(-1).item()) + 1  # 0-4 → 1-5


def predict_zero_shot(review: str) -> dict:
    return {"rating": _predict(review)}


def predict_few_shot(review: str, examples: list) -> dict:
    # Encoder model — few-shot examples don't change inference;
    # the rating comes purely from the fine-tuned weights.
    return {"rating": _predict(review)}
