# BERT from Scratch

A clean PyTorch implementation of BERT (bert-base-uncased) built from first principles, with pretrained weight loading and fine-tuning on Amazon product reviews for 1–5 star rating classification.

## Architecture

| Component | Detail |
|---|---|
| Layers | 12 transformer blocks |
| Hidden size | 768 |
| Attention heads | 12 |
| FFN size | 3072 (4×) |
| Max sequence length | 512 |
| Vocab size | 30 522 |
| Activation | GELU |
| Normalization | Post-LN (after residual) |
| Pooling | `[CLS]` token → Linear → Tanh |
| Classifier head | Linear(768 → 5) |

All components are implemented from scratch: multi-head bidirectional self-attention (padding mask only, no causal mask), feed-forward network, learned positional embeddings, token-type embeddings, and a custom LayerNorm.

## Project Structure

```
├── model/
│   ├── Bert.py              # top-level model
│   ├── BertEmbeddings.py    # word + position + token-type embeddings
│   ├── BertPooler.py        # [CLS] pooler
│   ├── attention.py         # MultiHeadAttention
│   ├── block.py             # TransformerBlock
│   ├── feedforward.py       # FeedForward (expand → GELU → project)
│   ├── layernorm.py         # custom LayerNorm (gamma/beta)
│   ├── gelu.py              # GELU activation
│   ├── ModelConfig.py       # architecture hyperparameters
│   └── TrainingConfig.py    # training hyperparameters
│
├── dataset/
│   └── AmazonReview.py      # PyTorch Dataset wrapping HF Amazon Reviews
│
├── training/
│   ├── optimizer.py         # AdamW with weight-decay param groups
│   ├── scheduler.py         # linear warmup + cosine decay
│   ├── loss.py              # cross-entropy loss
│   └── trainer.py           # training loop, validation, checkpointing
│
├── load_pretrained.py        # download HF weights, remap keys, verify
└── bert_finetune.ipynb       # self-contained Kaggle notebook
```

## Weight Loading

Pretrained `bert-base-uncased` weights are loaded from HuggingFace and remapped to the custom module names:

| HuggingFace key | Custom key |
|---|---|
| `encoder.layer.{i}.attention.self.query` | `h.{i}.attn.W_Q` |
| `encoder.layer.{i}.attention.self.key` | `h.{i}.attn.W_K` |
| `encoder.layer.{i}.attention.self.value` | `h.{i}.attn.W_V` |
| `encoder.layer.{i}.attention.output.dense` | `h.{i}.attn.final_linear` |
| `encoder.layer.{i}.attention.output.LayerNorm` | `h.{i}.ln1.{gamma,beta}` |
| `encoder.layer.{i}.intermediate.dense` | `h.{i}.mlp.c_fc` |
| `encoder.layer.{i}.output.dense` | `h.{i}.mlp.c_proj` |
| `encoder.layer.{i}.output.LayerNorm` | `h.{i}.ln2.{gamma,beta}` |
| `pooler.dense` | `pooler.dense` |

All 199 pretrained parameters load without skips or shape mismatches.

```python
python load_pretrained.py
# Loaded : 199 / 199
# logits shape: (1, 5)
# forward pass OK
```

## Fine-tuning

The `bert_finetune.ipynb` notebook is designed to run on Kaggle (T4 GPU). It is self-contained — no local file imports.

**Dataset:** `McAuley-Lab/Amazon-Reviews-2023` (`raw_review_Books`, 25% split, ~5 GB)  
**Task:** 5-class sentiment classification (1–5 stars, labels 0–4)  
**Sequence length:** 256 tokens with padding and truncation

Training setup:
- AdamW, lr = 2e-5, weight decay = 0.01 (no decay on LayerNorm and biases)
- Linear warmup (6% of steps) → cosine decay to 1e-6
- Mixed precision (AMP), gradient accumulation ×2, gradient clipping = 1.0
- Early stopping with patience = 5 epochs

## Requirements

```
torch >= 2.0
transformers
datasets == 2.21.0   # 3.x removed support for script-based datasets
```
