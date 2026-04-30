"""
Download bert-base-uncased weights, remap keys to our custom architecture,
load them, and verify with a forward pass.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import torch
from transformers import BertModel, BertTokenizer

from model.Bert import Bert
from model.ModelConfig import ModelConfig
from model.TrainingConfig import TrainingConfig


# ---------------------------------------------------------------------------
# Task 1 — download and print all HF keys
# ---------------------------------------------------------------------------

def get_hf_state_dict(model_name: str = "bert-base-uncased"):
    print(f"Downloading {model_name} ...")
    hf_model = BertModel.from_pretrained(model_name)
    sd = hf_model.state_dict()
    print(f"\n{'='*60}")
    print(f"HuggingFace state dict — {len(sd)} keys:")
    print(f"{'='*60}")
    for k, v in sd.items():
        print(f"  {k:60s}  {tuple(v.shape)}")
    return sd


# ---------------------------------------------------------------------------
# Task 2 — build the HF-key → custom-key remapping dict
# ---------------------------------------------------------------------------

def build_hf_to_custom_map(n_layers: int = 12) -> dict:
    mapping = {}

    # Embeddings — names match exactly (nn.LayerNorm uses 'weight'/'bias')
    for suffix in (
        "embeddings.word_embeddings.weight",
        "embeddings.position_embeddings.weight",
        "embeddings.token_type_embeddings.weight",
        "embeddings.LayerNorm.weight",
        "embeddings.LayerNorm.bias",
    ):
        mapping[suffix] = suffix

    for i in range(n_layers):
        hf = f"encoder.layer.{i}"
        us = f"h.{i}"

        # Attention projections
        mapping[f"{hf}.attention.self.query.weight"] = f"{us}.attn.W_Q.weight"
        mapping[f"{hf}.attention.self.query.bias"]   = f"{us}.attn.W_Q.bias"
        mapping[f"{hf}.attention.self.key.weight"]   = f"{us}.attn.W_K.weight"
        mapping[f"{hf}.attention.self.key.bias"]     = f"{us}.attn.W_K.bias"
        mapping[f"{hf}.attention.self.value.weight"] = f"{us}.attn.W_V.weight"
        mapping[f"{hf}.attention.self.value.bias"]   = f"{us}.attn.W_V.bias"

        # Attention output projection
        mapping[f"{hf}.attention.output.dense.weight"] = f"{us}.attn.final_linear.weight"
        mapping[f"{hf}.attention.output.dense.bias"]   = f"{us}.attn.final_linear.bias"

        # Post-attention LayerNorm  (our custom LN uses gamma/beta)
        mapping[f"{hf}.attention.output.LayerNorm.weight"] = f"{us}.ln1.gamma"
        mapping[f"{hf}.attention.output.LayerNorm.bias"]   = f"{us}.ln1.beta"

        # FFN: intermediate (expansion) and output (projection)
        mapping[f"{hf}.intermediate.dense.weight"] = f"{us}.mlp.c_fc.weight"
        mapping[f"{hf}.intermediate.dense.bias"]   = f"{us}.mlp.c_fc.bias"
        mapping[f"{hf}.output.dense.weight"]       = f"{us}.mlp.c_proj.weight"
        mapping[f"{hf}.output.dense.bias"]         = f"{us}.mlp.c_proj.bias"

        # Post-FFN LayerNorm
        mapping[f"{hf}.output.LayerNorm.weight"] = f"{us}.ln2.gamma"
        mapping[f"{hf}.output.LayerNorm.bias"]   = f"{us}.ln2.beta"

    # Pooler
    mapping["pooler.dense.weight"] = "pooler.dense.weight"
    mapping["pooler.dense.bias"]   = "pooler.dense.bias"

    return mapping


# ---------------------------------------------------------------------------
# Task 3 — load weights
# ---------------------------------------------------------------------------

def load_pretrained_weights(model: Bert, hf_sd: dict, mapping: dict):
    our_sd = model.state_dict()
    loaded, skipped_missing, skipped_shape = [], [], []

    with torch.no_grad():
        for hf_key, our_key in mapping.items():
            if hf_key not in hf_sd:
                skipped_missing.append(hf_key)
                continue
            if our_key not in our_sd:
                skipped_missing.append(f"[our] {our_key}")
                continue

            hf_tensor  = hf_sd[hf_key]
            our_tensor = our_sd[our_key]

            assert hf_tensor.shape == our_tensor.shape, (
                f"Shape mismatch: HF {hf_key} {tuple(hf_tensor.shape)} "
                f"!= ours {our_key} {tuple(our_tensor.shape)}"
            )

            our_sd[our_key].copy_(hf_tensor)
            loaded.append(our_key)

    model.load_state_dict(our_sd)

    print(f"\n{'='*60}")
    print(f"Weight loading summary")
    print(f"{'='*60}")
    print(f"  Loaded : {len(loaded)}")
    print(f"  Skipped (missing): {len(skipped_missing)}")
    if skipped_missing:
        for k in skipped_missing:
            print(f"    - {k}")
    if skipped_shape:
        for k in skipped_shape:
            print(f"    [shape mismatch] {k}")

    return loaded, skipped_missing


# ---------------------------------------------------------------------------
# Task 4 — verify forward pass
# ---------------------------------------------------------------------------

def verify(model: Bert, device: str = "cpu"):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    sample = "The quick brown fox jumps over the lazy dog."
    enc = tokenizer(sample, return_tensors="pt")
    input_ids      = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    model.eval()
    model.to(device)
    with torch.no_grad():
        logits = model(input_ids, attention_mask=attention_mask)

    print(f"\n{'='*60}")
    print(f"Forward pass verification")
    print(f"{'='*60}")
    print(f"  Input shape  : {tuple(input_ids.shape)}")
    print(f"  Logits shape : {tuple(logits.shape)}  (expected (1, 5))")
    assert logits.shape == (1, 5), f"Wrong output shape: {logits.shape}"
    print("  PASSED")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cfg        = ModelConfig()
    train_cfg  = TrainingConfig()
    model      = Bert(cfg, train_cfg)

    hf_sd   = get_hf_state_dict()
    mapping = build_hf_to_custom_map(n_layers=cfg.n_layer)

    print(f"\n{'='*60}")
    print(f"Remapping dict — {len(mapping)} entries")
    print(f"{'='*60}")
    for hf_k, our_k in mapping.items():
        print(f"  {hf_k:60s}  ->  {our_k}")

    loaded, _ = load_pretrained_weights(model, hf_sd, mapping)
    print(f"\n  {len(loaded)} / {len(mapping)} weights loaded.")

    verify(model)
