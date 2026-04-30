import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from torch.optim import AdamW
from model.Bert import Bert

def configure_optimizer(model : Bert, lr=5e-5, weight_decay=0.01):
    decay_params = []
    not_decay_params = []

    for name , param in model.named_parameters():
        if "bias" in name or "ln" in name:
            not_decay_params.append(param)

        elif 'weight' in name:
            decay_params.append(param)

    optimizer = AdamW([
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': not_decay_params, 'weight_decay': 0.0}
    ], lr=lr, betas=(0.9, 0.95), eps=1e-8)

    return optimizer
