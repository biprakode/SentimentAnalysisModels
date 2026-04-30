from torch import nn
from torch.nn import functional as F
import torch
from model.ModelConfig import ModelConfig
from model.layernorm import LayerNorm
from model.attention import MultiHeadAttention
from model.feedforward import FeedForward
from model.TrainingConfig import TrainingConfig

class TransformerBlock(nn.Module):
    def __init__(self, modelconfig:ModelConfig , trainconfig:TrainingConfig):
        super().__init__()
        self.ln1 = LayerNorm(modelconfig)
        self.attn = MultiHeadAttention(modelconfig , trainconfig)
        self.ln2 = LayerNorm(modelconfig)
        self.mlp = FeedForward(modelconfig , trainconfig)

    def forward(self, x , attention_mask = None):
        x = self.ln1(x  + self.attn(x , attention_mask)) # post LayerNorm
        x = self.ln2(x + self.mlp(x))
        return x

