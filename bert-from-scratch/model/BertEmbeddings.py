import torch
from torch import nn
from model.ModelConfig import ModelConfig
from torch import Tensor
from typing import Optional

from model.layernorm import LayerNorm


class BertEmbeddings(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embeddings = nn.Embedding(config.block_size, config.n_embd)
        self.token_type_embeddings = nn.Embedding(2 , config.n_embd)
        self.LayerNorm = nn.LayerNorm(config.n_embd, eps=config.eps)  # fix 2
        self.dropout = nn.Dropout(config.embd_pdrop)

    def forward(self, input_ids: Tensor , token_type_ids: Optional[Tensor] = None , position_ids : Optional[Tensor] = None) -> Tensor:
        T = input_ids.shape[1]

        if position_ids is None:
            position_ids = torch.arange(T, device=input_ids.device).unsqueeze(0)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids, device=input_ids.device)

        token_embd = self.word_embeddings(input_ids)
        position_embd = self.position_embeddings(position_ids)
        token_type_embd = self.token_type_embeddings(token_type_ids)

        hidden_states = token_embd + position_embd + token_type_embd
        hidden_states = self.LayerNorm(hidden_states)
        return self.dropout(hidden_states)


