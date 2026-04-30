from torch import nn

from model.ModelConfig import ModelConfig


class BertPooler(nn.Module):
    def __init__(self , config:ModelConfig):
        super().__init__()
        self.dense = nn.Linear(config.n_embd , config.n_embd)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        cls_token = hidden_states[:, 0, :] # only extracting first cls token of every sentence
        return self.activation(self.dense(cls_token))
