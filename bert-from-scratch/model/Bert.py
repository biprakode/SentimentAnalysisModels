from torch import nn

from model.BertEmbeddings import BertEmbeddings
from model.BertPooler import BertPooler
from model.ModelConfig import ModelConfig
from model.TrainingConfig import TrainingConfig
from model.block import TransformerBlock


class Bert(nn.Module):
    def __init__(self , modelconfig:ModelConfig , trainingconfig:TrainingConfig):
        super().__init__()
        self.embeddings = BertEmbeddings(modelconfig)
        self.h = nn.ModuleList([TransformerBlock(modelconfig , trainingconfig) for _ in range(modelconfig.n_layer)])
        self.pooler = BertPooler(modelconfig)
        self.classifier = nn.Linear(768 , modelconfig.num_labels) # 5 for amazon review
        self.config = modelconfig
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)


    def forward(self , input_ids , attention_mask= None , token_type_ids = None):
        hidden_states = self.embeddings(input_ids , token_type_ids)

        for block in self.h:
            hidden_states = block(hidden_states , attention_mask)

        pooled = self.pooler(hidden_states)
        logits = self.classifier(pooled)

        return logits





