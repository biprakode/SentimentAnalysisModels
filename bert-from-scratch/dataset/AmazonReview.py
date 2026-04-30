import torch
from datasets import load_dataset, Dataset
from transformers import BertTokenizer


class AmazonReview(Dataset):
    def __init__(self, hf_dataset , tokenizer):
        self.dataset =  hf_dataset
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        text = item['text']
        label = int(item['rating']) - 1

        tokens = self.tokenizer(text, max_length=256, padding='max_length', truncation=True, return_tensors='pt')

        return {
            'input_ids': tokens['input_ids'].squeeze(0),      # (1,256) → (256,)
            'attention_mask': tokens['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long)
        }