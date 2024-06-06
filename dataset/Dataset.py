import torch
from torch.utils.data import Dataset

class SentimentAnalysisDataset(Dataset):
    def __init__(self, tokenizer, sentences, labels):
        self.tokenizer = tokenizer
        self.input = self.tokenizer(sentences, return_tensors='pt', padding="max_length", truncation=True, max_length=100)
        self.labels = torch.tensor(labels, data=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        input = {key: self.input[key][index] for key in self.input}
        label = self.labels[index]

        return input, label