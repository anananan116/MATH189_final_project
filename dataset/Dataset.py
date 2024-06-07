import torch
from torch.utils.data import Dataset
def map_to_discrete_labels(label):
    if label == 0.0:
        return 0
    elif label == 5.0:
        return 1
    else:
        return 2
class SentimentAnalysisDataset(Dataset):
    def __init__(self, tokenizer, sentences, labels):
        self.tokenizer = tokenizer
        self.input = self.tokenizer(list(sentences), return_tensors='pt', padding="max_length", truncation=True, max_length=100)
        self.labels = torch.tensor(list(labels.apply(map_to_discrete_labels)), dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        input = {key: self.input[key][index] for key in self.input}
        label = self.labels[index]

        return input, label