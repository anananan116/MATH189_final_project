import torch
from torch import nn
from torch.nn import functional as F
from transformers import BertModel


class SentimentBert(nn.Module):
    def __init__(self, config):
        super().__init__()
        #self.tokenizer = tokenizer
        self.model_setup()

        self.dropout = nn.Dropout(config['dropout'])
        self.regressor = Regressor()

    def model_setup(self):
        self.encoder = BertModel.from_pretrained("bert-large-uncased")
        # self.encoder.resize_token_embeddings(len(self.tokenizer))

    def forward(self, inputs):
        encoder_outputs = self.encoder(**inputs)
        last_hidden_states = encoder_outputs.last_hidden_state
        cls_tok = last_hidden_states[:, 0, :]
        dropout_outputs = self.dropout(cls_tok)
        output = self.regressor(dropout_outputs)
        return output


class Regressor(nn.Module):
    def __init__(self):
        super().__init__()
        input_dim = 1024
        self.top = nn.Linear(input_dim, input_dim * 4)
        self.relu = nn.ReLU(inplace=True)
        self.bottom = nn.Linear(input_dim * 4, 1)

    def forward(self, hidden):
        middle = self.relu(self.top(hidden))
        logit = self.bottom(middle)
        return logit
