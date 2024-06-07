import torch
from torch import nn
from transformers import BertModel


class SentimentBert(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model_setup()

        self.dropout = nn.Dropout(config['dropout'])
        self.classifier = Classifier()

    def model_setup(self):
        self.encoder = BertModel.from_pretrained("bert-large-uncased")

        # Reinitialize the last two layers of the encoder
        # self._init_weights(self.encoder.encoder.layer[-1])
        # self._init_weights(self.encoder.encoder.layer[-2])

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.encoder.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.encoder.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, inputs):
        encoder_outputs = self.encoder(**inputs)
        last_hidden_states = encoder_outputs.last_hidden_state
        cls_tok = last_hidden_states[:, 0, :]
        dropout_outputs = self.dropout(cls_tok)
        output = self.classifier(dropout_outputs)
        return output


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        input_dim = 1024
        self.top = nn.Linear(input_dim, input_dim)
        self.relu = nn.ReLU(inplace=True)
        self.bottom = nn.Linear(input_dim, 3)

    def forward(self, hidden):
        middle = self.relu(self.top(hidden))
        logit = self.bottom(middle)
        return logit
