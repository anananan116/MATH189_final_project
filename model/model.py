import torch
from torch import nn
from torch.nn import functional as F
from transformers import BertModel


class IntentModel(nn.Module):
    def __init__(self, args, tokenizer, target_size):
        super().__init__()
        self.tokenizer = tokenizer
        self.model_setup(args)
        self.target_size = target_size

        # task1: add necessary class variables as you wish.
        self.optimizer = None
        self.scheduler = None

        # task2: initilize the dropout and classify layers
        self.dropout = nn.Dropout(p=args.drop_rate)
        self.classify = Classifier(args, target_size)

    def model_setup(self, args):
        print(f"Setting up {args.model} model")

        # task1: get a pretrained model of 'bert-base-uncased'
        self.encoder = BertModel.from_pretrained("bert-base-uncased")
        self.encoder.resize_token_embeddings(len(self.tokenizer))  # transformer_check

    def forward(self, inputs, targets):
        """
        task1:
            feeding the input to the encoder,
        task2:
            take the last_hidden_state's <CLS> token as output of the
            encoder, feed it to a drop_out layer with the preset dropout rate in the argparse argument,
        task3:
            feed the output of the dropout layer to the Classifier which is provided for you.
        """
        encoder_outputs = self.encoder(**inputs)
        last_hidden_states = encoder_outputs.last_hidden_state
        cls_tok = last_hidden_states[:, 0, :]
        dropout_outputs = self.dropout(cls_tok)
        classifier_outputs = self.classify(dropout_outputs)
        return classifier_outputs


class Classifier(nn.Module):
    def __init__(self, args, target_size):
        super().__init__()
        input_dim = args.embed_dim
        self.top = nn.Linear(input_dim, args.hidden_dim)
        self.relu = nn.ReLU(inplace=True)
        self.bottom = nn.Linear(args.hidden_dim, 1)

    def forward(self, hidden):
        middle = self.relu(self.top(hidden))
        logit = self.bottom(middle)
        return logit


class CustomModel(IntentModel):
    def __init__(self, args, tokenizer, target_size):
        super().__init__(args, tokenizer, target_size)
        if args.reinit_n_layers > 0:
            self.reinit_n_layers = args.reinit_n_layers
            self.reinitilization()
        # task1: use initialization for setting different strategies/techniques to better fine-tune the BERT model

    def LLRD_optimizer(self, args):
        parameters = []
        named_parameters = list(self.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        init_lr = args.learning_rate
        decay_rate = args.LLRD
        lr = init_lr
        params_0 = [
            p
            for n, p in named_parameters
            if ("pooler" in n or "classify" in n) and any(nd in n for nd in no_decay)
        ]
        params_1 = [
            p
            for n, p in named_parameters
            if ("pooler" in n or "classify" in n)
            and not any(nd in n for nd in no_decay)
        ]

        head_params = {"params": params_0, "lr": init_lr, "weight_decay": 0.0}
        parameters.append(head_params)

        head_params = {"params": params_1, "lr": init_lr, "weight_decay": 0.01}
        parameters.append(head_params)

        # === 12 Hidden layers ==========================================================

        for layer in range(11, -1, -1):
            params_0 = [
                p
                for n, p in named_parameters
                if f"encoder.layer.{layer}." in n and any(nd in n for nd in no_decay)
            ]
            params_1 = [
                p
                for n, p in named_parameters
                if f"encoder.layer.{layer}." in n
                and not any(nd in n for nd in no_decay)
            ]

            layer_params = {"params": params_0, "lr": lr, "weight_decay": 0.0}
            parameters.append(layer_params)

            layer_params = {"params": params_1, "lr": lr, "weight_decay": 0.01}
            parameters.append(layer_params)

            lr *= decay_rate

        # === Embeddings layer ==========================================================

        params_0 = [
            p
            for n, p in named_parameters
            if "embeddings" in n and any(nd in n for nd in no_decay)
        ]
        params_1 = [
            p
            for n, p in named_parameters
            if "embeddings" in n and not any(nd in n for nd in no_decay)
        ]

        embed_params = {"params": params_0, "lr": lr, "weight_decay": 0.0}
        parameters.append(embed_params)

        embed_params = {"params": params_1, "lr": lr, "weight_decay": 0.01}
        parameters.append(embed_params)

        return torch.optim.AdamW(params=parameters, lr=init_lr)

    def reinitilization(self):
        # Re-init pooler.
        self.encoder.pooler.dense.weight.data.normal_(
            mean=0.0, std=self.encoder.config.initializer_range
        )
        self.encoder.pooler.dense.bias.data.zero_()
        for param in self.encoder.pooler.parameters():
            param.requires_grad = True

        for n in range(self.reinit_n_layers):
            self.encoder.encoder.layer[-(n + 1)].apply(self._init_weight_and_bias)

    def _init_weight_and_bias(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(
                mean=0.0, std=self.encoder.config.initializer_range
            )
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)