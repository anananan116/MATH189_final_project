from model.model import SentimentBert
from trainer.Trainer import Trainer
from dataset.Dataset import SentimentAnalysisDataset
import pandas as pd
import torch
from transformers import BertTokenizer
import yaml

if __name__ == "__main__":
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)
    tokenizer = BertTokenizer.from_pretrained("bert-large-uncased")
    model = SentimentBert(config)
    train_data = pd.read_csv("data/cleaned_train.csv")
    val_data = pd.read_csv("data/cleaned_val.csv")
    test_data = pd.read_csv("data/cleaned_test.csv")
    train_dataset = SentimentAnalysisDataset(tokenizer, train_data['text'], train_data['label'])
    val_dataset = SentimentAnalysisDataset(tokenizer, val_data['text'], val_data['label'])
    test_dataset = SentimentAnalysisDataset(tokenizer, test_data['text'], test_data['label'])
    trainer = Trainer(config['trainer'], model)
    trainer.train(train_dataset, val_dataset)
    mse, output = trainer.evaluate(test_dataset)
    with open(f"{trainer.save_location}/results_output.txt", "w") as file:
        file.write(f"Mean Squared Error: {mse}\n")
        file.write(f"Output: {output}")