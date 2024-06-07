from model.model import SentimentBert
from trainer.Trainer import Trainer
from dataset.Dataset import SentimentAnalysisDataset
import pandas as pd
import torch
from transformers import BertTokenizer
import yaml
import pickle
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config['device'] = device
    trainer = Trainer(config, model)
    trainer.train(train_dataset, val_dataset)
    with torch.no_grad():
        mse, output = trainer.evaluate(test_dataset, return_output=True)
    with open(f"{trainer.save_location}/results.pkl", "wb") as file:
        pickle.dump(output, file)