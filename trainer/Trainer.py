import os
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader
import torch
from transformers.optimization import get_scheduler
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
import pickle

def LLRD_optimizer(model):
    parameters = []
    named_parameters = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    init_lr = 0.0001
    decay_rate = 0.85
    lr = init_lr

    # Pooler and classification layers
    params_0 = [
        p
        for n, p in named_parameters
        if ("pooler" in n or "classifier" in n) and any(nd in n for nd in no_decay)
    ]
    params_1 = [
        p
        for n, p in named_parameters
        if ("pooler" in n or "classifier" in n) and not any(nd in n for nd in no_decay)
    ]

    parameters.append({"params": params_0, "lr": init_lr, "weight_decay": 0.0})
    parameters.append({"params": params_1, "lr": init_lr, "weight_decay": 0.01})

    # 12 Hidden layers
    for layer in range(23, -1, -1):
        params_0 = [
            p
            for n, p in named_parameters
            if f"encoder.layer.{layer}." in n and any(nd in n for nd in no_decay)
        ]
        params_1 = [
            p
            for n, p in named_parameters
            if f"encoder.layer.{layer}." in n and not any(nd in n for nd in no_decay)
        ]

        parameters.append({"params": params_0, "lr": lr, "weight_decay": 0.0})
        parameters.append({"params": params_1, "lr": lr, "weight_decay": 0.01})

        lr *= decay_rate

    # Embeddings layer
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

    parameters.append({"params": params_0, "lr": lr, "weight_decay": 0.0})
    parameters.append({"params": params_1, "lr": lr, "weight_decay": 0.01})

    return torch.optim.AdamW(parameters)

class Trainer(object):
    def __init__(self, trainer_config, model):
        self.total_steps = trainer_config['steps']
        self.batch_size = trainer_config['batch_size']
        self.learning_rate = trainer_config['lr']
        self.eval_batch_size = trainer_config['eval_batch_size']
        self.optimizer = LLRD_optimizer(model)
        self.model = model
        self.device = trainer_config['device']
        model.to(trainer_config['device'])
        self.scheduler = get_scheduler(
        name="cosine",
        optimizer=self.optimizer,
        num_warmup_steps=trainer_config['warmup_steps'],
        num_training_steps=self.total_steps,
        )
        model.train()
        print(model)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total number of parameters: {total_params}")
        self.writer = SummaryWriter(log_dir=f"./logs/exp_{trainer_config['exp_id']}")
        self.save_location = f"./results/exp_{trainer_config['exp_id']}"
        self.no_output = False
        if not os.path.exists(self.save_location):
            os.makedirs(self.save_location)
        self.expid = trainer_config['exp_id']
        self.scaler = GradScaler(init_scale=2**14)
        self.patience = trainer_config['patience']
        self.auto_save_epochs = trainer_config['auto_save_epochs']
        self.epochs_per_eval = trainer_config['epochs_per_eval']
        self.loss = torch.nn.CrossEntropyLoss()
        
    def train(self, train_dataset, validation_dataset):
        
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)
        print(f"Total number of epochs{int(np.ceil(self.total_steps / len(train_dataloader)))}")
        best_epoch = 0
        best_mse = float('inf')
        for epoch in range(int(np.ceil(self.total_steps / len(train_dataloader)))):
            if not self.no_output:
                progress_bar = tqdm(range(len(train_dataloader)))
            total_loss = 0.0
            batch_num = 0
            self.model.train()
            for batch in train_dataloader:
                self.optimizer.zero_grad()
                input = {k: v.to(self.device) for k,v in batch[0].items()}
                labels = batch[1].to(self.device)
                with autocast():
                    outputs = self.model(input)
                    loss = self.loss(outputs, labels)
                # Scale the loss as necessary
                self.scaler.scale(loss).backward()
                clip_grad_norm_(self.model.parameters(), 1.0)
                # Update optimizer and scheduler using scaled gradients
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                total_loss += loss.item()
                batch_num += 1
                if not self.no_output:
                    progress_bar.set_description(f"Epoch {epoch+1}, Loss: {(total_loss/batch_num):.4f}")
                    progress_bar.update(1)
            avg_loss = total_loss / len(train_dataloader)
            self.writer.add_scalar('Loss/training_loss', avg_loss, epoch)
            if not self.no_output:
                progress_bar.close()
            torch.cuda.empty_cache()
            if (epoch + 1) % self.epochs_per_eval == 0:
                with torch.no_grad():
                    mse = self.evaluate(validation_dataset)
                    self.writer.add_scalar('Loss/Validation_loss', mse, epoch)
                    if mse < best_mse:
                        best_mse = mse
                        best_epoch = epoch
                        torch.save(self.model.state_dict(), self.save_location + f"/best_model.pt")
                    if epoch - best_epoch > self.patience:
                        break
            torch.cuda.empty_cache()
        self.model.load_state_dict(torch.load(self.save_location + f"/best_model.pt"))
        self.writer.close()
        
    def evaluate(self, eval_dataset, return_output = False):
        eval_dataloader = DataLoader(eval_dataset, batch_size=self.eval_batch_size, shuffle=False, num_workers=4)
        self.model.eval()
        losses = []
        accs = []
        if return_output:
            print("loading best model")
            self.model.load_state_dict(torch.load(self.save_location + f"/best_model.pt"))
            outputs = []
        for batch in tqdm(eval_dataloader):
            input = {k: v.to(self.device) for k,v in batch[0].items()}
            labels = batch[1].to(self.device)
            output = self.model(input)
            max_index = torch.argmax(output, dim=1)
            acc = (max_index == labels).sum().item() / self.eval_batch_size
            accs.append(acc)
            if return_output:
                print(max_index.cpu().detach().numpy())
                outputs.extend(output.cpu().detach().numpy())
            loss = self.loss(output, labels)
            losses.append(loss.item())
        mean_loss = np.mean(losses)
        acc = np.mean(accs)
        self.model.train()
        print(f"Validation Loss: {mean_loss}")
        print(f"Validation Accuracy: {acc}")
        if return_output:
            return mean_loss, outputs
        return mean_loss