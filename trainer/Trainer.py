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
class Trainer(object):
    def __init__(self, trainer_config, model):
        self.total_steps = trainer_config['steps']
        self.batch_size = trainer_config['batch_size']
        self.learning_rate = trainer_config['lr']
        self.eval_batch_size = trainer_config['eval_batch_size']
        self.optimizer = AdamW(model.parameters(), lr=self.learning_rate, weight_decay=trainer_config['weight_decay'])
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
        self.loss = torch.nn.MSELoss()
        
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
                    outputs = self.model(**input)
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
        with open(self.save_location + f"/results.txt", 'w') as f:
            f.write("Best Validation Results:\n")
            for key, value in self.best_performance.items():
                f.write(f"{key}: {value:.4f}\n")
        
    def evaluate(self, eval_dataset, return_output = False):
        eval_dataloader = DataLoader(eval_dataset, batch_size=self.eval_batch_size, shuffle=False, num_workers=4)
        self.model.eval()
        losses = []
        if return_output:
            outputs = []
        for batch in tqdm(eval_dataloader):
            input = {k: v.to(self.device) for k,v in batch[0].items()}
            labels = batch[1].to(self.device)
            output = self.model(**input)
            if return_output:
                outputs.extend(list(output.cpu().detach().numpy()))
            loss = self.loss(outputs, labels)
            losses.append(loss.item())
        mse = np.mean(losses)
        self.model.train()
        print(f"Validation MSE: {mse}")
        if return_output:
            return mse, outputs
        return mse