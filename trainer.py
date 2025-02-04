import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import wandb
from tqdm import tqdm
from typing import Dict, Any, Tuple

class ModelTrainer:
    def __init__(self, 
                 model: torch.nn.Module, 
                 device: torch.device, 
                 config: Dict[str, Any]):
        self.model = model
        self.device = device
        self.config = config
        
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config.get('weight_decay', 0.01)
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=config.get('scheduler_patience', 2),
            verbose=True
        )
        
    def train_epoch(self, 
                    train_loader: DataLoader) -> Tuple[float, float, float]:
        self.model.train()
        total_loss = 0
        predictions = []
        true_labels = []
        
        progress_bar = tqdm(train_loader, desc='Training')
        
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['label'].to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(input_ids, attention_mask)
            loss = self.criterion(outputs, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            predictions.extend(outputs.argmax(dim=1).cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
            
            progress_bar.set_postfix({'loss': loss.item()})
            
        epoch_loss = total_loss / len(train_loader)
        epoch_accuracy = accuracy_score(true_labels, predictions)
        epoch_f1 = f1_score(true_labels, predictions, average='weighted')
        
        return epoch_loss, epoch_accuracy, epoch_f1
    
    def evaluate(self, 
                val_loader: DataLoader) -> Tuple[float, float, float]:
        self.model.eval()
        total_loss = 0
        predictions = []
        true_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                predictions.extend(outputs.argmax(dim=1).cpu().numpy())
                true_labels.extend(labels.cpu().numpy())
        
        val_loss = total_loss / len(val_loader)
        val_accuracy = accuracy_score(true_labels, predictions)
        val_f1 = f1_score(true_labels, predictions, average='weighted')
        
        return val_loss, val_accuracy, val_f1

class GenerationTrainer(ModelTrainer):
    def __init__(self, 
                 model: torch.nn.Module, 
                 device: torch.device, 
                 config: Dict[str, Any]):
        super().__init__(model, device, config)
        
    def train_epoch(self, train_loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0
        
        progress_bar = tqdm(train_loader, desc='Training')
        
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
            
        return total_loss / len(train_loader)
