"""
Training utilities and loss functions for breast cancer detection.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
from datetime import datetime, timedelta


class CBFocalLoss(nn.Module):
    """Class-Balanced Focal Loss for handling class imbalance."""
    
    def __init__(self, class_counts, beta=0.9999, gamma=2.0, drw=True, total_epochs=50):
        super().__init__()
        self.register_buffer('class_counts', torch.tensor(class_counts, dtype=torch.float32))
        self.beta = beta
        self.gamma = gamma
        self.drw = drw
        self.total_epochs = total_epochs
        self.num_classes = self.class_counts.numel()

    def _compute_class_weights(self, current_epoch=None):
        if self.drw and self.total_epochs is not None:
            use_beta = 0.0 if (current_epoch is not None and current_epoch < (self.total_epochs // 2)) else self.beta
        else:
            use_beta = self.beta
        effective_num = 1.0 - torch.pow(use_beta * torch.ones_like(self.class_counts), self.class_counts)
        weights = (1.0 - use_beta) / torch.clamp(effective_num, min=1e-8)
        weights = weights / weights.sum() * self.num_classes
        return weights

    def forward(self, logits, targets, epoch=None):
        class_weights = self._compute_class_weights(epoch).to(logits.device)
        probs = torch.softmax(logits, dim=1)
        pt = probs.gather(1, targets.view(-1,1)).clamp(min=1e-8, max=1.0)
        focal_factor = torch.pow(1.0 - pt, self.gamma).squeeze(1)
        per_class_weight = class_weights.gather(0, targets)
        loss = - per_class_weight * focal_factor * pt.log().squeeze(1)
        return loss.mean()


def mixup_data(x, y, alpha=0.2):
    """Mixup data augmentation."""
    if alpha is None or alpha <= 0:
        return x, y, y, 1.0
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(loss_fn, pred, y_a, y_b, lam, epoch=None):
    """Mixup criterion for loss calculation."""
    loss_a = loss_fn(pred, y_a, epoch=epoch)
    loss_b = loss_fn(pred, y_b, epoch=epoch)
    return lam * loss_a + (1 - lam) * loss_b


class Trainer:
    """Advanced trainer with progress tracking and mixup."""
    
    def __init__(self, model, device, save_dir='results/models'):
        self.model = model
        self.device = device
        self.save_dir = save_dir
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
    def train_epoch(self, train_loader, criterion, optimizer, mixup_alpha=0.2, epoch=0):
        """Train for one epoch."""
        self.model.train()
        correct = 0
        total = 0
        running_loss = 0.0
        
        for images, labels in train_loader:
            images, labels = images.to(self.device), labels.to(self.device)
            optimizer.zero_grad()
            
            # Apply Mixup
            mixed_images, y_a, y_b, lam = mixup_data(images, labels, alpha=mixup_alpha)
            outputs = self.model(mixed_images)
            loss = mixup_criterion(criterion, outputs, y_a, y_b, lam, epoch=epoch)
            
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            # Calculate accuracy (approximate for mixup)
            _, preds = torch.max(outputs, 1)
            correct_a = (preds == y_a).float()
            correct_b = (preds == y_b).float()
            mixed_correct = lam * correct_a + (1 - lam) * correct_b
            correct += mixed_correct.sum().item()
            total += labels.size(0)
        
        train_loss = running_loss / len(train_loader)
        train_acc = correct / total
        return train_loss, train_acc
    
    def validate(self, val_loader, criterion, epoch=0):
        """Validate model."""
        self.model.eval()
        correct = 0
        total = 0
        running_loss = 0.0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = criterion(outputs, labels, epoch=epoch)
                running_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        
        val_loss = running_loss / len(val_loader)
        val_acc = correct / total
        return val_loss, val_acc
    
    def train(self, train_loader, val_loader, epochs=50, lr=0.0001, mixup_alpha=0.2, 
              class_counts=None, model_name='model'):
        """Complete training loop with progress tracking."""
        criterion = CBFocalLoss(
            class_counts=class_counts if class_counts is not None else [1,1], 
            beta=0.9999, gamma=2.0, drw=True, total_epochs=epochs
        )
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        best_val_acc = 0.0
        train_start = time.time()
        epoch_times = []
        
        print(f"🚀 Starting {epochs}-epoch training for {model_name}...")
        
        for epoch in range(epochs):
            epoch_start = time.time()
            
            # Training
            train_loss, train_acc = self.train_epoch(
                train_loader, criterion, optimizer, mixup_alpha, epoch
            )
            
            # Validation
            val_loss, val_acc = self.validate(val_loader, criterion, epoch)
            
            # Store history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.model.state_dict(), f'{self.save_dir}/{model_name}_best.pth')
            
            # Calculate epoch time and estimate remaining time
            epoch_time = time.time() - epoch_start
            epoch_times.append(epoch_time)
            
            # Progress bar animation
            progress = (epoch + 1) / epochs
            bar_length = 30
            filled_length = int(bar_length * progress)
            bar = '█' * filled_length + '░' * (bar_length - filled_length)
            
            # Time estimation
            avg_epoch_time = sum(epoch_times) / len(epoch_times)
            remaining_epochs = epochs - (epoch + 1)
            eta_seconds = avg_epoch_time * remaining_epochs
            eta = str(timedelta(seconds=int(eta_seconds)))
            
            # Current time
            current_time = datetime.now().strftime("%H:%M:%S")
            
            # Print animated progress
            print(f"⏰ [{current_time}] Epoch {epoch+1:2d}/{epochs} |{bar}| {progress*100:5.1f}% | "
                  f"Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | Val: {val_acc:.4f} | "
                  f"Time: {epoch_time:.1f}s | ETA: {eta}")
        
        train_time = time.time() - train_start
        return best_val_acc, train_time, self.history
