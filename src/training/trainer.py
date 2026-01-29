"""
Multi-Modal Breast Cancer Detection Training Pipeline
====================================================

Professional training pipeline for multi-modal breast cancer detection models
with advanced optimization, logging, and evaluation capabilities.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union
import logging
from tqdm import tqdm

from ..models.multi_modal import MultiModalFusionModel, BreastCancerClassifier
from ..utils.metrics import calculate_metrics, plot_confusion_matrix
from ..utils.logger import setup_logger


class MultiModalTrainer:
    """
    Professional trainer for multi-modal breast cancer detection models.
    
    Features:
    - Mixed precision training
    - Advanced optimization strategies
    - Comprehensive evaluation
    - Experiment tracking
    - Model checkpointing
    """
    
    def __init__(self,
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 test_loader: Optional[DataLoader] = None,
                 config: Optional[Dict] = None,
                 experiment_dir: Optional[str] = None):
        """
        Initialize trainer.
        
        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            test_loader: Test data loader
            config: Training configuration
            experiment_dir: Directory to save experiments
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.config = config or {}
        self.experiment_dir = Path(experiment_dir) if experiment_dir else Path("experiments")
        
        # Setup experiment directory
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.experiment_dir / f"run_{timestamp}"
        self.run_dir.mkdir(exist_ok=True)
        
        # Setup logging
        self.logger = setup_logger("trainer", log_file=str(self.run_dir / "training.log"))
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Multi-GPU support
        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)
            self.logger.info(f"Using {torch.cuda.device_count()} GPUs")
        
        # Training components
        self._setup_training_components()
        
        # Training history
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': [],
            'train_f1': [], 'val_f1': [],
            'train_auc': [], 'val_auc': []
        }
        
        # Best model tracking
        self.best_val_metric = 0.0
        self.best_epoch = 0
        
        self.logger.info(f"Trainer initialized. Run directory: {self.run_dir}")
    
    def _setup_training_components(self):
        """Setup optimizer, scheduler, loss function, and scaler."""
        # Optimizer
        optimizer_name = self.config.get('optimizer', 'adam')
        learning_rate = self.config.get('learning_rate', 1e-4)
        weight_decay = self.config.get('weight_decay', 1e-5)
        
        if optimizer_name.lower() == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
        elif optimizer_name.lower() == 'adamw':
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
        elif optimizer_name.lower() == 'sgd':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=learning_rate,
                momentum=0.9,
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")
        
        # Learning rate scheduler
        scheduler_name = self.config.get('scheduler', 'cosine')
        if scheduler_name == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.config.get('epochs', 100)
            )
        elif scheduler_name == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, step_size=30, gamma=0.1
            )
        elif scheduler_name == 'plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='max', factor=0.5, patience=10
            )
        else:
            self.scheduler = None
        
        # Loss function
        loss_name = self.config.get('loss', 'cross_entropy')
        if loss_name == 'cross_entropy':
            self.criterion = nn.CrossEntropyLoss()
        elif loss_name == 'focal':
            from ..utils.losses import FocalLoss
            self.criterion = FocalLoss(gamma=2.0)
        elif loss_name == 'weighted_cross_entropy':
            # Calculate class weights if needed
            class_weights = self._calculate_class_weights()
            self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            raise ValueError(f"Unsupported loss: {loss_name}")
        
        # Mixed precision scaler
        self.scaler = GradScaler() if self.config.get('mixed_precision', True) else None
    
    def _calculate_class_weights(self) -> torch.Tensor:
        """Calculate class weights for imbalanced dataset."""
        # Simple implementation - can be improved
        return torch.tensor([1.0, 1.0]).to(self.device)
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        epoch_loss = 0.0
        all_preds = []
        all_labels = []
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            batch = self._move_batch_to_device(batch)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            if self.scaler:
                with autocast():
                    outputs = self.model(batch)
                    loss = self.criterion(outputs, batch['label'])
                
                # Backward pass with mixed precision
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(batch)
                loss = self.criterion(outputs, batch['label'])
                loss.backward()
                self.optimizer.step()
            
            # Update metrics
            epoch_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch['label'].cpu().numpy())
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': loss.item(),
                'avg_loss': epoch_loss / (batch_idx + 1)
            })
        
        # Calculate epoch metrics
        epoch_metrics = self._calculate_epoch_metrics(all_preds, all_labels, epoch_loss)
        
        return epoch_metrics
    
    def validate_epoch(self, epoch: int) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        epoch_loss = 0.0
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                batch = self._move_batch_to_device(batch)
                
                if self.scaler:
                    with autocast():
                        outputs = self.model(batch)
                        loss = self.criterion(outputs, batch['label'])
                else:
                    outputs = self.model(batch)
                    loss = self.criterion(outputs, batch['label'])
                
                epoch_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                probs = torch.softmax(outputs, dim=1)[:, 1]
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch['label'].cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        # Calculate epoch metrics
        epoch_metrics = self._calculate_epoch_metrics(all_preds, all_labels, epoch_loss, all_probs)
        
        return epoch_metrics
    
    def _move_batch_to_device(self, batch: Dict) -> Dict:
        """Move batch to device."""
        device_batch = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                device_batch[key] = value.to(self.device)
            else:
                device_batch[key] = value
        return device_batch
    
    def _calculate_epoch_metrics(self, preds: List, labels: List, loss: float, probs: Optional[List] = None) -> Dict[str, float]:
        """Calculate epoch metrics."""
        preds = np.array(preds)
        labels = np.array(labels)
        
        metrics = {
            'loss': loss / len(self.train_loader),
            'accuracy': accuracy_score(labels, preds),
            'precision': precision_score(labels, preds, average='weighted', zero_division=0),
            'recall': recall_score(labels, preds, average='weighted', zero_division=0),
            'f1': f1_score(labels, preds, average='weighted', zero_division=0)
        }
        
        if probs is not None:
            try:
                metrics['auc'] = roc_auc_score(labels, probs)
            except ValueError:
                metrics['auc'] = 0.0
        
        return metrics
    
    def train(self, num_epochs: int) -> Dict[str, List]:
        """Main training loop."""
        self.logger.info(f"Starting training for {num_epochs} epochs")
        
        for epoch in range(num_epochs):
            self.logger.info(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            # Training
            train_metrics = self.train_epoch(epoch)
            
            # Validation
            val_metrics = self.validate_epoch(epoch)
            
            # Update learning rate scheduler
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['f1'])
                else:
                    self.scheduler.step()
            
            # Update history
            self._update_history(train_metrics, val_metrics)
            
            # Log metrics
            self._log_epoch_metrics(epoch, train_metrics, val_metrics)
            
            # Save best model
            self._save_best_model(val_metrics, epoch)
            
            # Save checkpoint
            if (epoch + 1) % self.config.get('checkpoint_freq', 10) == 0:
                self._save_checkpoint(epoch)
        
        # Final evaluation
        if self.test_loader:
            self.logger.info("\nFinal evaluation on test set...")
            test_metrics = self.evaluate(self.test_loader)
            self.logger.info(f"Test metrics: {test_metrics}")
        
        # Save training history
        self._save_history()
        
        return self.history
    
    def _update_history(self, train_metrics: Dict, val_metrics: Dict):
        """Update training history."""
        for key in train_metrics:
            self.history[f'train_{key}'].append(train_metrics[key])
        
        for key in val_metrics:
            self.history[f'val_{key}'].append(val_metrics[key])
    
    def _log_epoch_metrics(self, epoch: int, train_metrics: Dict, val_metrics: Dict):
        """Log epoch metrics."""
        self.logger.info(f"Train Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}, F1: {train_metrics['f1']:.4f}")
        self.logger.info(f"Val Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}, F1: {val_metrics['f1']:.4f}")
        
        if 'auc' in val_metrics:
            self.logger.info(f"Val AUC: {val_metrics['auc']:.4f}")
    
    def _save_best_model(self, val_metrics: Dict, epoch: int):
        """Save best model based on validation metric."""
        metric_name = self.config.get('val_metric', 'f1')
        current_metric = val_metrics.get(metric_name, 0.0)
        
        if current_metric > self.best_val_metric:
            self.best_val_metric = current_metric
            self.best_epoch = epoch
            
            # Save model
            model_path = self.run_dir / "best_model.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
                'best_metric': self.best_val_metric,
                'config': self.config
            }, model_path)
            
            self.logger.info(f"New best model saved with {metric_name}: {current_metric:.4f}")
    
    def _save_checkpoint(self, epoch: int):
        """Save training checkpoint."""
        checkpoint_path = self.run_dir / f"checkpoint_epoch_{epoch}.pth"
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'history': self.history,
            'config': self.config
        }, checkpoint_path)
    
    def _save_history(self):
        """Save training history."""
        history_path = self.run_dir / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        # Plot training curves
        self._plot_training_curves()
    
    def _plot_training_curves(self):
        """Plot training curves."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss
        axes[0, 0].plot(self.history['train_loss'], label='Train')
        axes[0, 0].plot(self.history['val_loss'], label='Validation')
        axes[0, 0].set_title('Loss')
        axes[0, 0].legend()
        
        # Accuracy
        axes[0, 1].plot(self.history['train_acc'], label='Train')
        axes[0, 1].plot(self.history['val_acc'], label='Validation')
        axes[0, 1].set_title('Accuracy')
        axes[0, 1].legend()
        
        # F1 Score
        axes[1, 0].plot(self.history['train_f1'], label='Train')
        axes[1, 0].plot(self.history['val_f1'], label='Validation')
        axes[1, 0].set_title('F1 Score')
        axes[1, 0].legend()
        
        # AUC (if available)
        if 'val_auc' in self.history and any(self.history['val_auc']):
            axes[1, 1].plot(self.history['val_auc'], label='Validation AUC')
            axes[1, 1].set_title('AUC')
            axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig(self.run_dir / "training_curves.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def evaluate(self, data_loader: DataLoader) -> Dict[str, float]:
        """Evaluate model on given data loader."""
        self.model.eval()
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Evaluation"):
                batch = self._move_batch_to_device(batch)
                
                outputs = self.model(batch)
                preds = torch.argmax(outputs, dim=1)
                probs = torch.softmax(outputs, dim=1)[:, 1]
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch['label'].cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        # Calculate comprehensive metrics
        metrics = calculate_metrics(all_labels, all_preds, all_probs)
        
        # Save confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        plot_confusion_matrix(cm, self.run_dir / "confusion_matrix.png")
        
        return metrics
    
    def load_model(self, model_path: str):
        """Load trained model."""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        if isinstance(self.model, nn.DataParallel):
            self.model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        self.logger.info(f"Model loaded from {model_path}")
