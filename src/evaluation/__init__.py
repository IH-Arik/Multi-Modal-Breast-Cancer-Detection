"""
Evaluation utilities and metrics for breast cancer detection.
"""

import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, recall_score, precision_score, 
    roc_auc_score, classification_report, confusion_matrix, roc_curve,
    mean_absolute_error
)
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os


def get_probabilities(model, loader, device):
    """Get prediction probabilities from model."""
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels.numpy())
    return np.array(all_probs), np.array(all_labels)


def comprehensive_evaluation(model, test_loader, device, model_name=''):
    """Comprehensive evaluation with all metrics."""
    probs, true_labels = get_probabilities(model, test_loader, device)
    preds = np.argmax(probs, axis=1)
    
    # Calculate all metrics
    accuracy = accuracy_score(true_labels, preds)
    precision = precision_score(true_labels, preds, average='macro', zero_division=0)
    recall = recall_score(true_labels, preds, average='macro', zero_division=0)
    f1 = f1_score(true_labels, preds, average='macro', zero_division=0)
    
    # Per-class recall
    per_class_recall = recall_score(true_labels, preds, average=None, zero_division=0)
    
    # Calculate confusion matrix
    cm = confusion_matrix(true_labels, preds)
    
    # Calculate sensitivity and specificity
    if len(np.unique(true_labels)) == 2:
        # Binary classification
        if cm.size == 4:  # 2x2 confusion matrix
            tn, fp, fn, tp = cm.ravel()
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        else:
            sensitivity = per_class_recall[1] if len(per_class_recall) > 1 else 0.0
            specificity = per_class_recall[0] if len(per_class_recall) > 0 else 0.0
    else:
        # Multi-class case
        if len(per_class_recall) >= 3:
            sensitivity = per_class_recall[2]  # Malignant class recall
            specificity = (per_class_recall[0] + per_class_recall[1]) / 2
        elif len(per_class_recall) == 2:
            sensitivity = per_class_recall[1]
            specificity = per_class_recall[0]
        else:
            sensitivity = per_class_recall[0] if len(per_class_recall) > 0 else 0.0
            specificity = per_class_recall[0] if len(per_class_recall) > 0 else 0.0
    
    # MAE and RMSE
    mae = np.mean(np.abs(true_labels - preds))
    rmse = np.sqrt(np.mean((true_labels - preds) ** 2))
    
    # Cross-entropy loss
    eps = 1e-7
    probs_clipped = np.clip(probs, eps, 1 - eps)
    ce_loss = -np.mean([np.log(probs_clipped[i, true_labels[i]]) for i in range(len(true_labels))])
    
    # AUC calculation
    try:
        if probs.shape[1] > 2:
            auc = roc_auc_score(true_labels, probs, multi_class='ovr', average='macro')
        else:
            pos_probs = probs[:, 1]
            auc = roc_auc_score(true_labels, pos_probs)
    except:
        auc = float('nan')
    
    # ROC curve
    try:
        if probs.shape[1] == 2:
            fpr, tpr, _ = roc_curve(true_labels, probs[:, 1])
        else:
            fpr, tpr = np.array([0.0, 1.0]), np.array([0.0, 1.0])
    except:
        fpr, tpr = np.array([0.0, 1.0]), np.array([0.0, 1.0])
    
    print(f"\n{'='*60}")
    print(f"🔬 {model_name.upper()} MODEL EVALUATION")
    print(f"{'='*60}")
    print(f"📊 Test Results:")
    print(f"   • Total Samples: {len(true_labels)}")
    print(f"   • Accuracy: {accuracy:.4f}")
    print(f"   • Precision: {precision:.4f}")
    print(f"   • Recall: {recall:.4f}")
    print(f"   • F1-Score: {f1:.4f}")
    print(f"   • AUC-ROC: {auc:.4f}")
    print(f"   • Sensitivity: {sensitivity:.4f}")
    print(f"   • Specificity: {specificity:.4f}")
    print(f"   • MAE: {mae:.4f}")
    print(f"   • RMSE: {rmse:.4f}")
    
    print(f"\n📋 Classification Report:")
    if len(np.unique(true_labels)) == 2:
        print(classification_report(true_labels, preds, target_names=['Benign', 'Malignant']))
    else:
        print(classification_report(true_labels, preds))
    
    print(f"\n🔢 Confusion Matrix:")
    print("                Predicted")
    if len(np.unique(true_labels)) == 2:
        print("              Benign  Malignant")
        print(f"Actual Benign    {cm[0,0]:4d}      {cm[0,1]:4d}")
        print(f"       Malignant {cm[1,0]:4d}      {cm[1,1]:4d}")
    else:
        print(f"Confusion Matrix:\n{cm}")
    
    return {
        'model_name': model_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'mae': mae,
        'rmse': rmse,
        'loss': ce_loss,
        'confusion_matrix': cm.tolist(),
        'fpr': fpr.tolist(),
        'tpr': tpr.tolist(),
        'true_labels': true_labels.tolist(),
        'preds': preds.tolist(),
        'probs': probs.tolist()
    }


def save_roc_curve(fpr, tpr, label='Model', out_path='roc_curve.json'):
    """Save ROC curve data."""
    roc_data = {
        'fpr': fpr.tolist() if hasattr(fpr, 'tolist') else fpr,
        'tpr': tpr.tolist() if hasattr(tpr, 'tolist') else tpr,
        'label': label
    }
    with open(out_path, 'w') as f:
        json.dump(roc_data, f, indent=2)


def plot_confusion_matrix(cm, class_names=None, title='Confusion Matrix', save_path=None):
    """Plot confusion matrix."""
    if class_names is None:
        class_names = [f'Class {i}' for i in range(len(cm))]
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_training_curves(history, model_name='Model', save_path=None):
    """Plot training curves."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Accuracy curves
    axes[0].plot(history['train_acc'], label='Train Accuracy', linewidth=2)
    axes[0].plot(history['val_acc'], label='Validation Accuracy', linewidth=2, linestyle='--')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title(f'{model_name} - Accuracy Curves')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([0, 1.05])
    
    # Loss curves
    axes[1].plot(history['train_loss'], label='Train Loss', linewidth=2)
    axes[1].plot(history['val_loss'], label='Validation Loss', linewidth=2, linestyle='--')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].set_title(f'{model_name} - Loss Curves')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def compare_models(results_list, metric='accuracy', save_path=None):
    """Compare multiple models results."""
    models = [r['model_name'] for r in results_list]
    scores = [r[metric] for r in results_list]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(models, scores, alpha=0.7)
    plt.xlabel('Models')
    plt.ylabel(metric.capitalize())
    plt.title(f'Model Comparison - {metric.capitalize()}')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, score in zip(bars, scores):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{score:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
