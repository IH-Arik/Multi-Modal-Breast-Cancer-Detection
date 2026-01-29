"""
Metrics and Evaluation Utilities
================================

Comprehensive metrics calculation and visualization for breast cancer detection.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve, precision_recall_curve,
    average_precision_score
)
from typing import Dict, List, Optional, Tuple
import torch
from pathlib import Path


def calculate_metrics(y_true: np.ndarray, 
                     y_pred: np.ndarray, 
                     y_prob: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Calculate comprehensive evaluation metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Prediction probabilities (optional)
        
    Returns:
        Dictionary of metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'specificity': _calculate_specificity(y_true, y_pred),
        'sensitivity': recall_score(y_true, y_pred, zero_division=0),
        'ppv': precision_score(y_true, y_pred, zero_division=0),
        'npv': _calculate_npv(y_true, y_pred)
    }
    
    # Add AUC if probabilities are provided
    if y_prob is not None:
        try:
            metrics['auc_roc'] = roc_auc_score(y_true, y_prob)
            metrics['auc_pr'] = average_precision_score(y_true, y_prob)
        except ValueError:
            metrics['auc_roc'] = 0.0
            metrics['auc_pr'] = 0.0
    
    return metrics


def _calculate_specificity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate specificity (true negative rate)."""
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    return tn / (tn + fp) if (tn + fp) > 0 else 0.0


def _calculate_npv(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate negative predictive value."""
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return tn / (tn + fn) if (tn + fn) > 0 else 0.0


def plot_confusion_matrix(cm: np.ndarray, 
                         save_path: Optional[str] = None,
                         class_names: Optional[List[str]] = None,
                         normalize: bool = False) -> plt.Figure:
    """
    Plot confusion matrix.
    
    Args:
        cm: Confusion matrix
        save_path: Path to save the plot
        class_names: Names of classes
        normalize: Whether to normalize the matrix
        
    Returns:
        Matplotlib figure
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        title = 'Normalized Confusion Matrix'
        fmt = '.2f'
    else:
        title = 'Confusion Matrix'
        fmt = 'd'
    
    if class_names is None:
        class_names = ['Benign', 'Malignant']
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_title(title)
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_roc_curve(y_true: np.ndarray, 
                   y_prob: np.ndarray,
                   save_path: Optional[str] = None,
                   model_name: str = 'Model') -> plt.Figure:
    """
    Plot ROC curve.
    
    Args:
        y_true: True labels
        y_prob: Prediction probabilities
        save_path: Path to save the plot
        model_name: Name of the model
        
    Returns:
        Matplotlib figure
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    auc_score = roc_auc_score(y_true, y_prob)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, linewidth=2, label=f'{model_name} (AUC = {auc_score:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_precision_recall_curve(y_true: np.ndarray,
                                y_prob: np.ndarray,
                                save_path: Optional[str] = None,
                                model_name: str = 'Model') -> plt.Figure:
    """
    Plot Precision-Recall curve.
    
    Args:
        y_true: True labels
        y_prob: Prediction probabilities
        save_path: Path to save the plot
        model_name: Name of the model
        
    Returns:
        Matplotlib figure
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    ap_score = average_precision_score(y_true, y_prob)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision, linewidth=2, label=f'{model_name} (AP = {ap_score:.3f})')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')
    ax.legend(loc='lower left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def calculate_class_weights(y_true: np.ndarray) -> torch.Tensor:
    """
    Calculate class weights for imbalanced datasets.
    
    Args:
        y_true: True labels
        
    Returns:
        Class weights tensor
    """
    unique, counts = np.unique(y_true, return_counts=True)
    total_samples = len(y_true)
    num_classes = len(unique)
    
    weights = total_samples / (num_classes * counts)
    weights = weights / weights.sum() * num_classes  # Normalize
    
    return torch.FloatTensor(weights)


def bootstrap_metrics(y_true: np.ndarray,
                     y_pred: np.ndarray,
                     y_prob: Optional[np.ndarray] = None,
                     n_bootstrap: int = 1000,
                     confidence_level: float = 0.95) -> Dict[str, Tuple[float, float]]:
    """
    Calculate confidence intervals using bootstrap.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Prediction probabilities
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level for intervals
        
    Returns:
        Dictionary of (lower_bound, upper_bound) for each metric
    """
    n_samples = len(y_true)
    alpha = 1 - confidence_level
    
    # Bootstrap samples
    bootstrap_metrics = []
    for _ in range(n_bootstrap):
        indices = np.random.choice(n_samples, n_samples, replace=True)
        bs_true = y_true[indices]
        bs_pred = y_pred[indices]
        bs_prob = y_prob[indices] if y_prob is not None else None
        
        metrics = calculate_metrics(bs_true, bs_pred, bs_prob)
        bootstrap_metrics.append(metrics)
    
    # Calculate confidence intervals
    ci_intervals = {}
    metric_names = bootstrap_metrics[0].keys()
    
    for metric in metric_names:
        values = [m[metric] for m in bootstrap_metrics]
        lower = np.percentile(values, 100 * alpha / 2)
        upper = np.percentile(values, 100 * (1 - alpha / 2))
        ci_intervals[metric] = (lower, upper)
    
    return ci_intervals


def compare_models(model_metrics: Dict[str, Dict[str, float]],
                  save_path: Optional[str] = None) -> plt.Figure:
    """
    Compare multiple models using bar plots.
    
    Args:
        model_metrics: Dictionary of model name to metrics
        save_path: Path to save the plot
        
    Returns:
        Matplotlib figure
    """
    metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc']
    model_names = list(model_metrics.keys())
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics_to_plot):
        if i >= len(axes):
            break
            
        values = [model_metrics[model].get(metric, 0) for model in model_names]
        
        axes[i].bar(model_names, values, alpha=0.7)
        axes[i].set_title(f'{metric.replace("_", " ").title()}')
        axes[i].set_ylabel('Score')
        axes[i].tick_params(axis='x', rotation=45)
        axes[i].grid(True, alpha=0.3)
    
    # Remove empty subplot
    if len(metrics_to_plot) < len(axes):
        axes[-1].remove()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def generate_classification_report(y_true: np.ndarray,
                                  y_pred: np.ndarray,
                                  class_names: Optional[List[str]] = None,
                                  save_path: Optional[str] = None) -> str:
    """
    Generate detailed classification report.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Names of classes
        save_path: Path to save the report
        
    Returns:
        Classification report string
    """
    if class_names is None:
        class_names = ['Benign', 'Malignant']
    
    report = classification_report(y_true, y_pred, target_names=class_names)
    
    if save_path:
        with open(save_path, 'w') as f:
            f.write(report)
    
    return report


def calculate_calibration_metrics(y_true: np.ndarray,
                                  y_prob: np.ndarray,
                                  n_bins: int = 10) -> Dict[str, float]:
    """
    Calculate calibration metrics.
    
    Args:
        y_true: True labels
        y_prob: Prediction probabilities
        n_bins: Number of bins for calibration
        
    Returns:
        Dictionary of calibration metrics
    """
    # Expected Calibration Error (ECE)
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    mce = 0.0
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find samples in this bin
        in_bin = ((y_prob > bin_lower) & (y_prob <= bin_upper))
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            # Calculate accuracy and confidence in this bin
            accuracy_in_bin = y_true[in_bin].mean()
            avg_confidence_in_bin = y_prob[in_bin].mean()
            
            # Update ECE
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            
            # Update MCE
            mce = max(mce, np.abs(avg_confidence_in_bin - accuracy_in_bin))
    
    return {
        'ece': ece,
        'mce': mce
    }
