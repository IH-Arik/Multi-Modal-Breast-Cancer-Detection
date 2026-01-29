#!/usr/bin/env python3
"""
Multi-Modal Breast Cancer Detection - Training Script

This script trains models for breast cancer detection across multiple modalities:
- Ultrasound (BUSI Dataset)
- Mammography (CBIS-DDSM Dataset) 
- Histology (BreakHis Dataset)

Usage:
    python scripts/train_all_modalities.py --config configs/config.yaml
"""

import os
import sys
import argparse
import yaml
import json
import pandas as pd
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from utils.common import set_device, json_default, convert_results_to_native
from data.dataset_utils import collect_images, stratified_split_indices, create_balanced_dataloaders
from models import get_model, get_available_backbones
from training import Trainer
from evaluation import comprehensive_evaluation, save_roc_curve, plot_training_curves


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def train_ultrasound_models(config):
    """Train ultrasound models on BUSI dataset."""
    print("\n" + "="*80)
    print("🔬 TRAINING ULTRASOUND MODELS (BUSI Dataset)")
    print("="*80)
    
    # Configuration
    busi_dir = config['dataset_paths']['busi_dataset']
    epochs = config['training']['epochs']
    batch_size = config['training']['batch_size']
    lr = config['training']['learning_rate']
    device = set_device()
    
    # Data collection
    print("\n📁 Loading BUSI dataset...")
    ultra_paths, ultra_labels = collect_images(
        busi_dir, 
        exclude_mask=config['modalities']['ultrasound']['exclude_masks'],
        exclude_normal=config['modalities']['ultrasound']['exclude_normal'],
        normal_as_benign=config['modalities']['ultrasound']['normal_as_benign']
    )
    print(f"Total images: {len(ultra_paths)}")
    
    # Class distribution
    class_names = ['Normal', 'Benign', 'Malignant'] if not config['modalities']['ultrasound']['normal_as_benign'] else ['Benign', 'Malignant']
    for i, name in enumerate(class_names):
        count = sum(1 for l in ultra_labels if l == i)
        print(f"{name}: {count}")
    
    # Split data
    ultra_tr, ultra_val, ultra_te = stratified_split_indices(ultra_labels, 0.7, 0.2, 42)
    subset = lambda lst, idxs: [lst[i] for i in idxs]
    
    train_paths = subset(ultra_paths, ultra_tr)
    train_labels = [ultra_labels[i] for i in ultra_tr]
    val_paths = subset(ultra_paths, ultra_val)
    val_labels = [ultra_labels[i] for i in ultra_val]
    test_paths = subset(ultra_paths, ultra_te)
    test_labels = [ultra_labels[i] for i in ultra_te]
    
    print(f"Data split: Train: {len(train_labels)}, Val: {len(val_labels)}, Test: {len(test_labels)}")
    
    # Create dataloaders
    train_loader, val_loader, test_loader, class_counts = create_balanced_dataloaders(
        train_paths, train_labels, val_paths, val_labels, test_paths, test_labels, 
        batch_size, modality='ultrasound'
    )
    
    # Train models
    backbones = get_available_backbones()[:5]  # Train top 5 for demo
    all_results = []
    all_histories = {}
    
    for backbone in backbones:
        print(f"\n{'='*60}")
        print(f"🚀 Training {backbone.upper()}")
        print(f"{'='*60}")
        
        try:
            model = get_model(backbone, num_classes=len(class_names)).to(device)
            
            # Train
            trainer = Trainer(model, device, config['results']['model_save_dir'])
            best_val_acc, train_time, history = trainer.train(
                train_loader, val_loader, epochs, lr, 
                config['training']['mixup_alpha'], class_counts, f'ultrasound_{backbone}'
            )
            
            # Store history
            all_histories[backbone] = history
            
            # Evaluate
            results = comprehensive_evaluation(model, test_loader, device, f'Ultrasound-{backbone}')
            results['train_time'] = train_time
            results['best_val_acc'] = best_val_acc
            results['backbone'] = backbone
            results['modality'] = 'ultrasound'
            
            all_results.append(results)
            
            # Save ROC curve
            save_roc_curve(results['fpr'], results['tpr'], f'Ultrasound-{backbone}', 
                          f"results/roc_ultrasound_{backbone}.json")
            
            # Plot training curves
            plot_training_curves(history, f'Ultrasound-{backbone}', 
                               f"results/figures/ultrasound_{backbone}_curves.png")
            
        except Exception as e:
            print(f"❌ Error training {backbone}: {str(e)}")
            continue
    
    return all_results, all_histories


def train_mammography_model(config):
    """Train mammography model on CBIS-DDSM dataset."""
    print("\n" + "="*80)
    print("🔬 TRAINING MAMMOGRAPHY MODEL (CBIS-DDSM Dataset)")
    print("="*80)
    
    # This would implement the mammography training from mammo_final.py
    # For now, return placeholder results
    print("📝 Mammography training implementation pending...")
    return []


def train_histology_models(config):
    """Train histology models on BreakHis dataset."""
    print("\n" + "="*80)
    print("🔬 TRAINING HISTOLOGY MODELS (BreakHis Dataset)")
    print("="*80)
    
    # This would implement the histology training from histology.py
    # For now, return placeholder results
    print("📝 Histology training implementation pending...")
    return []


def save_results(all_results, config):
    """Save all results to files."""
    results_dir = Path(config['results']['save_dir'])
    results_dir.mkdir(exist_ok=True)
    
    # Convert to native types
    native_results = [convert_results_to_native(r) for r in all_results]
    
    # Save as JSON
    with open(results_dir / 'all_results.json', 'w') as f:
        json.dump(native_results, f, indent=2, default=json_default)
    
    # Save as CSV
    if native_results:
        df = pd.DataFrame(native_results)
        df.to_csv(results_dir / 'results_summary.csv', index=False)
        print(f"\n📊 Results saved to {results_dir}")
        print(df[['modality', 'backbone', 'accuracy', 'f1', 'auc']].to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description='Train multi-modal breast cancer detection models')
    parser.add_argument('--config', type=str, default='configs/config.yaml', 
                       help='Path to configuration file')
    parser.add_argument('--modality', type=str, choices=['ultrasound', 'mammography', 'histology', 'all'],
                       default='all', help='Modality to train')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Create results directories
    for dir_path in [config['results']['model_save_dir'], 
                     config['results']['figure_save_dir'],
                     config['results']['log_dir']]:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    all_results = []
    
    # Train based on modality
    if args.modality in ['ultrasound', 'all']:
        ultrasound_results, _ = train_ultrasound_models(config)
        all_results.extend(ultrasound_results)
    
    if args.modality in ['mammography', 'all']:
        mammography_results = train_mammography_model(config)
        all_results.extend(mammography_results)
    
    if args.modality in ['histology', 'all']:
        histology_results = train_histology_models(config)
        all_results.extend(histology_results)
    
    # Save results
    if all_results:
        save_results(all_results, config)
    
    print("\n" + "="*80)
    print("🎉 TRAINING COMPLETED!")
    print("="*80)


if __name__ == "__main__":
    main()
