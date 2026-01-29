"""
Multi-Modal Breast Cancer Detection Datasets
============================================

Professional dataset classes for loading and preprocessing breast cancer data
from multiple modalities (ultrasound, mammography, histology).
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import os


class BreastCancerDataset(Dataset):
    """
    Base dataset class for breast cancer detection.
    
    Supports single-modality and multi-modal data loading with
    comprehensive preprocessing and augmentation.
    """
    
    def __init__(self,
                 data_paths: Dict[str, Union[str, Path]],
                 labels: Union[List[int], pd.Series],
                 transform: Optional[transforms.Compose] = None,
                 modality_transforms: Optional[Dict[str, transforms.Compose]] = None,
                 target_size: Tuple[int, int] = (224, 224)):
        """
        Initialize breast cancer dataset.
        
        Args:
            data_paths: Dictionary mapping modalities to image paths
            labels: List of labels (0: benign, 1: malignant)
            transform: General transform pipeline
            modality_transforms: Modality-specific transforms
            target_size: Target image size
        """
        self.data_paths = data_paths
        self.labels = labels
        self.transform = transform
        self.modality_transforms = modality_transforms or {}
        self.target_size = target_size
        self.modalities = list(data_paths.keys())
        
        # Validate data
        self._validate_data()
    
    def _validate_data(self):
        """Validate dataset consistency."""
        num_samples = len(self.labels)
        for modality, paths in self.data_paths.items():
            if len(paths) != num_samples:
                raise ValueError(f"Length mismatch for {modality}: {len(paths)} vs {num_samples} labels")
    
    def __len__(self) -> int:
        return len(self.labels)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get sample by index."""
        sample = {}
        
        # Load images for each modality
        for modality in self.modalities:
            img_path = self.data_paths[modality][idx]
            
            try:
                image = Image.open(img_path).convert('RGB')
                
                # Apply modality-specific transforms
                if modality in self.modality_transforms:
                    image = self.modality_transforms[modality](image)
                # Apply general transforms
                elif self.transform:
                    image = self.transform(image)
                else:
                    # Default resize and normalize
                    image = transforms.Resize(self.target_size)(image)
                    image = transforms.ToTensor()(image)
                
                sample[modality] = image
                
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
                # Return zero tensor as fallback
                sample[modality] = torch.zeros(3, *self.target_size)
        
        # Add label
        sample['label'] = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return sample


class UltrasoundDataset(BreastCancerDataset):
    """Specialized dataset for ultrasound breast cancer detection."""
    
    def __init__(self,
                 ultrasound_paths: List[str],
                 labels: List[int],
                 transform: Optional[transforms.Compose] = None,
                 intensity_transform: Optional[transforms.Compose] = None,
                 target_size: Tuple[int, int] = (224, 224)):
        """
        Initialize ultrasound dataset.
        
        Args:
            ultrasound_paths: List of ultrasound image paths
            labels: List of labels
            transform: General transform pipeline
            intensity_transform: Ultrasound-specific intensity transforms
            target_size: Target image size
        """
        data_paths = {'ultrasound': ultrasound_paths}
        modality_transforms = {}
        
        if intensity_transform:
            modality_transforms['ultrasound'] = intensity_transform
        
        super().__init__(data_paths, labels, transform, modality_transforms, target_size)


class MammographyDataset(BreastCancerDataset):
    """Specialized dataset for mammography breast cancer detection."""
    
    def __init__(self,
                 mammography_paths: List[str],
                 labels: List[int],
                 transform: Optional[transforms.Compose] = None,
                 target_size: Tuple[int, int] = (224, 224)):
        """
        Initialize mammography dataset.
        
        Args:
            mammography_paths: List of mammography image paths
            labels: List of labels
            transform: Transform pipeline
            target_size: Target image size
        """
        data_paths = {'mammography': mammography_paths}
        super().__init__(data_paths, labels, transform, target_size=target_size)


class HistologyDataset(BreastCancerDataset):
    """Specialized dataset for histology breast cancer detection."""
    
    def __init__(self,
                 histology_paths: List[str],
                 labels: List[int],
                 transform: Optional[transforms.Compose] = None,
                 stain_normalization: bool = True,
                 target_size: Tuple[int, int] = (224, 224)):
        """
        Initialize histology dataset.
        
        Args:
            histology_paths: List of histology image paths
            labels: List of labels
            transform: Transform pipeline
            stain_normalization: Whether to apply stain normalization
            target_size: Target image size
        """
        data_paths = {'histology': histology_paths}
        self.stain_normalization = stain_normalization
        
        super().__init__(data_paths, labels, transform, target_size=target_size)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get sample with optional stain normalization."""
        sample = super().__getitem__(idx)
        
        if self.stain_normalization:
            # Apply stain normalization (simplified version)
            histology_img = sample['histology']
            sample['histology'] = self._normalize_stain(histology_img)
        
        return sample
    
    def _normalize_stain(self, image: torch.Tensor) -> torch.Tensor:
        """Simple stain normalization (placeholder implementation)."""
        # In practice, this would implement Macenko or Reinhard normalization
        return image


class MultiModalBreastCancerDataset(BreastCancerDataset):
    """Multi-modal dataset combining ultrasound, mammography, and histology."""
    
    def __init__(self,
                 data_paths: Dict[str, List[str]],
                 labels: List[int],
                 transform: Optional[transforms.Compose] = None,
                 modality_transforms: Optional[Dict[str, transforms.Compose]] = None,
                 missing_modalities: Optional[List[bool]] = None,
                 target_size: Tuple[int, int] = (224, 224)):
        """
        Initialize multi-modal dataset.
        
        Args:
            data_paths: Dictionary mapping modalities to image paths
            labels: List of labels
            transform: General transform pipeline
            modality_transforms: Modality-specific transforms
            missing_modalities: List indicating which modalities are missing for each sample
            target_size: Target image size
        """
        self.missing_modalities = missing_modalities
        super().__init__(data_paths, labels, transform, modality_transforms, target_size)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get multi-modal sample."""
        sample = {}
        
        for modality in self.modalities:
            # Check if modality is missing for this sample
            if self.missing_modalities and self.missing_modalities[idx].get(modality, False):
                sample[modality] = torch.zeros(3, *self.target_size)
                continue
            
            img_path = self.data_paths[modality][idx]
            
            try:
                image = Image.open(img_path).convert('RGB')
                
                # Apply transforms
                if modality in self.modality_transforms:
                    image = self.modality_transforms[modality](image)
                elif self.transform:
                    image = self.transform(image)
                else:
                    image = transforms.Resize(self.target_size)(image)
                    image = transforms.ToTensor()(image)
                
                sample[modality] = image
                
            except Exception as e:
                print(f"Error loading {modality} image {img_path}: {e}")
                sample[modality] = torch.zeros(3, *self.target_size)
        
        sample['label'] = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return sample


def create_data_loaders(datasets: Dict[str, Dataset],
                       batch_sizes: Dict[str, int],
                       num_workers: int = 4,
                       pin_memory: bool = True) -> Dict[str, DataLoader]:
    """
    Create data loaders for train, validation, and test sets.
    
    Args:
        datasets: Dictionary of datasets
        batch_sizes: Dictionary of batch sizes
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory
        
    Returns:
        Dictionary of data loaders
    """
    loaders = {}
    
    for split, dataset in datasets.items():
        shuffle = (split == 'train')
        batch_size = batch_sizes.get(split, 32)
        
        loaders[split] = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=(split == 'train')
        )
    
    return loaders


def get_transforms(modality: str = 'general',
                  augment: bool = True,
                  target_size: Tuple[int, int] = (224, 224)) -> transforms.Compose:
    """
    Get transform pipeline for specific modality.
    
    Args:
        modality: Modality type
        augment: Whether to apply augmentation
        target_size: Target image size
        
    Returns:
        Transform pipeline
    """
    if augment and modality == 'train':
        transform_list = [
            transforms.RandomResizedCrop(target_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
    else:
        transform_list = [
            transforms.Resize(target_size),
            transforms.CenterCrop(target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
    
    return transforms.Compose(transform_list)


def split_dataset(dataset: Dataset,
                 train_ratio: float = 0.7,
                 val_ratio: float = 0.15,
                 test_ratio: float = 0.15,
                 random_seed: int = 42) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Split dataset into train, validation, and test sets.
    
    Args:
        dataset: Dataset to split
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"
    
    total_size = len(dataset)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size
    
    return torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(random_seed)
    )
