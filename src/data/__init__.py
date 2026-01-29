"""
Data loading and preprocessing utilities for breast cancer detection.
"""

import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from torchvision.transforms import functional as F
from PIL import Image
from pathlib import Path
import pandas as pd


class SharedGeometricTransform:
    """Apply the same geometric augmentations to images."""
    
    def __init__(self, use_aug=True):
        self.use_aug = use_aug

    def __call__(self, img):
        if self.use_aug:
            i, j, h, w = transforms.RandomResizedCrop.get_params(
                img, scale=(0.8, 1.0), ratio=(3./4., 4./3.)
            )
            img = F.resized_crop(img, i, j, h, w, (224, 224))
            if random.random() < 0.5:
                img = F.hflip(img)
            angle = transforms.RandomRotation.get_params([-15, 15])
            img = F.rotate(img, angle, fill=(0, 0, 0))
        else:
            img = transforms.Resize(256)(img)
            img = transforms.CenterCrop(224)(img)
        return img


def build_intensity_transform(modality='ultrasound'):
    """Build modality-specific intensity transforms."""
    if modality == 'ultrasound':
        return transforms.Compose([
            transforms.RandomAutocontrast(p=0.3),
            transforms.RandomAdjustSharpness(sharpness_factor=1.2, p=0.2),
            transforms.RandomAffine(degrees=0, translate=None, scale=(0.95, 1.05), shear=None, fill=0),
        ])
    elif modality == 'mammography':
        return transforms.Compose([
            transforms.RandomAutocontrast(p=0.3),
            transforms.RandomAdjustSharpness(sharpness_factor=1.2, p=0.2),
            transforms.RandomAffine(degrees=0, translate=None, scale=(0.95, 1.05), shear=None, fill=0),
        ])
    elif modality == 'histology':
        return transforms.Compose([
            transforms.RandomAutocontrast(p=0.3),
            transforms.RandomAdjustSharpness(sharpness_factor=1.2, p=0.2),
            transforms.RandomAffine(degrees=0, translate=None, scale=(0.95, 1.05), shear=None, fill=0),
        ])
    else:
        return transforms.Compose([])


# ImageNet normalization
imagenet_norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
final_tensor_transform = transforms.Compose([transforms.ToTensor(), imagenet_norm])
random_erasing_train = transforms.RandomErasing(p=0.25, scale=(0.02, 0.2), ratio=(0.3, 3.3), value='random')


class BaseMedicalDataset(Dataset):
    """Base dataset class for medical imaging."""
    
    def __init__(self, image_paths, labels, transform=None, pair_transform=None,
                 intensity_transform=None, final_transform=None, random_erasing=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.pair_transform = pair_transform
        self.intensity_transform = intensity_transform
        self.final_transform = final_transform
        self.random_erasing = random_erasing

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.image_paths[idx]).convert('RGB')
            
            # Legacy single transform path
            if self.transform is not None and self.pair_transform is None:
                img = self.transform(img)
                if self.random_erasing is not None:
                    img = self.random_erasing(img)
                return img, self.labels[idx]

            # Synchronized geometric transforms
            if self.pair_transform is not None:
                img = self.pair_transform(img)

            # Modality-specific intensity transforms
            if self.intensity_transform is not None:
                img = self.intensity_transform(img)

            # Final transforms to tensor + normalize
            if self.final_transform is not None:
                img = self.final_transform(img)

            # Optional Random Erasing
            if self.random_erasing is not None:
                img = self.random_erasing(img)

            return img, self.labels[idx]
        except Exception as e:
            print(f"Error loading image {self.image_paths[idx]}: {e}")
            # Return dummy image if loading fails
            dummy_img = torch.zeros(3, 224, 224)
            return dummy_img, self.labels[idx]


def collect_images(base_dir, benign_word='benign', malignant_word='malignant', 
                  exclude_mask=False, exclude_normal=False, normal_as_benign=False):
    """Generic image collection function."""
    paths = []
    labels = []
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.png') or file.endswith('.jpg'):
                if exclude_mask and '_mask' in file:
                    continue
                if exclude_normal and 'normal' in root.lower():
                    continue
                path = os.path.join(root, file)
                rl = root.lower()
                
                # Three-class mapping when normal_as_benign is False
                if (not normal_as_benign) and ('normal' in rl):
                    labels.append(0)  # Normal
                elif benign_word in rl:
                    labels.append(1 if not normal_as_benign else 0)  # Benign or merged into benign
                elif malignant_word in rl:
                    labels.append(2 if not normal_as_benign else 1)  # Malignant or 1 in binary
                else:
                    continue
                paths.append(path)
    return paths, labels


def stratified_split_indices(labels, train_ratio=0.7, test_ratio=0.2, seed=42):
    """Stratified split ensuring class balance."""
    rng = random.Random(seed)
    labels = np.array(labels)
    classes = np.unique(labels)
    per_class_indices = {c: np.where(labels == c)[0].tolist() for c in classes}
    for idxs in per_class_indices.values():
        rng.shuffle(idxs)
    train_idx, test_idx, val_idx = [], [], []
    for c, idxs in per_class_indices.items():
        n = len(idxs)
        n_train = int(round(train_ratio * n))
        n_test = int(round(test_ratio * n))
        train_idx.extend(idxs[:n_train])
        test_idx.extend(idxs[n_train:n_train+n_test])
        val_idx.extend(idxs[n_train+n_test:])
    rng.shuffle(train_idx); rng.shuffle(test_idx); rng.shuffle(val_idx)
    return train_idx, val_idx, test_idx


def create_balanced_dataloaders(train_paths, train_labels, val_paths, val_labels, 
                               test_paths, test_labels, batch_size=16, modality='ultrasound'):
    """Create data loaders with class balancing."""
    num_classes = int(max(train_labels)) + 1
    class_counts = np.bincount(np.array(train_labels), minlength=num_classes)
    sample_weights = [1.0 / class_counts[y] if class_counts[y] > 0 else 0.0 for y in train_labels]
    sampler = WeightedRandomSampler(weights=torch.DoubleTensor(sample_weights), 
                                   num_samples=len(train_labels), replacement=True)

    intensity_transform = build_intensity_transform(modality)
    eval_pair_tf = SharedGeometricTransform(use_aug=False)

    train_dataset = BaseMedicalDataset(
        train_paths, train_labels, 
        pair_transform=SharedGeometricTransform(use_aug=True),
        intensity_transform=intensity_transform,
        final_transform=final_tensor_transform, 
        random_erasing=random_erasing_train
    )
    val_dataset = BaseMedicalDataset(
        val_paths, val_labels, 
        pair_transform=eval_pair_tf, 
        final_transform=final_tensor_transform
    )
    test_dataset = BaseMedicalDataset(
        test_paths, test_labels, 
        pair_transform=eval_pair_tf, 
        final_transform=final_tensor_transform
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, 
                            sampler=sampler, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                          num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                           num_workers=0, pin_memory=True)
    
    return train_loader, val_loader, test_loader, class_counts
