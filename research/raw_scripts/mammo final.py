import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import WeightedRandomSampler
from torchvision import transforms, models
from torchvision.transforms import functional as F
import random
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
    mean_absolute_error,
    precision_score,
    recall_score,
    f1_score,
)
import json
import os
import matplotlib.pyplot as plt

def filter_roi_images(folder_path):
    """Get only ROI images (cropped, smaller size)"""
    jpgs = list(folder_path.glob('*.jpg'))
    roi_images = []
    
    for jpg in jpgs:
        try:
            img = Image.open(jpg)
            width, height = img.size
            if width < 1000 and height < 1000:
                roi_images.append(jpg)
        except Exception:
            continue
    
    return roi_images

def create_combined_mass_calc_mapping():
    """Combine both mass and calcification cases for larger dataset"""
    
    base_dir = Path(r'E:\Journal\Breast Cancer\cbis-ddsm-breast-cancer-image-dataset')
    csv_dir = base_dir / 'csv'
    jpeg_dir = base_dir / 'jpeg'
    
    # Load ALL descriptions
    mass_train = pd.read_csv(csv_dir / 'mass_case_description_train_set.csv')
    mass_test = pd.read_csv(csv_dir / 'mass_case_description_test_set.csv')
    calc_train = pd.read_csv(csv_dir / 'calc_case_description_train_set.csv')
    calc_test = pd.read_csv(csv_dir / 'calc_case_description_test_set.csv')
    
    # Add markers
    mass_train['abnormality_type'] = 'mass'
    mass_test['abnormality_type'] = 'mass'
    calc_train['abnormality_type'] = 'calcification'
    calc_test['abnormality_type'] = 'calcification'
    
    mass_train['split'] = 'train'
    mass_test['split'] = 'test'
    calc_train['split'] = 'train'
    calc_test['split'] = 'test'
    
    # Combine
    all_cases = pd.concat([mass_train, mass_test, calc_train, calc_test], ignore_index=True)
    
    print(f"Combined Dataset:")
    print(f"  Total cases: {len(all_cases)}")
    print(f"  Mass: {len(mass_train) + len(mass_test)}")
    print(f"  Calc: {len(calc_train) + len(calc_test)}")
    print(f"\nPathology distribution:")
    print(all_cases['pathology'].value_counts())
    
    # Load dicom_info
    dicom_info = pd.read_csv(csv_dir / 'dicom_info.csv')
    series_to_folder = {str(row['SeriesInstanceUID']): str(row['PatientID']) 
                        for _, row in dicom_info.iterrows()}
    
    print(f"Series mappings: {len(series_to_folder)}")
    
    # Get JPEG folders
    jpeg_folders = {f.name: f for f in jpeg_dir.iterdir() if f.is_dir()}
    print(f"JPEG folders: {len(jpeg_folders)}")
    
    # Match process
    mappings = []
    matched = 0
    missing = 0
    roi_count = 0
    full_count = 0
    
    for idx, row in all_cases.iterrows():
        pathology_str = str(row['pathology']).strip().upper()
        
        if 'BENIGN' in pathology_str:
            label = 0
        elif 'MALIGNANT' in pathology_str:
            label = 1
        else:
            continue
        
        img_path = row['image file path']
        folder_prefix = img_path.split('/')[0]
        
        matching_series = [uid for uid, folder in series_to_folder.items() 
                          if folder.startswith(folder_prefix)]
        
        if len(matching_series) == 0:
            missing += 1
            continue
        
        found = False
        for series_uid in matching_series:
            if series_uid in jpeg_folders:
                folder = jpeg_folders[series_uid]
                roi_jpgs = filter_roi_images(folder)
                
                all_jpgs = list(folder.glob('*.jpg'))
                full_count += len(all_jpgs) - len(roi_jpgs)
                roi_count += len(roi_jpgs)
                
                if len(roi_jpgs) > 0:
                    mappings.append({
                        'image_path': str(roi_jpgs[0]),
                        'patient_id': row['patient_id'],
                        'pathology': label,
                        'pathology_text': pathology_str,
                        'abnormality_type': row['abnormality_type'],
                        'breast_side': row['left or right breast'],
                        'view': row['image view'],
                        'assessment': row['assessment'],
                        'series_uid': series_uid
                    })
                    matched += 1
                    found = True
                    break
        
        if not found:
            missing += 1
    
    print(f"\nImage statistics:")
    print(f"  ROI: {roi_count}")
    print(f"  Full: {full_count}")
    
    print(f"\nMatching: {matched}/{matched+missing} ({matched/(matched+missing)*100:.1f}%)")
    
    mapping_df = pd.DataFrame(mappings)
    
    print(f"\nFinal dataset:")
    print(f"  Total images: {len(mapping_df)}")
    print(f"  Benign: {sum(mapping_df['pathology']==0)}")
    print(f"  Malignant: {sum(mapping_df['pathology']==1)}")
    print(f"  Mass: {sum(mapping_df['abnormality_type']=='mass')}")
    print(f"  Calc: {sum(mapping_df['abnormality_type']=='calcification')}")
    
    # Save
    processed_dir = base_dir / 'processed'
    processed_dir.mkdir(exist_ok=True)
    mapping_df.to_csv(processed_dir / 'combined_mapping_roi.csv', index=False)
    
    # Patient-level 80/10/10 split
    patients_series = mapping_df.groupby('patient_id')['pathology'].first()
    all_patient_ids = patients_series.index.tolist()
    all_labels = patients_series.values.tolist()

    train_pats, temp_pats, _, temp_labels = train_test_split(
        all_patient_ids, all_labels, test_size=0.2, stratify=all_labels, random_state=42
    )
    val_pats, test_pats = train_test_split(
        temp_pats, test_size=0.5, stratify=temp_labels, random_state=42
    )

    train_df = mapping_df[mapping_df['patient_id'].isin(train_pats)]
    val_df = mapping_df[mapping_df['patient_id'].isin(val_pats)]
    test_df = mapping_df[mapping_df['patient_id'].isin(test_pats)]

    print(f"\nSplits (80/10/10):")
    print(f"Train: {len(train_df)} images from {len(train_pats)} patients")
    print(f"  Benign: {sum(train_df['pathology']==0)}, Malignant: {sum(train_df['pathology']==1)}")
    print(f"Val: {len(val_df)} images from {len(val_pats)} patients")
    print(f"  Benign: {sum(val_df['pathology']==0)}, Malignant: {sum(val_df['pathology']==1)}")
    print(f"Test: {len(test_df)} images from {len(test_pats)} patients")
    print(f"  Benign: {sum(test_df['pathology']==0)}, Malignant: {sum(test_df['pathology']==1)}")
    
    train_df.to_csv(processed_dir / 'combined_train_split.csv', index=False)
    val_df.to_csv(processed_dir / 'combined_val_split.csv', index=False)
    test_df.to_csv(processed_dir / 'combined_test_split.csv', index=False)
    
    print("\n" + "="*70)
    print("SUCCESS! Combined dataset (mass + calc) ready")
    print("="*70)
    return mapping_df

# Training utilities
class MammogramROIDataset(Dataset):
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
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform is not None and self.pair_transform is None:
            image = self.transform(image)
            if self.random_erasing is not None:
                image = self.random_erasing(image)
            return image, self.labels[idx]

        if self.pair_transform is not None:
            image = self.pair_transform(image)
        if self.intensity_transform is not None:
            image = self.intensity_transform(image)
        if self.final_transform is not None:
            image = self.final_transform(image)
        if self.random_erasing is not None:
            image = self.random_erasing(image)
        return image, self.labels[idx]

class SharedGeometricTransform:
    def __init__(self, use_aug=True):
        self.use_aug = use_aug

    def __call__(self, img):
        if self.use_aug:
            i, j, h, w = transforms.RandomResizedCrop.get_params(img, scale=(0.8, 1.0), ratio=(3./4., 4./3.))
            img = F.resized_crop(img, i, j, h, w, (224, 224))
            if random.random() < 0.5:
                img = F.hflip(img)
            angle = transforms.RandomRotation.get_params([-15, 15])
            img = F.rotate(img, angle, fill=(0, 0, 0))
        else:
            img = transforms.Resize(256)(img)
            img = transforms.CenterCrop(224)(img)
        return img

def build_mammo_intensity_transform():
    return transforms.Compose([
        transforms.RandomAutocontrast(p=0.3),
        transforms.RandomAdjustSharpness(sharpness_factor=1.2, p=0.2),
        transforms.RandomAffine(degrees=0, translate=None, scale=(0.95, 1.05), shear=None, fill=0),
    ])

imagenet_norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
final_tensor_transform = transforms.Compose([transforms.ToTensor(), imagenet_norm])

class CBFocalLoss(nn.Module):
    def __init__(self, class_counts, beta=0.9999, gamma=2.0, drw=True, total_epochs=20):
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
        pt = probs.gather(1, targets.view(-1, 1)).clamp(min=1e-8, max=1.0)
        focal_factor = torch.pow(1.0 - pt, self.gamma).squeeze(1)
        per_class_weight = class_weights.gather(0, targets)
        loss = - per_class_weight * focal_factor * pt.log().squeeze(1)
        return loss.mean()

def _mixup_data(x, y, alpha=0.2):
    if alpha is None or alpha <= 0:
        return x, y, y, 1.0
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def _mixup_criterion_two_targets(loss_fn, pred, y_a, y_b, lam, epoch=None):
    loss_a = loss_fn(pred, y_a, epoch=epoch)
    loss_b = loss_fn(pred, y_b, epoch=epoch)
    return lam * loss_a + (1 - lam) * loss_b

def _read_combined_splits(processed_dir: Path):
    train_df = pd.read_csv(processed_dir / 'combined_train_split.csv')
    val_df = pd.read_csv(processed_dir / 'combined_val_split.csv')
    test_df = pd.read_csv(processed_dir / 'combined_test_split.csv')

    def extract(df):
        paths = df['image_path'].tolist()
        labels = df['pathology'].astype(int).tolist()
        return paths, labels

    return extract(train_df), extract(val_df), extract(test_df)

def train_and_evaluate_combined():
    base_dir = Path(r'E:\Journal\Breast Cancer\cbis-ddsm-breast-cancer-image-dataset')
    processed_dir = base_dir / 'processed'

    (train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels) = _read_combined_splits(processed_dir)

    print(f"Total images: train={len(train_paths)}, val={len(val_paths)}, test={len(test_paths)}")
    print(f"Train: {np.bincount(np.array(train_labels), minlength=2)}")
    print(f"Val: {np.bincount(np.array(val_labels), minlength=2)}")
    print(f"Test: {np.bincount(np.array(test_labels), minlength=2)}")

    batch_size = 32
    num_classes = 2
    class_counts = np.bincount(np.array(train_labels), minlength=num_classes)
    sample_weights = [1.0 / class_counts[y] if class_counts[y] > 0 else 0.0 for y in train_labels]
    sampler = WeightedRandomSampler(torch.DoubleTensor(sample_weights), len(train_labels), replacement=True)

    train_pair_tf = SharedGeometricTransform(use_aug=True)
    eval_pair_tf = SharedGeometricTransform(use_aug=False)
    intensity_tf = build_mammo_intensity_transform()
    random_erasing_train = transforms.RandomErasing(p=0.25, scale=(0.02, 0.2), ratio=(0.3, 3.3), value='random')

    train_dataset = MammogramROIDataset(train_paths, train_labels, pair_transform=train_pair_tf,
                                        intensity_transform=intensity_tf, final_transform=final_tensor_transform,
                                        random_erasing=random_erasing_train)
    val_dataset = MammogramROIDataset(val_paths, val_labels, pair_transform=eval_pair_tf,
                                      final_transform=final_tensor_transform)
    test_dataset = MammogramROIDataset(test_paths, test_labels, pair_transform=eval_pair_tf,
                                       final_transform=final_tensor_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    num_ftrs = model.classifier[1].in_features
    model.classifier = nn.Sequential(nn.Linear(num_ftrs, 512), nn.ReLU(), nn.Dropout(0.3), nn.Linear(512, 2))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    criterion = CBFocalLoss(class_counts=class_counts, beta=0.9999, gamma=2.0, drw=True, total_epochs=20)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    num_epochs = 20
    best_val_acc = 0.0
    mixup_alpha = 0.2

    for epoch in range(num_epochs):
        model.train()
        correct, total = 0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            mixed_inputs, y_a, y_b, lam = _mixup_data(inputs, labels, alpha=mixup_alpha)
            outputs = model(mixed_inputs)
            loss = _mixup_criterion_two_targets(criterion, outputs, y_a, y_b, lam, epoch=epoch)
            loss.backward()
            optimizer.step()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / max(1, total)

        model.eval()
        val_preds, val_true = [], []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                val_preds.extend(preds.cpu().numpy())
                val_true.extend(labels.cpu().numpy())
        val_acc = accuracy_score(val_true, val_preds)
        scheduler.step(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), str(processed_dir / 'combined_best_model.pth'))

        print(f"Epoch {epoch+1}: Train={train_acc:.4f}, Val={val_acc:.4f} (Best={best_val_acc:.4f})")

    # Load best and evaluate
    model.load_state_dict(torch.load(processed_dir / 'combined_best_model.pth'))
    model.eval()
    test_preds, test_true, test_probs = [], [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, 1)
            _, preds = torch.max(outputs, 1)
            test_preds.extend(preds.cpu().numpy())
            test_true.extend(labels.cpu().numpy())
            test_probs.extend(probs[:, 1].cpu().numpy())

    acc = accuracy_score(test_true, test_preds)
    precision = precision_score(test_true, test_preds)
    recall = recall_score(test_true, test_preds)
    f1 = f1_score(test_true, test_preds)
    auc = roc_auc_score(test_true, test_probs)
    fpr, tpr, _ = roc_curve(test_true, test_probs)
    cm = confusion_matrix(test_true, test_preds)

    print("\n" + "="*60)
    print("COMBINED DATASET RESULTS (Mass + Calc)")
    print("="*60)
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1: {f1:.4f}")
    print(f"AUC: {auc:.4f}")
    print(f"\n{classification_report(test_true, test_preds, target_names=['Benign', 'Malignant'])}")
    print("="*60)

    # Save ROC
    with open(processed_dir / 'combined_roc.json', 'w') as f:
        json.dump({'fpr': fpr.tolist(), 'tpr': tpr.tolist(), 'auc': float(auc)}, f)
    print(f"ROC saved: {processed_dir / 'combined_roc.json'}")

if __name__ == "__main__":
    # Run combined dataset mapping
    mapping_df = create_combined_mass_calc_mapping()
    
    # Train if splits exist
    base_dir = Path(r'E:\Journal\Breast Cancer\cbis-ddsm-breast-cancer-image-dataset')
    processed_dir = base_dir / 'processed'
    if (processed_dir / 'combined_train_split.csv').exists():
        train_and_evaluate_combined()