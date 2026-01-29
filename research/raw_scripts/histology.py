import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from torchvision.transforms import functional as F
from torchvision import transforms, models
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score, classification_report, confusion_matrix, roc_curve
from new import (
    save_roc_curve,
    load_roc_curves,
    plot_roc_curves,
    save_single_roc_pdf,
    save_confusion_matrix_pdf,
    save_roc_comparison_pdf,
)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -------- Advanced Augmentation Classes (from bussi.py) --------
class SharedGeometricTransform:
    """Apply the same geometric augmentations to image (PIL-level)."""
    def __init__(self, use_aug=True):
        self.use_aug = use_aug

    def __call__(self, img):
        if self.use_aug:
            # RandomResizedCrop params
            i, j, h, w = transforms.RandomResizedCrop.get_params(img, scale=(0.8, 1.0), ratio=(3./4., 4./3.))
            img = F.resized_crop(img, i, j, h, w, (224, 224))
            # Horizontal flip
            if random.random() < 0.5:
                img = F.hflip(img)
            # Small rotation
            angle = transforms.RandomRotation.get_params([-15, 15])
            img = F.rotate(img, angle, fill=(0, 0, 0))
        else:
            # Eval: Resize + CenterCrop
            img = transforms.Resize(256)(img)
            img = transforms.CenterCrop(224)(img)
        return img

# Histology intensity (PIL) - keep mild and grayscale-friendly
def build_histo_intensity_transform():
    return transforms.Compose([
        transforms.RandomAutocontrast(p=0.3),
        transforms.RandomAdjustSharpness(sharpness_factor=1.2, p=0.2),
        transforms.RandomAffine(degrees=0, translate=None, scale=(0.95, 1.05), shear=None, fill=0),
    ])

# Final tensor transform
imagenet_norm = transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
final_tensor_transform = transforms.Compose([
    transforms.ToTensor(),
    imagenet_norm,
])

random_erasing_train = transforms.RandomErasing(p=0.25, scale=(0.02, 0.2), ratio=(0.3, 3.3), value='random')

# -------- Advanced Histology Dataset Class --------
class HistologyDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None, pair_transform=None,
                 histo_intensity_transform=None, final_transform=None, random_erasing=None):
        self.image_paths = image_paths
        self.labels = labels
        # Back-compat: if transform is provided, apply to all images equally
        self.transform = transform
        # New: synchronized geometric ops over the pair
        self.pair_transform = pair_transform
        # New: modality-specific intensity ops (PIL-level)
        self.histo_intensity_transform = histo_intensity_transform
        # New: final tensor transform (ToTensor + Normalize)
        self.final_transform = final_transform
        # New: random erasing (tensor-level, training only)
        self.random_erasing = random_erasing
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        try:
            img = Image.open(self.image_paths[idx]).convert('RGB')
            
            # Legacy single transform path
            if self.transform is not None and self.pair_transform is None and \
               self.histo_intensity_transform is None and self.final_transform is None:
                img = self.transform(img)
                return img, self.labels[idx]

            # Synchronized geometric transforms (PIL)
            if self.pair_transform is not None:
                img = self.pair_transform(img)

            # Modality-specific intensity transforms (PIL)
            if self.histo_intensity_transform is not None:
                img = self.histo_intensity_transform(img)

            # Final transforms to tensor + normalize
            if self.final_transform is not None:
                img = self.final_transform(img)

            # Optional Random Erasing (tensor-level)
            if self.random_erasing is not None:
                img = self.random_erasing(img)

            return img, self.labels[idx]
        except Exception as e:
            print(f"Error loading image {self.image_paths[idx]}: {e}")
            # Return dummy image if loading fails
            dummy_img = torch.zeros(3, 224, 224)
            return dummy_img, self.labels[idx]

# -------- BreakHis Collection Functions with Patient-Level Split --------
def collect_images_by_patient(base_dir, benign_word='benign', malignant_word='malignant'):
    """Collect BreakHis histology images by patient to avoid data leakage"""
    patient_data = {}  # {patient_id: {'paths': [], 'label': 0/1, 'magnifications': []}}
    
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.png'):
                path = os.path.join(root, file)
                
                # Extract patient ID from path (e.g., SOB_B_A_14-22549AB)
                path_parts = path.split(os.sep)
                patient_id = None
                magnification = None
                
                for part in path_parts:
                    if part.startswith('SOB_') and len(part) > 10 and not part.endswith('.png'):  # Directory name, not file
                        patient_id = part
                    elif part in ['40X', '100X', '200X', '400X']:
                        magnification = part
                
                if patient_id and magnification:
                    # Determine label
                    if benign_word in root.lower():
                        label = 0
                    elif malignant_word in root.lower():
                        label = 1
                    else:
                        continue
                    
                    # Store patient data
                    if patient_id not in patient_data:
                        patient_data[patient_id] = {
                            'paths': [],
                            'label': label,
                            'magnifications': set()
                        }
                    
                    patient_data[patient_id]['paths'].append(path)
                    patient_data[patient_id]['magnifications'].add(magnification)
    
    return patient_data

def collect_images(base_dir, benign_word='benign', malignant_word='malignant', exclude_mask=False, exclude_normal=False):
    """Legacy function for backward compatibility"""
    paths = []
    labels = []
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.png'):
                if exclude_mask and '_mask' in file:
                    continue
                if exclude_normal and 'normal' in root:
                    continue
                path = os.path.join(root, file)
                if benign_word in root.lower():
                    labels.append(0)
                elif malignant_word in root.lower():
                    labels.append(1)
                else:
                    continue
                paths.append(path)
    return paths, labels

# -------- Patient-Level Stratified Split --------
def patient_level_split(patient_data, train_ratio=0.7, test_ratio=0.2, seed=42):
    """Patient-level split to avoid data leakage"""
    rng = random.Random(seed)
    
    # Group patients by class
    benign_patients = []
    malignant_patients = []
    
    for patient_id, data in patient_data.items():
        if data['label'] == 0:
            benign_patients.append(patient_id)
        else:
            malignant_patients.append(patient_id)
    
    # Shuffle patients
    rng.shuffle(benign_patients)
    rng.shuffle(malignant_patients)
    
    # Split patients
    def split_patients(patients, train_ratio, test_ratio):
        n = len(patients)
        n_train = int(round(train_ratio * n))
        n_test = int(round(test_ratio * n))
        n_val = n - n_train - n_test
        
        train_patients = patients[:n_train]
        test_patients = patients[n_train:n_train+n_test]
        val_patients = patients[n_train+n_test:]
        
        return train_patients, val_patients, test_patients
    
    train_benign, val_benign, test_benign = split_patients(benign_patients, train_ratio, test_ratio)
    train_malignant, val_malignant, test_malignant = split_patients(malignant_patients, train_ratio, test_ratio)
    
    # Combine splits
    train_patients = train_benign + train_malignant
    val_patients = val_benign + val_malignant
    test_patients = test_benign + test_malignant
    
    # Shuffle final splits
    rng.shuffle(train_patients)
    rng.shuffle(val_patients)
    rng.shuffle(test_patients)
    
    return train_patients, val_patients, test_patients

# (Removed) Patient-Level K-Fold Split

# -------- Legacy Stratified Split (Same as main code) --------
def stratified_split_indices(labels, train_ratio=0.7, test_ratio=0.2, seed=42):
    """Stratified split ensuring no data leakage"""
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
        n_val = n - n_train - n_test
        train_idx.extend(idxs[:n_train])
        test_idx.extend(idxs[n_train:n_train+n_test])
        val_idx.extend(idxs[n_train+n_test:])
    rng.shuffle(train_idx); rng.shuffle(test_idx); rng.shuffle(val_idx)
    return train_idx, val_idx, test_idx

# -------- Histology Transfer Learning Model --------
class HistologyTransferLearningModel(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super(HistologyTransferLearningModel, self).__init__()
        # Use same EfficientNet-B0 as main code
        self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None)
        # Replace the classifier head
        num_ftrs = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)

# -------- Class-Balanced Focal Loss (Same as main code) --------
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
        pt = probs.gather(1, targets.view(-1,1)).clamp(min=1e-8, max=1.0)
        focal_factor = torch.pow(1.0 - pt, self.gamma).squeeze(1)
        per_class_weight = class_weights.gather(0, targets)
        loss = - per_class_weight * focal_factor * pt.log().squeeze(1)
        return loss.mean()

# -------- Mixup Training Functions (from bussi.py) --------
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

# -------- Advanced Training Function --------
def train_model(model, train_loader, val_loader, epochs=20, lr=0.0001, mixup_alpha=0.2, class_counts=None, best_model_path='best_histology_model.pth'):
    """Advanced training with mixup and focal loss"""
    criterion = CBFocalLoss(class_counts=class_counts if class_counts is not None else [1,1], 
                           beta=0.9999, gamma=2.0, drw=True, total_epochs=epochs)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    
    best_val_acc = 0.0
    
    print(f"Training for {epochs} epochs with Mixup (alpha={mixup_alpha})...")
    
    for epoch in range(epochs):
        # Training phase with Mixup
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            
            # Apply Mixup
            mixed_images, y_a, y_b, lam = _mixup_data(images, labels, alpha=mixup_alpha)
            outputs = model(mixed_images)
            loss = _mixup_criterion_two_targets(criterion, outputs, y_a, y_b, lam, epoch=epoch)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            # Calculate accuracy (approximate for mixup)
            _, preds = torch.max(outputs, 1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)
        
        # Validation phase
        val_acc = evaluate_accuracy(model, val_loader)
        scheduler.step(val_acc)
        
        # Track best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # Save best model
            torch.save(model.state_dict(), best_model_path)
        
        print(f"Epoch {epoch+1}/{epochs}: Train Loss: {train_loss/len(train_loader):.4f}, "
              f"Train Acc: {train_correct/train_total:.4f}, Val Acc: {val_acc:.4f}")
        
        # No early stopping; continue training for all epochs
    
    # Load best model
    model.load_state_dict(torch.load(best_model_path))
    return model

def evaluate_accuracy(model, loader):
    """Evaluate model accuracy"""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total

def get_probabilities(model, loader):
    """Get prediction probabilities"""
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels.numpy())
    return np.array(all_probs), np.array(all_labels)

def comprehensive_evaluation(model, test_loader):
    """Comprehensive evaluation with metrics and visualizations"""
    print(f"\n{'='*60}")
    print("🔬 HISTOLOGY MODEL EVALUATION")
    print(f"{'='*60}")
    
    # Get predictions
    probs, true_labels = get_probabilities(model, test_loader)
    preds = (probs > 0.5).astype(int)
    
    # Calculate metrics
    accuracy = accuracy_score(true_labels, preds)
    precision = precision_score(true_labels, preds)
    recall = recall_score(true_labels, preds)
    f1 = f1_score(true_labels, preds)
    
    try:
        auc = roc_auc_score(true_labels, probs)
        fpr, tpr, _ = roc_curve(true_labels, probs)
    except ValueError:
        auc = float('nan')
        fpr, tpr = np.array([0.0, 1.0]), np.array([0.0, 1.0])
    
    print(f"📊 Test Results:")
    print(f"   • Total Samples: {len(true_labels)}")
    print(f"   • Benign Samples: {sum(true_labels == 0)}")
    print(f"   • Malignant Samples: {sum(true_labels == 1)}")
    print(f"   • Accuracy: {accuracy:.4f}")
    print(f"   • Precision: {precision:.4f}")
    print(f"   • Recall: {recall:.4f}")
    print(f"   • F1-Score: {f1:.4f}")
    print(f"   • AUC-ROC: {auc:.4f}")
    
    # Classification Report
    print(f"\n📋 Classification Report:")
    print(classification_report(true_labels, preds, target_names=['Benign', 'Malignant']))
    
    # Confusion Matrix
    cm = confusion_matrix(true_labels, preds)
    print(f"\n🔢 Confusion Matrix:")
    print("                Predicted")
    print("              Benign  Malignant")
    print(f"Actual Benign    {cm[0,0]:4d}      {cm[0,1]:4d}")
    print(f"       Malignant {cm[1,0]:4d}      {cm[1,1]:4d}")
    
    # Visualize Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Benign', 'Malignant'], 
                yticklabels=['Benign', 'Malignant'])
    plt.title('Histology Model - Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.show()
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'confusion_matrix': cm.tolist(),
        'fpr': fpr.tolist(),
        'tpr': tpr.tolist()
    }

# ROC utilities are imported from new.py

# -------- Main Execution --------
if __name__ == "__main__":
    # Create output directories
    os.makedirs('plots', exist_ok=True)
    os.makedirs('roc_curves', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    print("🔬 Histology Transfer Learning Model")
    print("=" * 50)
    
    # Data collection with patient-level split (updated path)
    breakhis_dir = r'E:\Journal\Breast Cancer\breast\BreaKHis_v1\BreaKHis_v1\histology_slides\breast'
    print("Collecting BreakHis histology images by patient...")
    
    # Collect patient data
    patient_data = collect_images_by_patient(breakhis_dir)
    print(f"Found {len(patient_data)} patients")
    
    # Count images and patients by class
    total_images = sum(len(data['paths']) for data in patient_data.values())
    benign_patients = sum(1 for data in patient_data.values() if data['label'] == 0)
    malignant_patients = sum(1 for data in patient_data.values() if data['label'] == 1)
    benign_images = sum(len(data['paths']) for data in patient_data.values() if data['label'] == 0)
    malignant_images = sum(len(data['paths']) for data in patient_data.values() if data['label'] == 1)
    
    print(f"Total images: {total_images}")
    print(f"Benign patients: {benign_patients} ({benign_images} images)")
    print(f"Malignant patients: {malignant_patients} ({malignant_images} images)")
    
    # Debug: Show sample patient data
    if patient_data:
        sample_patient = list(patient_data.keys())[0]
        sample_data = patient_data[sample_patient]
        print(f"\nSample patient: {sample_patient}")
        print(f"  Label: {sample_data['label']}")
        print(f"  Images: {len(sample_data['paths'])}")
        print(f"  Magnifications: {sample_data['magnifications']}")
        print(f"  Sample path: {sample_data['paths'][0] if sample_data['paths'] else 'None'}")
    
    if len(patient_data) == 0:
        print("❌ No patients found!")
        exit()
    
    # Patient-level split to avoid data leakage
    train_patients, val_patients, test_patients = patient_level_split(patient_data, 0.7, 0.2, 42)
    
    print(f"Patient split: Train: {len(train_patients)}, Val: {len(val_patients)}, Test: {len(test_patients)}")
    
    # Create datasets from patient splits with magnification control
    def get_paths_and_labels(patient_list, magnification=None):
        paths = []
        labels = []
        for patient_id in patient_list:
            patient_info = patient_data[patient_id]
            
            if magnification:
                # Filter by specific magnification (use both forward and backward slashes)
                filtered_paths = [p for p in patient_info['paths'] if f'/{magnification}/' in p or f'\\{magnification}\\' in p]
                paths.extend(filtered_paths)
                labels.extend([patient_info['label']] * len(filtered_paths))
            else:
                # Use all magnifications
                paths.extend(patient_info['paths'])
                labels.extend([patient_info['label']] * len(patient_info['paths']))
        return paths, labels
    
    # Magnification-wise split approach
    print("Using magnification-wise split approach...")
    
    # Step 1: Split by magnification first
    magnifications = ['40X', '100X', '200X', '400X']
    
    # Step 2: For each magnification, do patient-level split
    all_train_paths, all_train_labels = [], []
    all_val_paths, all_val_labels = [], []
    all_test_paths, all_test_labels = [], []
    
    for mag in magnifications:
        print(f"\nProcessing {mag} magnification...")
        
        # Get patient data for this magnification only
        mag_patient_data = {}
        for patient_id, data in patient_data.items():
            # Use both forward and backward slashes for cross-platform compatibility
            mag_paths = [p for p in data['paths'] if f'/{mag}/' in p or f'\\{mag}\\' in p]
            if mag_paths:  # Only if patient has this magnification
                mag_patient_data[patient_id] = {
                    'paths': mag_paths,
                    'label': data['label'],
                    'magnifications': {mag}
                }
        
        print(f"  Found {len(mag_patient_data)} patients with {mag} images")
        if len(mag_patient_data) > 0:
            sample_patient = list(mag_patient_data.keys())[0]
            sample_paths = mag_patient_data[sample_patient]['paths']
            print(f"  Sample {mag} path: {sample_paths[0] if sample_paths else 'None'}")
        
        if len(mag_patient_data) == 0:
            print(f"  No patients found for {mag}")
            continue
            
        # Patient-level split for this magnification
        mag_train_patients, mag_val_patients, mag_test_patients = patient_level_split(
            mag_patient_data, 0.7, 0.2, 42
        )
        
        print(f"  {mag} patient split: Train={len(mag_train_patients)}, Val={len(mag_val_patients)}, Test={len(mag_test_patients)}")
        
        # Get paths and labels for this magnification
        mag_train_paths, mag_train_labels = get_paths_and_labels(mag_train_patients, mag)
        mag_val_paths, mag_val_labels = get_paths_and_labels(mag_val_patients, mag)
        mag_test_paths, mag_test_labels = get_paths_and_labels(mag_test_patients, mag)
        
        print(f"  {mag}: Train={len(mag_train_paths)}, Val={len(mag_val_paths)}, Test={len(mag_test_paths)}")
        
        # Add to overall datasets
        all_train_paths.extend(mag_train_paths)
        all_train_labels.extend(mag_train_labels)
        all_val_paths.extend(mag_val_paths)
        all_val_labels.extend(mag_val_labels)
        all_test_paths.extend(mag_test_paths)
        all_test_labels.extend(mag_test_labels)
    
    # Final datasets
    train_paths, train_labels = all_train_paths, all_train_labels
    val_paths, val_labels = all_val_paths, all_val_labels
    test_paths, test_labels = all_test_paths, all_test_labels
    
    print(f"Data split: Train: {len(train_paths)}, Val: {len(val_paths)}, Test: {len(test_paths)}")
    
    # Show magnification distribution
    def count_magnifications(paths):
        mag_counts = {'40X': 0, '100X': 0, '200X': 0, '400X': 0}
        for path in paths:
            for mag in mag_counts.keys():
                if f'/{mag}/' in path or f'\\{mag}\\' in path:
                    mag_counts[mag] += 1
                    break
        return mag_counts
    
    train_mag = count_magnifications(train_paths)
    val_mag = count_magnifications(val_paths)
    test_mag = count_magnifications(test_paths)
    
    print(f"\n📊 Magnification Distribution:")
    print(f"Train: 40X={train_mag['40X']}, 100X={train_mag['100X']}, 200X={train_mag['200X']}, 400X={train_mag['400X']}")
    print(f"Val:   40X={val_mag['40X']}, 100X={val_mag['100X']}, 200X={val_mag['200X']}, 400X={val_mag['400X']}")
    print(f"Test:  40X={test_mag['40X']}, 100X={test_mag['100X']}, 200X={test_mag['200X']}, 400X={test_mag['400X']}")
    
    # Check if we have any training data
    if len(train_paths) == 0:
        print("❌ No training data found! Please check the dataset path and structure.")
        exit()
    
    # ---------------- Merged training and evaluation (all magnifications) ----------------
    print(f"\n{'='*80}")
    print("🔬 MERGED TRAINING & EVALUATION (All Magnifications)")
    print(f"{'='*80}")

    # Create class-balanced sampler for merged training
    merged_num_classes = int(max(train_labels)) + 1 if len(train_labels) > 0 else 2
    merged_class_counts = np.bincount(np.array(train_labels, dtype=int), minlength=merged_num_classes)
    merged_sample_weights = [1.0 / merged_class_counts[y] if merged_class_counts[y] > 0 else 0.0 for y in train_labels]
    merged_sampler = WeightedRandomSampler(weights=torch.DoubleTensor(merged_sample_weights), num_samples=len(train_labels), replacement=True)

    # Datasets/loaders for merged training
    merged_train_dataset = HistologyDataset(
        train_paths, train_labels,
        pair_transform=SharedGeometricTransform(use_aug=True),
        histo_intensity_transform=build_histo_intensity_transform(),
        final_transform=final_tensor_transform,
        random_erasing=random_erasing_train,
    )
    eval_pair_tf = SharedGeometricTransform(use_aug=False)
    merged_val_dataset = HistologyDataset(
        val_paths, val_labels,
        pair_transform=eval_pair_tf,
        final_transform=final_tensor_transform,
    )
    merged_test_dataset = HistologyDataset(
        test_paths, test_labels,
        pair_transform=eval_pair_tf,
        final_transform=final_tensor_transform,
    )
    
    merged_train_loader = DataLoader(merged_train_dataset, batch_size=8, shuffle=False, sampler=merged_sampler, num_workers=0, pin_memory=False)
    merged_val_loader = DataLoader(merged_val_dataset, batch_size=8, shuffle=False, num_workers=0, pin_memory=False)
    merged_test_loader = DataLoader(merged_test_dataset, batch_size=8, shuffle=False, num_workers=0, pin_memory=False)

    # Model for merged training
    merged_model = HistologyTransferLearningModel().to(device)
    print(f"  Model parameters: {sum(p.numel() for p in merged_model.parameters()):,}")

    # Train merged model
    merged_best_path = 'best_histology_model_merged.pth'
    merged_model = train_model(merged_model, merged_train_loader, merged_val_loader, epochs=20, lr=0.0001, mixup_alpha=0.2, class_counts=merged_class_counts, best_model_path=merged_best_path)

    # Save final merged model
    torch.save(merged_model.state_dict(), 'histology_transfer_model_merged.pth')
    print("  💾 Saved merged model")

    # Evaluate merged model on merged test set
    print("  📊 Evaluating MERGED model on MERGED test set")
    merged_results = comprehensive_evaluation(merged_model, merged_test_loader)
    # Save merged ROC
    save_roc_curve(merged_results['fpr'], merged_results['tpr'], label='Merged', out_path=os.path.join('roc_curves', 'roc_merged.json'))
    # Save PDFs
    save_single_roc_pdf('Merged', merged_results['fpr'], merged_results['tpr'], out_pdf=os.path.join('plots', 'roc_merged.pdf'))
    save_confusion_matrix_pdf(np.array(merged_results['confusion_matrix']), out_pdf=os.path.join('plots', 'cm_merged.pdf'))

    # ---------------- Per-magnification training and evaluation ----------------
    print(f"\n{'='*80}")
    print("🔬 PER-MAGNIFICATION TRAINING & EVALUATION")
    print(f"{'='*80}")
    
    # (Removed) Patient-level 5-Fold CV (Merged)

    per_mag_results = {}
    
    for mag in ['40X', '100X', '200X', '400X']:
        print(f"\n⚙️ Training for magnification: {mag}")

        # Build per-magnification paths/labels
        mag_train_paths, mag_train_labels = get_paths_and_labels(train_patients, mag)
        mag_val_paths, mag_val_labels = get_paths_and_labels(val_patients, mag)
        mag_test_paths, mag_test_labels = get_paths_and_labels(test_patients, mag)

        print(f"  {mag}: Train={len(mag_train_paths)}, Val={len(mag_val_paths)}, Test={len(mag_test_paths)}")

        if len(mag_train_paths) == 0 or len(mag_val_paths) == 0 or len(mag_test_paths) == 0:
            print(f"  ⚠️ Skipping {mag} due to insufficient data in one of the splits")
            continue
            
        # Sampler and class counts
        mag_num_classes = int(max(mag_train_labels)) + 1 if len(mag_train_labels) > 0 else 2
        mag_class_counts = np.bincount(np.array(mag_train_labels, dtype=int), minlength=mag_num_classes)
        mag_sample_weights = [1.0 / mag_class_counts[y] if mag_class_counts[y] > 0 else 0.0 for y in mag_train_labels]
        mag_sampler = WeightedRandomSampler(weights=torch.DoubleTensor(mag_sample_weights), num_samples=len(mag_train_labels), replacement=True)

        # Datasets/loaders
        mag_train_dataset = HistologyDataset(
            mag_train_paths, mag_train_labels,
            pair_transform=SharedGeometricTransform(use_aug=True),
            histo_intensity_transform=build_histo_intensity_transform(),
            final_transform=final_tensor_transform,
            random_erasing=random_erasing_train,
        )
        eval_pair_tf = SharedGeometricTransform(use_aug=False)
        mag_val_dataset = HistologyDataset(
            mag_val_paths, mag_val_labels,
            pair_transform=eval_pair_tf,
            final_transform=final_tensor_transform,
        )
        mag_test_dataset = HistologyDataset(
            mag_test_paths, mag_test_labels,
            pair_transform=eval_pair_tf,
            final_transform=final_tensor_transform,
        )

        mag_train_loader = DataLoader(mag_train_dataset, batch_size=8, shuffle=False, sampler=mag_sampler, num_workers=0, pin_memory=False)
        mag_val_loader = DataLoader(mag_val_dataset, batch_size=8, shuffle=False, num_workers=0, pin_memory=False)
        mag_test_loader = DataLoader(mag_test_dataset, batch_size=8, shuffle=False, num_workers=0, pin_memory=False)

        # Model
        mag_model = HistologyTransferLearningModel().to(device)
        print(f"  Model parameters: {sum(p.numel() for p in mag_model.parameters()):,}")

        # Train
        best_path = f'best_histology_model_{mag}.pth'
        mag_model = train_model(mag_model, mag_train_loader, mag_val_loader, epochs=20, lr=0.0001, mixup_alpha=0.2, class_counts=mag_class_counts, best_model_path=best_path)

        # Save final
        torch.save(mag_model.state_dict(), f'histology_transfer_model_{mag}.pth')
        print(f"  💾 Saved model for {mag}")
        
        # Evaluate
        print(f"  📊 Evaluating {mag} on {mag} test set")
        per_mag_results[mag] = comprehensive_evaluation(mag_model, mag_test_loader)
        # Save per-mag ROC
        save_roc_curve(per_mag_results[mag]['fpr'], per_mag_results[mag]['tpr'], label=f'{mag}', out_path=os.path.join('roc_curves', f'roc_{mag}.json'))
        # Save PDFs
        save_single_roc_pdf(f'{mag}', per_mag_results[mag]['fpr'], per_mag_results[mag]['tpr'], out_pdf=os.path.join('plots', f'roc_{mag}.pdf'))
        save_confusion_matrix_pdf(np.array(per_mag_results[mag]['confusion_matrix']), out_pdf=os.path.join('plots', f'cm_{mag}.pdf'))

    # Save per-mag results
    import json
    with open('histology_per_magnification_results.json', 'w') as f:
        json.dump(per_mag_results, f, indent=2)
    print(f"\n💾 Per-magnification results saved to histology_per_magnification_results.json")
    print(f"🎯 Completed per-magnification training & evaluation")

    # (Removed) Per-magnification 5-Fold CV

    # Final combined ROC comparison
    print(f"\n{'='*80}")
    print("📊 GENERATING FINAL ROC COMPARISON")
    print(f"{'='*80}")
    
    all_roc_files = [
        os.path.join('roc_curves', 'roc_merged.json'),
        os.path.join('roc_curves', 'roc_40X.json'),
        os.path.join('roc_curves', 'roc_100X.json'),
        os.path.join('roc_curves', 'roc_200X.json'),
        os.path.join('roc_curves', 'roc_400X.json'),
    ]
    existing = [f for f in all_roc_files if os.path.exists(f)]
    if len(existing) > 1:
        save_roc_comparison_pdf(
            existing,
            out_pdf=os.path.join('plots', 'all_models_roc_comparison.pdf'),
            title='BreakHis Dataset - ROC Comparison (All Models)'
        )
        print("✅ ROC comparison saved: plots/all_models_roc_comparison.pdf")
    
    print("\n🎉 ALL DONE! Check 'plots/' folder.")
