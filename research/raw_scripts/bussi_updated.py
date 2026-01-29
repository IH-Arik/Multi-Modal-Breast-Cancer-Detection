import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score, classification_report, confusion_matrix, roc_curve
import numpy as np
from torchvision.transforms import functional as F
from torch.utils.data.sampler import WeightedRandomSampler
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import time
import json
from datetime import datetime, timedelta


def _json_default(obj):
    """Convert numpy / torch types into JSON-serializable Python objects."""
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if torch.is_tensor(obj):
        return obj.detach().cpu().tolist() if obj.numel() != 1 else obj.item()
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==================== Dataset & Data Loading ====================
class BUSIDataset(Dataset):
    def __init__(self, ultra_paths, labels, transform=None, pair_transform=None,
                 ultra_intensity_transform=None, final_transform=None, random_erasing=None):
        self.ultra_paths = ultra_paths
        self.labels = labels
        self.transform = transform
        self.pair_transform = pair_transform
        self.ultra_intensity_transform = ultra_intensity_transform
        self.final_transform = final_transform
        self.random_erasing = random_erasing

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        ultra_img = Image.open(self.ultra_paths[idx]).convert('RGB')
        label = self.labels[idx]

        if self.transform is not None and self.pair_transform is None:
            ultra_img = self.transform(ultra_img)
            return ultra_img, label

        if self.pair_transform is not None:
            ultra_img = self.pair_transform(ultra_img)
        if self.ultra_intensity_transform is not None:
            ultra_img = self.ultra_intensity_transform(ultra_img)
        if self.final_transform is not None:
            ultra_img = self.final_transform(ultra_img)
        if self.random_erasing is not None:
            ultra_img = self.random_erasing(ultra_img)

        return ultra_img, label

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

def build_ultra_intensity_transform():
    return transforms.Compose([
        transforms.RandomAutocontrast(p=0.3),
        transforms.RandomAdjustSharpness(sharpness_factor=1.2, p=0.2),
        transforms.RandomAffine(degrees=0, translate=None, scale=(0.95, 1.05), shear=None, fill=0),
    ])

imagenet_norm = transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
final_tensor_transform = transforms.Compose([transforms.ToTensor(), imagenet_norm])
random_erasing_train = transforms.RandomErasing(p=0.25, scale=(0.02, 0.2), ratio=(0.3, 3.3), value='random')

def collect_images(base_dir, benign_word='benign', malignant_word='malignant', exclude_mask=False, exclude_normal=False, normal_as_benign=False):
    paths = []
    labels = []
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.png'):
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

def create_dataloaders(train_paths, train_labels, val_paths, val_labels, test_paths, test_labels, batch_size=16):
    num_classes = int(max(train_labels)) + 1
    class_counts = np.bincount(np.array(train_labels), minlength=num_classes)
    sample_weights = [1.0 / class_counts[y] if class_counts[y] > 0 else 0.0 for y in train_labels]
    sampler = WeightedRandomSampler(weights=torch.DoubleTensor(sample_weights), num_samples=len(train_labels), replacement=True)

    train_ds = BUSIDataset(train_paths, train_labels, pair_transform=SharedGeometricTransform(use_aug=True),
                           ultra_intensity_transform=build_ultra_intensity_transform(),
                           final_transform=final_tensor_transform, random_erasing=random_erasing_train)
    eval_pair_tf = SharedGeometricTransform(use_aug=False)
    val_ds = BUSIDataset(val_paths, val_labels, pair_transform=eval_pair_tf, final_transform=final_tensor_transform)
    test_ds = BUSIDataset(test_paths, test_labels, pair_transform=eval_pair_tf, final_transform=final_tensor_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False, sampler=sampler, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    return train_loader, val_loader, test_loader, class_counts

# ==================== 10 Backbone Models ====================
class BUSIBackboneModel(nn.Module):
    """Universal model class for all backbones"""
    def __init__(self, backbone_name='resnet50', num_classes=2):
        super(BUSIBackboneModel, self).__init__()
        self.backbone_name = backbone_name
        
        if backbone_name == 'resnet50':
            self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            feat_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
            
        elif backbone_name == 'efficientnet_b4':
            self.backbone = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.DEFAULT)
            feat_dim = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
            
        elif backbone_name == 'efficientnet_b0':
            self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
            feat_dim = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
            
        elif backbone_name == 'densenet169':
            self.backbone = models.densenet169(weights=models.DenseNet169_Weights.DEFAULT)
            feat_dim = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Identity()
            
        elif backbone_name == 'swin_t':
            self.backbone = models.swin_t(weights=models.Swin_T_Weights.DEFAULT)
            feat_dim = self.backbone.head.in_features
            self.backbone.head = nn.Identity()
            
        elif backbone_name == 'vit_b_16':
            self.backbone = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
            feat_dim = self.backbone.heads.head.in_features
            self.backbone.heads.head = nn.Identity()
            
        elif backbone_name == 'regnet_y_400mf':
            self.backbone = models.regnet_y_400mf(weights=models.RegNet_Y_400MF_Weights.DEFAULT)
            feat_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
            
        elif backbone_name == 'efficientnet_v2_s':
            self.backbone = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.DEFAULT)
            feat_dim = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
            
        elif backbone_name == 'resnext50_32x4d':
            self.backbone = models.resnext50_32x4d(weights=models.ResNeXt50_32X4D_Weights.DEFAULT)
            feat_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
            
        elif backbone_name == 'wide_resnet50_2':
            self.backbone = models.wide_resnet50_2(weights=models.Wide_ResNet50_2_Weights.DEFAULT)
            feat_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        
        else:
            raise ValueError(f"Unknown backbone: {backbone_name}")
        
        # Classifier head
        self.fc = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        feat = self.backbone(x)
        out = self.fc(feat)
        return out

# ==================== Loss Function ====================
class CBFocalLoss(nn.Module):
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

# ==================== Training & Evaluation ====================
def train_model(model, train_loader, val_loader, epochs=50, lr=0.0001, mixup_alpha=0.2, class_counts=None, backbone_name=''):
    criterion = CBFocalLoss(class_counts=class_counts if class_counts is not None else [1,1], 
                            beta=0.9999, gamma=2.0, drw=True, total_epochs=epochs)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    def _mixup_data(x, y, alpha=0.2):
        if alpha is None or alpha <= 0:
            return x, y, y, 1.0
        lam = np.random.beta(alpha, alpha)
        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(x.device)
        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam

    def _mixup_criterion(loss_fn, pred, y_a, y_b, lam, epoch=None):
        loss_a = loss_fn(pred, y_a, epoch=epoch)
        loss_b = loss_fn(pred, y_b, epoch=epoch)
        return lam * loss_a + (1 - lam) * loss_b

    # Training history tracking
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    best_val_acc = 0.0
    train_start = time.time()
    epoch_times = []
    
    print(f"🚀 Starting {epochs}-epoch training for {backbone_name}...")
    
    for epoch in range(epochs):
        epoch_start = time.time()
        # Training phase
        model.train()
        correct = 0
        total = 0
        running_loss = 0.0
        for ultra_img, labels in train_loader:
            ultra_img, labels = ultra_img.to(device), labels.to(device)
            optimizer.zero_grad()
            mixed_ultra, y_a, y_b, lam = _mixup_data(ultra_img, labels, alpha=mixup_alpha)
            outputs = model(mixed_ultra)
            loss = _mixup_criterion(criterion, outputs, y_a, y_b, lam, epoch=epoch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            # Skip accuracy calculation during mixup training as it's misleading
            # Using original labels with mixed inputs gives incorrect accuracy
            # _, preds = torch.max(outputs, 1)
            # correct += (preds==labels).sum().item()
            # total += labels.size(0)
            
            # For mixup, we'll calculate accuracy differently
            with torch.no_grad():
                _, preds = torch.max(outputs, 1)
                # Calculate accuracy as weighted combination based on lambda
                correct_a = (preds == y_a).float()
                correct_b = (preds == y_b).float()
                mixed_correct = lam * correct_a + (1 - lam) * correct_b
                correct += mixed_correct.sum().item()
                total += labels.size(0)
        
        train_loss = running_loss / len(train_loader)
        train_acc = correct / total
        
        # Validation phase
        val_acc, val_loss = evaluate_with_loss(model, val_loader, criterion, epoch)
        
        # Store history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
        
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
        
        # Print animated progress with all metrics
        print(f"⏰ [{current_time}] Epoch {epoch+1:2d}/{epochs} |{bar}| {progress*100:5.1f}% | "
              f"Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | Val: {val_acc:.4f} | "
              f"Time: {epoch_time:.1f}s | ETA: {eta}")
    
    train_time = time.time() - train_start
    return best_val_acc, train_time, history

def evaluate_with_loss(model, loader, criterion, epoch):
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0
    with torch.no_grad():
        for ultra_img, labels in loader:
            ultra_img, labels = ultra_img.to(device), labels.to(device)
            outputs = model(ultra_img)
            loss = criterion(outputs, labels, epoch=epoch)
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds==labels).sum().item()
            total += labels.size(0)
    return correct/total, running_loss/len(loader)

def evaluate_accuracy(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for ultra_img, labels in loader:
            ultra_img, labels = ultra_img.to(device), labels.to(device)
            outputs = model(ultra_img)
            _, preds = torch.max(outputs,1)
            correct += (preds==labels).sum().item()
            total += labels.size(0)
    return correct/total

def get_probabilities(model, loader):
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for ultra_img, labels in loader:
            ultra_img = ultra_img.to(device)
            out = model(ultra_img)
            probs = torch.softmax(out, dim=1).cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels.numpy())
    return np.array(all_probs), np.array(all_labels)

def comprehensive_evaluation(model, test_loader, backbone_name=''):
    probs, true_labels = get_probabilities(model, test_loader)
    preds = np.argmax(probs, axis=1)
    
    # Calculate all metrics
    accuracy = accuracy_score(true_labels, preds)
    precision = precision_score(true_labels, preds, average='macro', zero_division=0)
    recall = recall_score(true_labels, preds, average='macro', zero_division=0)
    f1 = f1_score(true_labels, preds, average='macro', zero_division=0)
    # For multiclass, calculate per-class recall (sensitivity/specificity)
    per_class_recall = recall_score(true_labels, preds, average=None, zero_division=0)
    
    # Calculate confusion matrix first
    cm = confusion_matrix(true_labels, preds)
    
    # Calculate sensitivity and specificity properly from confusion matrix
    if len(np.unique(true_labels)) == 2:
        # Binary classification case
        if cm.size == 4:  # 2x2 confusion matrix
            tn, fp, fn, tp = cm.ravel()
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        else:
            sensitivity = per_class_recall[1] if len(per_class_recall) > 1 else 0.0
            specificity = per_class_recall[0] if len(per_class_recall) > 0 else 0.0
    else:
        # Multi-class case: use macro-averaged sensitivity and specificity
        # For medical applications, often focus on malignant class (class 2)
        if len(per_class_recall) >= 3:
            # Sensitivity: ability to correctly identify malignant cases (class 2)
            sensitivity = per_class_recall[2]  # Malignant class recall
            # Specificity: average ability to correctly identify non-malignant cases
            specificity = (per_class_recall[0] + per_class_recall[1]) / 2  # Average of Normal and Benign
        elif len(per_class_recall) == 2:
            sensitivity = per_class_recall[1]  # Second class (typically positive)
            specificity = per_class_recall[0]  # First class (typically negative)
        else:
            sensitivity = per_class_recall[0] if len(per_class_recall) > 0 else 0.0
            specificity = per_class_recall[0] if len(per_class_recall) > 0 else 0.0
    
    # MAE and RMSE
    mae = np.mean(np.abs(true_labels - preds))
    rmse = np.sqrt(np.mean((true_labels - preds) ** 2))
    
    # Loss calculation
    eps = 1e-7
    probs_clipped = np.clip(probs, eps, 1 - eps)
    ce_loss = -np.mean([np.log(probs_clipped[i, true_labels[i]]) for i in range(len(true_labels))])
    
    try:
        # For multiclass, use macro-average AUC
        if probs.shape[1] > 2:
            auc = roc_auc_score(true_labels, probs, multi_class='ovr', average='macro')
        else:
            # Binary case
            pos_probs = probs[:, 1]
            auc = roc_auc_score(true_labels, pos_probs)
    except:
        auc = float('nan')
    
    return {
        'backbone': backbone_name,
        'accuracy': accuracy,
        'loss': ce_loss,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'mae': mae,
        'rmse': rmse,
        'auc': auc,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'confusion_matrix': cm.tolist()
    }

# ==================== Main Execution ====================
if __name__ == "__main__":
    # Configuration
    BUSI_DIR = r'E:\Journal\Breast Cancer\Dataset_BUSI_with_GT'
    EPOCHS = 50
    BATCH_SIZE = 16
    LR = 0.0001
    
    # 10 Optimized Backbones for Research Paper
    BACKBONES = [
        'resnet50',           # Baseline standard
        'efficientnet_b4',    # Highest accuracy potential
        'efficientnet_b0',    # Best efficiency
        'densenet169',        # Good for small datasets
        'swin_t',             # Transformer (Swin-Tiny)
        'vit_b_16',           # Vision Transformer
        'regnet_y_400mf',     # Efficient RegNet
        'efficientnet_v2_s',  # Modern EfficientNet V2
        'resnext50_32x4d',    # Better than ResNet101
        'wide_resnet50_2'     # Wide ResNet variant
    ]
    
    print("="*80)
    print("🔬 BUSI MULTI-BACKBONE TRAINING SYSTEM")
    print("="*80)
    
    # Data preparation
    print("\n📁 Loading dataset...")
    ultra_paths, ultra_labels = collect_images(BUSI_DIR, exclude_mask=True, exclude_normal=False, normal_as_benign=False)
    print(f"Total images: {len(ultra_paths)}")
    print(f"Normal: {sum(1 for l in ultra_labels if l==0)}, Benign: {sum(1 for l in ultra_labels if l==1)}, Malignant: {sum(1 for l in ultra_labels if l==2)}")
    
    # Split data
    ultra_tr, ultra_val, ultra_te = stratified_split_indices(ultra_labels, 0.7, 0.2, 42)
    subset = lambda lst, idxs: [lst[i] for i in idxs]
    
    train_paths = subset(ultra_paths, ultra_tr)
    train_labels = [ultra_labels[i] for i in ultra_tr]
    val_paths = subset(ultra_paths, ultra_val)
    val_labels = [ultra_labels[i] for i in ultra_val]
    test_paths = subset(ultra_paths, ultra_te)
    test_labels = [ultra_labels[i] for i in ultra_te]
    
    print(f"Train: {len(train_labels)}, Val: {len(val_labels)}, Test: {len(test_labels)}")
    
    # Create dataloaders
    train_loader, val_loader, test_loader, class_counts = create_dataloaders(
        train_paths, train_labels, val_paths, val_labels, test_paths, test_labels, BATCH_SIZE
    )
    
    # Train all backbones and store histories
    all_results = []
    all_histories = {}
    
    for i, backbone in enumerate(BACKBONES, 1):
        print(f"\n{'='*80}")
        print(f"🚀 Training {i}/{len(BACKBONES)}: {backbone.upper()}")
        print(f"{'='*80}")
        
        try:
            model = BUSIBackboneModel(backbone_name=backbone, num_classes=3).to(device)
            
            # Count parameters
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            print(f"Model parameters: {total_params:,} (Trainable: {trainable_params:,})")
            
            # Train and get history
            best_val_acc, train_time, history = train_model(
                model, train_loader, val_loader, 
                epochs=EPOCHS, lr=LR, 
                class_counts=class_counts,
                backbone_name=backbone
            )
            
            # Store history for plotting
            all_histories[backbone] = history
            
            # Evaluate
            results = comprehensive_evaluation(model, test_loader, backbone)
            results['train_time'] = train_time
            results['best_val_acc'] = best_val_acc
            results['total_params'] = total_params
            results['trainable_params'] = trainable_params
            
            all_results.append(results)
            
            print(f"\n✅ {backbone} Results:")
            print(f"   Accuracy: {results['accuracy']:.4f}")
            print(f"   Loss: {results['loss']:.4f}")
            print(f"   F1-Score: {results['f1']:.4f}")
            print(f"   Precision: {results['precision']:.4f}")
            print(f"   Recall: {results['recall']:.4f}")
            print(f"   MAE: {results['mae']:.4f}")
            print(f"   RMSE: {results['rmse']:.4f}")
            print(f"   AUC: {results['auc']:.4f}")
            print(f"   Sensitivity: {results['sensitivity']:.4f}")
            print(f"   Specificity: {results['specificity']:.4f}")
            print(f"   Training Time: {train_time:.2f}s")
            
            # Save checkpoint
            torch.save(model.state_dict(), f'busi_{backbone}_best.pth')
            
        except Exception as e:
            print(f"❌ Error training {backbone}: {str(e)}")
            continue
    
    # ==================== Generate Research Paper Results ====================
    print(f"\n{'='*80}")
    print("📊 FINAL RESULTS - RESEARCH PAPER TABLE")
    print(f"{'='*80}\n")
    
    # Create results DataFrame
    df_results = pd.DataFrame(all_results)
    df_results = df_results.sort_values('accuracy', ascending=False)
    
    # Convert all numpy types to native Python types before JSON serialization
    all_results_native = []
    for result in all_results:
        native_result = {}
        for key, value in result.items():
            if isinstance(value, np.generic):
                native_result[key] = value.item()
            elif isinstance(value, np.ndarray):
                native_result[key] = value.tolist()
            else:
                native_result[key] = value
        all_results_native.append(native_result)
    
    # Convert all numpy types in training histories
    all_histories_native = {}
    for backbone, history in all_histories.items():
        native_history = {}
        for metric, values in history.items():
            if isinstance(values, list) and len(values) > 0 and isinstance(values[0], np.generic):
                native_history[metric] = [v.item() for v in values]
            else:
                native_history[metric] = values
        all_histories_native[backbone] = native_history
    
    # Display table
    print("\n📋 Complete Results Table:")
    display_cols = ['backbone', 'accuracy', 'loss', 'f1', 'precision', 'recall', 
                    'mae', 'rmse', 'auc', 'sensitivity', 'specificity']
    print(df_results[display_cols].to_string(index=False))
    
    # Save results
    df_results.to_csv('busi_all_backbones_results.csv', index=False)
    
    with open('busi_all_backbones_results.json', 'w') as f:
        json.dump(all_results_native, f, indent=2)
    
    # Save training histories
    with open('busi_training_histories.json', 'w') as f:
        json.dump(all_histories_native, f, indent=2)
    
    # ==================== Training Curves Comparison ====================
    print(f"\n{'='*80}")
    print("📈 GENERATING TRAINING CURVES COMPARISON")
    print(f"{'='*80}\n")
    
    # Generate comprehensive comparison plots
    with PdfPages('busi_comprehensive_results.pdf') as pdf:
        
        # ========== PAGE 1: Training & Validation Accuracy Curves (All 10 Models) ==========
        plt.figure(figsize=(16, 10))
        
        plt.subplot(2, 1, 1)
        for backbone, history in all_histories.items():
            epochs_range = range(1, len(history['train_acc']) + 1)
            plt.plot(epochs_range, history['train_acc'], label=f'{backbone}', linewidth=2)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Training Accuracy', fontsize=12)
        plt.title('Training Accuracy Curves - All 10 Backbones Comparison', fontsize=14, fontweight='bold')
        plt.legend(loc='lower right', fontsize=9, ncol=2)
        plt.grid(True, alpha=0.3)
        plt.ylim([0, 1.05])
        
        plt.subplot(2, 1, 2)
        for backbone, history in all_histories.items():
            epochs_range = range(1, len(history['val_acc']) + 1)
            plt.plot(epochs_range, history['val_acc'], label=f'{backbone}', linewidth=2, linestyle='--')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Validation Accuracy', fontsize=12)
        plt.title('Validation Accuracy Curves - All 10 Backbones Comparison', fontsize=14, fontweight='bold')
        plt.legend(loc='lower right', fontsize=9, ncol=2)
        plt.grid(True, alpha=0.3)
        plt.ylim([0, 1.05])
        
        plt.tight_layout()
        pdf.savefig()
        plt.close()
        
        # ========== PAGE 2: Training & Validation Loss Curves (All 10 Models) ==========
        plt.figure(figsize=(16, 10))
        
        plt.subplot(2, 1, 1)
        for backbone, history in all_histories.items():
            epochs_range = range(1, len(history['train_loss']) + 1)
            plt.plot(epochs_range, history['train_loss'], label=f'{backbone}', linewidth=2)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Training Loss', fontsize=12)
        plt.title('Training Loss Curves - All 10 Backbones Comparison', fontsize=14, fontweight='bold')
        plt.legend(loc='upper right', fontsize=9, ncol=2)
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 1, 2)
        for backbone, history in all_histories.items():
            epochs_range = range(1, len(history['val_loss']) + 1)
            plt.plot(epochs_range, history['val_loss'], label=f'{backbone}', linewidth=2, linestyle='--')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Validation Loss', fontsize=12)
        plt.title('Validation Loss Curves - All 10 Backbones Comparison', fontsize=14, fontweight='bold')
        plt.legend(loc='upper right', fontsize=9, ncol=2)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        pdf.savefig()
        plt.close()
        
        # ========== PAGE 3: Combined 4-in-1 Training Curves ==========
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        
        # Train Accuracy
        for backbone, history in all_histories.items():
            epochs_range = range(1, len(history['train_acc']) + 1)
            axes[0, 0].plot(epochs_range, history['train_acc'], label=backbone, linewidth=2)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Training Accuracy')
        axes[0, 0].set_title('Training Accuracy', fontweight='bold')
        axes[0, 0].legend(fontsize=8, ncol=2)
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_ylim([0, 1.05])
        
        # Train Loss
        for backbone, history in all_histories.items():
            epochs_range = range(1, len(history['train_loss']) + 1)
            axes[0, 1].plot(epochs_range, history['train_loss'], label=backbone, linewidth=2)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Training Loss')
        axes[0, 1].set_title('Training Loss', fontweight='bold')
        axes[0, 1].legend(fontsize=8, ncol=2)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Val Accuracy
        for backbone, history in all_histories.items():
            epochs_range = range(1, len(history['val_acc']) + 1)
            axes[1, 0].plot(epochs_range, history['val_acc'], label=backbone, linewidth=2, linestyle='--')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Validation Accuracy')
        axes[1, 0].set_title('Validation Accuracy', fontweight='bold')
        axes[1, 0].legend(fontsize=8, ncol=2)
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_ylim([0, 1.05])
        
        # Val Loss
        for backbone, history in all_histories.items():
            epochs_range = range(1, len(history['val_loss']) + 1)
            axes[1, 1].plot(epochs_range, history['val_loss'], label=backbone, linewidth=2, linestyle='--')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Validation Loss')
        axes[1, 1].set_title('Validation Loss', fontweight='bold')
        axes[1, 1].legend(fontsize=8, ncol=2)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle('BUSI Dataset - Complete Training Curves (All 10 Backbones)', fontsize=16, fontweight='bold')
        plt.tight_layout()
        pdf.savefig()
        plt.close()
        
        # ========== PAGE 4: Final Metrics Comparison Bar Charts ==========
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        metrics = ['accuracy', 'f1', 'precision', 'recall', 'auc', 'loss']
        titles = ['Accuracy', 'F1-Score', 'Precision', 'Recall', 'AUC-ROC', 'Loss']
        colors = ['steelblue', 'coral', 'mediumseagreen', 'orange', 'purple', 'indianred']
        
        for idx, (metric, title, color) in enumerate(zip(metrics, titles, colors)):
            ax = axes[idx // 3, idx % 3]
            if metric == 'loss':
                sorted_df = df_results.sort_values(metric, ascending=True)
                ax.barh(sorted_df['backbone'], sorted_df[metric], color=color)
                ax.set_title(f'{title} (Lower is Better)', fontweight='bold')
            else:
                ax.barh(df_results['backbone'], df_results[metric], color=color)
                ax.set_xlim([0, 1])
                ax.set_title(title, fontweight='bold')
            ax.grid(axis='x', alpha=0.3)
        
        plt.suptitle('BUSI Dataset - Final Test Metrics Comparison', fontsize=16, fontweight='bold')
        plt.tight_layout()
        pdf.savefig()
        plt.close()
        
        # ========== PAGE 5: MAE & RMSE + Sensitivity vs Specificity ==========
        fig = plt.figure(figsize=(18, 8))
        
        # MAE
        ax1 = plt.subplot(1, 3, 1)
        ax1.barh(df_results['backbone'], df_results['mae'], color='teal')
        ax1.set_xlabel('MAE')
        ax1.set_title('Mean Absolute Error', fontweight='bold')
        ax1.grid(axis='x', alpha=0.3)
        
        # RMSE
        ax2 = plt.subplot(1, 3, 2)
        ax2.barh(df_results['backbone'], df_results['rmse'], color='darkslateblue')
        ax2.set_xlabel('RMSE')
        ax2.set_title('Root Mean Squared Error', fontweight='bold')
        ax2.grid(axis='x', alpha=0.3)
        
        # Sensitivity vs Specificity Scatter
        ax3 = plt.subplot(1, 3, 3)
        for idx, row in df_results.iterrows():
            ax3.scatter(row['specificity'], row['sensitivity'], s=200, alpha=0.6)
            ax3.annotate(row['backbone'], 
                        (row['specificity'], row['sensitivity']),
                        fontsize=8, ha='center')
        ax3.set_xlabel('Specificity (TNR)')
        ax3.set_ylabel('Sensitivity (TPR)')
        ax3.set_title('Sensitivity vs Specificity Trade-off', fontweight='bold')
        ax3.set_xlim([0, 1.05])
        ax3.set_ylim([0, 1.05])
        ax3.grid(True, alpha=0.3)
        ax3.axhline(y=0.5, color='r', linestyle='--', alpha=0.3)
        ax3.axvline(x=0.5, color='r', linestyle='--', alpha=0.3)
        
        plt.suptitle('BUSI Dataset - Error Metrics & Classification Balance', fontsize=16, fontweight='bold')
        plt.tight_layout()
        pdf.savefig()
        plt.close()
        
        # ========== PAGE 6: Model Efficiency Analysis ==========
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Parameters vs Accuracy
        axes[0].scatter(df_results['total_params'] / 1e6, df_results['accuracy'], 
                       s=200, alpha=0.6, c=range(len(df_results)), cmap='viridis')
        for idx, row in df_results.iterrows():
            axes[0].annotate(row['backbone'], 
                           (row['total_params'] / 1e6, row['accuracy']),
                           fontsize=8, ha='right')
        axes[0].set_xlabel('Total Parameters (Millions)')
        axes[0].set_ylabel('Test Accuracy')
        axes[0].set_title('Model Size vs Accuracy', fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        
        # Training Time vs Accuracy
        axes[1].scatter(df_results['train_time'] / 60, df_results['accuracy'], 
                       s=200, alpha=0.6, c=range(len(df_results)), cmap='plasma')
        for idx, row in df_results.iterrows():
            axes[1].annotate(row['backbone'], 
                           (row['train_time'] / 60, row['accuracy']),
                           fontsize=8, ha='right')
        axes[1].set_xlabel('Training Time (Minutes)')
        axes[1].set_ylabel('Test Accuracy')
        axes[1].set_title('Training Time vs Accuracy', fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        plt.suptitle('BUSI Dataset - Model Efficiency Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        pdf.savefig()
        plt.close()
        
        # ========== PAGE 7: Confusion Matrices for All 10 Models ==========
        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        axes = axes.flatten()
        
        for idx, result in enumerate(all_results):
            if idx < 10:  # Ensure we don't exceed subplot count
                cm = np.array(result['confusion_matrix'])
                backbone_name = result['backbone']
                
                # Create heatmap
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                           xticklabels=['Normal', 'Benign', 'Malignant'], 
                           yticklabels=['Normal', 'Benign', 'Malignant'],
                           ax=axes[idx], cbar=False)
                axes[idx].set_title(f'{backbone_name}\nAcc: {result["accuracy"]:.3f}', 
                                  fontsize=10, fontweight='bold')
                axes[idx].set_xlabel('Predicted')
                axes[idx].set_ylabel('Actual')
        
        # Hide any unused subplots
        for idx in range(len(all_results), 10):
            axes[idx].axis('off')
        
        plt.suptitle('BUSI Dataset - Confusion Matrices (All 10 Backbones)', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        pdf.savefig()
        plt.close()
    
    print(f"\n✅ Results saved:")
    print(f"   - busi_all_backbones_results.csv (Metrics table)")
    print(f"   - busi_all_backbones_results.json (Detailed results)")
    print(f"   - busi_training_histories.json (Training curves data)")
    print(f"   - busi_comprehensive_results.pdf (7-page visual report)")
    print(f"   - Individual checkpoints: busi_<backbone>_best.pth (×10)")
    
    print(f"\n{'='*80}")
    print("🎉 TRAINING COMPLETE!")
    print(f"{'='*80}")
    print("\n📊 PDF Report Contents:")
    print("   Page 1: Training Accuracy Curves (All 10 models)")
    print("   Page 2: Training Loss Curves (All 10 models)")
    print("   Page 3: Combined 4-in-1 Training Curves")
    print("   Page 4: Final Test Metrics Comparison")
    print("   Page 5: Error Metrics & Classification Balance")
    print("   Page 6: Model Efficiency Analysis")
    print("   Page 7: Confusion Matrices (All 10 models)")
    print(f"{'='*80}")