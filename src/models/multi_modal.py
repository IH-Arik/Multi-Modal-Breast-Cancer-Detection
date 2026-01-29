"""
Multi-Modal Breast Cancer Detection Models
==========================================

Professional implementation of multi-modal breast cancer detection models
supporting ultrasound, mammography, and histology data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Dict, List, Optional, Tuple
import numpy as np


class MultiModalFusionModel(nn.Module):
    """
    Advanced multi-modal fusion model for breast cancer detection.
    
    Supports fusion of ultrasound, mammography, and histology features
    with attention mechanisms for optimal performance.
    """
    
    def __init__(self, 
                 modality_configs: Dict[str, Dict],
                 fusion_method: str = 'attention',
                 num_classes: int = 2,
                 dropout: float = 0.3):
        """
        Initialize multi-modal fusion model.
        
        Args:
            modality_configs: Configuration for each modality
            fusion_method: Method for fusing modalities ('attention', 'concat', 'gate')
            num_classes: Number of output classes
            dropout: Dropout rate
        """
        super().__init__()
        
        self.modality_configs = modality_configs
        self.fusion_method = fusion_method
        self.modalities = list(modality_configs.keys())
        
        # Build modality-specific encoders
        self.encoders = nn.ModuleDict()
        for modality, config in modality_configs.items():
            self.encoders[modality] = self._build_encoder(config)
        
        # Fusion layer
        if fusion_method == 'attention':
            self.fusion = AttentionFusion(len(self.modalities))
        elif fusion_method == 'gate':
            self.fusion = GatedFusion(len(self.modalities))
        else:  # concat
            self.fusion = ConcatFusion()
        
        # Classifier
        self.classifier = self._build_classifier(len(self.modalities), num_classes, dropout)
        
    def _build_encoder(self, config: Dict) -> nn.Module:
        """Build encoder for specific modality."""
        backbone = config.get('backbone', 'resnet50')
        pretrained = config.get('pretrained', True)
        
        if backbone == 'resnet50':
            model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
            feat_dim = model.fc.in_features
            model.fc = nn.Identity()
        elif backbone == 'efficientnet_b4':
            model = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.DEFAULT if pretrained else None)
            feat_dim = model.classifier[1].in_features
            model.classifier = nn.Identity()
        elif backbone == 'densenet169':
            model = models.densenet169(weights=models.DenseNet169_Weights.DEFAULT if pretrained else None)
            feat_dim = model.classifier.in_features
            model.classifier = nn.Identity()
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        return model
    
    def _build_classifier(self, num_modalities: int, num_classes: int, dropout: float) -> nn.Module:
        """Build classifier head."""
        input_dim = 512 * num_modalities  # Assuming 512 features per modality
        
        return nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            inputs: Dictionary of modality inputs
            
        Returns:
            Classification logits
        """
        # Extract features from each modality
        features = []
        for modality in self.modalities:
            if modality in inputs:
                feat = self.encoders[modality](inputs[modality])
                features.append(feat)
        
        if not features:
            raise ValueError("No valid modalities provided")
        
        # Fuse features
        fused_features = self.fusion(features)
        
        # Classify
        output = self.classifier(fused_features)
        
        return output
    
    def get_modality_features(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Extract features from each modality separately."""
        features = {}
        for modality in self.modalities:
            if modality in inputs:
                features[modality] = self.encoders[modality](inputs[modality])
        return features


class AttentionFusion(nn.Module):
    """Attention-based fusion mechanism."""
    
    def __init__(self, num_modalities: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Softmax(dim=0)
        )
    
    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        weights = []
        for feat in features:
            w = self.attention(feat.mean(dim=(2, 3)))  # Global average pooling
            weights.append(w)
        
        weights = torch.stack(weights, dim=1)  # (batch, num_modalities, 1)
        weights = weights.transpose(1, 2)  # (batch, 1, num_modalities)
        
        # Apply attention weights
        weighted_features = []
        for i, feat in enumerate(features):
            weighted_feat = feat * weights[:, :, i:i+1]
            weighted_features.append(weighted_feat)
        
        return torch.cat(weighted_features, dim=1)


class GatedFusion(nn.Module):
    """Gated fusion mechanism."""
    
    def __init__(self, num_modalities: int):
        super().__init__()
        self.gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 512),
                nn.Sigmoid()
            ) for _ in range(num_modalities)
        ])
    
    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        gated_features = []
        for feat, gate in zip(features, self.gates):
            gate_weights = gate(feat.mean(dim=(2, 3))).unsqueeze(-1).unsqueeze(-1)
            gated_feat = feat * gate_weights
            gated_features.append(gated_feat)
        
        return torch.cat(gated_features, dim=1)


class ConcatFusion(nn.Module):
    """Simple concatenation fusion."""
    
    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        return torch.cat(features, dim=1)


class BreastCancerClassifier(nn.Module):
    """
    Specialized breast cancer classifier with clinical feature integration.
    """
    
    def __init__(self, 
                 image_backbone: str = 'resnet50',
                 num_clinical_features: int = 0,
                 num_classes: int = 2,
                 dropout: float = 0.3):
        super().__init__()
        
        # Image encoder
        self.image_encoder = self._build_image_encoder(image_backbone)
        
        # Clinical feature processor
        self.num_clinical_features = num_clinical_features
        if num_clinical_features > 0:
            self.clinical_encoder = nn.Sequential(
                nn.Linear(num_clinical_features, 128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, 64)
            )
            fusion_dim = 512 + 64  # Image features + clinical features
        else:
            fusion_dim = 512
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
    
    def _build_image_encoder(self, backbone: str) -> nn.Module:
        """Build image encoder backbone."""
        if backbone == 'resnet50':
            model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            feat_dim = model.fc.in_features
            model.fc = nn.Identity()
        elif backbone == 'efficientnet_b4':
            model = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.DEFAULT)
            feat_dim = model.classifier[1].in_features
            model.classifier = nn.Identity()
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        return model
    
    def forward(self, image: torch.Tensor, clinical: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass."""
        # Extract image features
        image_features = self.image_encoder(image)
        
        # Process clinical features if available
        if clinical is not None and self.num_clinical_features > 0:
            clinical_features = self.clinical_encoder(clinical)
            features = torch.cat([image_features, clinical_features], dim=1)
        else:
            features = image_features
        
        # Classify
        output = self.classifier(features)
        return output


def create_multi_modal_model(modalities: List[str], 
                           backbones: Dict[str, str],
                           fusion_method: str = 'attention',
                           num_classes: int = 2) -> MultiModalFusionModel:
    """
    Factory function to create multi-modal model.
    
    Args:
        modalities: List of modality names
        backbones: Dictionary mapping modalities to backbone names
        fusion_method: Fusion method
        num_classes: Number of classes
        
    Returns:
        Multi-modal model
    """
    modality_configs = {}
    for modality in modalities:
        modality_configs[modality] = {
            'backbone': backbones.get(modality, 'resnet50'),
            'pretrained': True
        }
    
    return MultiModalFusionModel(
        modality_configs=modality_configs,
        fusion_method=fusion_method,
        num_classes=num_classes
    )
