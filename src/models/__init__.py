"""
Model architectures for breast cancer detection.
"""

import torch
import torch.nn as nn
from torchvision import models


class MultiModalBackboneModel(nn.Module):
    """Universal model class for all backbones across modalities."""
    
    def __init__(self, backbone_name='resnet50', num_classes=2, pretrained=True):
        super(MultiModalBackboneModel, self).__init__()
        self.backbone_name = backbone_name
        
        if backbone_name == 'resnet50':
            self.backbone = models.resnet50(
                weights=models.ResNet50_Weights.DEFAULT if pretrained else None
            )
            feat_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
            
        elif backbone_name == 'efficientnet_b4':
            self.backbone = models.efficientnet_b4(
                weights=models.EfficientNet_B4_Weights.DEFAULT if pretrained else None
            )
            feat_dim = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
            
        elif backbone_name == 'efficientnet_b0':
            self.backbone = models.efficientnet_b0(
                weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            )
            feat_dim = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
            
        elif backbone_name == 'densenet169':
            self.backbone = models.densenet169(
                weights=models.DenseNet169_Weights.DEFAULT if pretrained else None
            )
            feat_dim = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Identity()
            
        elif backbone_name == 'swin_t':
            self.backbone = models.swin_t(
                weights=models.Swin_T_Weights.DEFAULT if pretrained else None
            )
            feat_dim = self.backbone.head.in_features
            self.backbone.head = nn.Identity()
            
        elif backbone_name == 'vit_b_16':
            self.backbone = models.vit_b_16(
                weights=models.ViT_B_16_Weights.DEFAULT if pretrained else None
            )
            feat_dim = self.backbone.heads.head.in_features
            self.backbone.heads.head = nn.Identity()
            
        elif backbone_name == 'regnet_y_400mf':
            self.backbone = models.regnet_y_400mf(
                weights=models.RegNet_Y_400MF_Weights.DEFAULT if pretrained else None
            )
            feat_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
            
        elif backbone_name == 'efficientnet_v2_s':
            self.backbone = models.efficientnet_v2_s(
                weights=models.EfficientNet_V2_S_Weights.DEFAULT if pretrained else None
            )
            feat_dim = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
            
        elif backbone_name == 'resnext50_32x4d':
            self.backbone = models.resnext50_32x4d(
                weights=models.ResNeXt50_32X4D_Weights.DEFAULT if pretrained else None
            )
            feat_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
            
        elif backbone_name == 'wide_resnet50_2':
            self.backbone = models.wide_resnet50_2(
                weights=models.Wide_ResNet50_2_Weights.DEFAULT if pretrained else None
            )
            feat_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        
        else:
            raise ValueError(f"Unknown backbone: {backbone_name}")
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        feat = self.backbone(x)
        out = self.classifier(feat)
        return out


# Available backbones for research
AVAILABLE_BACKBONES = [
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


def get_model(backbone_name, num_classes=2, pretrained=True):
    """Get model by name."""
    return MultiModalBackboneModel(backbone_name, num_classes, pretrained)


def get_available_backbones():
    """Get list of available backbone models."""
    return AVAILABLE_BACKBONES.copy()
