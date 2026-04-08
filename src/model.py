"""
CNN model architecture for image sequence classification
"""

import torch
import torch.nn as nn

from .config import DEFAULT_FILTERS, DEFAULT_KERNEL_SIZE, DEFAULT_DROPOUT, DEFAULT_BATCH_NORM


class ImageSequenceClassifier(nn.Module):
    """
    3-layer CNN for classifying image position in a sequence.
    
    Architecture:
        - Conv1 -> (BatchNorm) -> ReLU -> MaxPool
        - Conv2 -> (BatchNorm) -> ReLU -> MaxPool
        - Conv3 -> (BatchNorm) -> ReLU -> AdaptiveAvgPool
        - Flatten -> FC -> ReLU -> Dropout -> FC (5 classes)
    """
    
    def __init__(
        self,
        dropout=DEFAULT_DROPOUT,
        filters=DEFAULT_FILTERS,
        kernel=DEFAULT_KERNEL_SIZE,
        batchnorm=DEFAULT_BATCH_NORM
    ):
        """
        Args:
            dropout: Dropout rate (0.0 = none, 0.3 = default)
            filters: Base number of filters (doubles at each layer)
            kernel: Convolutional kernel size (3 or 5)
            batchnorm: Whether to use batch normalization
        """
        super().__init__()
        
        layers = []
        
        # Conv Layer 1: 3 -> filters channels
        layers.append(nn.Conv2d(3, filters, kernel, padding=1))
        if batchnorm:
            layers.append(nn.BatchNorm2d(filters))
        layers.append(nn.ReLU())
        layers.append(nn.MaxPool2d(2))
        
        # Conv Layer 2: filters -> filters*2 channels
        layers.append(nn.Conv2d(filters, filters*2, kernel, padding=1))
        if batchnorm:
            layers.append(nn.BatchNorm2d(filters*2))
        layers.append(nn.ReLU())
        layers.append(nn.MaxPool2d(2))
        
        # Conv Layer 3: filters*2 -> filters*4 channels
        layers.append(nn.Conv2d(filters*2, filters*4, kernel, padding=1))
        if batchnorm:
            layers.append(nn.BatchNorm2d(filters*4))
        layers.append(nn.ReLU())
        
        # Adaptive pooling to fixed size (4x8)
        layers.append(nn.AdaptiveAvgPool2d((4, 8)))
        
        self.conv = nn.Sequential(*layers)
        
        # Fully connected classification head
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(filters*4*4*8, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 5)  # 5 classes (positions 1-5)
        )
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, 3, height, width)
            
        Returns:
            Output logits of shape (batch_size, 5)
        """
        x = self.conv(x)
        x = self.fc(x)
        return x


def count_parameters(model):
    """
    Count total and trainable parameters in the model.
    
    Args:
        model: PyTorch model
        
    Returns:
        total_params, trainable_params
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return total_params, trainable_params


def print_model_summary(model):
    """
    Print model architecture summary.
    
    Args:
        model: PyTorch model
    """
    total, trainable = count_parameters(model)
    
    print("\n" + "="*60)
    print("MODEL ARCHITECTURE SUMMARY")
    print("="*60)
    print(model)
    print("="*60)
    print(f"Total parameters: {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    print("="*60)
