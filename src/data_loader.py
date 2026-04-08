"""
Dataset loading and preprocessing for image sequence classification
"""

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from datasets import load_dataset
import numpy as np

from .config import (
    DATASET_NAME, DATASET_SUBSET_SIZE, TRAIN_SPLIT_RATIO,
    IMAGE_HEIGHT, IMAGE_WIDTH, BATCH_SIZE
)


class ImagePositionDataset(Dataset):
    """
    Custom dataset for image position classification.
    Extracts individual images from story sequences and assigns position labels.
    """
    
    def __init__(self, dataset):
        """
        Args:
            dataset: HuggingFace dataset containing stories with 5 images each
        """
        self.samples = []
        
        # Resize images and convert to tensor
        self.transform = transforms.Compose([
            transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
            transforms.ToTensor()
        ])
        
        # Extract individual images with position labels
        for story in dataset:
            images = story["images"]
            
            # Assign labels 0–4 (positions 1–5)
            for pos in range(5):
                self.samples.append((images[pos], pos))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        image, label = self.samples[idx]
        
        # Apply histogram equalization for brightness normalization
        image = TF.equalize(image)
        
        # Apply transforms
        image = self.transform(image)
        
        return image, label


def load_story_dataset(subset_size=DATASET_SUBSET_SIZE):
    """
    Load StoryReasoning dataset from HuggingFace.
    
    Args:
        subset_size: Number of stories to load (default from config)
        
    Returns:
        train_dataset, test_dataset: HuggingFace datasets
    """
    print(f"Loading dataset: {DATASET_NAME}")
    
    train_dataset = load_dataset(DATASET_NAME, split="train")
    test_dataset = load_dataset(DATASET_NAME, split="test")
    
    print(f"Train dataset loaded: {len(train_dataset)} stories")
    print(f"Test dataset loaded: {len(test_dataset)} stories")
    
    return train_dataset, test_dataset


def create_data_loaders(train_dataset, batch_size=BATCH_SIZE, train_split=TRAIN_SPLIT_RATIO):
    """
    Create train and validation data loaders.
    
    Args:
        train_dataset: HuggingFace dataset
        batch_size: Batch size for DataLoader
        train_split: Ratio for train/validation split
        
    Returns:
        train_loader, val_loader: PyTorch DataLoaders
        dataset: ImagePositionDataset instance
    """
    # Create image position dataset
    dataset = ImagePositionDataset(train_dataset)
    
    # Split into train and validation
    train_size = int(train_split * len(dataset))
    val_size = len(dataset) - train_size
    
    train_data, val_data = random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True
    )
    
    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False
    )
    
    print(f"\nDataset Statistics:")
    print(f"  Total samples: {len(dataset)}")
    print(f"  Training samples: {train_size}")
    print(f"  Validation samples: {val_size}")
    print(f"  Split ratio: {train_split*100:.0f}% / {(1-train_split)*100:.0f}%")
    print(f"  Batch size: {batch_size}")
    
    return train_loader, val_loader, dataset


def check_class_distribution(dataset):
    """
    Check and display class distribution in the dataset.
    
    Args:
        dataset: ImagePositionDataset instance
    """
    labels = [s[1] for s in dataset.samples]
    unique, counts = np.unique(labels, return_counts=True)
    
    print("\n" + "="*50)
    print("CLASS DISTRIBUTION")
    print("="*50)
    for u, c in zip(unique, counts):
        print(f"Position {u+1}: {c} samples ({c/len(dataset)*100:.1f}%)")
    print("="*50)
    print(f"Dataset is {'BALANCED' if len(set(counts)) == 1 else 'IMBALANCED'}")
    print("="*50)
