import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
import numpy as np
import os
from collections import Counter

def get_dataloaders(dataset_path, batch_size=32, train_ratio=0.8):
    """
    Load the dataset and return train and validation DataLoaders with enhanced augmentation.
    """
    # Enhanced data augmentation for training
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomRotation(15),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.2)
        ]),
        'val': transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # Load datasets with the appropriate transforms
    full_dataset_train = datasets.ImageFolder(root=dataset_path, transform=data_transforms['train'])
    full_dataset_val = datasets.ImageFolder(root=dataset_path, transform=data_transforms['val'])

    # Split indices for training and validation
    dataset_size = len(full_dataset_train)
    indices = list(range(dataset_size))
    np.random.seed(42)  # Set seed for reproducibility
    np.random.shuffle(indices)
    split = int(np.floor(train_ratio * dataset_size))
    train_idx, val_idx = indices[:split], indices[split:]

    # Create dataset subsets
    train_dataset = Subset(full_dataset_train, train_idx)
    val_dataset = Subset(full_dataset_val, val_idx)

    # Check for class imbalance and create a weighted sampler if needed
    targets = [full_dataset_train.targets[i] for i in train_idx]
    class_counts = Counter(targets)
    class_weights = {cls: 1.0 / count for cls, count in class_counts.items()}
    sample_weights = [class_weights[targets[i]] for i in range(len(targets))]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        sampler=sampler,  # Use weighted sampler for balanced training
        num_workers=2,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=2,
        pin_memory=True
    )

    print(f"Dataset loaded: {len(train_idx)} training samples, {len(val_idx)} validation samples")
    print(f"Number of classes: {len(full_dataset_train.classes)}")
    
    return train_loader, val_loader, full_dataset_train.classes