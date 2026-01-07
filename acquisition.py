"""
Acquisition functions and data management for active learning.
"""

import torch
import random
import numpy as np
from torchvision import datasets, transforms


class FastActiveLearningMNIST:
    """Data manager for active learning experiments."""
    
    def __init__(self, initial_labeled=20, validation_size=100):
        self.initial_labeled = initial_labeled
        self.validation_size = validation_size
        
        # Data transformations
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        # Load datasets
        self.train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
        self.test_dataset = datasets.MNIST('./data', train=False, transform=transform)
        
        # Create splits
        self._create_splits()
    
    def _create_splits(self):
        indices = list(range(len(self.train_dataset)))
        random.shuffle(indices)
        
        # Balanced initial set (2 per class)
        class_counts = [0] * 10
        self.labeled_indices = []
        
        for idx in indices:
            _, label = self.train_dataset[idx]
            if class_counts[label] < 2:
                self.labeled_indices.append(idx)
                class_counts[label] += 1
            if len(self.labeled_indices) >= self.initial_labeled:
                break
        
        # Remaining for validation and pool
        remaining = [i for i in indices if i not in self.labeled_indices]
        self.val_indices = remaining[:self.validation_size]
        self.pool_indices = remaining[self.validation_size:]
    
    def get_datasets(self):
        """Get current dataset splits."""
        return {
            'train': torch.utils.data.Subset(self.train_dataset, self.labeled_indices),
            'val': torch.utils.data.Subset(self.train_dataset, self.val_indices),
            'pool': torch.utils.data.Subset(self.train_dataset, self.pool_indices),
            'test': self.test_dataset
        }
    
    def acquire_points(self, selected_indices):
        """Acquire points from pool and add to labeled set."""
        actual_indices = [self.pool_indices[i] for i in selected_indices]
        self.labeled_indices.extend(actual_indices)
        
        # Remove from pool
        for idx in sorted(actual_indices, reverse=True):
            if idx in self.pool_indices:
                self.pool_indices.remove(idx)
        
        return len(actual_indices)


# Acquisition functions
def fast_bald(predictions):
    """Bayesian Active Learning by Disagreement."""
    T, N, C = predictions.shape
    probs = torch.exp(predictions)
    predictive_probs = probs.mean(dim=0)
    predictive_entropy = -torch.sum(predictive_probs * torch.log(predictive_probs + 1e-10), dim=1)
    entropy_per_sample = -torch.sum(probs * torch.log(probs + 1e-10), dim=2)
    expected_entropy = entropy_per_sample.mean(dim=0)
    return predictive_entropy - expected_entropy


def fast_variation_ratios(predictions):
    """Variation Ratios acquisition."""
    probs = torch.exp(predictions.mean(dim=0))
    max_probs = probs.max(dim=1)[0]
    return 1 - max_probs


def fast_max_entropy(predictions):
    """Maximum Entropy acquisition."""
    probs = torch.exp(predictions.mean(dim=0))
    entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)
    return entropy


def fast_mean_std(predictions):
    """Mean Standard Deviation acquisition."""
    probs = torch.exp(predictions)
    std_per_class = probs.std(dim=0)
    return std_per_class.mean(dim=1)


def random_acquisition(predictions):
    """Random acquisition baseline."""
    return torch.rand(predictions.shape[1])


# Dictionary of acquisition functions
ACQUISITION_FUNCTIONS = {
    'bald': fast_bald,
    'variation_ratios': fast_variation_ratios,
    'max_entropy': fast_max_entropy,
    'mean_std': fast_mean_std,
    'random': random_acquisition
}