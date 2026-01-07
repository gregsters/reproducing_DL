"""
Model definitions for Active Learning Project.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import DEVICE


class FastBayesianCNN(nn.Module):
    """Bayesian CNN with MC Dropout for uncertainty estimation."""
    
    def __init__(self, dropout=True, dropout_p1=0.25, dropout_p2=0.5):
        super(FastBayesianCNN, self).__init__()
        self.dropout = dropout
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=4, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=4, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Dropout layers
        self.dropout1 = nn.Dropout2d(p=dropout_p1)
        self.dropout2 = nn.Dropout(p=dropout_p2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(32 * 13 * 13, 128)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x, mc_dropout=False):
        # First convolutional block
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        
        # Apply dropout if enabled
        if self.dropout and (self.training or mc_dropout):
            x = self.dropout1(x)
        
        # Flatten and fully connected layers
        x = x.view(-1, 32 * 13 * 13)
        x = F.relu(self.fc1(x))
        
        # Apply dropout if enabled
        if self.dropout and (self.training or mc_dropout):
            x = self.dropout2(x)
        
        # Output layer
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class DeterministicCNN(nn.Module):
    """Deterministic CNN without dropout for baseline comparison."""
    
    def __init__(self):
        super(DeterministicCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=4, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=4, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(32 * 13 * 13, 128)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        # Convolutional block
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        
        # Flatten and fully connected
        x = x.view(-1, 32 * 13 * 13)
        x = F.relu(self.fc1(x))
        
        # Output layer
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class FrozenFeatureExtractor(nn.Module):
    """Feature extractor with frozen weights for transfer learning."""
    
    def __init__(self, device=DEVICE):
        super(FrozenFeatureExtractor, self).__init__()
        self.device = device
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=4, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=4, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout2d(p=0.25)
        self.fc1 = nn.Linear(32 * 13 * 13, 128)
        self.dropout2 = nn.Dropout(p=0.5)
        
        # Freeze all parameters
        for param in self.parameters():
            param.requires_grad = False
        
        # Move to device
        self.to(device)
    
    def forward(self, x):
        # Move to device if necessary
        if x.device != self.device:
            x = x.to(self.device)
        
        # Feature extraction
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout1(x)
        x = x.view(-1, 32 * 13 * 13)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        
        return x