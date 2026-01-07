"""
Training utilities for active learning.
"""

import torch
import torch.nn.functional as F
import torch.optim as optim
from config import DEVICE, LEARNING_RATE, WEIGHT_DECAY, TRAIN_EPOCHS


def fast_train(model, train_loader, val_loader, device=DEVICE, lr=LEARNING_RATE, 
               weight_decay=WEIGHT_DECAY, epochs=TRAIN_EPOCHS):
    """Fast training function for active learning."""
    
    model.to(device)
    model.train()
    
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    for epoch in range(epochs):
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
    
    model.eval()
    return model


def evaluate_model(model, test_loader, model_type='bayesian', device=DEVICE):
    """Evaluate model on test set."""
    
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            
            if model_type == 'bayesian':
                # Bayesian: average over MC samples
                outputs = []
                for _ in range(5):
                    out = model(data, mc_dropout=True)
                    outputs.append(torch.exp(out).unsqueeze(0))
                
                avg_probs = torch.mean(torch.cat(outputs, dim=0), dim=0)
            else:
                # Deterministic: single forward pass
                out = model(data)
                avg_probs = torch.exp(out)
            
            # Predictions
            pred = avg_probs.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)
    
    accuracy = 100. * correct / total
    return accuracy