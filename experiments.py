"""
Experiment runners for active learning project.
"""

import torch
import random
import numpy as np
from torchvision import datasets, transforms
import torch.utils.data

from config import *
from models import FastBayesianCNN, DeterministicCNN, FrozenFeatureExtractor
from inference import AnalyticGaussianInference, MFVIDiagonalInference, MFVIFullCovarianceInference, HeteroscedasticLastLayer
from acquisition import FastActiveLearningMNIST, ACQUISITION_FUNCTIONS
from training import fast_train, evaluate_model


def run_reproduction_experiment(model_type='bayesian', acquisition_name='bald', 
                               n_acquisitions=REPRO_ACQ_STEPS_BAYES, device=DEVICE):
    """Run reproduction experiment (Bayesian vs Deterministic CNN)."""
    
    # Initialize data manager
    al_manager = FastActiveLearningMNIST(
        initial_labeled=INITIAL_LABELED,
        validation_size=VALIDATION_SIZE
    )
    datasets_dict = al_manager.get_datasets()
    
    # Initialize model
    if model_type == 'bayesian':
        model = FastBayesianCNN(dropout=True).to(device)
    else:
        model = DeterministicCNN().to(device)
    
    # Get acquisition function
    acquisition_func = ACQUISITION_FUNCTIONS[acquisition_name]
    
    # Storage for test accuracies
    test_accuracies = []
    
    # Main active learning loop
    for step in range(n_acquisitions + 1):
        # Train/fine-tune
        train_loader = torch.utils.data.DataLoader(
            datasets_dict['train'],
            batch_size=TRAIN_BATCH,
            shuffle=True
        )
        val_loader = torch.utils.data.DataLoader(
            datasets_dict['val'],
            batch_size=VAL_BATCH,
            shuffle=False
        )
        model = fast_train(model, train_loader, val_loader, device=device, epochs=TRAIN_EPOCHS)
        
        # Evaluate on test set
        test_loader = torch.utils.data.DataLoader(
            datasets_dict['test'],
            batch_size=1000,
            shuffle=False
        )
        accuracy = evaluate_model(model, test_loader, model_type, device)
        test_accuracies.append(accuracy)
        
        # Acquisition step (skip on last iteration)
        if step < n_acquisitions and len(al_manager.pool_indices) > 0:
            pool_size = len(datasets_dict['pool'])
            subsample_size = min(REPRO_POOL_SUBSAMPLE, pool_size)
            pool_sample_rel = random.sample(range(pool_size), subsample_size)
            
            # Create subset for efficiency
            pool_subset = torch.utils.data.Subset(datasets_dict['pool'], pool_sample_rel)
            pool_loader = torch.utils.data.DataLoader(
                pool_subset,
                batch_size=512,
                shuffle=False
            )
            
            # Collect predictions
            all_predictions = []
            with torch.no_grad():
                for data, _ in pool_loader:
                    data = data.to(device)
                    batch_preds = []
                    
                    # Multiple forward passes
                    for _ in range(MC_DROPOUT_SAMPLES):
                        if model_type == 'bayesian':
                            output = model(data, mc_dropout=True)
                        else:
                            output = model(data)
                        batch_preds.append(output.unsqueeze(0))
                    
                    batch_preds = torch.cat(batch_preds, dim=0)
                    all_predictions.append(batch_preds.cpu())
            
            if all_predictions:
                pool_predictions = torch.cat(all_predictions, dim=1)
                scores = acquisition_func(pool_predictions)
                k = min(ACQUIRE_K, len(scores))
                _, selected_rel = torch.topk(scores, k)
                
                # Convert to absolute indices
                selected_abs_in_pool = [pool_sample_rel[r] for r in selected_rel.cpu().tolist()]
                al_manager.acquire_points(selected_abs_in_pool)
                
                # Update datasets
                datasets_dict = al_manager.get_datasets()
    
    return test_accuracies


def run_minimal_extension_experiment(inference_method='analytic',
                                    n_acquisitions=MINIMAL_ACQ_STEPS,
                                    device=DEVICE):
    """Run minimal extension experiment (Bayesian last-layer inference)."""
    
    # Data transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Load datasets
    train_dataset = datasets.MNIST(DATA_DIR, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(DATA_DIR, train=False, transform=transform)
    
    def to_regression(dataset):
        """Convert to one-hot regression format."""
        images, targets = [], []
        for img, label in dataset:
            images.append(img)
            t = torch.zeros(10, dtype=torch.float32)
            t[label] = 1.0
            targets.append(t)
        return torch.stack(images), torch.stack(targets)
    
    # Convert to regression
    X_train_full, y_train_full = to_regression(train_dataset)
    X_test, y_test = to_regression(test_dataset)
    
    # Move to device
    X_train_full = X_train_full.to(device)
    y_train_full = y_train_full.to(device)
    X_test = X_test.to(device)
    y_test = y_test.to(device)
    
    # Initialize feature extractor
    feature_extractor = FrozenFeatureExtractor(device=device)
    
    # Initialize inference method
    if inference_method == 'analytic':
        inference = AnalyticGaussianInference(
            prior_var=PRIOR_VARIANCE,
            noise_var=NOISE_VARIANCE,
            device=device
        )
    elif inference_method == 'mfvi_diagonal':
        inference = MFVIDiagonalInference(
            feature_dim=128,
            output_dim=10,
            prior_var=PRIOR_VARIANCE,
            noise_var=NOISE_VARIANCE,
            device=device
        )
    elif inference_method == 'mfvi_full':
        inference = MFVIFullCovarianceInference(
            feature_dim=128,
            output_dim=10,
            prior_var=PRIOR_VARIANCE,
            noise_var=NOISE_VARIANCE,
            device=device
        )
    else:
        raise ValueError(f"Unknown inference method: {inference_method}")
    
    # Create initial splits
    N = len(X_train_full)
    indices = torch.randperm(N).to(device)
    labeled_indices = indices[:INITIAL_LABELED]
    pool_indices = indices[INITIAL_LABELED:POOL_LIMIT]
    
    # Storage for RMSE history
    rmse_history = []
    
    # Active learning loop
    for step in range(n_acquisitions + 1):
        if step % 5 == 0:
            print(f"  Step {step}: Labeled = {len(labeled_indices)}")
        
        # Extract features for labeled data
        X_labeled = X_train_full[labeled_indices]
        y_labeled = y_train_full[labeled_indices]
        with torch.no_grad():
            features_labeled = feature_extractor(X_labeled)
        
        # Fit inference model
        inference.fit(features_labeled, y_labeled)
        
        # Evaluate on test subset
        with torch.no_grad():
            features_test = feature_extractor(
                X_test[:MINIMAL_IMAGES_PER_STEP * n_acquisitions]
            )
            rmse = inference.rmse(
                features_test,
                y_test[:MINIMAL_IMAGES_PER_STEP * n_acquisitions]
            )
            rmse_history.append(rmse)
        
        # Acquisition step
        if step < n_acquisitions and len(pool_indices) > 0:
            X_pool = X_train_full[pool_indices]
            with torch.no_grad():
                features_pool = feature_extractor(X_pool)
            
            # Predict and get uncertainty (variance)
            _, pred_var = inference.predict(features_pool, n_samples=PREDICTION_SAMPLES)
            
            # Select points with highest uncertainty
            k = min(ACQUIRE_K, len(pred_var))
            _, selected_rel = torch.topk(pred_var, k)
            selected_abs = pool_indices[selected_rel]
            
            # Update sets
            labeled_indices = torch.cat([labeled_indices, selected_abs])
            
            # Remove from pool
            mask = torch.ones(len(pool_indices), dtype=torch.bool, device=device)
            mask[selected_rel] = False
            pool_indices = pool_indices[mask]
    
    return rmse_history


def run_novel_extension_experiment(n_acquisitions=NOVEL_ACQ_STEPS, device=DEVICE):
    """Run novel extension experiment (heteroscedastic last layer)."""
    
    # Data transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Load datasets
    train_dataset = datasets.MNIST(DATA_DIR, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(DATA_DIR, train=False, transform=transform)
    
    def to_regression(dataset):
        """Convert to one-hot regression format."""
        images, targets = [], []
        for img, label in dataset:
            images.append(img)
            t = torch.zeros(10, dtype=torch.float32)
            t[label] = 1.0
            targets.append(t)
        return torch.stack(images), torch.stack(targets)
    
    # Convert to regression
    X_train_full, y_train_full = to_regression(train_dataset)
    X_test, y_test = to_regression(test_dataset)
    
    # Move to device
    X_train_full = X_train_full.to(device)
    y_train_full = y_train_full.to(device)
    X_test = X_test.to(device)
    y_test = y_test.to(device)
    
    # Initialize models
    feature_extractor = FrozenFeatureExtractor(device=device)
    model = HeteroscedasticLastLayer(
        feature_dim=128,
        output_dim=10,
        prior_var=PRIOR_VARIANCE,
        device=device
    )
    
    # Create initial splits
    N = len(X_train_full)
    indices = torch.randperm(N).to(device)
    labeled_indices = indices[:INITIAL_LABELED]
    pool_indices = indices[INITIAL_LABELED:POOL_LIMIT]
    
    # Storage for metrics
    rmse_history = []
    aleatoric_history = []
    epistemic_history = []
    
    # Active learning loop
    for step in range(n_acquisitions + 1):
        if step % 5 == 0:
            print(f"  Step {step}: Labeled = {len(labeled_indices)}")
        
        # Extract features for labeled data
        X_labeled = X_train_full[labeled_indices]
        y_labeled = y_train_full[labeled_indices]
        with torch.no_grad():
            features_labeled = feature_extractor(X_labeled)
        
        # Fit heteroscedastic model
        model.fit(features_labeled, y_labeled, n_iter=MFVI_ITERATIONS, lr=MFVI_LR)
        
        # Evaluate on test subset
        with torch.no_grad():
            features_test = feature_extractor(
                X_test[:NOVEL_IMAGES_PER_STEP * n_acquisitions]
            )
            
            # Compute RMSE
            rmse = model.rmse(
                features_test,
                y_test[:NOVEL_IMAGES_PER_STEP * n_acquisitions]
            )
            rmse_history.append(rmse)
            
            # Compute uncertainty decomposition
            _, _, aleatoric, epistemic = model.predict(
                features_test[:min(100, features_test.size(0))],
                n_samples=PREDICTION_SAMPLES
            )
            aleatoric_history.append(aleatoric.mean().item())
            epistemic_history.append(epistemic.mean().item())
        
        # Acquisition step
        if step < n_acquisitions and len(pool_indices) > 0:
            X_pool = X_train_full[pool_indices]
            with torch.no_grad():
                features_pool = feature_extractor(X_pool)
            
            # Compute predictive entropy (total uncertainty)
            entropy = model.predictive_entropy(features_pool, n_samples=PREDICTION_SAMPLES)
            
            # Select points with highest entropy
            k = min(ACQUIRE_K, len(entropy))
            _, selected_rel = torch.topk(entropy, k)
            selected_abs = pool_indices[selected_rel]
            
            # Update sets
            labeled_indices = torch.cat([labeled_indices, selected_abs])
            
            # Remove from pool
            mask = torch.ones(len(pool_indices), dtype=torch.bool, device=device)
            mask[selected_rel] = False
            pool_indices = pool_indices[mask]
    
    return {
        'rmse': rmse_history,
        'aleatoric': aleatoric_history,
        'epistemic': epistemic_history
    }