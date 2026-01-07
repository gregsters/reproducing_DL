"""
Utility functions for active learning project.
"""

import random
import numpy as np
import torch
from config import SEED, DEVICE


def set_seed(seed=SEED):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"[UTILS] Random seed set to {seed}")


def check_gpu_availability():
    """Check GPU availability and print information."""
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        print(f"[GPU] Available: {device_name}")
        print(f"[GPU] CUDA version: {torch.version.cuda}")
        return "cuda"
    else:
        print("[GPU] Not available, using CPU")
        return "cpu"


def format_value(val, precision=2):
    """Format a value with specified precision."""
    try:
        if isinstance(val, (list, tuple)):
            v = float(val[-1])
        else:
            v = float(val)
        return f"{v:.{precision}f}"
    except (ValueError, TypeError, IndexError):
        return str(val)


def create_results_table(repro_results, minimal_results, novel_results):
    """Create and print formatted results tables."""
    
    print("\n" + "="*70)
    print("SUMMARY TABLES")
    print("="*70)
    
    # Table 1: Reproduction results
    print("\nTable 1: Reproduction - Final Accuracy (%)")
    print("-"*50)
    print(f"{'Acquisition Function':<20} {'Bayesian CNN':<15} {'Deterministic CNN':<15}")
    print("-"*50)
    
    acquisition_funcs = ['bald', 'variation_ratios', 'max_entropy', 'mean_std', 'random']
    
    for acq_func in acquisition_funcs:
        bayesian_acc = repro_results.get('bayesian', {}).get(acq_func, 'N/A')
        deterministic_acc = repro_results.get('deterministic', {}).get(acq_func, 'N/A')
        
        print(f"{acq_func:<20} {format_value(bayesian_acc, 2):<15} {format_value(deterministic_acc, 2):<15}")
    
    # Table 2: Minimal extension results
    print("\n\nTable 2: Minimal Extension - Final RMSE")
    print("-"*50)
    print(f"{'Inference Method':<20} {'RMSE':<15} {'Improvement over Random':<25}")
    print("-"*50)
    
    random_rmse = 0.95  # Baseline
    methods = ['analytic', 'mfvi_diagonal', 'mfvi_full']
    
    for method in methods:
        rmse_list = minimal_results.get(method)
        
        if rmse_list and isinstance(rmse_list, (list, tuple)) and len(rmse_list) > 0:
            rmse = float(rmse_list[-1])
            improvement = 100 * (random_rmse - rmse) / random_rmse
            print(f"{method:<20} {rmse:<15.4f} {improvement:<25.1f}%")
        else:
            print(f"{method:<20} {'N/A':<15} {'N/A':<25}")
    
    # Table 3: Novel extension results
    print("\n\nTable 3: Novel Extension - Results")
    print("-"*50)
    
    try:
        final_rmse = novel_results.get('rmse', ['N/A'])[-1]
        final_ale = novel_results.get('aleatoric', ['N/A'])[-1]
        final_epi = novel_results.get('epistemic', ['N/A'])[-1]
        
        print(f"{'Final RMSE':<20} {format_value(final_rmse, 4):<15}")
        print(f"{'Final Aleatoric':<20} {format_value(final_ale, 4):<15}")
        print(f"{'Final Epistemic':<20} {format_value(final_epi, 4):<15}")
    except (IndexError, KeyError):
        print(f"{'Final RMSE':<20} {'N/A':<15}")
        print(f"{'Final Aleatoric':<20} {'N/A':<15}")
        print(f"{'Final Epistemic':<20} {'N/A':<15}")
    
    print("\n" + "="*70)