"""
Visualization functions for active learning project.
"""

import matplotlib.pyplot as plt
import numpy as np
from config import *


def plot_reproduction_results(results, images_per_step=REPRO_IMAGES_PER_STEP,
                             save_path='reproduction_results.png'):
    """Plot reproduction experiment results."""
    
    plt.figure(figsize=(12, 6))
    
    colors = {
        'bald': 'blue',
        'variation_ratios': 'green',
        'max_entropy': 'red',
        'mean_std': 'orange',
        'random': 'purple'
    }
    
    labels = {
        'bald': 'BALD',
        'variation_ratios': 'Variation Ratios',
        'max_entropy': 'Max Entropy',
        'mean_std': 'Mean STD',
        'random': 'Random'
    }
    
    for acq_name, color in colors.items():
        if acq_name in results:
            accuracies = results[acq_name]
            x = np.arange(len(accuracies)) * images_per_step
            plt.plot(x, accuracies, label=labels[acq_name], linewidth=2, color=color)
    
    plt.xlabel('Number of Acquired Images', fontsize=12)
    plt.ylabel('Test Accuracy (%)', fontsize=12)
    plt.title('Figure 1: MNIST Test Accuracy vs Acquired Images (Bayesian CNN)', fontsize=14)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    max_steps = max((len(v) for v in results.values()), default=0)
    plt.xlim(0, max_steps * images_per_step)
    plt.ylim(0, 100)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
    
    plt.show()


def plot_bayesian_vs_deterministic(results_dict, images_per_step=REPRO_IMAGES_PER_STEP,
                                  save_path='bayesian_vs_deterministic.png'):
    """Plot Bayesian vs Deterministic comparison."""
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    acq_funcs = ['bald', 'variation_ratios', 'max_entropy']
    titles = ['BALD', 'Variation Ratios', 'Max Entropy']
    
    for idx, (acq_func, title) in enumerate(zip(acq_funcs, titles)):
        ax = axes[idx]
        
        bayesian_acc = results_dict.get('bayesian', {}).get(acq_func)
        deterministic_acc = results_dict.get('deterministic', {}).get(acq_func)
        
        if bayesian_acc is not None:
            x_b = np.arange(len(bayesian_acc)) * images_per_step
            ax.plot(x_b, bayesian_acc, label='Bayesian CNN', linewidth=2, color='red')
        
        if deterministic_acc is not None:
            x_d = np.arange(len(deterministic_acc)) * images_per_step
            ax.plot(x_d, deterministic_acc, label='Deterministic CNN', linewidth=2, color='blue')
        
        ax.set_xlabel('Number of Acquired Images', fontsize=10)
        ax.set_ylabel('Test Accuracy (%)', fontsize=10)
        ax.set_title(title, fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        
        max_len = 0
        if bayesian_acc is not None:
            max_len = max(max_len, len(bayesian_acc))
        if deterministic_acc is not None:
            max_len = max(max_len, len(deterministic_acc))
        ax.set_xlim(0, max_len * images_per_step)
        ax.set_ylim(0, 100)
    
    plt.suptitle('Figure 2: Bayesian vs Deterministic CNN Comparison', fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
    
    plt.show()


def plot_minimal_results_comparison(results_dict, images_per_step=MINIMAL_IMAGES_PER_STEP,
                                   save_path='minimal_comparison.png'):
    """Plot minimal extension results comparison."""
    
    plt.figure(figsize=(12, 6))
    
    methods = ['analytic', 'mfvi_diagonal', 'mfvi_full']
    colors = {'analytic': 'blue', 'mfvi_diagonal': 'green', 'mfvi_full': 'red'}
    labels = {
        'analytic': 'Analytic (Full Cov)',
        'mfvi_diagonal': 'MFVI (Diagonal)',
        'mfvi_full': 'MFVI (Full Cov)'
    }
    
    max_len = 0
    
    for method in methods:
        rmse_values = results_dict.get(method)
        if rmse_values:
            x = np.arange(len(rmse_values)) * images_per_step
            plt.plot(x, rmse_values, label=labels[method], linewidth=2, color=colors[method])
            max_len = max(max_len, len(rmse_values))
    
    plt.xlabel('Number of Acquired Images', fontsize=12)
    plt.ylabel('RMSE', fontsize=12)
    plt.title('Minimal Extension: Comparison of Three Inference Methods', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    plt.xlim(0, max_len * images_per_step)
    
    y_max = 0
    for v in results_dict.values():
        if isinstance(v, list) and len(v) > 0:
            y_max = max(y_max, max(v))
    plt.ylim(0, max(1.0, y_max))
    
    plt.axhline(y=0.95, color='gray', linestyle='--', alpha=0.5, label='Random baseline')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
    
    plt.show()


def plot_novel_results(novel_results, images_per_step=NOVEL_IMAGES_PER_STEP,
                      save_path='novel_results.png'):
    """Plot novel extension results."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    x = np.arange(len(novel_results['rmse'])) * images_per_step
    
    # Subplot 1: RMSE
    ax1 = axes[0, 0]
    ax1.plot(x, novel_results['rmse'], color='blue', linewidth=2)
    ax1.set_xlabel('Number of Acquired Images')
    ax1.set_ylabel('RMSE')
    ax1.set_title('Heteroscedastic Model: RMSE')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, max(1, len(novel_results['rmse'])) * images_per_step)
    
    if novel_results['rmse']:
        ax1.set_ylim(0, max(1.0, max(novel_results['rmse'])))
    
    # Subplot 2: Uncertainty decomposition
    ax2 = axes[0, 1]
    ax2.plot(x, novel_results['aleatoric'], label='Aleatoric', linewidth=2)
    ax2.plot(x, novel_results['epistemic'], label='Epistemic', linewidth=2)
    ax2.set_xlabel('Number of Acquired Images')
    ax2.set_ylabel('Uncertainty')
    ax2.set_title('Uncertainty Decomposition')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, max(1, len(novel_results['aleatoric'])) * images_per_step)
    
    # Subplot 3: Uncertainty ratio
    ax3 = axes[1, 0]
    aleatoric = np.array(novel_results['aleatoric'])
    epistemic = np.array(novel_results['epistemic'])
    ratio = aleatoric / (aleatoric + epistemic + 1e-10)
    
    ax3.plot(x, ratio, color='brown', linewidth=2)
    ax3.set_xlabel('Number of Acquired Images')
    ax3.set_ylabel('Aleatoric / Total Ratio')
    ax3.set_title('Uncertainty Ratio Evolution')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 1)
    ax3.set_xlim(0, max(1, len(ratio)) * images_per_step)
    
    # Subplot 4: RMSE vs Epistemic
    ax4 = axes[1, 1]
    ax4.scatter(novel_results['rmse'], novel_results['epistemic'], alpha=0.6, color='red')
    ax4.set_xlabel('RMSE')
    ax4.set_ylabel('Epistemic Uncertainty')
    ax4.set_title('RMSE vs Epistemic Uncertainty')
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle('Novel Extension: Heteroscedastic Model Results', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
    
    plt.show()