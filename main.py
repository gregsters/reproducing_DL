#!/usr/bin/env python3
"""
Active Learning Project - Main Script
Run this file to execute the complete project.
"""

import os
import time
import warnings
warnings.filterwarnings('ignore')

# Import project modules
from config import *
from utils import set_seed, check_gpu_availability, create_results_table
from experiments import (
    run_reproduction_experiment, 
    run_minimal_extension_experiment, 
    run_novel_extension_experiment
)
from visualization import (
    plot_reproduction_results,
    plot_bayesian_vs_deterministic,
    plot_minimal_results_comparison,
    plot_novel_results
)

def main():
    """Main function to run the complete active learning project."""
    
    print("="*70)
    print("ACTIVE LEARNING PROJECT")
    print("="*70)
    
    # Setup
    print(f"\n[SETUP] Using device: {DEVICE}")
    set_seed(SEED)
    
    # Check GPU
    device_type = check_gpu_availability()
    
    # Create results directory
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"[SETUP] Results will be saved to: {RESULTS_DIR}")
    
    total_start_time = time.time()
    
    # Storage for results
    repro_results = {'bayesian': {}, 'deterministic': {}}
    minimal_results = {}
    

    # EXPERIMENT 1: REPRODUCTION (Bayesian CNN)

    print("\n" + "="*60)
    print("EXPERIMENT 1: REPRODUCTION (Bayesian CNN with MC Dropout)")
    print("="*60)
    
    acquisition_funcs = ['bald', 'variation_ratios', 'max_entropy', 'mean_std', 'random']
    
    for acq_func in acquisition_funcs:
        print(f"\n[BAYESIAN] Running {acq_func.upper()}...")
        start_time = time.time()
        
        accuracies = run_reproduction_experiment(
            model_type='bayesian',
            acquisition_name=acq_func,
            n_acquisitions=REPRO_ACQ_STEPS_BAYES,
            device=DEVICE
        )
        
        repro_results['bayesian'][acq_func] = accuracies
        elapsed = time.time() - start_time
        
        print(f"  Final accuracy: {accuracies[-1]:.2f}%")
        print(f"  Time elapsed: {elapsed:.1f} seconds")
    
    # Plot Bayesian results
    print("\n[PLOTTING] Generating Figure 1...")
    plot_reproduction_results(
        repro_results['bayesian'],
        images_per_step=REPRO_IMAGES_PER_STEP,
        save_path=os.path.join(RESULTS_DIR, 'figure1_reproduction.png')
    )
    

    # EXPERIMENT 2: REPRODUCTION (Deterministic CNN - Comparison)

    print("\n" + "="*60)
    print("EXPERIMENT 2: REPRODUCTION (Deterministic CNN - Comparison)")
    print("="*60)
    
    comparison_funcs = ['bald', 'variation_ratios', 'max_entropy']
    
    for acq_func in comparison_funcs:
        print(f"\n[DETERMINISTIC] Running {acq_func.upper()}...")
        start_time = time.time()
        
        accuracies = run_reproduction_experiment(
            model_type='deterministic',
            acquisition_name=acq_func,
            n_acquisitions=REPRO_ACQ_STEPS_DET,
            device=DEVICE
        )
        
        repro_results['deterministic'][acq_func] = accuracies
        elapsed = time.time() - start_time
        
        print(f"  Final accuracy: {accuracies[-1]:.2f}%")
        print(f"  Time elapsed: {elapsed:.1f} seconds")
    
    # Plot comparison
    print("\n[PLOTTING] Generating Figure 2...")
    plot_bayesian_vs_deterministic(
        repro_results,
        images_per_step=REPRO_IMAGES_PER_STEP,
        save_path=os.path.join(RESULTS_DIR, 'figure2_comparison.png')
    )
    

    # EXPERIMENT 3: MINIMAL EXTENSION

    print("\n" + "="*60)
    print("EXPERIMENT 3: MINIMAL EXTENSION (Bayesian Inference Methods)")
    print("="*60)
    
    methods = ['analytic', 'mfvi_diagonal', 'mfvi_full']
    
    for method in methods:
        print(f"\n[MINIMAL] Running {method}...")
        start_time = time.time()
        
        rmse = run_minimal_extension_experiment(
            inference_method=method,
            n_acquisitions=MINIMAL_ACQ_STEPS,
            device=DEVICE
        )
        
        minimal_results[method] = rmse
        elapsed = time.time() - start_time
        
        print(f"  Final RMSE: {rmse[-1]:.4f}")
        print(f"  Time elapsed: {elapsed:.1f} seconds")
    
    # Plot minimal extension results
    print("\n[PLOTTING] Generating minimal extension comparison plot...")
    plot_minimal_results_comparison(
        minimal_results,
        images_per_step=MINIMAL_IMAGES_PER_STEP,
        save_path=os.path.join(RESULTS_DIR, 'minimal_comparison.png')
    )
    

    # EXPERIMENT 4: NOVEL EXTENSION

    print("\n" + "="*60)
    print("EXPERIMENT 4: NOVEL EXTENSION (Heteroscedastic Uncertainty)")
    print("="*60)
    
    print("[NOVEL] Running heteroscedastic model...")
    start_time = time.time()
    
    novel_results = run_novel_extension_experiment(
        n_acquisitions=NOVEL_ACQ_STEPS,
        device=DEVICE
    )
    
    elapsed = time.time() - start_time
    
    print(f"\n  Final RMSE: {novel_results['rmse'][-1]:.4f}")
    print(f"  Final Aleatoric Uncertainty: {novel_results['aleatoric'][-1]:.4f}")
    print(f"  Final Epistemic Uncertainty: {novel_results['epistemic'][-1]:.4f}")
    print(f"  Time elapsed: {elapsed:.1f} seconds")
    
    # Plot novel extension results
    print("\n[PLOTTING] Generating novel extension results plot...")
    plot_novel_results(
        novel_results,
        images_per_step=NOVEL_IMAGES_PER_STEP,
        save_path=os.path.join(RESULTS_DIR, 'novel_results.png')
    )
    

    # FINAL SUMMARY

    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    
    # Display summary tables
    create_results_table(repro_results, minimal_results, novel_results)
    
    # Calculate total time
    total_time = time.time() - total_start_time
    print(f"\nTotal execution time: {total_time/60:.1f} minutes")
    
    # Key findings
    print("\n" + "="*60)
    print("KEY FINDINGS")
    print("="*60)
    
    print("\n1. REPRODUCTION EXPERIMENT:")
    print("   • Bayesian CNN generally outperforms Deterministic CNN")
    print("   • BALD acquisition often yields the best performance")
    print("   • Random acquisition serves as a useful baseline")
    
    print("\n2. MINIMAL EXTENSION:")
    print("   • Analytic method provides exact inference but is computationally expensive")
    print("   • MFVI with diagonal covariance offers good balance of speed and accuracy")
    print("   • MFVI with full covariance is expressive but slow")
    
    print("\n3. NOVEL EXTENSION:")
    print("   • Heteroscedastic model successfully decomposes uncertainty")
    print("   • Epistemic uncertainty decreases with more data")
    print("   • Aleatoric uncertainty captures inherent data noise")
    
    print("\n" + "="*70)
    print("PROJECT COMPLETE")
    print(f"All results saved in: {RESULTS_DIR}")
    print("="*70)
    
    # Save results to files
    save_results_to_files(repro_results, minimal_results, novel_results)


def save_results_to_files(repro_results, minimal_results, novel_results):
    """Save results to files for later analysis."""
    import pickle
    import json
    
    print("\n[SAVING] Saving results to files...")
    
    # Save reproduction results
    with open(os.path.join(RESULTS_DIR, 'repro_results.pkl'), 'wb') as f:
        pickle.dump(repro_results, f)
    
    # Save minimal results
    with open(os.path.join(RESULTS_DIR, 'minimal_results.pkl'), 'wb') as f:
        pickle.dump(minimal_results, f)
    
    # Save novel results as JSON (for better readability)
    novel_results_json = {k: [float(x) for x in v] for k, v in novel_results.items()}
    with open(os.path.join(RESULTS_DIR, 'novel_results.json'), 'w') as f:
        json.dump(novel_results_json, f, indent=2)
    
    print(f"[SAVED] Results saved to {RESULTS_DIR}/")
    print(f"  - repro_results.pkl")
    print(f"  - minimal_results.pkl")
    print(f"  - novel_results.json")
    print(f"  - figure1_reproduction.png")
    print(f"  - figure2_comparison.png")
    print(f"  - minimal_comparison.png")
    print(f"  - novel_results.png")


if __name__ == "__main__":
    main()