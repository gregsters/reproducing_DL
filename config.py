"""
Configuration file for Active Learning Project.
Contains all global constants and hyperparameters.
"""

import torch

# EXPERIMENT PARAMETERS

# Reproduction Experiment
REPRO_ACQ_STEPS_BAYES = 100      # Acquisition steps for Bayesian CNN
REPRO_ACQ_STEPS_DET = 100        # Acquisition steps for Deterministic CNN
REPRO_IMAGES_PER_STEP = 10       # Images acquired per step

# Extension Experiments
MINIMAL_ACQ_STEPS = 50           # Minimal extension acquisition steps
NOVEL_ACQ_STEPS = 50             # Novel extension acquisition steps
MINIMAL_IMAGES_PER_STEP = 10     # Images per step for minimal extension
NOVEL_IMAGES_PER_STEP = 10       # Images per step for novel extension

# DATA AND ACQUISITION PARAMETERS

REPRO_POOL_SUBSAMPLE = 1000      # Pool subsample for scoring
POOL_LIMIT = 2000                # Limit pool size for speed
ACQUIRE_K = 10                   # Points to acquire per acquisition
INITIAL_LABELED = 20             # Initial labeled samples
VALIDATION_SIZE = 100            # Validation set size

# TRAINING PARAMETERS

TRAIN_BATCH = 64                 # Training batch size
VAL_BATCH = 64                   # Validation batch size
LEARNING_RATE = 0.001            # Learning rate
WEIGHT_DECAY = 1e-4              # Weight decay
TRAIN_EPOCHS = 3                 # Epochs per acquisition

# BAYESIAN PARAMETERS

MC_DROPOUT_SAMPLES = 3           # MC dropout samples for uncertainty
PREDICTION_SAMPLES = 5           # Samples for Bayesian predictions
PRIOR_VARIANCE = 1.0             # Prior variance for Bayesian inference
NOISE_VARIANCE = 0.1             # Observation noise variance

# INFERENCE PARAMETERS

MFVI_ITERATIONS = 30             # MFVI optimization iterations
MFVI_LR = 0.01                   # Learning rate for MFVI

# SYSTEM PARAMETERS

# Device configuration
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Random seed for reproducibility
SEED = 42

# Paths
RESULTS_DIR = 'results'          # Directory for saving results
DATA_DIR = './data'              # Directory for dataset