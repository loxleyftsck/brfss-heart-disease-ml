"""
Configuration file for ADAPTA project.
Contains all global constants, file paths, and model parameters.
"""

import os
from pathlib import Path

# ============================================================================
# PROJECT PATHS
# ============================================================================

# Get project root directory
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"

# ============================================================================
# DATA FILES
# ============================================================================

RAW_DATASET_PATH = RAW_DATA_DIR / "brfss2020.csv"
PROCESSED_TRAIN_PATH = PROCESSED_DATA_DIR / "X_train.pkl"
PROCESSED_TEST_PATH = PROCESSED_DATA_DIR / "X_test.pkl"
PROCESSED_LABELS_PATH = PROCESSED_DATA_DIR / "y_train.pkl"

# ============================================================================
# REPRODUCIBILITY
# ============================================================================

RANDOM_SEED = 42

# ============================================================================
# DATA PREPROCESSING
# ============================================================================

# Target variable
TARGET_COLUMN = "_MICHD"

# Selected features for modeling
SELECTED_FEATURES = [
    "_SEX",      # Gender
    "_AGEG5YR",  # Age group (5-year intervals)
    "_RFSMOK3",  # Smoking status
    "_RFBMI5",   # BMI category
    "_TOTINDA",  # Physical activity
]

# Missing value codes in BRFSS
MISSING_VALUE_CODES = [7, 9]  # 7 = Don't know, 9 = Refused

# Train/test split ratio
TEST_SIZE = 0.3
VALIDATION_SIZE = 0.2  # For hyperparameter tuning

# ============================================================================
# MODEL PARAMETERS
# ============================================================================

# Random Forest default parameters
RF_PARAMS_DEFAULT = {
    "n_estimators": 100,
    "class_weight": "balanced",
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
}

# Logistic Regression default parameters
LR_PARAMS_DEFAULT = {
    "class_weight": "balanced",
    "max_iter": 1000,
    "solver": "lbfgs",
    "random_state": RANDOM_SEED,
}

# ============================================================================
# HYPERPARAMETER TUNING
# ============================================================================

# Random Forest hyperparameter grid
RF_PARAM_GRID = {
    "n_estimators": [100, 200],
    "max_depth": [10, 20, None],
    "min_samples_split": [2, 10],
    "class_weight": ["balanced"],
}

# Logistic Regression hyperparameter grid
LR_PARAM_GRID = {
    "C": [0.01, 0.1, 1, 10],
    "penalty": ["l2"],
    "solver": ["lbfgs"],
}

# GridSearchCV parameters
CV_FOLDS = 5
SCORING_METRIC = "f1"  # Focus on F1-score for imbalanced data

# ============================================================================
# EVALUATION
# ============================================================================

# Class labels
CLASS_LABELS = {
    0: "No Heart Disease",
    1: "Has Heart Disease"
}

# Metrics to track
EVALUATION_METRICS = [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
]

# ============================================================================
# VISUALIZATION
# ============================================================================

# Plot settings
FIGURE_DPI = 300
FIGURE_FORMAT = "png"

# Color schemes
COLOR_PALETTE = {
    "rf_baseline": "#f9844a",
    "rf_tuned": "#f3722c",
    "lr_baseline": "#577590",
    "lr_tuned": "#277da1",
}

# ============================================================================
# LOGGING
# ============================================================================

LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
