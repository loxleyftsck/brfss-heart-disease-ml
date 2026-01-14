"""
Data preprocessing module for ADAPTA project.
Handles cleaning, transformation, and train/test splitting.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional
import logging
import pickle
from pathlib import Path

from src.config import (
    TARGET_COLUMN,
    SELECTED_FEATURES,
    MISSING_VALUE_CODES,
    TEST_SIZE,
    RANDOM_SEED,
    PROCESSED_DATA_DIR,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Preprocess BRFSS data for machine learning.
    
    Steps:
    1. Handle missing values
    2. Convert target variable to binary
    3. Train/test split with stratification
    4. Feature scaling
    """
    
    def __init__(self, random_seed: int = RANDOM_SEED):
        """
        Initialize preprocessor.
        
        Parameters
        ----------
        random_seed : int
            Random seed for reproducibility
        """
        self.random_seed = random_seed
        self.scaler = StandardScaler()
        self.feature_names = None
        
    def handle_missing_values(
        self, 
        df: pd.DataFrame,
        strategy: str = "drop"
    ) -> pd.DataFrame:
        """
        Handle missing values in dataset.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe
        strategy : str, default="drop"
            Strategy for handling missing values:
            - "drop": Remove rows with missing values
            - "impute": Fill with median (not implemented yet)
            
        Returns
        -------
        pd.DataFrame
            Cleaned dataframe
        """
        logger.info("Handling missing values...")
        
        initial_rows = len(df)
        
        # Replace BRFSS missing value codes with NaN
        df_clean = df.replace(MISSING_VALUE_CODES, pd.NA)
        
        if strategy == "drop":
            # Drop rows with any missing values
            df_clean = df_clean.dropna()
            
        elif strategy == "impute":
            # TODO: Implement imputation strategy
            raise NotImplementedError("Imputation not yet implemented")
        
        rows_removed = initial_rows - len(df_clean)
        logger.info(
            f"✓ Removed {rows_removed:,} rows with missing values "
            f"({rows_removed/initial_rows*100:.1f}%)"
        )
        logger.info(f"✓ Remaining: {len(df_clean):,} rows")
        
        return df_clean
    
    def prepare_target_variable(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert target variable to binary format.
        
        BRFSS _MICHD encoding:
        - 1 = Yes (has heart disease) → 1
        - 2 = No (no heart disease) → 0
        
        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe
            
        Returns
        -------
        pd.DataFrame
            DataFrame with binary target
        """
        if TARGET_COLUMN not in df.columns:
            raise ValueError (f"Target column '{TARGET_COLUMN}' not found")
        
        logger.info("Preparing target variable...")
        
        df = df.copy()
        
        # Convert to binary: 1 → 1 (positive), 2 → 0 (negative)
        df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int).apply(
            lambda x: 1 if x == 1 else 0
        )
        
        # Check class distribution
        class_counts = df[TARGET_COLUMN].value_counts().sort_index()
        total = len(df)
        
        logger.info("Class distribution:")
        logger.info(f"  Negative (0): {class_counts[0]:,} ({class_counts[0]/total*100:.1f}%)")
        logger.info(f"  Positive (1): {class_counts[1]:,} ({class_counts[1]/total*100:.1f}%)")
        
        return df
    
    def split_features_target(
        self, 
        df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Separate features (X) and target (y).
        
        Parameters
        ----------
        df : pd.DataFrame
            Preprocessed dataframe
            
        Returns
        -------
        Tuple[pd.DataFrame, pd.Series]
            (X, y) features and target
        """
        if TARGET_COLUMN not in df.columns:
            raise ValueError(f"Target column '{TARGET_COLUMN}' not found")
        
        X = df.drop(columns=[TARGET_COLUMN])
        y = df[TARGET_COLUMN]
        
        logger.info(f"✓ Features shape: {X.shape}")
        logger.info(f"✓ Target shape: {y.shape}")
        
        self.feature_names = X.columns.tolist()
        
        return X, y
    
    def create_train_test_split(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = TEST_SIZE,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split data into train and test sets with stratification.
        
        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix
        y : pd.Series
            Target vector
        test_size : float
            Proportion of test set
            
        Returns
        -------
        Tuple
            (X_train, X_test, y_train, y_test)
        """
        logger.info(f"Splitting data: {int((1-test_size)*100)}% train, {int(test_size*100)}% test...")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=self.random_seed,
            stratify=y  # Preserve class distribution
        )
        
        logger.info(f"✓ Train set: {len(X_train):,} samples")
        logger.info(f"✓ Test set: {len(X_test):,} samples")
        
        # Verify stratification
        train_dist = y_train.value_counts(normalize=True)
        test_dist = y_test.value_counts(normalize=True)
        
        logger.info("\nClass distribution preserved:")
        logger.info(f"  Train: {train_dist[1]:.1%} positive")
        logger.info(f"  Test:  {test_dist[1]:.1%} positive")
        
        return X_train, X_test, y_train, y_test
    
    def scale_features(
        self,
        X_train: pd.DataFrame,
        X_test: Optional[pd.DataFrame] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Apply StandardScaler normalization.
        
        IMPORTANT: Fit scaler on training data only to prevent data leakage.
        
        Parameters
        ----------
        X_train : pd.DataFrame
            Training features
        X_test : pd.DataFrame, optional
            Test features
            
        Returns
        -------
        Tuple
            (X_train_scaled, X_test_scaled)
        """
        logger.info("Scaling features with StandardScaler...")
        
        # Fit scaler on training data only
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Transform test data with fitted scaler
        X_test_scaled = None
        if X_test is not None:
            X_test_scaled = self.scaler.transform(X_test)
        
        logger.info("✓ Features scaled (mean=0, std=1)")
        
        return X_train_scaled, X_test_scaled
    
    def full_pipeline(
        self, 
        df: pd.DataFrame,
        save_processed: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, pd.Series, pd.Series]:
        """
        Run complete preprocessing pipeline.
        
        Parameters
        ----------
        df : pd.DataFrame
            Raw dataframe
        save_processed : bool
            Whether to save processed data to disk
            
        Returns
        -------
        Tuple
            (X_train_scaled, X_test_scaled, y_train, y_test)
        """
        logger.info("=" * 60)
        logger.info("Starting preprocessing pipeline...")
        logger.info("=" * 60)
        
        # Step 1: Clean missing values
        df_clean = self.handle_missing_values(df)
        
        # Step 2: Prepare target variable
        df_clean = self.prepare_target_variable(df_clean)
        
        # Step 3: Split features and target
        X, y = self.split_features_target(df_clean)
        
        # Step 4: Train/test split
        X_train, X_test, y_train, y_test = self.create_train_test_split(X, y)
        
        # Step 5: Scale features
        X_train_scaled, X_test_scaled = self.scale_features(X_train, X_test)
        
        # Save processed data
        if save_processed:
            self.save_processed_data(
                X_train_scaled, X_test_scaled, 
                y_train, y_test
            )
        
        logger.info("=" * 60)
        logger.info("✓ Preprocessing complete!")
        logger.info("=" * 60)
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def save_processed_data(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: pd.Series,
        y_test: pd.Series
    ) -> None:
        """
        Save processed data to disk.
        
        Parameters
        ----------
        X_train, X_test : np.ndarray
            Scaled feature matrices
        y_train, y_test : pd.Series
            Target vectors
        """
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        
        save_path = PROCESSED_DATA_DIR / "train_test_split.pkl"
        
        data = {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
            "feature_names": self.feature_names,
            "scaler": self.scaler,
        }
        
        with open(save_path, "wb") as f:
            pickle.dump(data, f)
        
        logger.info(f"✓ Saved processed data to {save_path}")


if __name__ == "__main__":
    # Test preprocessing
    from src.data_loader import DataLoader
    
    print("\nTesting Data Preprocessor...")
    print("=" * 60)
    
    # Load sample data
    loader = DataLoader()
    df = loader.load_selected_features(nrows=50000)
    
    # Run preprocessing
    preprocessor = DataPreprocessor()
    X_train, X_test, y_train, y_test = preprocessor.full_pipeline(df)
    
    print(f"\nFinal shapes:")
    print(f"  X_train: {X_train.shape}")
    print(f"  X_test: {X_test.shape}")
    print(f"  y_train: {y_train.shape}")
    print(f"  y_test: {y_test.shape}")
