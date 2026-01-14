"""
Data loading module for ADAPTA project.
Handles reading raw data and initial validation.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import logging

from src.config import (
    RAW_DATASET_PATH,
    TARGET_COLUMN,
    SELECTED_FEATURES,
    MISSING_VALUE_CODES,
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    """
    Load and perform initial validation of BRFSS 2020 dataset.
    """
    
    def __init__(self, data_path: Optional[Path] = None):
        """
        Initialize DataLoader.
        
        Parameters
        ----------
        data_path : Path, optional
            Path to raw data file. If None, uses config default.
        """
        self.data_path = data_path or RAW_DATASET_PATH
        
    def load_raw_data(self, nrows: Optional[int] = None) -> pd.DataFrame:
        """
        Load raw BRFSS dataset from CSV.
        
        Parameters
        ----------
        nrows : int, optional
            Number of rows to load (for testing). If None, loads all data.
            
        Returns
        -------
        pd.DataFrame
            Raw dataset with all columns
            
        Raises
        ------
        FileNotFoundError
            If data file doesn't exist
        """
        if not self.data_path.exists():
            raise FileNotFoundError(
                f"Dataset not found at {self.data_path}. "
                "Please download from CDC BRFSS website."
            )
        
        logger.info(f"Loading data from {self.data_path}...")
        
        try:
            df = pd.read_csv(self.data_path, nrows=nrows)
            
            # Clean column names: strip whitespace and convert to uppercase
            df.columns = df.columns.str.strip().str.upper()
            
            logger.info(f"✓ Loaded {len(df):,} records with {len(df.columns)} columns")
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def load_selected_features(
        self, 
        include_target: bool = True,
        nrows: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Load only the selected features for modeling.
        
        Parameters
        ----------
        include_target : bool, default=True
            Whether to include target column
        nrows : int, optional
            Number of rows to load
            
        Returns
        -------
        pd.DataFrame
            DataFrame with selected columns only
        """
        df = self.load_raw_data(nrows=nrows)
        
        # Determine columns to keep
        columns_to_keep = SELECTED_FEATURES.copy()
        if include_target:
            columns_to_keep.append(TARGET_COLUMN)
        
        # Validate all columns exist
        missing_cols = [col for col in columns_to_keep if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing columns in dataset: {missing_cols}")
        
        # Select only needed columns
        df_selected = df[columns_to_keep].copy()
        
        logger.info(f"✓ Selected {len(columns_to_keep)} features/target from dataset")
        
        return df_selected
    
    def get_data_info(self) -> dict:
        """
        Get summary information about the dataset.
        
        Returns
        -------
        dict
            Dictionary containing dataset statistics
        """
        df = self.load_raw_data(nrows=1000)  # Sample for speed
        
        info = {
            "total_columns": len(df.columns),
            "selected_features": len(SELECTED_FEATURES),
            "target_column": TARGET_COLUMN,
            "dtypes": df.dtypes.value_counts().to_dict(),
        }
        
        return info
    
    def validate_data_integrity(self, df: pd.DataFrame) -> Tuple[bool, list]:
        """
        Validate loaded data for common issues.
        
        Parameters
        ----------
        df : pd.DataFrame
            Loaded dataset
            
        Returns
        -------
        Tuple[bool, list]
            (is_valid, list_of_issues)
        """
        issues = []
        
        # Check for required columns
        required_cols = SELECTED_FEATURES + [TARGET_COLUMN]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            issues.append(f"Missing columns: {missing_cols}")
        
        # Check for empty DataFrame
        if df.empty:
            issues.append("DataFrame is empty")
        
        # Check for duplicate rows
        n_duplicates = df.duplicated().sum()
        if n_duplicates > 0:
            issues.append(f"Found {n_duplicates} duplicate rows")
        
        # Check data types
        for col in SELECTED_FEATURES:
            if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
                issues.append(f"Column {col} is not numeric")
        
        is_valid = len(issues) == 0
        
        if is_valid:
            logger.info("✓ Data validation passed")
        else:
            logger.warning(f"⚠ Data validation found {len(issues)} issues")
            for issue in issues:
                logger.warning(f"  - {issue}")
        
        return is_valid, issues


def quick_load() -> pd.DataFrame:
    """
    Convenience function to quickly load selected features.
    
    Returns
    -------
    pd.DataFrame
        DataFrame ready for preprocessing
    """
    loader = DataLoader()
    return loader.load_selected_features()


if __name__ == "__main__":
    # Test data loader
    loader = DataLoader()
    
    print("=" * 60)
    print("ADAPTA Data Loader - Test Run")
    print("=" * 60)
    
    # Load sample data
    df = loader.load_selected_features(nrows=10000)
    
    print(f"\nDataset shape: {df.shape}")
    print(f"\nColumns:\n{df.columns.tolist()}")
    print(f"\nFirst 5 rows:\n{df.head()}")
    print(f"\nData types:\n{df.dtypes}")
    print(f"\nMissing values:\n{df.isnull().sum()}")
    
    # Validate
    is_valid, issues = loader.validate_data_integrity(df)
    print(f"\nValidation: {'PASSED' if is_valid else 'FAILED'}")
    
    if not is_valid:
        print("\nIssues found:")
        for issue in issues:
            print(f"  - {issue}")
