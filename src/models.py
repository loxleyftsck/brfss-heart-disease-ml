"""
Machine learning models module for ADAPTA project.
Defines model classes and training procedures.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from typing import Dict, Any, Optional, Tuple
import logging
import pickle
from pathlib import Path

from src.config import (
    RF_PARAMS_DEFAULT,
    LR_PARAMS_DEFAULT,
    RF_PARAM_GRID,
    LR_PARAM_GRID,
    CV_FOLDS,
    SCORING_METRIC,
    RANDOM_SEED,
    EXPERIMENTS_DIR,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HeartDiseaseModel:
    """
    Base class for heart disease prediction models.
    """
    
    def __init__(self, model_name: str, random_seed: int = RANDOM_SEED):
        """
        Initialize model.
        
        Parameters
        ----------
        model_name : str
            Name identifier for the model
        random_seed : int
            Random seed for reproducibility
        """
        self.model_name = model_name
        self.random_seed = random_seed
        self.model = None
        self.is_trained = False
        self.best_params = None
        
    def train(self, X_train: np.ndarray, y_train: pd.Series) -> None:
        """
        Train the model.
        
        Parameters
        ----------
        X_train : np.ndarray
            Training features
        y_train : pd.Series
            Training labels
        """
        if self.model is None:
            raise ValueError("Model not initialized")
        
        logger.info(f"Training {self.model_name}...")
        self.model.fit(X_train, y_train)
        self.is_trained = True
        logger.info(f"✓ {self.model_name} trained successfully")
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix
            
        Returns
        -------
        np.ndarray
            Predicted labels
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix
            
        Returns
        -------
        np.ndarray
            Predicted probabilities
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        return self.model.predict_proba(X)
    
    def save_model(self, filepath: Optional[Path] = None) -> None:
        """
        Save trained model to disk.
        
        Parameters
        ----------
        filepath : Path, optional
            Save location. If None, uses default experiments directory.
        """
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        
        if filepath is None:
            EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
            filepath = EXPERIMENTS_DIR / f"{self.model_name}.pkl"
        
        with open(filepath, "wb") as f:
            pickle.dump(self.model, f)
        
        logger.info(f"✓ Model saved to {filepath}")
    
    def load_model(self, filepath: Path) -> None:
        """
        Load trained model from disk.
        
        Parameters
        ----------
        filepath : Path
            Path to saved model
        """
        with open(filepath, "rb") as f:
            self.model = pickle.load(f)
        
        self.is_trained = True
        logger.info(f"✓ Model loaded from {filepath}")


class RandomForestModel(HeartDiseaseModel):
    """
    Random Forest classifier for heart disease prediction.
    """
    
    def __init__(
        self, 
        params: Optional[Dict[str, Any]] = None,
        random_seed: int = RANDOM_SEED
    ):
        """
        Initialize Random Forest model.
        
        Parameters
        ----------
        params : dict, optional
            Model hyperparameters. Uses defaults if None.
        random_seed : int
            Random seed
        """
        super().__init__("RandomForest", random_seed)
        
        model_params = RF_PARAMS_DEFAULT.copy()
        if params:
            model_params.update(params)
        
        self.model = RandomForestClassifier(**model_params)
        
    def get_feature_importance(self, feature_names: list) -> pd.DataFrame:
        """
        Get feature importance scores.
        
        Parameters
        ----------
        feature_names : list
            List of feature names
            
        Returns
        -------
        pd.DataFrame
            Feature importance ranked
        """
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        importance_df = pd.DataFrame({
            "feature": feature_names,
            "importance": self.model.feature_importances_
        }).sort_values("importance", ascending=False)
        
        return importance_df


class LogisticRegressionModel(HeartDiseaseModel):
    """
    Logistic Regression classifier for heart disease prediction.
    """
    
    def __init__(
        self, 
        params: Optional[Dict[str, Any]] = None,
        random_seed: int = RANDOM_SEED
    ):
        """
        Initialize Logistic Regression model.
        
        Parameters
        ----------
        params : dict, optional
            Model hyperparameters. Uses defaults if None.
        random_seed : int
            Random seed
        """
        super().__init__("LogisticRegression", random_seed)
        
        model_params = LR_PARAMS_DEFAULT.copy()
        if params:
            model_params.update(params)
        
        self.model = LogisticRegression(**model_params)
    
    def get_coefficients(self, feature_names: list) -> pd.DataFrame:
        """
        Get model coefficients.
        
        Parameters
        ----------
        feature_names : list
            List of feature names
            
        Returns
        -------
        pd.DataFrame
            Feature coefficients
        """
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        coef_df = pd.DataFrame({
            "feature": feature_names,
            "coefficient": self.model.coef_[0]
        }).sort_values("coefficient", key=abs, ascending=False)
        
        return coef_df


class ModelTuner:
    """
    Hyperparameter tuning using GridSearchCV.
    """
    
    def __init__(
        self,
        model_type: str,
        cv_folds: int = CV_FOLDS,
        scoring: str = SCORING_METRIC,
        random_seed: int = RANDOM_SEED
    ):
        """
        Initialize model tuner.
        
        Parameters
        ----------
        model_type : str
            Either "rf" or "lr"
        cv_folds : int
            Number of cross-validation folds
        scoring : str
            Scoring metric for optimization
        random_seed : int
            Random seed
        """
        self.model_type = model_type.lower()
        self.cv_folds = cv_folds
        self.scoring = scoring
        self.random_seed = random_seed
        self.best_model = None
        self.best_params = None
        self.cv_results = None
        
    def tune(
        self, 
        X_train: np.ndarray, 
        y_train: pd.Series,
        param_grid: Optional[Dict] = None
    ) -> Tuple[Any, Dict]:
        """
        Perform hyperparameter tuning.
        
        Parameters
        ----------
        X_train : np.ndarray
            Training features
        y_train : pd.Series
            Training labels
        param_grid : dict, optional
            Custom parameter grid. Uses defaults if None.
            
        Returns
        -------
        Tuple
            (best_model, best_params)
        """
        logger.info(f"Starting hyperparameter tuning for {self.model_type.upper()}...")
        
        # Initialize base model
        if self.model_type == "rf":
            base_model = RandomForestClassifier(random_state=self.random_seed)
            param_grid = param_grid or RF_PARAM_GRID
        elif self.model_type == "lr":
            base_model = LogisticRegression(
                random_state=self.random_seed,
                max_iter=1000
            )
            param_grid = param_grid or LR_PARAM_GRID
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        # GridSearch
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            scoring=self.scoring,
            cv=self.cv_folds,
            verbose=2,
            n_jobs=-1,
            return_train_score=True
        )
        
        grid_search.fit(X_train, y_train)
        
        self.best_model = grid_search.best_estimator_
        self.best_params = grid_search.best_params_
        self.cv_results = pd.DataFrame(grid_search.cv_results_)
        
        logger.info(f"✓ Tuning complete!")
        logger.info(f"Best {self.scoring} score: {grid_search.best_score_:.4f}")
        logger.info(f"Best parameters: {self.best_params}")
        
        return self.best_model, self.best_params


if __name__ == "__main__":
    # Test models
    print("\nTesting Models...")
    print("=" * 60)
    
    # Create dummy data
    np.random.seed(42)
    X_train = np.random.randn(1000, 5)
    y_train = pd.Series(np.random.randint(0, 2, 1000))
    X_test = np.random.randn(300, 5)
    
    # Test Random Forest
    print("\n1. Testing Random Forest...")
    rf_model = RandomForestModel()
    rf_model.train(X_train, y_train)
    predictions = rf_model.predict(X_test)
    print(f"✓ Predictions shape: {predictions.shape}")
    
    # Test Logistic Regression
    print("\n2. Testing Logistic Regression...")
    lr_model = LogisticRegressionModel()
    lr_model.train(X_train, y_train)
    predictions = lr_model.predict(X_test)
    print(f"✓ Predictions shape: {predictions.shape}")
    
    print("\n✓ All tests passed!")
