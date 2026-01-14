"""
Model evaluation module for ADAPTA project.
Calculate metrics, generate reports, and create visualizations.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
)
from typing import Dict, Optional, Tuple
import logging
from pathlib import Path

from src.config import (
    CLASS_LABELS,
    FIGURES_DIR,
    EXPERIMENTS_DIR,
    FIGURE_DPI,
    FIGURE_FORMAT,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Comprehensive model evaluation and reporting.
    """
    
    def __init__(self, model_name: str):
        """
        Initialize evaluator.
        
        Parameters
        ----------
        model_name : str
            Name of the model being evaluated
        """
        self.model_name = model_name
        self.metrics = {}
        
    def calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Calculate all evaluation metrics.
        
        Parameters
        ----------
        y_true : np.ndarray
            True labels
        y_pred : np.ndarray
            Predicted labels
        y_pred_proba : np.ndarray, optional
            Predicted probabilities (for ROC-AUC)
            
        Returns
        -------
        Dict[str, float]
            Dictionary of metrics
        """
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1_score": f1_score(y_true, y_pred, zero_division=0),
        }
        
        # Add ROC-AUC if probabilities provided
        if y_pred_proba is not None:
            metrics["roc_auc"] = roc_auc_score(y_true, y_pred_proba[:, 1])
        
        self.metrics = metrics
        return metrics
    
    def print_metrics(self, metrics: Optional[Dict] = None) -> None:
        """
        Print metrics in formatted table.
        
        Parameters
        ----------
        metrics : dict, optional
            Metrics dictionary. Uses self.metrics if None.
        """
        metrics = metrics or self.metrics
        
        print(f"\n{'=' * 60}")
        print(f"{self.model_name} - Evaluation Metrics")
        print(f"{'=' * 60}")
        
        for metric_name, value in metrics.items():
            print(f"{metric_name.replace('_', ' ').title():.<30} {value:.4f}")
        
        print(f"{'=' * 60}\n")
    
    def get_classification_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> str:
        """
        Generate sklearn classification report.
        
        Parameters
        ----------
        y_true : np.ndarray
            True labels
        y_pred : np.ndarray
            Predicted labels
            
        Returns
        -------
        str
            Classification report
        """
        report = classification_report(
            y_true, y_pred,
            target_names=CLASS_LABELS.values(),
            digits=4
        )
        
        return report
    
    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        save_path: Optional[Path] = None,
        show_plot: bool = True
    ) -> None:
        """
        Plot confusion matrix heatmap.
        
        Parameters
        ----------
        y_true : np.ndarray
            True labels
        y_pred : np.ndarray
            Predicted labels
        save_path : Path, optional
            Path to save figure
        show_plot : bool
            Whether to display plot
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='YlGnBu',
            xticklabels=CLASS_LABELS.values(),
            yticklabels=CLASS_LABELS.values(),
            cbar_kws={'label': 'Count'}
        )
        
        plt.title(f'Confusion Matrix - {self.model_name}', fontsize=14, fontweight='bold')
        plt.ylabel('Actual Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        
        if save_path:
            FIGURES_DIR.mkdir(parents=True, exist_ok=True)
            save_file = save_path or FIGURES_DIR / f"{self.model_name}_confusion_matrix.{FIGURE_FORMAT}"
            plt.savefig(save_file, dpi=FIGURE_DPI, bbox_inches='tight')
            logger.info(f"✓ Saved confusion matrix to {save_file}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    def plot_roc_curve(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        save_path: Optional[Path] = None,
        show_plot: bool = True
    ) -> None:
        """
        Plot ROC curve.
        
        Parameters
        ----------
        y_true : np.ndarray
            True labels
        y_pred_proba : np.ndarray
            Predicted probabilities
        save_path : Path, optional
            Path to save figure
        show_plot : bool
            Whether to display plot
        """
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba[:, 1])
        roc_auc = roc_auc_score(y_true, y_pred_proba[:, 1])
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                 label=f'{self.model_name} (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
                 label='Random Classifier')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title(f'ROC Curve - {self.model_name}', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            FIGURES_DIR.mkdir(parents=True, exist_ok=True)
            save_file = save_path or FIGURES_DIR / f"{self.model_name}_roc_curve.{FIGURE_FORMAT}"
            plt.savefig(save_file, dpi=FIGURE_DPI, bbox_inches='tight')
            logger.info(f"✓ Saved ROC curve to {save_file}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    def full_evaluation(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: Optional[np.ndarray] = None,
        save_figures: bool = True
    ) -> Dict[str, float]:
        """
        Run complete evaluation pipeline.
        
        Parameters
        ----------
        y_true : np.ndarray
            True labels
        y_pred : np.ndarray
            Predicted labels
        y_pred_proba : np.ndarray, optional
            Predicted probabilities
        save_figures : bool
            Whether to save plots
            
        Returns
        -------
        Dict[str, float]
            Evaluation metrics
        """
        logger.info(f"Evaluating {self.model_name}...")
        
        # Calculate metrics
        metrics = self.calculate_metrics(y_true, y_pred, y_pred_proba)
        self.print_metrics(metrics)
        
        # Classification report
        print(f"\n{self.model_name} - Classification Report:")
        print(self.get_classification_report(y_true, y_pred))
        
        # Plot confusion matrix
        self.plot_confusion_matrix(
            y_true, y_pred,
            save_path=FIGURES_DIR / f"{self.model_name}_confusion_matrix.{FIGURE_FORMAT}" if save_figures else None,
            show_plot=False
        )
        
        # Plot ROC curve if probabilities available
        if y_pred_proba is not None:
            self.plot_roc_curve(
                y_true, y_pred_proba,
                save_path=FIGURES_DIR / f"{self.model_name}_roc_curve.{FIGURE_FORMAT}" if save_figures else None,
                show_plot=False
            )
        
        logger.info(f"✓ Evaluation complete for {self.model_name}")
        
        return metrics


def compare_models(
    evaluations: Dict[str, Dict[str, float]],
    save_path: Optional[Path] = None
) -> None:
    """
    Create comparison chart for multiple models.
    
    Parameters
    ----------
    evaluations : dict
        Dictionary of {model_name: metrics_dict}
    save_path : Path, optional
        Path to save figure
    """
    metrics_df = pd.DataFrame(evaluations).T
    
    # Plot comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Bar plot
    metrics_df[['precision', 'recall', 'f1_score']].plot(
        kind='bar',
        ax=axes[0],
        color=['#577590', '#43aa8b', '#f9c74f']
    )
    axes[0].set_title('Model Comparison - Key Metrics', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Score', fontsize=12)
    axes[0].set_xlabel('Model', fontsize=12)
    axes[0].legend(['Precision', 'Recall', 'F1-Score'])
    axes[0].set_ylim(0, 1)
    axes[0].grid(axis='y', alpha=0.3)
    
    # Heatmap
    sns.heatmap(
        metrics_df,
        annot=True,
        fmt='.3f',
        cmap='YlGnBu',
        ax=axes[1],
        cbar_kws={'label': 'Score'}
    )
    axes[1].set_title('Model Metrics Heatmap', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight')
        logger.info(f"✓ Saved comparison to {save_path}")
    
    plt.show()


if __name__ == "__main__":
    # Test evaluator
    print("\nTesting Model Evaluator...")
    
    # Create dummy predictions
    np.random.seed(42)
    y_true = np.random.randint(0, 2, 1000)
    y_pred = np.random.randint(0, 2, 1000)
    y_pred_proba = np.random.rand(1000, 2)
    y_pred_proba /= y_pred_proba.sum(axis=1, keepdims=True)
    
    # Evaluate
    evaluator = ModelEvaluator("TestModel")
    metrics = evaluator.full_evaluation(y_true, y_pred, y_pred_proba, save_figures=False)
    
    print("\n✓ Evaluation test passed!")
