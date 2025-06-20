"""
Model Evaluation Module for IoT Intrusion Detection System

This module provides comprehensive evaluation capabilities for the neural network
model including performance metrics, visualizations, and analysis tools.

Author: Devon Cardoso Elias
Institution: Vistula University
Year: 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    log_loss, accuracy_score, precision_recall_fscore_support,
    roc_curve, auc, precision_recall_curve, average_precision_score
)
from sklearn.model_selection import StratifiedKFold
import os
import datetime
from typing import Dict, List, Tuple, Optional, Any
import warnings

warnings.filterwarnings('ignore')


class IoTIDSEvaluator:
    """
    Comprehensive evaluation system for IoT Intrusion Detection System
    
    This class provides:
    - Performance metrics calculation
    - Visualization tools
    - Cross-validation evaluation
    - Model comparison utilities
    - Detailed analysis reports
    """
    
    def __init__(self, model=None, label_encoder=None):
        """
        Initialize the evaluator
        
        Args:
            model: Trained Keras model
            label_encoder: LabelEncoder used for target encoding
        """
        self.model = model
        self.label_encoder = label_encoder
        self.evaluation_results = {}
        self.cross_validation_results = {}
        
    def evaluate_model_performance(self, X_test: np.ndarray, y_test: np.ndarray, 
                                 model=None, verbose: bool = True) -> Dict[str, Any]:
        """
        Comprehensive model performance evaluation
        
        Args:
            X_test: Test features
            y_test: Test labels
            model: Model to evaluate (uses self.model if None)
            verbose: Whether to print results
            
        Returns:
            Dict containing all evaluation metrics
        """
        if model is None:
            model = self.model
            
        if model is None:
            raise ValueError("No model provided for evaluation")
        
        print("Evaluating model performance...")
        
        # Get predictions
        y_pred_proba = model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        # Multi-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            y_test, y_pred, average=None, zero_division=0
        )
        
        # Macro and weighted averages
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            y_test, y_pred, average='macro', zero_division=0
        )
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
            y_test, y_pred, average='weighted', zero_division=0
        )
        
        # ROC-AUC (multi-class)
        try:
            roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovo', average='macro')
        except ValueError:
            roc_auc = 0.0
            
        # Log loss
        try:
            logloss = log_loss(y_test, y_pred_proba)
        except ValueError:
            logloss = float('inf')
        
        # Store results
        results = {
            'accuracy': accuracy,
            'precision_per_class': precision,
            'recall_per_class': recall,
            'f1_per_class': f1,
            'support_per_class': support,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro,
            'precision_weighted': precision_weighted,
            'recall_weighted': recall_weighted,
            'f1_weighted': f1_weighted,
            'roc_auc_score': roc_auc,
            'log_loss': logloss,
            'predictions': y_pred,
            'prediction_probabilities': y_pred_proba,
            'true_labels': y_test
        }
        
        self.evaluation_results = results
        
        if verbose:
            self._print_evaluation_summary(results)
            
        return results
    
    def cross_validate_model(self, X: np.ndarray, y: np.ndarray, model_builder,
                           n_splits: int = 5, random_state: int = 42, 
                           verbose: bool = True) -> Dict[str, Any]:
        """
        Perform stratified k-fold cross-validation
        
        Args:
            X: Features
            y: Labels
            model_builder: Function that builds and returns a compiled model
            n_splits: Number of CV folds
            random_state: Random state for reproducibility
            verbose: Whether to print progress
            
        Returns:
            Dict containing cross-validation results
        """
        print(f"Performing {n_splits}-fold cross-validation...")
        
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        
        # Storage for results
        cv_scores = {
            'accuracy': [],
            'precision_macro': [],
            'recall_macro': [],
            'f1_macro': [],
            'roc_auc': [],
            'log_loss': [],
            'fold_predictions': [],
            'fold_true_labels': [],
            'fold_probabilities': []
        }
        
        fold = 1
        for train_idx, val_idx in skf.split(X, y):
            if verbose:
                print(f"Evaluating fold {fold}/{n_splits}...")
                
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Build and train model for this fold
            model = model_builder()
            
            # Train the model (you may want to add callbacks here)
            model.fit(X_train, y_train, epochs=50, batch_size=32, 
                     validation_data=(X_val, y_val), verbose=0)
            
            # Evaluate on validation set
            fold_results = self.evaluate_model_performance(
                X_val, y_val, model=model, verbose=False
            )
            
            # Store results
            cv_scores['accuracy'].append(fold_results['accuracy'])
            cv_scores['precision_macro'].append(fold_results['precision_macro'])
            cv_scores['recall_macro'].append(fold_results['recall_macro'])
            cv_scores['f1_macro'].append(fold_results['f1_macro'])
            cv_scores['roc_auc'].append(fold_results['roc_auc_score'])
            cv_scores['log_loss'].append(fold_results['log_loss'])
            cv_scores['fold_predictions'].append(fold_results['predictions'])
            cv_scores['fold_true_labels'].append(fold_results['true_labels'])
            cv_scores['fold_probabilities'].append(fold_results['prediction_probabilities'])
            
            fold += 1
        
        # Calculate summary statistics
        cv_summary = {
            'mean_accuracy': np.mean(cv_scores['accuracy']),
            'std_accuracy': np.std(cv_scores['accuracy']),
            'mean_precision_macro': np.mean(cv_scores['precision_macro']),
            'std_precision_macro': np.std(cv_scores['precision_macro']),
            'mean_recall_macro': np.mean(cv_scores['recall_macro']),
            'std_recall_macro': np.std(cv_scores['recall_macro']),
            'mean_f1_macro': np.mean(cv_scores['f1_macro']),
            'std_f1_macro': np.std(cv_scores['f1_macro']),
            'mean_roc_auc': np.mean(cv_scores['roc_auc']),
            'std_roc_auc': np.std(cv_scores['roc_auc']),
            'mean_log_loss': np.mean(cv_scores['log_loss']),
            'std_log_loss': np.std(cv_scores['log_loss']),
            'detailed_scores': cv_scores
        }
        
        self.cross_validation_results = cv_summary
        
        if verbose:
            self._print_cv_summary(cv_summary)
            
        return cv_summary
    
    def generate_classification_report(self, y_true: np.ndarray, y_pred: np.ndarray,
                                     class_names: Optional[List[str]] = None,
                                     save_path: Optional[str] = None) -> pd.DataFrame:
        """
        Generate detailed classification report
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: List of class names
            save_path: Path to save the report
            
        Returns:
            DataFrame containing the classification report
        """
        if class_names is None and self.label_encoder is not None:
            class_names = self.label_encoder.classes_
            
        # Generate classification report
        report = classification_report(
            y_true, y_pred, 
            target_names=class_names,
            output_dict=True,
            zero_division=0
        )
        
        # Convert to DataFrame
        report_df = pd.DataFrame(report).transpose()
        
        if save_path:
            report_df.to_csv(save_path)
            print(f"Classification report saved to: {save_path}")
            
        return report_df
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray,
                            class_names: Optional[List[str]] = None,
                            normalize: bool = False, 
                            save_path: Optional[str] = None,
                            figsize: Tuple[int, int] = (10, 8)) -> None:
        """
        Plot confusion matrix with customizable options
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: List of class names
            normalize: Whether to normalize the confusion matrix
            save_path: Path to save the plot
            figsize: Figure size
        """
        if class_names is None and self.label_encoder is not None:
            class_names = self.label_encoder.classes_
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
        else:
            fmt = 'd'
        
        # Create plot
        plt.figure(figsize=figsize)
        sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names,
                   cbar_kws={'label': 'Count' if not normalize else 'Proportion'})
        
        plt.title('Confusion Matrix' + (' (Normalized)' if normalize else ''))
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to: {save_path}")
        
        plt.show()
    
    def plot_roc_curves(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                       class_names: Optional[List[str]] = None,
                       save_path: Optional[str] = None,
                       figsize: Tuple[int, int] = (12, 8)) -> None:
        """
        Plot ROC curves for multi-class classification
        
        Args:
            y_true: True labels (integer encoded)
            y_pred_proba: Prediction probabilities
            class_names: List of class names
            save_path: Path to save the plot
            figsize: Figure size
        """
        if class_names is None and self.label_encoder is not None:
            class_names = self.label_encoder.classes_
        
        n_classes = y_pred_proba.shape[1]
        
        # Binarize the output
        from sklearn.preprocessing import label_binarize
        y_true_bin = label_binarize(y_true, classes=range(n_classes))
        
        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        # Plot ROC curves
        plt.figure(figsize=figsize)
        colors = plt.cm.Set3(np.linspace(0, 1, n_classes))
        
        for i, color in zip(range(n_classes), colors):
            class_name = class_names[i] if class_names else f'Class {i}'
            plt.plot(fpr[i], tpr[i], color=color, lw=2,
                    label=f'{class_name} (AUC = {roc_auc[i]:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Multi-class ROC Curves')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"ROC curves saved to: {save_path}")
        
        plt.show()
    
    def plot_precision_recall_curves(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                                   class_names: Optional[List[str]] = None,
                                   save_path: Optional[str] = None,
                                   figsize: Tuple[int, int] = (12, 8)) -> None:
        """
        Plot Precision-Recall curves for multi-class classification
        
        Args:
            y_true: True labels (integer encoded)
            y_pred_proba: Prediction probabilities
            class_names: List of class names
            save_path: Path to save the plot
            figsize: Figure size
        """
        if class_names is None and self.label_encoder is not None:
            class_names = self.label_encoder.classes_
        
        n_classes = y_pred_proba.shape[1]
        
        # Binarize the output
        from sklearn.preprocessing import label_binarize
        y_true_bin = label_binarize(y_true, classes=range(n_classes))
        
        # Compute Precision-Recall curve and average precision for each class
        precision = dict()
        recall = dict()
        avg_precision = dict()
        
        for i in range(n_classes):
            precision[i], recall[i], _ = precision_recall_curve(
                y_true_bin[:, i], y_pred_proba[:, i]
            )
            avg_precision[i] = average_precision_score(y_true_bin[:, i], y_pred_proba[:, i])
        
        # Plot Precision-Recall curves
        plt.figure(figsize=figsize)
        colors = plt.cm.Set3(np.linspace(0, 1, n_classes))
        
        for i, color in zip(range(n_classes), colors):
            class_name = class_names[i] if class_names else f'Class {i}'
            plt.plot(recall[i], precision[i], color=color, lw=2,
                    label=f'{class_name} (AP = {avg_precision[i]:.3f})')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Multi-class Precision-Recall Curves')
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Precision-Recall curves saved to: {save_path}")
        
        plt.show()
    
    def plot_training_history(self, history, save_path: Optional[str] = None,
                            figsize: Tuple[int, int] = (15, 5)) -> None:
        """
        Plot training history (loss and accuracy curves)
        
        Args:
            history: Keras training history object
            save_path: Path to save the plot
            figsize: Figure size
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Plot training & validation accuracy
        axes[0].plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
        axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
        axes[0].set_title('Model Accuracy')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot training & validation loss
        axes[1].plot(history.history['loss'], label='Training Loss', linewidth=2)
        axes[1].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
        axes[1].set_title('Model Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training history plot saved to: {save_path}")
        
        plt.show()
    
    def generate_evaluation_report(self, X_test: np.ndarray, y_test: np.ndarray,
                                 model=None, save_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate comprehensive evaluation report with all metrics and visualizations
        
        Args:
            X_test: Test features
            y_test: Test labels
            model: Model to evaluate
            save_dir: Directory to save reports and plots
            
        Returns:
            Complete evaluation results
        """
        if model is None:
            model = self.model
            
        print("Generating comprehensive evaluation report...")
        
        # Create save directory if specified
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            save_dir = os.path.join(save_dir, f"evaluation_{timestamp}")
            os.makedirs(save_dir, exist_ok=True)
        
        # Evaluate model performance
        results = self.evaluate_model_performance(X_test, y_test, model)
        
        # Generate classification report
        report_df = self.generate_classification_report(
            results['true_labels'], 
            results['predictions'],
            save_path=os.path.join(save_dir, "classification_report.csv") if save_dir else None
        )
        
        # Generate visualizations
        if save_dir:
            # Confusion matrix
            self.plot_confusion_matrix(
                results['true_labels'], 
                results['predictions'],
                save_path=os.path.join(save_dir, "confusion_matrix.png")
            )
            
            # Normalized confusion matrix
            self.plot_confusion_matrix(
                results['true_labels'], 
                results['predictions'],
                normalize=True,
                save_path=os.path.join(save_dir, "confusion_matrix_normalized.png")
            )
            
            # ROC curves
            self.plot_roc_curves(
                results['true_labels'],
                results['prediction_probabilities'],
                save_path=os.path.join(save_dir, "roc_curves.png")
            )
            
            # Precision-Recall curves
            self.plot_precision_recall_curves(
                results['true_labels'],
                results['prediction_probabilities'],
                save_path=os.path.join(save_dir, "precision_recall_curves.png")
            )
        else:
            # Show plots without saving
            self.plot_confusion_matrix(results['true_labels'], results['predictions'])
            self.plot_roc_curves(results['true_labels'], results['prediction_probabilities'])
            self.plot_precision_recall_curves(results['true_labels'], results['prediction_probabilities'])
        
        # Combine all results
        complete_results = {
            'performance_metrics': results,
            'classification_report': report_df,
            'save_directory': save_dir
        }
        
        # Save summary report
        if save_dir:
            self._save_summary_report(complete_results, save_dir)
        
        print("Evaluation report generation completed!")
        return complete_results
    
    def compare_models(self, models_dict: Dict[str, Any], X_test: np.ndarray, 
                      y_test: np.ndarray) -> pd.DataFrame:
        """
        Compare multiple models on the same test set
        
        Args:
            models_dict: Dictionary of {model_name: model} pairs
            X_test: Test features
            y_test: Test labels
            
        Returns:
            DataFrame with comparison results
        """
        print("Comparing multiple models...")
        
        comparison_results = []
        
        for model_name, model in models_dict.items():
            print(f"Evaluating {model_name}...")
            results = self.evaluate_model_performance(X_test, y_test, model, verbose=False)
            
            comparison_results.append({
                'Model': model_name,
                'Accuracy': results['accuracy'],
                'Precision (Macro)': results['precision_macro'],
                'Recall (Macro)': results['recall_macro'],
                'F1-Score (Macro)': results['f1_macro'],
                'ROC-AUC': results['roc_auc_score'],
                'Log Loss': results['log_loss']
            })
        
        comparison_df = pd.DataFrame(comparison_results)
        comparison_df = comparison_df.round(4)
        
        print("\nModel Comparison Results:")
        print("=" * 80)
        print(comparison_df.to_string(index=False))
        
        return comparison_df
    
    def _print_evaluation_summary(self, results: Dict[str, Any]) -> None:
        """Print formatted evaluation summary"""
        print("\n" + "=" * 60)
        print("MODEL EVALUATION SUMMARY")
        print("=" * 60)
        print(f"Overall Accuracy: {results['accuracy']:.4f}")
        print(f"ROC-AUC Score: {results['roc_auc_score']:.4f}")
        print(f"Log Loss: {results['log_loss']:.4f}")
        print("\nMacro Averages:")
        print(f"  Precision: {results['precision_macro']:.4f}")
        print(f"  Recall: {results['recall_macro']:.4f}")
        print(f"  F1-Score: {results['f1_macro']:.4f}")
        print("\nWeighted Averages:")
        print(f"  Precision: {results['precision_weighted']:.4f}")
        print(f"  Recall: {results['recall_weighted']:.4f}")
        print(f"  F1-Score: {results['f1_weighted']:.4f}")
        print("=" * 60)
    
    def _print_cv_summary(self, cv_results: Dict[str, Any]) -> None:
        """Print formatted cross-validation summary"""
        print("\n" + "=" * 60)
        print("CROSS-VALIDATION SUMMARY")
        print("=" * 60)
        print(f"Accuracy: {cv_results['mean_accuracy']:.4f} ± {cv_results['std_accuracy']:.4f}")
        print(f"ROC-AUC: {cv_results['mean_roc_auc']:.4f} ± {cv_results['std_roc_auc']:.4f}")
        print(f"F1-Score (Macro): {cv_results['mean_f1_macro']:.4f} ± {cv_results['std_f1_macro']:.4f}")
        print(f"Precision (Macro): {cv_results['mean_precision_macro']:.4f} ± {cv_results['std_precision_macro']:.4f}")
        print(f"Recall (Macro): {cv_results['mean_recall_macro']:.4f} ± {cv_results['std_recall_macro']:.4f}")
        print(f"Log Loss: {cv_results['mean_log_loss']:.4f} ± {cv_results['std_log_loss']:.4f}")
        print("=" * 60)
    
    def _save_summary_report(self, results: Dict[str, Any], save_dir: str) -> None:
        """Save a text summary report"""
        report_path = os.path.join(save_dir, "evaluation_summary.txt")
        
        with open(report_path, 'w') as f:
            f.write("IoT Intrusion Detection System - Evaluation Report\n")
            f.write("=" * 60 + "\n")
            f.write(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            metrics = results['performance_metrics']
            f.write("Performance Metrics:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Overall Accuracy: {metrics['accuracy']:.4f}\n")
            f.write(f"ROC-AUC Score: {metrics['roc_auc_score']:.4f}\n")
            f.write(f"Log Loss: {metrics['log_loss']:.4f}\n")
            f.write(f"Precision (Macro): {metrics['precision_macro']:.4f}\n")
            f.write(f"Recall (Macro): {metrics['recall_macro']:.4f}\n")
            f.write(f"F1-Score (Macro): {metrics['f1_macro']:.4f}\n\n")
            
            f.write("Per-Class Performance:\n")
            f.write("-" * 30 + "\n")
            class_names = self.label_encoder.classes_ if self.label_encoder else None
            
            for i in range(len(metrics['precision_per_class'])):
                class_name = class_names[i] if class_names else f"Class {i}"
                f.write(f"{class_name}:\n")
                f.write(f"  Precision: {metrics['precision_per_class'][i]:.4f}\n")
                f.write(f"  Recall: {metrics['recall_per_class'][i]:.4f}\n")
                f.write(f"  F1-Score: {metrics['f1_per_class'][i]:.4f}\n")
                f.write(f"  Support: {metrics['support_per_class'][i]}\n\n")
        
        print(f"Summary report saved to: {report_path}")


def evaluate_iot_model(model, X_test: np.ndarray, y_test: np.ndarray,
                      label_encoder=None, save_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Convenience function for complete model evaluation
    
    Args:
        model: Trained Keras model
        X_test: Test features
        y_test: Test labels
        label_encoder: Label encoder for class names
        save_dir: Directory to save evaluation results
        
    Returns:
        Complete evaluation results
    """
    evaluator = IoTIDSEvaluator(model=model, label_encoder=label_encoder)
    return evaluator.generate_evaluation_report(X_test, y_test, save_dir=save_dir)


if __name__ == "__main__":
    print("IoT Intrusion Detection System - Evaluation Module")
    print("=" * 60)
    print("This module provides comprehensive evaluation capabilities for")
    print("the IoT IDS neural network model including:")
    print("- Performance metrics calculation")
    print("- Confusion matrices and classification reports")
    print("- ROC and Precision-Recall curves")
    print("- Cross-validation evaluation")
    print("- Model comparison utilities")
    print("- Comprehensive reporting and visualization")
    print("=" * 60) 