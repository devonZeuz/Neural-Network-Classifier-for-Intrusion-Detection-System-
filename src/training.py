"""
Training Module for IoT Intrusion Detection System

This module handles the complete training pipeline including cross-validation,
model training, and performance evaluation.

Author: Devon Cardoso Elias
Institution: Vistula University
Year: 2025
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, log_loss
from sklearn.utils.class_weight import compute_class_weight
from typing import Dict, List, Tuple, Optional
import os
import datetime
import pickle
from pathlib import Path

from .model_architecture import IoTIDSModel
from .data_preprocessing import IoTDataPreprocessor
from .evaluation import ModelEvaluator


class IoTIDSTrainer:
    """
    Complete training pipeline for IoT Intrusion Detection System
    
    Handles:
    - Cross-validation training
    - Model persistence
    - Training metrics collection
    - Performance evaluation
    """
    
    def __init__(self, config: Optional[Dict] = None, random_state: int = 42):
        """
        Initialize the trainer
        
        Args:
            config (Dict, optional): Training configuration
            random_state (int): Random state for reproducibility
        """
        self.random_state = random_state
        self.config = config or self._get_default_config()
        self.training_history = []
        self.cv_results = {}
        self.best_model = None
        self.evaluator = ModelEvaluator()
        
    def _get_default_config(self) -> Dict:
        """Get default training configuration"""
        return {
            'cross_validation': {
                'n_splits': 5,
                'shuffle': True,
                'stratify': True
            },
            'model': {
                'learning_rate': 0.001,
                'epochs': 50,
                'batch_size': 256,
                'early_stopping_patience': 10,
                'early_stopping_monitor': 'val_loss'
            },
            'preprocessing': {
                'apply_smote': True,
                'scale_features': True,
                'scaler_type': 'robust'
            },
            'output': {
                'save_model': True,
                'save_results': True,
                'results_dir': 'results',
                'models_dir': 'models/saved_models'
            }
        }
    
    def cross_validate_model(self, X: np.ndarray, y: np.ndarray, 
                           feature_names: List[str],
                           label_encoder) -> Dict:
        """
        Perform stratified k-fold cross-validation
        
        Args:
            X (np.ndarray): Features
            y (np.ndarray): Labels
            feature_names (List[str]): Feature names
            label_encoder: Label encoder for class names
            
        Returns:
            Dict: Cross-validation results
        """
        print("=" * 60)
        print("Starting Cross-Validation Training")
        print("=" * 60)
        
        # Setup cross-validation
        cv_config = self.config['cross_validation']
        skf = StratifiedKFold(
            n_splits=cv_config['n_splits'],
            shuffle=cv_config['shuffle'],
            random_state=self.random_state
        )
        
        # Initialize result tracking
        cv_scores = []
        roc_auc_scores = []
        log_losses = []
        fold_histories = []
        confusion_matrices = []
        classification_reports = []
        
        # Cross-validation loop
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            print(f"\n--- Fold {fold + 1}/{cv_config['n_splits']} ---")
            
            # Split data
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Create and train model
            model = self._create_model(X_train.shape[1], len(np.unique(y)))
            history = self._train_fold(model, X_train, y_train, X_val, y_val)
            
            # Evaluate fold
            fold_results = self._evaluate_fold(
                model, X_val, y_val, label_encoder, fold
            )
            
            # Store results
            cv_scores.append(fold_results['accuracy'])
            roc_auc_scores.append(fold_results['roc_auc'])
            log_losses.append(fold_results['log_loss'])
            fold_histories.append(history)
            confusion_matrices.append(fold_results['confusion_matrix'])
            classification_reports.append(fold_results['classification_report'])
            
            # Save best model
            if not self.best_model or fold_results['accuracy'] > max(cv_scores[:-1] + [0]):
                self.best_model = model
                print(f"New best model found in fold {fold + 1}")
        
        # Compile results
        self.cv_results = {
            'cv_scores': cv_scores,
            'roc_auc_scores': roc_auc_scores,
            'log_losses': log_losses,
            'mean_accuracy': np.mean(cv_scores),
            'std_accuracy': np.std(cv_scores),
            'mean_roc_auc': np.mean(roc_auc_scores),
            'std_roc_auc': np.std(roc_auc_scores),
            'mean_log_loss': np.mean(log_losses),
            'std_log_loss': np.std(log_losses),
            'fold_histories': fold_histories,
            'confusion_matrices': confusion_matrices,
            'classification_reports': classification_reports,
            'feature_names': feature_names,
            'label_encoder': label_encoder
        }
        
        self._print_cv_summary()
        return self.cv_results
    
    def _create_model(self, input_dim: int, num_classes: int) -> IoTIDSModel:
        """Create a new model instance"""
        model_config = self.config['model']
        
        ids_model = IoTIDSModel(
            input_shape=(input_dim,),
            num_classes=num_classes,
            learning_rate=model_config['learning_rate']
        )
        
        return ids_model.build_model()
    
    def _train_fold(self, model, X_train, y_train, X_val, y_val):
        """Train model for one fold"""
        model_config = self.config['model']
        
        # Compute class weights
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(y_train),
            y=y_train
        )
        class_weights_dict = dict(enumerate(class_weights))
        
        # Setup callbacks
        from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
        
        log_dir = os.path.join(
            "logs/fit", 
            datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        )
        
        callbacks = [
            EarlyStopping(
                monitor=model_config['early_stopping_monitor'],
                patience=model_config['early_stopping_patience'],
                restore_best_weights=True,
                verbose=1
            ),
            TensorBoard(log_dir=log_dir, histogram_freq=1)
        ]
        
        # Train model
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=model_config['epochs'],
            batch_size=model_config['batch_size'],
            class_weight=class_weights_dict,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def _evaluate_fold(self, model, X_val, y_val, label_encoder, fold):
        """Evaluate model performance for one fold"""
        # Predictions
        val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
        val_predictions = model.predict(X_val, verbose=0)
        y_pred_classes = np.argmax(val_predictions, axis=1)
        
        # Metrics
        cm = confusion_matrix(y_val, y_pred_classes)
        cr = classification_report(
            y_val, y_pred_classes,
            target_names=label_encoder.classes_,
            output_dict=True
        )
        roc_auc = roc_auc_score(y_val, val_predictions, multi_class='ovo')
        log_loss_val = log_loss(y_val, val_predictions)
        
        print(f"Fold {fold + 1} Results:")
        print(f"  Accuracy: {val_accuracy:.4f}")
        print(f"  ROC-AUC: {roc_auc:.4f}")
        print(f"  Log Loss: {log_loss_val:.4f}")
        
        return {
            'accuracy': val_accuracy,
            'roc_auc': roc_auc,
            'log_loss': log_loss_val,
            'confusion_matrix': cm,
            'classification_report': cr,
            'predictions': val_predictions,
            'predicted_classes': y_pred_classes
        }
    
    def _print_cv_summary(self):
        """Print cross-validation summary"""
        print("\n" + "=" * 60)
        print("Cross-Validation Results Summary")
        print("=" * 60)
        print(f"Mean Accuracy: {self.cv_results['mean_accuracy']:.4f} ± {self.cv_results['std_accuracy']:.4f}")
        print(f"Mean ROC-AUC: {self.cv_results['mean_roc_auc']:.4f} ± {self.cv_results['std_roc_auc']:.4f}")
        print(f"Mean Log Loss: {self.cv_results['mean_log_loss']:.4f} ± {self.cv_results['std_log_loss']:.4f}")
        print("=" * 60)
    
    def save_results(self, output_dir: str = "results"):
        """Save training results and model"""
        if not self.cv_results:
            print("No results to save. Run cross_validate_model first.")
            return
        
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Save results
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save CV results
        results_file = f"{output_dir}/cv_results_{timestamp}.pkl"
        with open(results_file, 'wb') as f:
            pickle.dump(self.cv_results, f)
        
        # Save model
        if self.best_model and self.config['output']['save_model']:
            models_dir = self.config['output']['models_dir']
            Path(models_dir).mkdir(parents=True, exist_ok=True)
            
            model_file = f"{models_dir}/best_model_{timestamp}.h5"
            self.best_model.save(model_file)
            print(f"Best model saved to: {model_file}")
        
        # Save summary report
        summary_file = f"{output_dir}/training_summary_{timestamp}.txt"
        self._save_summary_report(summary_file)
        
        print(f"Results saved to: {output_dir}")
    
    def _save_summary_report(self, filepath: str):
        """Save a human-readable summary report"""
        with open(filepath, 'w') as f:
            f.write("=== IoT IDS Training Summary ===\n\n")
            f.write(f"Timestamp: {datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}\n\n")
            
            f.write("Model Configuration:\n")
            f.write(f"  Architecture: Neural Network\n")
            f.write(f"  Learning Rate: {self.config['model']['learning_rate']}\n")
            f.write(f"  Batch Size: {self.config['model']['batch_size']}\n")
            f.write(f"  Max Epochs: {self.config['model']['epochs']}\n\n")
            
            f.write("Cross-Validation Results:\n")
            f.write(f"  Folds: {self.config['cross_validation']['n_splits']}\n")
            f.write(f"  Mean Accuracy: {self.cv_results['mean_accuracy']:.4f} ± {self.cv_results['std_accuracy']:.4f}\n")
            f.write(f"  Mean ROC-AUC: {self.cv_results['mean_roc_auc']:.4f} ± {self.cv_results['std_roc_auc']:.4f}\n")
            f.write(f"  Mean Log Loss: {self.cv_results['mean_log_loss']:.4f} ± {self.cv_results['std_log_loss']:.4f}\n\n")
            
            f.write("Individual Fold Results:\n")
            for i, (acc, auc, loss) in enumerate(zip(
                self.cv_results['cv_scores'],
                self.cv_results['roc_auc_scores'],
                self.cv_results['log_losses']
            )):
                f.write(f"  Fold {i+1}: Acc={acc:.4f}, AUC={auc:.4f}, Loss={loss:.4f}\n")


def train_model_from_config(config_path: str = None, data_path: str = None):
    """
    Train model from configuration file
    
    Args:
        config_path (str): Path to configuration file
        data_path (str): Path to dataset
    """
    # Load data
    if data_path is None:
        raise ValueError("data_path must be provided")
    
    preprocessor = IoTDataPreprocessor()
    processed_data = preprocessor.preprocess_pipeline(data_path)
    
    # Initialize trainer
    trainer = IoTIDSTrainer()
    
    # Train model
    results = trainer.cross_validate_model(
        processed_data['X'],
        processed_data['y'],
        processed_data['feature_names'],
        processed_data['label_encoder']
    )
    
    # Save results
    trainer.save_results()
    
    return results


if __name__ == "__main__":
    # Example usage
    print("IoT Intrusion Detection System - Training Module")
    print("=" * 60)
    
    # This would be called with actual data path
    # results = train_model_from_config(data_path="data/raw/CTU-IoT-Malware-Capture-34-1/bro/conn.log.labeled")
    print("Import this module and call train_model_from_config() with your data path") 