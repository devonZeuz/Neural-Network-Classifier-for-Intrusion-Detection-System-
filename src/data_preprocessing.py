"""
Data Preprocessing Pipeline for IoT Intrusion Detection System

This module handles the complete data preprocessing pipeline for IoT-23 dataset,
including data loading, feature engineering, class balancing, and scaling.

Author: Devon Cardoso Elias
Institution: Vistula University
Year: 2025
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE
import warnings
import os
from typing import Tuple, List, Dict, Optional

warnings.filterwarnings('ignore')


class IoTDataPreprocessor:
    """
    Complete data preprocessing pipeline for IoT-23 dataset
    
    This class handles:
    - Data loading from conn.log.labeled files
    - Feature engineering and categorical encoding
    - Class imbalance handling with SMOTE
    - Feature scaling with RobustScaler
    - Stratified K-Fold cross-validation setup
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the data preprocessor
        
        Args:
            random_state (int): Random state for reproducibility
        """
        self.random_state = random_state
        self.label_encoder = LabelEncoder()
        self.scaler = RobustScaler()
        self.smote = SMOTE(random_state=random_state)
        self.feature_columns = None
        self.original_labels = None
        self.processed_data = {}
        
    def load_iot23_data(self, file_path: str) -> pd.DataFrame:
        """
        Load IoT-23 conn.log.labeled dataset
        
        Args:
            file_path (str): Path to the conn.log.labeled.csv file
            
        Returns:
            pd.DataFrame: Loaded dataset with proper column names
        """
        print(f"Loading data from: {file_path}")
        
        try:
            # Load the dataset with tab delimiter
            df = pd.read_csv(
                file_path, 
                delimiter='\t', 
                comment='#', 
                na_values='-', 
                header=None, 
                dtype={'service': str},
                low_memory=False
            )
            
            # Define column names based on Zeek/Bro conn.log format
            df.columns = [
                'ts', 'uid', 'id.orig_h', 'id.orig_p', 'id.resp_h', 'id.resp_p', 
                'proto', 'service', 'duration', 'orig_bytes', 'resp_bytes', 
                'conn_state', 'local_orig', 'local_resp', 'missed_bytes', 
                'history', 'orig_pkts', 'orig_ip_bytes', 'resp_pkts', 
                'resp_ip_bytes', 'label'
            ]
            
            print(f"Dataset loaded successfully: {df.shape}")
            print(f"Unique labels: {df['label'].unique()}")
            print(f"Number of unique labels: {len(df['label'].unique())}")
            
            return df
            
        except Exception as e:
            print(f"Error loading dataset: {str(e)}")
            raise
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Perform feature engineering on the dataset
        
        Args:
            df (pd.DataFrame): Raw dataset
            
        Returns:
            pd.DataFrame: Dataset with engineered features
        """
        print("Starting feature engineering...")
        
        # Create a copy to avoid modifying original data
        df_processed = df.copy()
        
        # Store original labels
        self.original_labels = df_processed['label'].unique()
        
        # Encode labels to numerical values
        df_processed['label_encoded'] = self.label_encoder.fit_transform(df_processed['label'])
        
        # Drop columns that may cause data leakage or are not useful
        columns_to_drop = ['uid', 'ts', 'id.orig_h', 'id.resp_h']
        df_processed.drop(columns=columns_to_drop, inplace=True)
        print(f"Dropped columns: {columns_to_drop}")
        
        # Define categorical features for one-hot encoding
        categorical_features = ['proto', 'service', 'conn_state', 'local_orig', 'local_resp', 'history']
        
        # Apply one-hot encoding to categorical variables
        print(f"Applying one-hot encoding to: {categorical_features}")
        df_processed = pd.get_dummies(
            df_processed, 
            columns=categorical_features, 
            drop_first=True
        )
        
        # Fill any remaining NaN values
        df_processed.fillna(0, inplace=True)
        
        # Store feature columns (excluding labels)
        self.feature_columns = [col for col in df_processed.columns 
                               if col not in ['label', 'label_encoded']]
        
        print(f"Feature engineering completed:")
        print(f"  - Total features: {len(self.feature_columns)}")
        print(f"  - Dataset shape: {df_processed.shape}")
        
        return df_processed
    
    def split_features_labels(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Split dataset into features and labels
        
        Args:
            df (pd.DataFrame): Processed dataset
            
        Returns:
            Tuple[pd.DataFrame, pd.Series]: Features and labels
        """
        X = df[self.feature_columns]
        y = df['label_encoded']
        
        print(f"Features shape: {X.shape}")
        print(f"Labels shape: {y.shape}")
        print(f"Label distribution:\n{y.value_counts().sort_index()}")
        
        return X, y
    
    def handle_class_imbalance(self, X: pd.DataFrame, y: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """
        Handle class imbalance using SMOTE (Synthetic Minority Oversampling Technique)
        
        Args:
            X (pd.DataFrame): Features
            y (pd.Series): Labels
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Balanced features and labels
        """
        print("Handling class imbalance with SMOTE...")
        
        # Display original class distribution
        print("Original class distribution:")
        original_distribution = y.value_counts().sort_index()
        for label, count in original_distribution.items():
            print(f"  Class {label}: {count:,} samples")
        
        # Apply SMOTE
        X_resampled, y_resampled = self.smote.fit_resample(X, y)
        
        # Display new class distribution
        print("Resampled class distribution:")
        resampled_distribution = pd.Series(y_resampled).value_counts().sort_index()
        for label, count in resampled_distribution.items():
            print(f"  Class {label}: {count:,} samples")
        
        print(f"Dataset shape after SMOTE: {X_resampled.shape}")
        
        return X_resampled, y_resampled
    
    def scale_features(self, X_train: np.ndarray, X_val: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Scale features using RobustScaler (outlier-resistant)
        
        Args:
            X_train (np.ndarray): Training features
            X_val (np.ndarray, optional): Validation features
            
        Returns:
            Tuple[np.ndarray, Optional[np.ndarray]]: Scaled features
        """
        print("Scaling features with RobustScaler...")
        
        # Fit scaler on training data
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        X_val_scaled = None
        if X_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
        
        print("Feature scaling completed.")
        print(f"RobustScaler centers data at median and scales according to IQR")
        
        return X_train_scaled, X_val_scaled
    
    def setup_cross_validation(self, n_splits: int = 5, shuffle: bool = True) -> StratifiedKFold:
        """
        Setup stratified K-fold cross-validation
        
        Args:
            n_splits (int): Number of folds
            shuffle (bool): Whether to shuffle data
            
        Returns:
            StratifiedKFold: Cross-validation splitter
        """
        print(f"Setting up {n_splits}-fold stratified cross-validation...")
        
        skf = StratifiedKFold(
            n_splits=n_splits,
            shuffle=shuffle,
            random_state=self.random_state
        )
        
        print("Stratified K-Fold ensures balanced class distribution across folds")
        
        return skf
    
    def preprocess_pipeline(self, file_path: str, apply_smote: bool = True) -> Dict:
        """
        Complete preprocessing pipeline
        
        Args:
            file_path (str): Path to the dataset file
            apply_smote (bool): Whether to apply SMOTE for class balancing
            
        Returns:
            Dict: Processed data with all components
        """
        print("=" * 60)
        print("IoT-23 Data Preprocessing Pipeline")
        print("=" * 60)
        
        # Step 1: Load data
        df = self.load_iot23_data(file_path)
        
        # Step 2: Feature engineering
        df_processed = self.engineer_features(df)
        
        # Step 3: Split features and labels
        X, y = self.split_features_labels(df_processed)
        
        # Step 4: Handle class imbalance (optional)
        if apply_smote:
            X_resampled, y_resampled = self.handle_class_imbalance(X, y)
        else:
            X_resampled, y_resampled = X.values, y.values
            print("Skipping SMOTE - using original class distribution")
        
        # Store processed data
        self.processed_data = {
            'X': X_resampled,
            'y': y_resampled,
            'feature_names': self.feature_columns,
            'label_encoder': self.label_encoder,
            'scaler': self.scaler,
            'original_labels': self.original_labels,
            'num_features': len(self.feature_columns),
            'num_classes': len(np.unique(y_resampled))
        }
        
        print("=" * 60)
        print("Preprocessing Pipeline Completed Successfully!")
        print(f"Final dataset shape: {X_resampled.shape}")
        print(f"Number of features: {len(self.feature_columns)}")
        print(f"Number of classes: {len(np.unique(y_resampled))}")
        print("=" * 60)
        
        return self.processed_data
    
    def get_feature_importance_info(self) -> Dict:
        """
        Get information about features for importance analysis
        
        Returns:
            Dict: Feature information
        """
        if not self.processed_data:
            raise ValueError("Data must be processed first")
        
        feature_info = {
            'feature_names': self.feature_columns,
            'num_features': len(self.feature_columns),
            'categorical_features': [col for col in self.feature_columns if any(
                cat in col for cat in ['proto_', 'service_', 'conn_state_', 
                                     'local_orig_', 'local_resp_', 'history_']
            )],
            'numerical_features': [col for col in self.feature_columns if not any(
                cat in col for cat in ['proto_', 'service_', 'conn_state_', 
                                     'local_orig_', 'local_resp_', 'history_']
            )]
        }
        
        return feature_info
    
    def save_preprocessor(self, filepath: str):
        """
        Save the preprocessor state
        
        Args:
            filepath (str): Path to save the preprocessor
        """
        import joblib
        
        preprocessor_state = {
            'label_encoder': self.label_encoder,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'original_labels': self.original_labels,
            'random_state': self.random_state
        }
        
        joblib.dump(preprocessor_state, filepath)
        print(f"Preprocessor saved to: {filepath}")
    
    def load_preprocessor(self, filepath: str):
        """
        Load a saved preprocessor state
        
        Args:
            filepath (str): Path to the saved preprocessor
        """
        import joblib
        
        preprocessor_state = joblib.load(filepath)
        
        self.label_encoder = preprocessor_state['label_encoder']
        self.scaler = preprocessor_state['scaler']
        self.feature_columns = preprocessor_state['feature_columns']
        self.original_labels = preprocessor_state['original_labels']
        self.random_state = preprocessor_state['random_state']
        
        print(f"Preprocessor loaded from: {filepath}")


def analyze_dataset_info(file_path: str) -> Dict:
    """
    Quick analysis of dataset characteristics
    
    Args:
        file_path (str): Path to the dataset file
        
    Returns:
        Dict: Dataset analysis information
    """
    preprocessor = IoTDataPreprocessor()
    
    # Load and analyze
    df = preprocessor.load_iot23_data(file_path)
    
    analysis = {
        'total_samples': len(df),
        'features': len(df.columns) - 1,  # Excluding label
        'unique_labels': df['label'].unique().tolist(),
        'label_distribution': df['label'].value_counts().to_dict(),
        'missing_values': df.isnull().sum().sum(),
        'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024
    }
    
    return analysis


if __name__ == "__main__":
    # Example usage
    print("IoT Intrusion Detection System - Data Preprocessing")
    print("=" * 60)
    
    # Example file path (adjust as needed)
    example_file = "data/raw/CTU-IoT-Malware-Capture-34-1/bro/conn.log.labeled.csv"
    
    if os.path.exists(example_file):
        # Initialize preprocessor
        preprocessor = IoTDataPreprocessor(random_state=42)
        
        # Run complete pipeline
        processed_data = preprocessor.preprocess_pipeline(example_file)
        
        # Display results
        print("\nProcessing Results:")
        for key, value in processed_data.items():
            if key in ['X', 'y']:
                print(f"{key}: {value.shape}")
            elif isinstance(value, list):
                print(f"{key}: {len(value)} items")
            else:
                print(f"{key}: {value}")
    else:
        print(f"Example file not found: {example_file}")
        print("Please provide a valid path to IoT-23 conn.log.labeled.csv file")
