"""
Neural Network Model Architecture for IoT Intrusion Detection System

This module implements the deep learning architecture used for classifying
IoT network traffic as benign or malicious.

Author: Devon Cardoso Elias
Institution: Vistula University
Year: 2025
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import os
import datetime


class IoTIDSModel:
    """
    Neural Network Model for IoT Intrusion Detection System
    
    Architecture:
    - Input Layer: Accepts flattened network traffic features
    - Dense Layer 1: 128 units, ReLU activation, L2 regularization, 40% dropout
    - Dense Layer 2: 64 units, ReLU activation, L2 regularization, 30% dropout
    - Output Layer: Softmax activation for multi-class classification
    
    Features:
    - Lightweight architecture (18,246 trainable parameters)
    - L2 regularization to prevent overfitting
    - Dropout layers for improved generalization
    - Class weighting for handling imbalanced data
    - Early stopping mechanism
    """
    
    def __init__(self, input_shape, num_classes, learning_rate=0.001):
        """
        Initialize the IoT IDS Model
        
        Args:
            input_shape (tuple): Shape of input features
            num_classes (int): Number of output classes
            learning_rate (float): Learning rate for Adam optimizer
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.model = None
        self.history = None
        
    def build_model(self):
        """
        Build the neural network architecture
        
        Returns:
            tensorflow.keras.Model: Compiled model ready for training
        """
        self.model = Sequential([
            # Input layer - flatten the input features
            Flatten(input_shape=self.input_shape),
            
            # First dense layer with L2 regularization
            Dense(
                128, 
                activation='relu', 
                kernel_regularizer=l2(0.001),
                name='dense_1'
            ),
            
            # First dropout layer (40% dropout rate)
            Dropout(0.4, name='dropout_1'),
            
            # Second dense layer with L2 regularization  
            Dense(
                64, 
                activation='relu', 
                kernel_regularizer=l2(0.001),
                name='dense_2'
            ),
            
            # Second dropout layer (30% dropout rate)
            Dropout(0.3, name='dropout_2'),
            
            # Output layer with softmax for multi-class classification
            Dense(
                self.num_classes, 
                activation='softmax',
                name='output'
            )
        ])
        
        # Compile the model
        self.model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return self.model
    
    def get_model_summary(self):
        """
        Get detailed model architecture summary
        
        Returns:
            str: Model summary string
        """
        if self.model is None:
            self.build_model()
        
        # Print model summary
        self.model.summary()
        
        # Calculate and display parameter information
        total_params = self.model.count_params()
        trainable_params = sum([tf.keras.backend.count_params(w) for w in self.model.trainable_weights])
        
        print(f"\nModel Architecture Details:")
        print(f"Total Parameters: {total_params:,}")
        print(f"Trainable Parameters: {trainable_params:,}")
        print(f"Memory Footprint (approx): {total_params * 4 / 1024 / 1024:.2f} MB")
        
        return self.model.summary()
    
    def prepare_callbacks(self, patience=10, monitor='val_loss', log_dir=None):
        """
        Prepare training callbacks
        
        Args:
            patience (int): Early stopping patience
            monitor (str): Metric to monitor for early stopping
            log_dir (str): Directory for TensorBoard logs
            
        Returns:
            list: List of Keras callbacks
        """
        callbacks = []
        
        # Early stopping callback
        early_stopping = EarlyStopping(
            monitor=monitor,
            patience=patience,
            restore_best_weights=True,
            verbose=1
        )
        callbacks.append(early_stopping)
        
        # TensorBoard callback
        if log_dir is None:
            log_dir = os.path.join(
                "logs", 
                "fit", 
                datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            )
        
        tensorboard_callback = TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,
            write_graph=True,
            write_images=True
        )
        callbacks.append(tensorboard_callback)
        
        return callbacks
    
    def compute_class_weights(self, y_train):
        """
        Compute class weights for handling imbalanced data
        
        Args:
            y_train (array): Training labels
            
        Returns:
            dict: Class weights dictionary
        """
        # Get unique classes
        classes = np.unique(y_train)
        
        # Compute class weights
        class_weights = compute_class_weight(
            'balanced',
            classes=classes,
            y=y_train
        )
        
        # Convert to dictionary
        class_weight_dict = dict(zip(classes, class_weights))
        
        print("Class Weights:")
        for class_idx, weight in class_weight_dict.items():
            print(f"  Class {class_idx}: {weight:.4f}")
        
        return class_weight_dict
    
    def train(self, X_train, y_train, X_val, y_val, 
              epochs=50, batch_size=32, verbose=1):
        """
        Train the model with given data
        
        Args:
            X_train (array): Training features
            y_train (array): Training labels
            X_val (array): Validation features  
            y_val (array): Validation labels
            epochs (int): Number of training epochs
            batch_size (int): Training batch size
            verbose (int): Verbosity level
            
        Returns:
            tensorflow.keras.callbacks.History: Training history
        """
        if self.model is None:
            self.build_model()
        
        # Compute class weights
        class_weights = self.compute_class_weights(y_train)
        
        # Prepare callbacks
        callbacks = self.prepare_callbacks()
        
        # Train the model
        print(f"\nTraining model with {len(X_train)} samples...")
        print(f"Validation set: {len(X_val)} samples")
        print(f"Input shape: {X_train.shape}")
        print(f"Number of classes: {self.num_classes}")
        
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            class_weight=class_weights,
            callbacks=callbacks,
            verbose=verbose
        )
        
        return self.history
    
    def save_model(self, filepath):
        """
        Save the trained model
        
        Args:
            filepath (str): Path to save the model
        """
        if self.model is None:
            raise ValueError("Model must be built and trained before saving")
        
        self.model.save(filepath)
        print(f"Model saved to: {filepath}")
    
    def load_model(self, filepath):
        """
        Load a pre-trained model
        
        Args:
            filepath (str): Path to the saved model
        """
        self.model = tf.keras.models.load_model(filepath)
        print(f"Model loaded from: {filepath}")
    
    def predict(self, X_test):
        """
        Make predictions on test data
        
        Args:
            X_test (array): Test features
            
        Returns:
            tuple: (predictions, probabilities)
        """
        if self.model is None:
            raise ValueError("Model must be built and trained before prediction")
        
        # Get prediction probabilities
        probabilities = self.model.predict(X_test)
        
        # Get class predictions
        predictions = np.argmax(probabilities, axis=1)
        
        return predictions, probabilities


def create_model_config():
    """
    Create default model configuration
    
    Returns:
        dict: Model configuration parameters
    """
    config = {
        'learning_rate': 0.001,
        'dense_1_units': 128,
        'dense_2_units': 64,
        'dropout_1_rate': 0.4,
        'dropout_2_rate': 0.3,
        'l2_regularization': 0.001,
        'batch_size': 32,
        'epochs': 50,
        'early_stopping_patience': 10,
        'early_stopping_monitor': 'val_loss'
    }
    
    return config


if __name__ == "__main__":
    # Example usage
    print("IoT Intrusion Detection System - Model Architecture")
    print("=" * 60)
    
    # Example parameters
    input_shape = (74,)  # Based on preprocessed IoT-23 features
    num_classes = 6      # Number of traffic classes
    
    # Create and build model
    ids_model = IoTIDSModel(input_shape, num_classes)
    model = ids_model.build_model()
    
    # Display model summary
    ids_model.get_model_summary()
    
    # Display configuration
    config = create_model_config()
    print("\nDefault Model Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
