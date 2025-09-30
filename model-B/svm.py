"""
SVM Model for Stroke Prediction
===============================

This module implements a Support Vector Machine (SVM) model for predicting stroke
based on preprocessed healthcare data.

Author: Data Mining Project
Date: 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score, roc_curve
)
from sklearn.preprocessing import StandardScaler
import joblib
import warnings
warnings.filterwarnings('ignore')

class StrokeSVMModel:
    """
    SVM Model for Stroke Prediction
    
    This class implements a complete SVM pipeline for stroke prediction including
    data loading, model training, hyperparameter tuning, and evaluation.
    """
    
    def __init__(self, data_dir='../data-pre/'):
        """
        Initialize the SVM model
        
        Args:
            data_dir (str): Directory containing preprocessed data files
        """
        self.data_dir = data_dir
        self.model = None
        self.scaler = StandardScaler()
        self.best_params = None
        self.feature_names = None
        
    def load_data(self):
        """
        Load preprocessed training and test data
        
        Returns:
            tuple: (X_train, y_train, X_test, y_test)
        """
        print("Loading preprocessed data...")
        
        # Load feature names
        with open(f'{self.data_dir}feature_names.txt', 'r') as f:
            self.feature_names = [line.strip() for line in f.readlines()]
        
        # Load training data
        train_data = pd.read_csv(f'{self.data_dir}train_preprocessed.csv')
        X_train = train_data.drop('stroke', axis=1)
        y_train = train_data['stroke']
        
        # Load test data
        test_data = pd.read_csv(f'{self.data_dir}test_preprocessed.csv')
        X_test = test_data.drop('stroke', axis=1)
        y_test = test_data['stroke']
        
        print(f"Training data shape: {X_train.shape}")
        print(f"Test data shape: {X_test.shape}")
        print(f"Training target distribution:")
        print(y_train.value_counts(normalize=True))
        print(f"Test target distribution:")
        print(y_test.value_counts(normalize=True))
        
        return X_train, y_train, X_test, y_test
    
    def hyperparameter_tuning(self, X_train, y_train):
        """
        Perform hyperparameter tuning using GridSearchCV
        
        Args:
            X_train: Training features
            y_train: Training labels
            
        Returns:
            dict: Best parameters found
        """
        print("Performing hyperparameter tuning...")
        
        # Define parameter grid for SVM
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'kernel': ['linear', 'rbf', 'poly'],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
            'class_weight': ['balanced', None]
        }
        
        # Create SVM classifier
        svm = SVC(random_state=42, probability=True)
        
        # Perform grid search with cross-validation
        grid_search = GridSearchCV(
            svm, 
            param_grid, 
            cv=5, 
            scoring='f1',  # Use F1 score due to class imbalance
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        self.best_params = grid_search.best_params_
        print(f"Best parameters: {self.best_params}")
        print(f"Best cross-validation F1 score: {grid_search.best_score_:.4f}")
        
        return self.best_params
    
    def train_model(self, X_train, y_train, use_tuning=True):
        """
        Train the SVM model
        
        Args:
            X_train: Training features
            y_train: Training labels
            use_tuning (bool): Whether to use hyperparameter tuning
        """
        print("Training SVM model...")
        
        if use_tuning and self.best_params is None:
            self.hyperparameter_tuning(X_train, y_train)
        
        # Create and train the final model
        if self.best_params:
            self.model = SVC(**self.best_params, random_state=42, probability=True)
        else:
            # Default parameters with class balancing
            self.model = SVC(
                kernel='rbf', 
                C=1.0, 
                gamma='scale', 
                class_weight='balanced',
                random_state=42, 
                probability=True
            )
        
        self.model.fit(X_train, y_train)
        print("Model training completed!")
        
        # Perform cross-validation
        cv_scores = cross_val_score(self.model, X_train, y_train, cv=5, scoring='f1')
        print(f"Cross-validation F1 scores: {cv_scores}")
        print(f"Mean CV F1 score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    def evaluate_model(self, X_test, y_test):
        """
        Evaluate the trained model on test data
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            dict: Evaluation metrics
        """
        print("Evaluating model on test data...")
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }
        
        # Print metrics
        print("\n" + "="*50)
        print("MODEL EVALUATION RESULTS")
        print("="*50)
        for metric, value in metrics.items():
            print(f"{metric.upper()}: {value:.4f}")
        
        # Classification report
        print("\nCLASSIFICATION REPORT:")
        print(classification_report(y_test, y_pred))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print("\nCONFUSION MATRIX:")
        print(cm)
        
        return metrics, y_pred, y_pred_proba
    
    def plot_results(self, y_test, y_pred, y_pred_proba):
        """
        Create visualization plots for model results
        
        Args:
            y_test: True test labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0,0])
        axes[0,0].set_title('Confusion Matrix')
        axes[0,0].set_xlabel('Predicted')
        axes[0,0].set_ylabel('Actual')
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        axes[0,1].plot(fpr, tpr, color='darkorange', lw=2, 
                      label=f'ROC curve (AUC = {roc_auc:.4f})')
        axes[0,1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        axes[0,1].set_xlim([0.0, 1.0])
        axes[0,1].set_ylim([0.0, 1.05])
        axes[0,1].set_xlabel('False Positive Rate')
        axes[0,1].set_ylabel('True Positive Rate')
        axes[0,1].set_title('ROC Curve')
        axes[0,1].legend(loc="lower right")
        
        # Prediction Distribution
        axes[1,0].hist(y_pred_proba[y_test == 0], bins=30, alpha=0.7, 
                      label='No Stroke', color='blue')
        axes[1,0].hist(y_pred_proba[y_test == 1], bins=30, alpha=0.7, 
                      label='Stroke', color='red')
        axes[1,0].set_xlabel('Predicted Probability')
        axes[1,0].set_ylabel('Frequency')
        axes[1,0].set_title('Prediction Probability Distribution')
        axes[1,0].legend()
        
        # Feature Importance (for linear kernel)
        if hasattr(self.model, 'coef_') and self.model.coef_ is not None:
            feature_importance = np.abs(self.model.coef_[0])
            feature_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': feature_importance
            }).sort_values('importance', ascending=True).tail(10)
            
            axes[1,1].barh(feature_df['feature'], feature_df['importance'])
            axes[1,1].set_xlabel('Absolute Coefficient Value')
            axes[1,1].set_title('Top 10 Feature Importance (Linear SVM)')
        else:
            axes[1,1].text(0.5, 0.5, 'Feature importance not available\nfor non-linear kernels', 
                          ha='center', va='center', transform=axes[1,1].transAxes)
            axes[1,1].set_title('Feature Importance')
        
        plt.tight_layout()
        plt.savefig('svm_results.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_model(self, filepath='svm_stroke_model.joblib'):
        """
        Save the trained model to disk
        
        Args:
            filepath (str): Path to save the model
        """
        if self.model is not None:
            joblib.dump({
                'model': self.model,
                'scaler': self.scaler,
                'feature_names': self.feature_names,
                'best_params': self.best_params
            }, filepath)
            print(f"Model saved to {filepath}")
        else:
            print("No trained model to save!")
    
    def load_model(self, filepath='svm_stroke_model.joblib'):
        """
        Load a trained model from disk
        
        Args:
            filepath (str): Path to load the model from
        """
        try:
            saved_data = joblib.load(filepath)
            self.model = saved_data['model']
            self.scaler = saved_data['scaler']
            self.feature_names = saved_data['feature_names']
            self.best_params = saved_data['best_params']
            print(f"Model loaded from {filepath}")
        except FileNotFoundError:
            print(f"Model file {filepath} not found!")
    
    def predict_single(self, features):
        """
        Make prediction for a single sample
        
        Args:
            features (array-like): Feature values for prediction
            
        Returns:
            tuple: (prediction, probability)
        """
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        features = np.array(features).reshape(1, -1)
        prediction = self.model.predict(features)[0]
        probability = self.model.predict_proba(features)[0, 1]
        
        return prediction, probability

def main():
    """
    Main function to run the complete SVM pipeline
    """
    print("="*60)
    print("SVM MODEL FOR STROKE PREDICTION")
    print("="*60)
    
    # Initialize model
    svm_model = StrokeSVMModel()
    
    # Load data
    X_train, y_train, X_test, y_test = svm_model.load_data()
    
    # Train model with hyperparameter tuning
    svm_model.train_model(X_train, y_train, use_tuning=True)
    
    # Evaluate model
    metrics, y_pred, y_pred_proba = svm_model.evaluate_model(X_test, y_test)
    
    # Create visualizations
    svm_model.plot_results(y_test, y_pred, y_pred_proba)
    
    # Save model
    svm_model.save_model('svm_stroke_model.joblib')
    
    print("\n" + "="*60)
    print("SVM MODEL TRAINING AND EVALUATION COMPLETED!")
    print("="*60)
    
    return svm_model, metrics

if __name__ == "__main__":
    model, results = main()