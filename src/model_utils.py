"""
Model utilities for ESG claim analysis.

This module provides functions for model training, evaluation, and explainability.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer
import torch
import pickle
import json
import os
from typing import Dict, Any, Tuple, List
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

class BaselineModel:
    """Baseline model using TF-IDF and Logistic Regression."""
    
    def __init__(self, task: str):
        """
        Initialize baseline model.
        
        Args:
            task: Either 'category' or 'greenwash'
        """
        self.task = task
        self.model = None
        self.vectorizer = None
        self.scaler = None
        self.feature_names = None
        
    def fit(self, X: pd.DataFrame, y: pd.Series, text_column: str = 'esg_claim_text'):
        """
        Fit the baseline model.
        
        Args:
            X: Feature DataFrame
            y: Target series
            text_column: Name of text column
        """
        logger.info(f"Training baseline model for {self.task} task...")
        
        # Ensure text column is present and of string type
        if text_column not in X.columns:
            raise ValueError(f"Text column '{text_column}' not found in features.")
        X[text_column] = X[text_column].astype(str)
        
        # Separate text and numeric features
        text_features = X[[text_column]]
        numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
        logger.info(f"Text features shape: {text_features.shape}, Numeric features: {numeric_features}")
        
        # Create preprocessing pipeline
        preprocessors = []
        
        # Text preprocessing
        text_transformer = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=1000, stop_words='english'))
        ])
        preprocessors.append(('text', text_transformer, text_column))
        
        # Numeric preprocessing
        if numeric_features:
            numeric_transformer = Pipeline([
                ('scaler', StandardScaler())
            ])
            preprocessors.append(('numeric', numeric_transformer, numeric_features))
        
        # Combine preprocessors
        preprocessor = ColumnTransformer(preprocessors, remainder='drop')
        
        # Create full pipeline
        self.model = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression(random_state=RANDOM_SEED, max_iter=1000))
        ])
        
        # Fit model
        self.model.fit(X, y)
        # Store feature names for explainability
        if hasattr(self.model.named_steps['preprocessor'], 'get_feature_names_out'):
            self.feature_names = self.model.named_steps['preprocessor'].get_feature_names_out()
        else:
            self.feature_names = X.columns.tolist()
        
        logger.info("Baseline model training complete")
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get prediction probabilities."""
        return self.model.predict_proba(X)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores."""
        if hasattr(self.model, 'named_steps'):
            classifier = self.model.named_steps['classifier']
        else:
            classifier = self.model
        
        if hasattr(classifier, 'coef_'):
            importance = classifier.coef_[0] if classifier.coef_.shape[0] == 1 else classifier.coef_
            feature_names = self.feature_names or [f'feature_{i}' for i in range(len(importance))]
            return dict(zip(feature_names, importance))
        return {}

class TransformerModel:
    """Transformer-based model using sentence transformers."""
    
    def __init__(self, task: str, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize transformer model.
        
        Args:
            task: Either 'category' or 'greenwash'
            model_name: Name of the sentence transformer model
        """
        self.task = task
        self.model_name = model_name
        self.transformer = None
        self.classifier = None
        self.scaler = None
        
    def fit(self, X: pd.DataFrame, y: pd.Series, text_column: str = 'esg_claim_text'):
        """
        Fit the transformer model.
        
        Args:
            X: Feature DataFrame
            y: Target series
            text_column: Name of text column
        """
        logger.info(f"Training transformer model for {self.task} task...")
        
        # Load sentence transformer
        self.transformer = SentenceTransformer(self.model_name)
        
        # Get text embeddings
        texts = X[text_column].tolist()
        embeddings = self.transformer.encode(texts, show_progress_bar=True)
        
        # Combine with numeric features
        numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_features:
            numeric_data = X[numeric_features].values
            # Scale numeric features
            self.scaler = StandardScaler()
            numeric_data_scaled = self.scaler.fit_transform(numeric_data)
            # Combine embeddings with numeric features
            combined_features = np.hstack([embeddings, numeric_data_scaled])
        else:
            combined_features = embeddings
        
        # Train classifier
        self.classifier = LogisticRegression(random_state=RANDOM_SEED, max_iter=1000)
        self.classifier.fit(combined_features, y)
        
        logger.info("Transformer model training complete")
        
    def predict(self, X: pd.DataFrame, text_column: str = 'esg_claim_text') -> np.ndarray:
        """Make predictions."""
        texts = X[text_column].tolist()
        embeddings = self.transformer.encode(texts)
        
        numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_features and self.scaler:
            numeric_data = X[numeric_features].values
            numeric_data_scaled = self.scaler.transform(numeric_data)
            combined_features = np.hstack([embeddings, numeric_data_scaled])
        else:
            combined_features = embeddings
        
        return self.classifier.predict(combined_features)
    
    def predict_proba(self, X: pd.DataFrame, text_column: str = 'esg_claim_text') -> np.ndarray:
        """Get prediction probabilities."""
        texts = X[text_column].tolist()
        embeddings = self.transformer.encode(texts)
        
        numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_features and self.scaler:
            numeric_data = X[numeric_features].values
            numeric_data_scaled = self.scaler.transform(numeric_data)
            combined_features = np.hstack([embeddings, numeric_data_scaled])
        else:
            combined_features = embeddings
        
        return self.classifier.predict_proba(combined_features)

def evaluate_model(model, X: pd.DataFrame, y: pd.Series, cv_folds: int = 5) -> Dict[str, Any]:
    """
    Evaluate model using cross-validation.
    
    Args:
        model: Trained model
        X: Feature DataFrame
        y: Target series
        cv_folds: Number of CV folds
        
    Returns:
        Dictionary with evaluation metrics
    """
    logger.info(f"Evaluating model with {cv_folds}-fold cross-validation...")
    
    # Cross-validation
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_SEED)
    
    # Get predictions for full dataset
    # Handle different model types
    if hasattr(model, 'transformer') and model.transformer is not None:
        # TransformerModel
        y_pred = model.predict(X, text_column='esg_claim_text')
        y_pred_proba = model.predict_proba(X, text_column='esg_claim_text')
    else:
        # BaselineModel or other sklearn-compatible models
        y_pred = model.predict(X)
        y_pred_proba = model.predict_proba(X)
    
    # Calculate metrics
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, average='weighted' if len(np.unique(y)) > 2 else 'binary')
    f1 = f1_score(y, y_pred, average='weighted' if len(np.unique(y)) > 2 else 'binary')
    
    # Classification report
    report = classification_report(y, y_pred, output_dict=True)
    
    # Confusion matrix
    cm = confusion_matrix(y, y_pred)
    
    results = {
        'accuracy': accuracy,
        'precision': precision,
        'f1_score': f1,
        'classification_report': report,
        'confusion_matrix': cm.tolist(),
        'predictions': y_pred.tolist(),
        'probabilities': y_pred_proba.tolist()
    }
    
    logger.info(f"Evaluation complete. Accuracy: {accuracy:.3f}, F1: {f1:.3f}")
    return results

def save_model(model, filepath: str) -> None:
    """Save model to a file."""
    logger.info(f"Saving model to {filepath}...")
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    logger.info("Model saved successfully")

def load_model(filepath: str):
    """Load model from file."""
    logger.info(f"Loading model from {filepath}")
    with open(filepath, 'rb') as f:
        model = pickle.load(f)
    logger.info("Model loaded successfully")
    return model

def save_metrics(metrics: Dict[str, Any], filepath: str) -> None:
    """Save metrics to JSON file."""
    logger.info(f"Saving metrics to {filepath}")
    with open(filepath, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info("Metrics saved successfully")

def plot_confusion_matrix(cm: np.ndarray, labels: List[str], task: str) -> None:
    """Plot confusion matrix."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title(f'Confusion Matrix - {task}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f'reports/figures/confusion_matrix_{task}.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_feature_importance(importance: Dict[str, float], task: str, top_n: int = 20) -> None:
    """Plot feature importance."""
    # Sort by absolute importance
    sorted_importance = sorted(importance.items(), key=lambda x: abs(x[1]), reverse=True)[:top_n]
    features, scores = zip(*sorted_importance)
    
    plt.figure(figsize=(12, 8))
    colors = ['red' if score < 0 else 'blue' for score in scores]
    plt.barh(range(len(features)), scores, color=colors)
    plt.yticks(range(len(features)), features)
    plt.xlabel('Feature Importance')
    plt.title(f'Feature Importance - {task}')
    plt.tight_layout()
    plt.savefig(f'reports/figures/feature_importance_{task}.png', dpi=300, bbox_inches='tight')
    plt.close() 