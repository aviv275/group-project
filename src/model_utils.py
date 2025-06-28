"""
Model utilities for ESG claim analysis.

This module provides functions for model training, evaluation, and explainability.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import TruncatedSVD
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, precision_recall_curve, PrecisionRecallDisplay
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_random_state
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
            ('classifier', LogisticRegression(random_state=RANDOM_SEED, max_iter=1000, class_weight='balanced'))
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
        if self.model is None:
            raise RuntimeError("Model has not been trained. Call fit() first.")
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get prediction probabilities."""
        if self.model is None:
            raise RuntimeError("Model has not been trained. Call fit() first.")
        return self.model.predict_proba(X)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores."""
        if self.model is None:
            raise RuntimeError("Model has not been trained. Call fit() first.")
            
        if hasattr(self.model, 'named_steps') and hasattr(self.model.named_steps['classifier'], 'coef_'):
            classifier = self.model.named_steps['classifier']
            
            # Handle binary vs. multi-class coefficients
            if classifier.coef_.shape[0] == 1:
                importance = classifier.coef_[0]
            else:
                # For multi-class, we might need a different approach,
                # but for now, let's take the mean across classes.
                importance = classifier.coef_.mean(axis=0)

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
        self.classifier = LogisticRegression(random_state=RANDOM_SEED, max_iter=1000, class_weight='balanced')
        self.classifier.fit(combined_features, y)
        
        logger.info("Transformer model training complete")
        
    def predict(self, X: pd.DataFrame, text_column: str = 'esg_claim_text') -> np.ndarray:
        """Make predictions."""
        if self.transformer is None or self.classifier is None:
            raise RuntimeError("Model has not been trained. Call fit() first.")
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
        if self.transformer is None or self.classifier is None:
            raise RuntimeError("Model has not been trained. Call fit() first.")
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

class ThresholdWrapper(BaseEstimator, ClassifierMixin):
    """Wrapper for applying a custom threshold to calibrated probabilities."""
    
    def __init__(self, model, threshold):
        self.model = model
        self.threshold = threshold
        
    def fit(self, X, y):
        """Fit the underlying model."""
        self.model.fit(X, y)
        return self
        
    def predict_proba(self, X):
        """Get prediction probabilities."""
        return self.model.predict_proba(X)
        
    def predict(self, X):
        """Make predictions using the custom threshold."""
        p = self.predict_proba(X)[:, 1]
        return (p >= self.threshold).astype(int)

def make_balanced_group_folds(y, groups, n_splits=4, rng=42):
    """
    Build list of (train_idx, val_idx) tuples so that each validation fold
    contains at least one positive and one negative sample while keeping all
    rows of a given group together.
    """
    rng = check_random_state(rng)
    uniq_groups = np.unique(groups)

    for _ in range(200):  # up to 200 shuffle attempts
        rng.shuffle(uniq_groups)
        folds = np.array_split(uniq_groups, n_splits)
        splits, ok = [], True
        for fold_groups in folds:
            val_mask = np.isin(groups, fold_groups)
            if len(np.unique(y[val_mask])) < 2:
                ok = False
                break
            splits.append((np.where(~val_mask)[0], np.where(val_mask)[0]))
        if ok:
            return splits
    raise RuntimeError("Could not build balanced group folds.")

class AdvancedLogisticRegression:
    """Advanced Logistic Regression model with calibration and threshold optimization."""
    
    def __init__(self, target_recall: float = 0.80):
        """
        Initialize advanced logistic regression model.
        
        Args:
            target_recall: Target recall for threshold optimization
        """
        self.target_recall = target_recall
        self.calibrated_model = None
        self.threshold = None
        self.best_pipeline = None
        self.feature_names = None
        
    def fit(self, X: pd.DataFrame, y: pd.Series, groups: np.ndarray = None):
        """
        Fit the advanced model with calibration and threshold optimization.
        
        Args:
            X: Feature DataFrame
            y: Target series
            groups: Group identifiers for group-balanced CV (optional)
        """
        logger.info("Training advanced Logistic Regression model...")
        
        # Convert y to numpy array if it's a pandas Series
        if isinstance(y, pd.Series):
            y = y.values
        
        # Define features
        text_feature = "esg_claim_text"
        categorical_features = ["claim_category", "project_location", "claimed_metric_type"]
        numerical_features = [
            "claimed_value", "actual_measured_value", "report_year", 
            "abs_value_deviation", "rel_value_deviation"
        ]
        
        # Create preprocessing pipeline
        preprocessor = ColumnTransformer(transformers=[
            ("text", Pipeline([
                ("tfidf", TfidfVectorizer(max_features=2_000, ngram_range=(1, 2), stop_words="english")),
                ("svd", TruncatedSVD(n_components=100, random_state=RANDOM_SEED))
            ]), text_feature),
            
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("num", StandardScaler(), numerical_features)
        ])
        
        # Create base model
        base_lr = LogisticRegression(
            solver="liblinear",
            class_weight="balanced",
            max_iter=2_000
        )
        
        # Create full pipeline
        pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", base_lr),
        ])
        
        # Define hyperparameter grid
        param_grid = {
            "classifier__C": [0.01, 0.1, 1, 10, 100],
            "classifier__penalty": ["l1", "l2"],
        }
        
        # Define cross-validation
        inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
        
        # Perform grid search
        grid_search = GridSearchCV(
            estimator=pipe,
            param_grid=param_grid,
            cv=inner_cv,
            scoring="roc_auc",
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X, y)
        self.best_pipeline = grid_search.best_estimator_
        
        # Calibrate probabilities
        self.calibrated_model = CalibratedClassifierCV(
            self.best_pipeline, method="sigmoid", cv=inner_cv
        )
        self.calibrated_model.fit(X, y)
        
        # Choose optimal threshold for target recall
        oof_prob = cross_val_predict(
            self.calibrated_model, X, y,
            cv=inner_cv, method="predict_proba",
            n_jobs=-1
        )
        oof_prob = np.array(oof_prob)[:, 1]
        
        prec, rec, thr = precision_recall_curve(y, oof_prob)
        prec, rec, thr = prec[1:], rec[1:], thr
        
        # Find highest threshold with recall >= target
        ix = np.where(rec >= self.target_recall)[0][-1]
        self.threshold = float(thr[ix])
        
        logger.info(f"Optimal threshold: {self.threshold:.3f}")
        logger.info(f"OOF Recall @ threshold: {rec[ix]:.3f}")
        logger.info(f"OOF Precision @ threshold: {prec[ix]:.3f}")
        
        # Store feature names
        self.feature_names = {
            "text": text_feature,
            "categorical": categorical_features,
            "numerical": numerical_features
        }
        
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get prediction probabilities."""
        if self.calibrated_model is None:
            raise RuntimeError("Model has not been trained. Call fit() first.")
        return self.calibrated_model.predict_proba(X)
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using the optimized threshold."""
        if self.calibrated_model is None or self.threshold is None:
            raise RuntimeError("Model has not been trained. Call fit() first.")
        probs = self.predict_proba(X)[:, 1]
        return (probs >= self.threshold).astype(int)
    
    def get_threshold(self) -> float:
        """Get the optimized threshold."""
        if self.threshold is None:
            raise RuntimeError("Model has not been trained. Call fit() first.")
        return self.threshold 