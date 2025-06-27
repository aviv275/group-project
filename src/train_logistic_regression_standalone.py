#!/usr/bin/env python3
"""
Standalone Logistic Regression Training Pipeline for ESG Greenwashing Detection.

This module implements the exact Logistic Regression model specification
with TF-IDF preprocessing and hyperparameter tuning as specified.
"""

import pandas as pd
import numpy as np
import sys
import os
import logging
import json
import pickle
from typing import Dict, Any
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

def load_and_prepare_data(data_path: str) -> tuple:
    """
    Load and prepare data for training.
    
    Args:
        data_path: Path to the data file
        
    Returns:
        Tuple of (X, y) where X is the feature DataFrame and y is the target series
    """
    logger.info(f"Loading data from {data_path}")
    
    # Load data
    if data_path.endswith('.csv'):
        df = pd.read_csv(data_path)
    elif data_path.endswith('.parquet'):
        df = pd.read_parquet(data_path)
    else:
        raise ValueError(f"Unsupported file format: {data_path}")
    
    logger.info(f"Loaded {len(df)} rows and {len(df.columns)} columns")
    
    # Clean data
    df_clean = df.copy()
    
    # Convert timestamp to datetime if present
    if 'timestamp' in df_clean.columns:
        df_clean['timestamp'] = pd.to_datetime(df_clean['timestamp'], errors='coerce')
    
    # Ensure numeric columns are properly typed
    numeric_columns = [
        'claimed_value', 'actual_measured_value', 'value_deviation',
        'external_validation_score', 'report_sentiment_score', 
        'llm_claim_consistency_score'
    ]
    
    for col in numeric_columns:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    
    # Ensure flag columns are binary
    flag_columns = ['greenwashing_flag', 'controversy_flag']
    for col in flag_columns:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].astype(int)
    
    # Clean text columns
    if 'esg_claim_text' in df_clean.columns:
        df_clean['esg_claim_text'] = df_clean['esg_claim_text'].astype(str)
        df_clean['esg_claim_text'] = df_clean['esg_claim_text'].str.strip()
    
    # Remove duplicates
    initial_rows = len(df_clean)
    df_clean = df_clean.drop_duplicates()
    removed_duplicates = initial_rows - len(df_clean)
    logger.info(f"Removed {removed_duplicates} duplicates")
    
    # Add claim_length feature as specified
    df_clean["claim_length"] = df_clean["esg_claim_text"].apply(lambda x: len(str(x).split()))
    
    # Define features exactly as specified
    text_feature = "esg_claim_text"
    categorical_features = ["claim_category", "project_location", "claimed_metric_type"]
    numerical_features = ["claimed_value", "report_year", "report_sentiment_score", "claim_length"]
    target = "greenwashing_flag"
    
    # Check if all required columns exist
    required_columns = [text_feature] + categorical_features + numerical_features + [target]
    missing_columns = [col for col in required_columns if col not in df_clean.columns]
    
    if missing_columns:
        logger.warning(f"Missing columns: {missing_columns}")
        # Fill missing columns with default values
        for col in missing_columns:
            if col in categorical_features:
                df_clean[col] = "Unknown"
            elif col in numerical_features:
                df_clean[col] = 0.0
            elif col == target:
                logger.error(f"Target column '{target}' is missing!")
                raise ValueError(f"Target column '{target}' is required but not found")
    
    # Prepare input/output
    X = df_clean[[text_feature] + categorical_features + numerical_features]
    y = df_clean[target]
    
    # Remove rows with missing values
    missing_mask = X.isnull().any(axis=1) | y.isnull()
    X = X[~missing_mask]
    y = y[~missing_mask]
    
    logger.info(f"Final dataset shape: {X.shape}")
    logger.info(f"Greenwashing rate: {y.mean():.2%}")
    
    return X, y

def train_logistic_regression_model(data_path: str, output_dir: str = "models") -> Dict[str, Any]:
    """
    Train Logistic Regression model with exact specifications.
    
    Args:
        data_path: Path to the cleaned data
        output_dir: Directory to save models
        
    Returns:
        Dictionary with training results
    """
    logger.info("Training Logistic Regression model with exact specifications...")
    
    # Load and prepare data
    X, y = load_and_prepare_data(data_path)
    
    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=RANDOM_SEED
    )
    
    logger.info(f"Training set: {len(X_train)} samples")
    logger.info(f"Test set: {len(X_test)} samples")
    
    # Define features exactly as specified
    text_feature = "esg_claim_text"
    categorical_features = ["claim_category", "project_location", "claimed_metric_type"]
    numerical_features = ["claimed_value", "report_year", "report_sentiment_score", "claim_length"]
    
    # Create preprocessing pipeline exactly as specified
    preprocessor = ColumnTransformer(transformers=[
        ("text", Pipeline([
            ("tfidf", TfidfVectorizer(max_features=1000, ngram_range=(1, 2), stop_words="english")),
            ("svd", TruncatedSVD(n_components=100, random_state=RANDOM_SEED))
        ]), text_feature),
        
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("num", StandardScaler(), numerical_features)
    ])
    
    # Create full pipeline exactly as specified
    logreg_pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear"))
    ])
    
    # Define hyperparameter grid exactly as specified
    param_grid = {
        "classifier__C": [0.01, 0.1, 1, 10],
        "classifier__penalty": ["l1", "l2"]
    }
    
    # Define cross-validation exactly as specified
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    scoring = "roc_auc"
    
    # Perform grid search
    logger.info("Performing hyperparameter tuning...")
    grid_search = GridSearchCV(
        estimator=logreg_pipe,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        verbose=1
    )
    
    # Fit the grid search
    grid_search.fit(X_train, y_train)
    
    # Get best model
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_cv_score = grid_search.best_score_
    
    logger.info(f"Best parameters: {best_params}")
    logger.info(f"Best CV AUC: {best_cv_score:.4f}")
    logger.info(f"Best CV Gini: {2 * best_cv_score - 1:.4f}")
    
    # Evaluate on test set
    y_test_pred = best_model.predict(X_test)
    y_test_proba = best_model.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, y_test_proba)
    test_gini = 2 * test_auc - 1
    
    logger.info(f"Test AUC: {test_auc:.4f}")
    logger.info(f"Test Gini: {test_gini:.4f}")
    
    # Generate detailed metrics
    test_report = classification_report(y_test, y_test_pred, output_dict=True)
    test_cm = confusion_matrix(y_test, y_test_pred)
    
    # Save model
    model_path = os.path.join(output_dir, "logistic_regression_greenwashing.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(best_model, f)
    
    logger.info(f"Model saved to {model_path}")
    
    # Save metrics
    metrics = {
        "best_parameters": best_params,
        "cv_auc": float(best_cv_score),
        "cv_gini": float(2 * best_cv_score - 1),
        "test_auc": float(test_auc),
        "test_gini": float(test_gini),
        "classification_report": test_report,
        "confusion_matrix": test_cm.tolist(),
        "feature_columns": [text_feature] + categorical_features + numerical_features,
        "target_column": "greenwashing_flag"
    }
    
    metrics_path = os.path.join(output_dir.replace('models', 'metrics'), "logistic_regression_metrics.json")
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    logger.info(f"Metrics saved to {metrics_path}")
    
    # Create visualizations
    create_visualizations(y_test, y_test_pred, y_test_proba, output_dir)
    
    results = {
        "model_path": model_path,
        "metrics": metrics,
        "best_model": best_model
    }
    
    logger.info("Logistic Regression model training completed successfully")
    return results

def create_visualizations(y_true, y_pred, y_proba, output_dir: str):
    """Create visualization plots for the model results."""
    logger.info("Creating visualizations...")
    
    # Create figures directory
    figures_dir = os.path.join(output_dir.replace('models', 'reports/figures'))
    os.makedirs(figures_dir, exist_ok=True)
    
    # 1. Confusion Matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Legitimate', 'Greenwashing'],
                yticklabels=['Legitimate', 'Greenwashing'])
    plt.title('Logistic Regression - Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'logistic_regression_confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. ROC Curve
    from sklearn.metrics import roc_curve
    plt.figure(figsize=(8, 6))
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc_score = roc_auc_score(y_true, y_proba)
    
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_score:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Logistic Regression - ROC Curve')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'logistic_regression_roc_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Visualizations saved to {figures_dir}")

def main():
    """Main function to run the Logistic Regression training pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Logistic Regression Training Pipeline')
    parser.add_argument('--data', default='data/clean_claims.parquet', help='Path to cleaned data')
    parser.add_argument('--output', default='models', help='Output directory for models')
    
    args = parser.parse_args()
    
    # Create output directories
    os.makedirs(args.output, exist_ok=True)
    os.makedirs(args.output.replace('models', 'metrics'), exist_ok=True)
    os.makedirs(args.output.replace('models', 'reports/figures'), exist_ok=True)
    
    try:
        results = train_logistic_regression_model(args.data, args.output)
        
        # Print results summary
        print("\n=== LOGISTIC REGRESSION TRAINING RESULTS ===")
        print(f"Best Parameters: {results['metrics']['best_parameters']}")
        print(f"CV AUC: {results['metrics']['cv_auc']:.4f}")
        print(f"CV Gini: {results['metrics']['cv_gini']:.4f}")
        print(f"Test AUC: {results['metrics']['test_auc']:.4f}")
        print(f"Test Gini: {results['metrics']['test_gini']:.4f}")
        print(f"Model saved: {results['model_path']}")
        print(f"Metrics saved: {args.output.replace('models', 'metrics')}/logistic_regression_metrics.json")
        
    except Exception as e:
        logger.error(f"Training pipeline failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 