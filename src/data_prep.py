"""
Data preparation module for ESG claim analysis.

This module provides functions for cleaning, validating, and feature engineering
of ESG claim data.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Tuple, Dict, Any
import re
from sklearn.preprocessing import LabelEncoder
import logging
import json
from sentence_transformers import SentenceTransformer
from textblob import TextBlob
import pickle
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# At the top of src/data_prep.py, after the imports
EXPECTED_NUMERIC_FEATURES = [
    'report_year', 'claimed_value', 'actual_measured_value', 'value_deviation',
    'external_validation_score', 'controversy_flag', 'report_sentiment_score',
    'llm_claim_consistency_score', 'text_length', 'word_count',
    'deviation_abs', 'deviation_pct', 'avg_score', 'year', 'month',
    'claim_category_encoded', 'claimed_metric_type_encoded', 'project_location_encoded'
]

def load_data(filepath: str) -> pd.DataFrame:
    """
    Load ESG claims data from CSV or Parquet file.
    
    Args:
        filepath: Path to the CSV or Parquet file
        
    Returns:
        DataFrame with ESG claims data
    """
    logger.info(f"Loading data from {filepath}")
    if filepath.endswith('.csv'):
        df = pd.read_csv(filepath)
    elif filepath.endswith('.parquet'):
        df = pd.read_parquet(filepath)
    else:
        raise ValueError(f"Unsupported file format for {filepath}")
    logger.info(f"Loaded {len(df)} rows and {len(df.columns)} columns")
    return df

def validate_schema(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Validate data schema and return validation results.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Dictionary with validation results
    """
    logger.info("Validating data schema...")
    
    validation_results = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values': df.isnull().sum().to_dict(),
        'duplicates': df.duplicated().sum(),
        'data_types': df.dtypes.to_dict()
    }
    
    # Check required columns
    required_columns = [
        'project_id', 'organization_name', 'report_year', 'esg_claim_text',
        'claim_category', 'claimed_metric_type', 'claimed_value', 'measurement_unit',
        'project_location', 'actual_measured_value', 'value_deviation',
        'external_validation_score', 'greenwashing_flag', 'controversy_flag',
        'source_doc_link', 'report_sentiment_score', 'llm_claim_consistency_score',
        'timestamp'
    ]
    
    missing_columns = set(required_columns) - set(df.columns)
    validation_results['missing_columns'] = list(missing_columns)
    
    # Check data types
    expected_types = {
        'project_id': 'object',
        'organization_name': 'object',
        'report_year': 'int64',
        'esg_claim_text': 'object',
        'claim_category': 'object',
        'claimed_metric_type': 'object',
        'claimed_value': 'float64',
        'measurement_unit': 'object',
        'project_location': 'object',
        'actual_measured_value': 'float64',
        'value_deviation': 'float64',
        'external_validation_score': 'float64',
        'greenwashing_flag': 'int64',
        'controversy_flag': 'int64',
        'source_doc_link': 'object',
        'report_sentiment_score': 'float64',
        'llm_claim_consistency_score': 'float64',
        'timestamp': 'object'
    }
    
    type_mismatches = {}
    for col, expected_type in expected_types.items():
        if col in df.columns:
            actual_type = str(df[col].dtype)
            if actual_type != expected_type:
                type_mismatches[col] = {'expected': expected_type, 'actual': actual_type}
    
    validation_results['type_mismatches'] = type_mismatches
    
    logger.info(f"Schema validation complete. Found {len(missing_columns)} missing columns and {len(type_mismatches)} type mismatches")
    return validation_results

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and preprocess the ESG claims data.
    
    Args:
        df: Raw DataFrame
        
    Returns:
        Cleaned DataFrame
    """
    logger.info("Cleaning data...")
    df_clean = df.copy()
    
    # Convert timestamp to datetime
    df_clean['timestamp'] = pd.to_datetime(df_clean['timestamp'])
    
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
        # Remove extra whitespace
        df_clean['esg_claim_text'] = df_clean['esg_claim_text'].str.strip()
    
    # Remove duplicates
    initial_rows = len(df_clean)
    df_clean = df_clean.drop_duplicates()
    removed_duplicates = initial_rows - len(df_clean)
    
    logger.info(f"Data cleaning complete. Removed {removed_duplicates} duplicates")
    return df_clean

def add_text_features(texts):
    """Add text length (number of words and characters) and sentiment polarity"""
    lengths = np.array([len(t) for t in texts]).reshape(-1, 1)
    word_counts = np.array([len(t.split()) for t in texts]).reshape(-1, 1)
    sentiments = np.array([TextBlob(t).sentiment.polarity for t in texts]).reshape(-1, 1)
    return np.hstack([lengths, word_counts, sentiments])

def engineer_features_sentence_transformer(df: pd.DataFrame) -> np.ndarray:
    """
    Engineer features for sentence transformer models.
    
    This function:
    1. Loads the sentence transformer model
    2. Generates embeddings for the text
    3. Adds additional text features
    4. Returns the combined feature array
    
    Args:
        df: DataFrame with 'esg_claim_text' column
        
    Returns:
        numpy array with combined features
    """
    logger.info("Engineering features for sentence transformer models...")
    
    if 'esg_claim_text' not in df.columns:
        raise ValueError("DataFrame must contain 'esg_claim_text' column")
    
    # Get text data
    texts = df['esg_claim_text'].values
    
    # Load sentence transformer model from HuggingFace
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info("Loaded sentence transformer model from HuggingFace")
    except Exception as e:
        logger.error(f"Failed to load sentence transformer model: {e}")
        raise
    
    # Generate embeddings
    logger.info("Generating sentence embeddings...")
    embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    
    # Add extra text features
    logger.info("Adding text features...")
    extra_features = add_text_features(texts)
    
    # Combine features
    combined_features = np.hstack([embeddings, extra_features])
    
    logger.info(f"Feature engineering complete. Shape: {combined_features.shape}")
    return combined_features

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create engineered features for ESG claims data. It ensures that the output DataFrame
    has a consistent set of columns as expected by the trained model.
    """
    logger.info("Engineering features...")
    df_features = df.copy()

    # --- Step 1: Create features from all available source columns ---
    if 'esg_claim_text' in df_features.columns:
        df_features['text_length'] = df_features['esg_claim_text'].str.len().fillna(0)
        df_features['word_count'] = df_features['esg_claim_text'].str.split().str.len().fillna(0)

    if 'value_deviation' in df_features.columns and 'claimed_value' in df_features.columns:
        df_features['deviation_abs'] = abs(df_features['value_deviation']).fillna(0)
        safe_claimed_value = df_features['claimed_value'].copy()
        safe_claimed_value.loc[safe_claimed_value == 0] = np.nan
        df_features['deviation_pct'] = (df_features['value_deviation'] / safe_claimed_value).fillna(0) * 100

    score_cols = ['external_validation_score', 'report_sentiment_score', 'llm_claim_consistency_score']
    existing_score_cols = [col for col in score_cols if col in df_features.columns]
    if existing_score_cols:
        df_features['avg_score'] = df_features[existing_score_cols].mean(axis=1).fillna(0)

    if 'timestamp' in df_features.columns:
        dt_series = pd.to_datetime(df_features['timestamp'], errors='coerce')
        df_features['year'] = dt_series.dt.year.fillna(0)
        df_features['month'] = dt_series.dt.month.fillna(0)

    categorical_columns = ['claim_category', 'claimed_metric_type', 'project_location']
    for col in categorical_columns:
        if col in df_features.columns:
            le = LabelEncoder()
            df_features[f'{col}_encoded'] = le.fit_transform(df_features[col].astype(str))

    # --- Step 2: Fill any remaining missing columns with medians or 0 ---
    is_single_prediction = len(df_features) == 1 and 'claimed_value' not in df_features.columns
    
    # Load medians if available
    medians = {}
    if is_single_prediction:
        try:
            with open('models/feature_medians.json', 'r') as f:
                medians = json.load(f)
        except FileNotFoundError:
            logger.warning("feature_medians.json not found. Filling missing features with 0.")

    # Ensure all expected columns are present
    for col in EXPECTED_NUMERIC_FEATURES:
        if col not in df_features.columns:
            # Use median for single prediction if available, otherwise use 0
            df_features[col] = medians.get(col, 0)
            
    logger.info("Feature engineering complete")
    return df_features

def prepare_model_data(df: pd.DataFrame, task: str) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
    """
    Prepare data for model training.
    
    Args:
        df: DataFrame with features
        task: Either 'category' or 'greenwash'
        
    Returns:
        Tuple of (features, target, metadata)
    """
    logger.info(f"Preparing data for {task} task...")
    
    if task == 'category':
        target = df['claim_category']
        # Remove target and text columns from features
        feature_columns = [col for col in df.columns if col not in 
                          ['claim_category', 'esg_claim_text', 'project_id', 'organization_name', 
                           'source_doc_link', 'timestamp']]
    elif task == 'greenwash':
        target = df['greenwashing_flag']
        # Remove target and text columns from features
        feature_columns = [col for col in df.columns if col not in 
                          ['greenwashing_flag', 'esg_claim_text', 'project_id', 'organization_name', 
                           'source_doc_link', 'timestamp']]
    else:
        raise ValueError("Task must be either 'category' or 'greenwash'")
    
    features = df[feature_columns].copy()
    
    # Handle missing values
    numeric_features = features.select_dtypes(include=[np.number]).columns
    features[numeric_features] = features[numeric_features].fillna(features[numeric_features].median())
    
    categorical_features = features.select_dtypes(include=['object']).columns
    features[categorical_features] = features[categorical_features].fillna('unknown')
    
    metadata = {
        'feature_columns': feature_columns,
        'target_name': target.name,
        'n_samples': len(features),
        'n_features': len(feature_columns),
        'label_encoders': df.attrs.get('label_encoders', {})
    }
    
    logger.info(f"Data preparation complete. {len(features)} samples, {len(feature_columns)} features")
    return features, target, metadata

def save_clean_data(df: pd.DataFrame, filepath: str) -> None:
    """
    Save cleaned data to parquet format.
    
    Args:
        df: Cleaned DataFrame
        filepath: Output file path
    """
    logger.info(f"Saving cleaned data to {filepath}")
    
    # Create a copy to avoid modifying the original DataFrame
    df_to_save = df.copy()
    
    # Temporarily remove non-serializable objects from attrs
    if hasattr(df_to_save, 'attrs') and df_to_save.attrs:
        # Store the original attrs
        original_attrs = df_to_save.attrs.copy()
        
        # Remove non-serializable objects (like LabelEncoder instances)
        serializable_attrs = {}
        for key, value in original_attrs.items():
            try:
                # Test if the value is JSON serializable
                json.dumps(value)
                serializable_attrs[key] = value
            except (TypeError, ValueError):
                logger.warning(f"Skipping non-serializable attribute: {key}")
        
        # Update attrs with only serializable values
        df_to_save.attrs = serializable_attrs
    
    # Save to parquet
    df_to_save.to_parquet(filepath, index=False)
    
    # Restore original attrs if they were modified
    if hasattr(df, 'attrs') and df.attrs:
        df.attrs = original_attrs
    
    logger.info("Data saved successfully")

def main():
    """Main function for data preparation pipeline."""
    # Load data
    df = load_data('data/Synthetic_ESG_Greenwashing_Dataset_200_v2.csv')
    
    # Validate schema
    validation_results = validate_schema(df)
    print("Validation Results:", validation_results)
    
    # Clean data
    df_clean = clean_data(df)
    
    # Engineer features
    df_features = engineer_features(df_clean)
    
    # Save cleaned data
    save_clean_data(df_features, 'data/clean_claims.parquet')
    
    print("Data preparation pipeline completed successfully!")

if __name__ == "__main__":
    main() 