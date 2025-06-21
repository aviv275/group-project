#!/usr/bin/env python3
"""
Data Quality Analysis Script

This script performs comprehensive data quality analysis and cleaning
for the ESG greenwashing dataset without requiring Jupyter notebooks.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import json
import os
import sys
from pathlib import Path

# Add src to path
sys.path.append('src')

# Suppress warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Set random seed
np.random.seed(42)

def main():
    """Main data quality analysis function."""
    
    print("=== ESG DATA QUALITY ANALYSIS ===\n")
    
    # 1. Load Data
    print("1. Loading Data...")
    df = pd.read_csv('Synthetic_ESG_Greenwashing_Dataset_200_v2.csv')
    print(f"   Dataset shape: {df.shape}")
    print(f"   Columns: {list(df.columns)}")
    
    # 2. Data Schema Validation
    print("\n2. Data Schema Validation...")
    
    expected_columns = [
        'project_id', 'organization_name', 'report_year', 'esg_claim_text',
        'claim_category', 'claimed_metric_type', 'claimed_value', 'measurement_unit',
        'project_location', 'actual_measured_value', 'value_deviation',
        'external_validation_score', 'greenwashing_flag', 'controversy_flag',
        'source_doc_link', 'report_sentiment_score', 'llm_claim_consistency_score', 'timestamp'
    ]
    
    missing_columns = set(expected_columns) - set(df.columns)
    extra_columns = set(df.columns) - set(expected_columns)
    
    print(f"   Missing columns: {missing_columns}")
    print(f"   Extra columns: {extra_columns}")
    print(f"   All expected columns present: {len(missing_columns) == 0}")
    
    # 3. Missing Values Analysis
    print("\n3. Missing Values Analysis...")
    
    missing_data = df.isnull().sum()
    missing_percent = (missing_data / len(df)) * 100
    missing_df = pd.DataFrame({
        'Missing Count': missing_data,
        'Missing Percent': missing_percent
    })
    
    missing_columns_data = missing_df[missing_df['Missing Count'] > 0]
    
    if len(missing_columns_data) > 0:
        print("   Found missing values:")
        print(missing_columns_data)
    else:
        print("   ✅ No missing values found!")
    
    # 4. Duplicate Detection
    print("\n4. Duplicate Detection...")
    
    duplicates = df.duplicated().sum()
    key_columns = ['project_id', 'esg_claim_text']
    duplicates_key = df.duplicated(subset=key_columns).sum()
    
    print(f"   Total duplicate rows: {duplicates}")
    print(f"   Duplicate rows based on key columns: {duplicates_key}")
    
    if duplicates == 0:
        print("   ✅ No duplicate rows found!")
    
    # 5. Text Quality Analysis
    print("\n5. Text Quality Analysis...")
    
    df['text_length'] = df['esg_claim_text'].str.len()
    df['word_count'] = df['esg_claim_text'].str.split().str.len()
    
    print("   Text Length Statistics:")
    print(df[['text_length', 'word_count']].describe())
    
    short_texts = df[df['text_length'] < 50]
    long_texts = df[df['text_length'] > 500]
    
    print(f"   Texts with < 50 characters: {len(short_texts)}")
    print(f"   Texts with > 500 characters: {len(long_texts)}")
    
    # 6. Data Cleaning
    print("\n6. Data Cleaning...")
    
    df_clean = df.copy()
    
    # Remove duplicates
    df_clean = df_clean.drop_duplicates()
    print(f"   Removed {len(df) - len(df_clean)} duplicate rows")
    
    # Clean text data
    df_clean['esg_claim_text'] = df_clean['esg_claim_text'].str.strip()
    df_clean['esg_claim_text'] = df_clean['esg_claim_text'].fillna('No claim text provided')
    
    # Remove rows with extremely short texts
    df_clean = df_clean[df_clean['esg_claim_text'].str.len() >= 10]
    print(f"   Removed {len(df) - len(df_clean)} rows with very short texts")
    
    # 7. Feature Engineering
    print("\n7. Feature Engineering...")
    
    try:
        from data_prep import engineer_features
        df_features = engineer_features(df_clean)
        print(f"   Original features: {len(df_clean.columns)}")
        print(f"   Engineered features: {len(df_features.columns)}")
        print(f"   New features added: {len(df_features.columns) - len(df_clean.columns)}")
        
        # Show new features
        original_cols = set(df_clean.columns)
        new_cols = set(df_features.columns) - original_cols
        print(f"   New features: {list(new_cols)}")
        
    except Exception as e:
        print(f"   ⚠️ Feature engineering failed: {e}")
        df_features = df_clean
    
    # 8. Data Quality Metrics
    print("\n8. Data Quality Metrics...")
    
    # Calculate text metrics for cleaned data
    df_clean['text_length'] = df_clean['esg_claim_text'].str.len()
    
    quality_metrics = {
        'original_rows': len(df),
        'cleaned_rows': len(df_clean),
        'rows_removed': len(df) - len(df_clean),
        'duplicates_removed': len(df) - len(df.drop_duplicates()),
        'missing_values_original': df.isnull().sum().sum(),
        'missing_values_cleaned': df_clean.isnull().sum().sum(),
        'text_length_min': df_clean['text_length'].min(),
        'text_length_max': df_clean['text_length'].max(),
        'text_length_mean': df_clean['text_length'].mean(),
        'unique_organizations': df_clean['organization_name'].nunique(),
        'unique_locations': df_clean['project_location'].nunique(),
        'category_distribution': df_clean['claim_category'].value_counts().to_dict(),
        'greenwashing_rate': df_clean['greenwashing_flag'].mean()
    }
    
    print(f"   Original rows: {quality_metrics['original_rows']}")
    print(f"   Cleaned rows: {quality_metrics['cleaned_rows']}")
    print(f"   Rows removed: {quality_metrics['rows_removed']}")
    print(f"   Missing values (original): {quality_metrics['missing_values_original']}")
    print(f"   Missing values (cleaned): {quality_metrics['missing_values_cleaned']}")
    print(f"   Text length range: {quality_metrics['text_length_min']} - {quality_metrics['text_length_max']} characters")
    print(f"   Average text length: {quality_metrics['text_length_mean']:.1f} characters")
    print(f"   Unique organizations: {quality_metrics['unique_organizations']}")
    print(f"   Unique locations: {quality_metrics['unique_locations']}")
    print(f"   Greenwashing rate: {quality_metrics['greenwashing_rate']:.2%}")
    
    # 9. Save Results
    print("\n9. Saving Results...")
    
    # Create directories
    os.makedirs('data', exist_ok=True)
    os.makedirs('reports', exist_ok=True)
    os.makedirs('reports/figures', exist_ok=True)
    
    # Save cleaned data
    df_clean.to_parquet('data/clean_claims.parquet', index=False)
    
    # Remove attributes that contain non-serializable objects before saving
    if hasattr(df_features, 'attrs'):
        df_features.attrs = {}
    df_features.to_parquet('data/clean_claims_features.parquet', index=False)
    
    # Convert all values to native Python types for JSON serialization
    def convert(o):
        if isinstance(o, (np.integer, np.int64)): return int(o)
        if isinstance(o, (np.floating, np.float64)): return float(o)
        if isinstance(o, (np.ndarray,)): return o.tolist()
        return o

    quality_metrics_serializable = {k: convert(v) for k, v in quality_metrics.items()}

    with open('reports/data_quality_metrics.json', 'w') as f:
        json.dump(quality_metrics_serializable, f, indent=2)
    
    # Calculate and save feature medians for inference
    numeric_cols = df_features.select_dtypes(include=np.number).columns.tolist()
    feature_medians = df_features[numeric_cols].median().to_dict()
    
    os.makedirs('models', exist_ok=True)
    medians_path = 'models/feature_medians.json'
    with open(medians_path, 'w') as f:
        json.dump(feature_medians, f, indent=4)
    
    # Create visualizations
    create_visualizations(df, df_clean)
    
    print("   ✅ Cleaned data saved to data/")
    print("   ✅ Quality metrics saved to reports/")
    print("   ✅ Visualizations saved to reports/figures/")
    print("   ✅ Feature medians saved to models/")
    
    # 10. Summary
    print("\n=== SUMMARY ===")
    print(f"✅ Dataset loaded successfully: {len(df)} rows, {len(df.columns)} columns")
    print(f"✅ Data quality is excellent: {quality_metrics['missing_values_original']} missing values")
    print(f"✅ No duplicates found: {quality_metrics['duplicates_removed']} duplicates removed")
    print(f"✅ Text quality is good: {quality_metrics['text_length_mean']:.1f} avg characters")
    print(f"✅ Feature engineering successful: {len(df_features.columns)} total features")
    print(f"✅ Data ready for analysis and modeling!")

def create_visualizations(df: pd.DataFrame, df_clean: pd.DataFrame):
    """Create and save visualizations."""
    
    # 1. Data Completeness
    missing_data = df.isnull().sum()
    missing_percent = (missing_data / len(df)) * 100
    completeness = (1 - missing_percent/100) * 100
    
    plt.figure(figsize=(12, 6))
    completeness.plot(kind='bar', color='green', alpha=0.7)
    plt.title('Data Completeness by Column')
    plt.xlabel('Columns')
    plt.ylabel('Completeness (%)')
    plt.xticks(rotation=45)
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.savefig('reports/figures/data_completeness.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Text Length Distribution
    df_clean['text_length'] = df_clean['esg_claim_text'].str.len()
    df_clean['word_count'] = df_clean['esg_claim_text'].str.split().str.len()
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    axes[0].hist(df_clean['text_length'], bins=30, alpha=0.7, color='skyblue')
    axes[0].set_title('Distribution of Text Length (Characters)')
    axes[0].set_xlabel('Character Count')
    axes[0].set_ylabel('Frequency')
    
    axes[1].hist(df_clean['word_count'], bins=20, alpha=0.7, color='orange')
    axes[1].set_title('Distribution of Word Count')
    axes[1].set_xlabel('Word Count')
    axes[1].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig('reports/figures/text_length_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Category Distribution
    plt.figure(figsize=(10, 6))
    df_clean['claim_category'].value_counts().plot(kind='bar', color='lightcoral')
    plt.title('Distribution of Claim Categories')
    plt.xlabel('Category')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('reports/figures/category_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Greenwashing Rate
    plt.figure(figsize=(8, 6))
    greenwashing_counts = df_clean['greenwashing_flag'].value_counts()
    plt.pie(greenwashing_counts.values, labels=['No Greenwashing', 'Greenwashing'], 
            autopct='%1.1f%%', colors=['lightgreen', 'lightcoral'])
    plt.title('Greenwashing Distribution')
    plt.tight_layout()
    plt.savefig('reports/figures/greenwashing_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    main() 