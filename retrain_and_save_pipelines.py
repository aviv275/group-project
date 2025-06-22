#!/usr/bin/env python3
"""
Retrain and save ESG category and greenwashing classifiers as pipelines.
"""

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib

# Load data
DATA_PATH = 'Synthetic_ESG_Greenwashing_Dataset_200_v2.csv'
df = pd.read_csv(DATA_PATH)

# Check columns
print(f"Columns: {df.columns.tolist()}")

# Clean up column names if needed
if 'esg_claim_text' not in df.columns:
    # Try to find the right column
    for col in df.columns:
        if 'claim' in col.lower():
            df.rename(columns={col: 'esg_claim_text'}, inplace=True)

# Category classifier
if 'claim_category' in df.columns:
    print("\nTraining category classifier pipeline...")
    X_cat = df['esg_claim_text'].astype(str)
    y_cat = df['claim_category'].astype(str)
    X_train_cat, X_test_cat, y_train_cat, y_test_cat = train_test_split(X_cat, y_cat, test_size=0.2, random_state=42)
    
    cat_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=1000, stop_words='english')),
        ('clf', LogisticRegression(max_iter=1000, random_state=42))
    ])
    cat_pipeline.fit(X_train_cat, y_train_cat)
    print(f"Category pipeline train accuracy: {cat_pipeline.score(X_train_cat, y_train_cat):.3f}")
    print(f"Category pipeline test accuracy: {cat_pipeline.score(X_test_cat, y_test_cat):.3f}")
    joblib.dump(cat_pipeline, 'models/category_classifier.pkl')
    print("✅ Saved: models/category_classifier.pkl")
else:
    print("❌ 'claim_category' column not found in data.")

# Greenwashing classifier
if 'greenwashing_flag' in df.columns:
    print("\nTraining greenwashing classifier pipeline...")
    X_gw = df['esg_claim_text'].astype(str)
    y_gw = df['greenwashing_flag'].astype(int)
    X_train_gw, X_test_gw, y_train_gw, y_test_gw = train_test_split(X_gw, y_gw, test_size=0.2, random_state=42)
    
    gw_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=1000, stop_words='english')),
        ('clf', LogisticRegression(max_iter=1000, random_state=42))
    ])
    gw_pipeline.fit(X_train_gw, y_train_gw)
    print(f"Greenwashing pipeline train accuracy: {gw_pipeline.score(X_train_gw, y_train_gw):.3f}")
    print(f"Greenwashing pipeline test accuracy: {gw_pipeline.score(X_test_gw, y_test_gw):.3f}")
    joblib.dump(gw_pipeline, 'models/greenwashing_classifier.pkl')
    print("✅ Saved: models/greenwashing_classifier.pkl")
else:
    print("❌ 'greenwashing_flag' column not found in data.")

print("\nAll done!") 