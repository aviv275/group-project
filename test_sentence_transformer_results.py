#!/usr/bin/env python3
"""
Minimal test script for validating the tuned sentence transformer model.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
from src.data_prep import engineer_features_sentence_transformer
import pickle
import os

MODEL_PATH = "models/tuned_gradient_boosting_sentence_embeddings.pkl"
DATA_PATH = "data/clean_claims.parquet"

if not os.path.exists(MODEL_PATH):
    print(f"❌ Model not found: {MODEL_PATH}")
    exit(1)
if not os.path.exists(DATA_PATH):
    print(f"❌ Data not found: {DATA_PATH}")
    exit(1)

# Load model
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# Load data
print("Loading data...")
df = pd.read_parquet(DATA_PATH)
df = df.dropna(subset=["greenwashing_flag", "esg_claim_text"])

# Prepare test split (same as in training)
from sklearn.model_selection import train_test_split
X_text = df["esg_claim_text"].values
y = df["greenwashing_flag"].values
_, X_test, _, y_test = train_test_split(
    X_text, y, test_size=0.2, random_state=42, stratify=y
)

df_test = pd.DataFrame({"esg_claim_text": X_test})
X_test_features = engineer_features_sentence_transformer(df_test)

# Predict
print("Evaluating model on test set...")
y_pred = model.predict(X_test_features)
y_pred_proba = model.predict_proba(X_test_features)[:, 1]

# Metrics
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print(f"Accuracy: {acc:.3f}")
print(f"Precision: {prec:.3f}")
print(f"Recall: {rec:.3f}")
print(f"F1 Score: {f1:.3f}")
print(f"ROC-AUC: {roc_auc:.3f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred)) 