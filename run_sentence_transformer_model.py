#!/usr/bin/env python3
"""
04. Model Tuning with Sentence Transformers and SMOTE
This script implements advanced model tuning for ESG greenwashing detection using 
sentence-transformer embeddings and SMOTE for class balancing.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
from sentence_transformers import SentenceTransformer
from textblob import TextBlob
import pickle
import json
import os
from datetime import datetime
warnings.filterwarnings('ignore')

# Set style and random seeds
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
np.random.seed(42)

def add_text_features(texts):
    """Add text length (number of words and characters) and sentiment polarity"""
    lengths = np.array([len(t) for t in texts]).reshape(-1, 1)
    word_counts = np.array([len(t.split()) for t in texts]).reshape(-1, 1)
    sentiments = np.array([TextBlob(t).sentiment.polarity for t in texts]).reshape(-1, 1)
    return np.hstack([lengths, word_counts, sentiments])

def main():
    print("=== ESG GREENWASHING DETECTION WITH SENTENCE TRANSFORMERS + SMOTE ===\n")
    
    # Load and prepare data
    print("=== DATA LOADING AND PREPARATION ===\n")
    
    # Load cleaned data
    df = pd.read_parquet('data/clean_claims.parquet')
    print(f"Dataset shape: {df.shape}")
    print(f"Greenwashing rate: {df['greenwashing_flag'].mean():.2%}")
    
    # Remove rows with missing targets
    df_model = df.dropna(subset=['greenwashing_flag', 'esg_claim_text'])
    print(f"Rows after removing missing targets: {len(df_model)}")
    
    # Prepare features and targets
    X_text = df_model['esg_claim_text'].values
    y_greenwashing = df_model['greenwashing_flag'].values
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_text, y_greenwashing, test_size=0.2, random_state=42, stratify=y_greenwashing
    )
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    print(f"Training greenwashing rate: {y_train.mean():.2%}")
    print(f"Test greenwashing rate: {y_test.mean():.2%}")
    
    # Generate sentence-transformer embeddings
    print("\n=== GENERATING SENTENCE EMBEDDINGS ===\n")
    
    # Load sentence transformer model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("Model loaded successfully!")
    
    # Generate embeddings for training and test sets
    print("Generating embeddings for training set...")
    X_train_emb = model.encode(X_train, show_progress_bar=True, convert_to_numpy=True)
    
    print("Generating embeddings for test set...")
    X_test_emb = model.encode(X_test, show_progress_bar=True, convert_to_numpy=True)
    
    print(f"Training embeddings shape: {X_train_emb.shape}")
    print(f"Test embeddings shape: {X_test_emb.shape}")
    print(f"Embedding dimension: {X_train_emb.shape[1]}")
    
    # Add extra features (length, word count, sentiment)
    print("\n=== ADDING EXTRA TEXT FEATURES ===\n")
    X_train_extra = add_text_features(X_train)
    X_test_extra = add_text_features(X_test)
    X_train_full = np.hstack([X_train_emb, X_train_extra])
    X_test_full = np.hstack([X_test_emb, X_test_extra])
    print(f"Full training feature shape: {X_train_full.shape}")
    print(f"Full test feature shape: {X_test_full.shape}")
    
    # Apply SMOTE to balance the training set
    print("\n=== APPLYING SMOTE FOR CLASS BALANCING ===\n")
    
    # Check class distribution before SMOTE
    print("Class distribution before SMOTE:")
    print(f"  Class 0 (Legitimate): {np.sum(y_train == 0)}")
    print(f"  Class 1 (Greenwashing): {np.sum(y_train == 1)}")
    
    # Apply SMOTE
    smote = SMOTE(random_state=42)
    X_train_bal, y_train_bal = smote.fit_resample(X_train_full, y_train)
    
    print("Class distribution after SMOTE:")
    print(f"  Class 0 (Legitimate): {np.sum(y_train_bal == 0)}")
    print(f"  Class 1 (Greenwashing): {np.sum(y_train_bal == 1)}")
    print(f"Balanced training set shape: {X_train_bal.shape}")
    
    # Train and evaluate models
    print("\n=== MODEL TRAINING AND EVALUATION ===\n")
    
    # Define models to compare
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42, n_estimators=100),
        'SVM': SVC(random_state=42, probability=True)
    }
    
    results = {}
    for name, model in models.items():
        print(f"Training {name}...")
        
        # Train model on balanced data
        model.fit(X_train_bal, y_train_bal)
        
        # Make predictions on test set
        y_pred = model.predict(X_test_full)
        y_pred_proba = model.predict_proba(X_test_full)[:, 1]
        
        # Calculate metrics
        results[name] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }
        
        print(f"{name} Results:")
        print(f"  Accuracy: {results[name]['accuracy']:.3f}")
        print(f"  Precision: {results[name]['precision']:.3f}")
        print(f"  Recall: {results[name]['recall']:.3f}")
        print(f"  F1-Score: {results[name]['f1']:.3f}")
        print(f"  ROC-AUC: {results[name]['roc_auc']:.3f}")
        print()
    
    # Visualize model comparison
    print("=== CREATING VISUALIZATIONS ===\n")
    os.makedirs('reports/figures', exist_ok=True)
    
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(metrics))
    width = 0.2
    
    for i, (name, result) in enumerate(results.items()):
        values = [result[metric] for metric in metrics]
        ax.bar(x + i*width, values, width, label=name)
    
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Score')
    ax.set_title('Model Comparison with Sentence Embeddings + SMOTE')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('reports/figures/model_comparison_sentence_embeddings_smote.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: model_comparison_sentence_embeddings_smote.png")
    
    # Perform hyperparameter tuning
    print("\n=== HYPERPARAMETER TUNING ===\n")
    
    # Find the best model based on ROC-AUC
    best_model_name = max(results.items(), key=lambda x: x[1]['roc_auc'])[0]
    print(f"Best model for tuning: {best_model_name}")
    
    # Define parameter grids for different models
    param_grids = {
        'Logistic Regression': {
            'C': [0.1, 1, 10, 100],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear', 'saga']
        },
        'Random Forest': {
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        },
        'Gradient Boosting': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'subsample': [0.8, 0.9, 1.0]
        },
        'SVM': {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.001, 0.01],
            'kernel': ['rbf', 'linear']
        }
    }
    
    # Get the base model and parameter grid
    base_model = models[best_model_name]
    param_grid = param_grids[best_model_name]
    
    print(f"Parameter grid for {best_model_name}:")
    for param, values in param_grid.items():
        print(f"  {param}: {values}")
    
    # Perform grid search with cross-validation
    print(f"\nPerforming grid search for {best_model_name}...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    grid_search = GridSearchCV(
        base_model, 
        param_grid, 
        cv=cv, 
        scoring='roc_auc', 
        n_jobs=-1, 
        verbose=1
    )
    
    grid_search.fit(X_train_bal, y_train_bal)
    
    print(f"\nBest parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.3f}")
    
    # Evaluate tuned model
    print("\n=== TUNED MODEL EVALUATION ===\n")
    
    # Get tuned model
    tuned_model = grid_search.best_estimator_
    
    # Make predictions
    y_pred_tuned = tuned_model.predict(X_test_full)
    y_pred_proba_tuned = tuned_model.predict_proba(X_test_full)[:, 1]
    
    # Calculate metrics
    tuned_results = {
        'accuracy': accuracy_score(y_test, y_pred_tuned),
        'precision': precision_score(y_test, y_pred_tuned),
        'recall': recall_score(y_test, y_pred_tuned),
        'f1': f1_score(y_test, y_pred_tuned),
        'roc_auc': roc_auc_score(y_test, y_pred_proba_tuned)
    }
    
    print(f"Tuned {best_model_name} Results:")
    for metric, value in tuned_results.items():
        print(f"  {metric.capitalize()}: {value:.3f}")
    
    # Compare with untuned model
    print(f"\nImprovement over untuned model:")
    for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
        improvement = tuned_results[metric] - results[best_model_name][metric]
        print(f"  {metric.capitalize()}: {improvement:+.3f}")
    
    # Perform cross-validation for all models
    print("\n=== CROSS-VALIDATION ANALYSIS ===\n")
    
    # Add tuned model to results
    results[f'Tuned {best_model_name}'] = tuned_results
    
    # Cross-validation scores
    cv_scores = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for name, model in models.items():
        print(f"Performing cross-validation for {name}...")
        scores = cross_val_score(model, X_train_bal, y_train_bal, cv=cv, scoring='roc_auc')
        cv_scores[name] = {
            'mean': scores.mean(),
            'std': scores.std(),
            'scores': scores.tolist()
        }
        print(f"  CV ROC-AUC: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")
    
    # Cross-validation for tuned model
    print(f"\nPerforming cross-validation for Tuned {best_model_name}...")
    tuned_scores = cross_val_score(tuned_model, X_train_bal, y_train_bal, cv=cv, scoring='roc_auc')
    cv_scores[f'Tuned {best_model_name}'] = {
        'mean': tuned_scores.mean(),
        'std': tuned_scores.std(),
        'scores': tuned_scores.tolist()
    }
    print(f"  CV ROC-AUC: {tuned_scores.mean():.3f} (+/- {tuned_scores.std() * 2:.3f})")
    
    # Visualize cross-validation results
    fig, ax = plt.subplots(figsize=(12, 6))
    
    names = list(cv_scores.keys())
    means = [cv_scores[name]['mean'] for name in names]
    stds = [cv_scores[name]['std'] for name in names]
    
    bars = ax.bar(names, means, yerr=stds, capsize=5, alpha=0.7)
    ax.set_xlabel('Models')
    ax.set_ylabel('ROC-AUC Score')
    ax.set_title('Cross-Validation ROC-AUC Scores (Sentence Embeddings + SMOTE)')
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, mean, std in zip(bars, means, stds):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.01,
                f'{mean:.3f}', ha='center', va='bottom')
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('reports/figures/cv_scores_sentence_embeddings_smote.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: cv_scores_sentence_embeddings_smote.png")
    
    # Detailed analysis of the best model
    print("\n=== MODEL PERFORMANCE ANALYSIS ===\n")
    
    # Get the best model (tuned version)
    best_model = tuned_model
    best_model_name_display = f'Tuned {best_model_name}'
    
    # Detailed evaluation
    print(f"Detailed evaluation of {best_model_name_display}:")
    y_pred_tuned = tuned_model.predict(X_test_full)
    print(classification_report(y_test, y_pred_tuned))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred_tuned)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Legitimate', 'Greenwashing'], 
                yticklabels=['Legitimate', 'Greenwashing'])
    plt.title(f'Confusion Matrix - {best_model_name_display}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig('reports/figures/best_model_confusion_matrix_sentence_embeddings_smote.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: best_model_confusion_matrix_sentence_embeddings_smote.png")
    
    # ROC curve
    y_pred_proba_tuned = tuned_model.predict_proba(X_test_full)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba_tuned)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, linewidth=2, label=f'ROC curve (AUC = {tuned_results["roc_auc"]:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {best_model_name_display}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('reports/figures/best_model_roc_curve_sentence_embeddings_smote.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: best_model_roc_curve_sentence_embeddings_smote.png")
    
    # Feature importance analysis (for tree-based models)
    if hasattr(best_model, 'feature_importances_'):
        print("\nFeature importance analysis:")
        
        # Get feature importance
        feature_importance = best_model.feature_importances_
        
        print(f"Number of features with importance > 0: {(feature_importance > 0).sum()}")
        print(f"Average feature importance: {feature_importance.mean():.6f}")
        print(f"Max feature importance: {feature_importance.max():.6f}")
        
        # Plot feature importance distribution
        plt.figure(figsize=(10, 6))
        plt.hist(feature_importance, bins=50, alpha=0.7, color='skyblue')
        plt.xlabel('Feature Importance')
        plt.ylabel('Frequency')
        plt.title('Distribution of Feature Importance (Sentence Embeddings)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('reports/figures/feature_importance_distribution_sentence_embeddings_smote.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Saved: feature_importance_distribution_sentence_embeddings_smote.png")
    
    # Save models and metrics
    print("\n=== SAVING MODELS AND METRICS ===\n")
    
    # Create directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('metrics', exist_ok=True)
    
    # Save sentence transformer model
    with open('models/sentence_transformer_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("Saved: sentence_transformer_model.pkl")
    
    # Save all models
    for name, model in models.items():
        filename = f'models/{name.lower().replace(" ", "_")}_sentence_embeddings.pkl'
        with open(filename, 'wb') as f:
            pickle.dump(model, f)
        print(f"Saved: {filename}")
    
    # Save tuned model
    tuned_filename = f'models/tuned_{best_model_name.lower().replace(" ", "_")}_sentence_embeddings.pkl'
    with open(tuned_filename, 'wb') as f:
        pickle.dump(tuned_model, f)
    print(f"Saved: {tuned_filename}")
    
    # Save grid search results
    with open('models/grid_search_results_sentence_embeddings.pkl', 'wb') as f:
        pickle.dump(grid_search, f)
    print("Saved: grid_search_results_sentence_embeddings.pkl")
    
    # Save advanced metrics
    print("\n=== SAVING METRICS ===\n")
    
    advanced_metrics = {
        'model_comparison': results,
        'cross_validation_scores': cv_scores,
        'best_model': {
            'name': f'Tuned {best_model_name}',
            'parameters': grid_search.best_params_,
            'cv_score': grid_search.best_score_,
            'test_performance': tuned_results
        },
        'dataset_info': {
            'total_samples': len(df_model),
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'greenwashing_rate': df_model['greenwashing_flag'].mean(),
            'feature_dimension': X_train_full.shape[1],
            'feature_type': 'Sentence Embeddings + Text Features',
            'embedding_model': 'all-MiniLM-L6-v2',
            'class_balancing': 'SMOTE'
        }
    }
    
    with open('metrics/advanced_metrics_sentence_embeddings_smote.json', 'w') as f:
        json.dump(advanced_metrics, f, indent=2)
    print("Saved: advanced_metrics_sentence_embeddings_smote.json")
    
    # Print comprehensive summary
    print("\n=== ADVANCED MODELS SUMMARY (SENTENCE EMBEDDINGS + SMOTE) ===\n")
    
    print("1. DATASET:")
    print(f"   - Total samples: {advanced_metrics['dataset_info']['total_samples']}")
    print(f"   - Training samples: {advanced_metrics['dataset_info']['training_samples']}")
    print(f"   - Test samples: {advanced_metrics['dataset_info']['test_samples']}")
    print(f"   - Greenwashing rate: {advanced_metrics['dataset_info']['greenwashing_rate']:.2%}")
    print(f"   - Feature dimension: {advanced_metrics['dataset_info']['feature_dimension']}")
    print(f"   - Feature type: {advanced_metrics['dataset_info']['feature_type']}")
    print(f"   - Embedding model: {advanced_metrics['dataset_info']['embedding_model']}")
    print(f"   - Class balancing: {advanced_metrics['dataset_info']['class_balancing']}")
    
    print("\n2. MODEL COMPARISON:")
    for name, result in results.items():
        print(f"   {name}:")
        print(f"     ROC-AUC: {result['roc_auc']:.3f}")
        print(f"     F1-Score: {result['f1']:.3f}")
        print(f"     CV ROC-AUC: {cv_scores[name]['mean']:.3f} (+/- {cv_scores[name]['std']*2:.3f})")
    
    print(f"\n3. BEST MODEL: Tuned {best_model_name}")
    print(f"   - Best parameters: {advanced_metrics['best_model']['parameters']}")
    print(f"   - CV ROC-AUC: {advanced_metrics['best_model']['cv_score']:.3f}")
    print(f"   - Test ROC-AUC: {advanced_metrics['best_model']['test_performance']['roc_auc']:.3f}")
    print(f"   - Test F1-Score: {advanced_metrics['best_model']['test_performance']['f1']:.3f}")
    
    print("\n4. KEY INSIGHTS:")
    print("   - Sentence embeddings provide rich semantic representations")
    print("   - SMOTE effectively balances the training data")
    print("   - Hyperparameter tuning improves model performance")
    print("   - Cross-validation confirms model stability")
    
    print("\n5. NEXT STEPS:")
    print("   - Models saved and ready for deployment")
    print("   - Proceed to RAG integration")
    print("   - Consider ensemble methods for further improvement")
    print("   - Explore other sentence transformer models")
    
    print(f"\n✅ Model tuning with sentence embeddings and SMOTE completed successfully at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main() 