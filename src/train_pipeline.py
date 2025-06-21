#!/usr/bin/env python3
"""
Enhanced Training Pipeline for ESG Fraud Detection.

This module provides a comprehensive training pipeline for ESG claim analysis,
including baseline models, transformer models, and RAG system setup.
"""

import argparse
import sys
import os
import logging
import json
import pickle
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_prep import load_data, clean_data, engineer_features
from model_utils import BaselineModel, TransformerModel, evaluate_model, save_model, save_metrics
from rag_utils import create_rag_system

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

def train_baseline_models(data_path: str, output_dir: str = "models") -> Dict[str, Any]:
    """
    Train baseline models for category classification and greenwashing detection.
    
    Args:
        data_path: Path to the cleaned data
        output_dir: Directory to save models
        
    Returns:
        Dictionary with training results
    """
    logger.info("Training baseline models...")
    
    # Load and prepare data
    df = load_data(data_path)
    df_clean = clean_data(df)
    df_features = engineer_features(df_clean)
    
    # Prepare features and targets
    # Only drop the target columns, keep 'esg_claim_text' in X
    X = df_features.drop(['greenwashing_flag', 'claim_category'], axis=1, errors='ignore')
    y_category = pd.Series(df_clean['claim_category'])
    y_greenwash = pd.Series(df_clean['greenwashing_flag'])
    
    results = {}
    
    # Train category classification model
    logger.info("Training category classification model...")
    category_model = BaselineModel(task='category')
    category_model.fit(X, y_category)
    category_metrics = evaluate_model(category_model, X, y_category)
    
    # Save category model
    category_model_path = os.path.join(output_dir, 'category_classifier.pkl')
    save_model(category_model, category_model_path)
    save_metrics(category_metrics, os.path.join(output_dir.replace('models', 'metrics'), 'category_baseline.json'))
    
    results['category'] = {
        'model_path': category_model_path,
        'metrics': category_metrics
    }
    
    # Train greenwashing detection model
    logger.info("Training greenwashing detection model...")
    greenwash_model = BaselineModel(task='greenwash')
    greenwash_model.fit(X, y_greenwash)
    greenwash_metrics = evaluate_model(greenwash_model, X, y_greenwash)
    
    # Save greenwashing model
    greenwash_model_path = os.path.join(output_dir, 'greenwashing_classifier.pkl')
    save_model(greenwash_model, greenwash_model_path)
    save_metrics(greenwash_metrics, os.path.join(output_dir.replace('models', 'metrics'), 'greenwash_baseline.json'))
    
    results['greenwash'] = {
        'model_path': greenwash_model_path,
        'metrics': greenwash_metrics
    }
    
    logger.info("Baseline models training completed")
    return results

def train_transformer_models(data_path: str, output_dir: str = "models") -> Dict[str, Any]:
    """
    Train transformer-based models using sentence transformers.
    
    Args:
        data_path: Path to the cleaned data
        output_dir: Directory to save models
        
    Returns:
        Dictionary with training results
    """
    logger.info("Training transformer models...")
    
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        logger.error("Transformer dependencies not installed. Install with: pip install sentence-transformers")
        return {}
    
    # Load and prepare data
    df = load_data(data_path)
    df_clean = clean_data(df)
    df_features = engineer_features(df_clean)
    
    # Prepare features and targets
    X = df_features.drop(['greenwashing_flag', 'claim_category'], axis=1, errors='ignore')
    y_category = pd.Series(df_clean['claim_category'])
    y_greenwash = pd.Series(df_clean['greenwashing_flag'])
    
    results = {}
    
    # Train category classification model
    logger.info("Training transformer category classification model...")
    category_model = TransformerModel(task='category')
    category_model.fit(X, y_category)
    category_metrics = evaluate_model(category_model, X, y_category)
    
    # Save category model
    category_model_path = os.path.join(output_dir, 'category_transformer.pkl')
    save_model(category_model, category_model_path)
    save_metrics(category_metrics, os.path.join(output_dir.replace('models', 'metrics'), 'category_transformer.json'))
    
    results['category'] = {
        'model_path': category_model_path,
        'metrics': category_metrics
    }
    
    # Train greenwashing detection model
    logger.info("Training transformer greenwashing detection model...")
    greenwash_model = TransformerModel(task='greenwash')
    greenwash_model.fit(X, y_greenwash)
    greenwash_metrics = evaluate_model(greenwash_model, X, y_greenwash)
    
    # Save greenwashing model
    greenwash_model_path = os.path.join(output_dir, 'greenwashing_transformer.pkl')
    save_model(greenwash_model, greenwash_model_path)
    save_metrics(greenwash_metrics, os.path.join(output_dir.replace('models', 'metrics'), 'greenwash_transformer.json'))
    
    results['greenwash'] = {
        'model_path': greenwash_model_path,
        'metrics': greenwash_metrics
    }
    
    logger.info("Transformer models training completed")
    return results

def setup_rag_system(output_dir: str = "models") -> Dict[str, Any]:
    """
    Set up RAG system with ESG corpora.
    
    Args:
        output_dir: Directory to save RAG components
        
    Returns:
        Dictionary with RAG setup results
    """
    logger.info("Setting up RAG system...")
    
    try:
        from rag_utils import create_rag_system
        
        # Create RAG system
        rag_analyzer = create_rag_system(use_chroma=False)
        
        # Save RAG components
        rag_path = os.path.join(output_dir, 'rag_system')
        os.makedirs(rag_path, exist_ok=True)
        
        # Save vector store
        if hasattr(rag_analyzer.vector_store, 'save'):
            rag_analyzer.vector_store.save(os.path.join(rag_path, 'vector_store'))
        
        results = {
            'rag_path': rag_path,
            'status': 'success',
            'message': 'RAG system created successfully'
        }
        
        logger.info("RAG system setup completed")
        return results
        
    except Exception as e:
        logger.error(f"RAG system setup failed: {e}")
        return {
            'status': 'error',
            'message': str(e)
        }

def train_and_evaluate(args):
    # Create output directories
    os.makedirs(args.output, exist_ok=True)
    os.makedirs(args.metrics, exist_ok=True)
    
    try:
        results = {}
        
        # Train baseline models
        if args.model in ['baseline', 'both']:
            if args.task in ['category', 'greenwash', 'both']:
                baseline_results = train_baseline_models(args.data, args.output)
                results['baseline'] = baseline_results
        
        # Train transformer models
        if args.model in ['transformer', 'both']:
            if args.task in ['category', 'greenwash', 'both']:
                transformer_results = train_transformer_models(args.data, args.output)
                results['transformer'] = transformer_results
        
        # Setup RAG system
        if args.task == 'rag' or args.task == 'both':
            rag_results = setup_rag_system(args.output)
            results['rag'] = rag_results
        
        # Print results summary
        print("\n=== TRAINING RESULTS SUMMARY ===")
        
        if 'baseline' in results:
            print("\nBaseline Models:")
            for task, result in results['baseline'].items():
                metrics = result['metrics']
                print(f"  {task.title()}:")
                print(f"    Accuracy: {metrics.get('accuracy', 'N/A'):.3f}")
                print(f"    F1 Score: {metrics.get('f1_score', 'N/A'):.3f}")
                print(f"    Model saved: {result['model_path']}")
        
        if 'transformer' in results:
            print("\nTransformer Models:")
            for task, result in results['transformer'].items():
                metrics = result['metrics']
                print(f"  {task.title()}:")
                print(f"    Accuracy: {metrics.get('accuracy', 'N/A'):.3f}")
                print(f"    F1 Score: {metrics.get('f1_score', 'N/A'):.3f}")
                print(f"    Model saved: {result['model_path']}")
        
        if 'rag' in results:
            print("\nRAG System:")
            print(f"  Status: {results['rag']['status']}")
            print(f"  Message: {results['rag']['message']}")
            if 'rag_path' in results['rag']:
                print(f"  Components saved: {results['rag']['rag_path']}")
        
        # Save overall results
        with open(os.path.join(args.metrics, 'training_results.json'), 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nAll results saved to {args.metrics}/training_results.json")
        
    except Exception as e:
        logger.error(f"Training pipeline failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ESG Fraud Detection Training Pipeline')
    parser.add_argument('--data', default='../data/clean_claims.parquet', help='Path to cleaned data')
    parser.add_argument('--task', choices=['category', 'greenwash', 'both', 'rag'], default='both', 
                       help='Training task')
    parser.add_argument('--model', choices=['baseline', 'transformer', 'both'], default='both',
                       help='Model type to train')
    parser.add_argument('--output', default='../models', help='Output directory for models')
    parser.add_argument('--metrics', default='../metrics', help='Output directory for metrics')
    parser.add_argument("--google_key", type=str, default=None, help="Google API key for Gemini.")
    args = parser.parse_args()
    train_and_evaluate(args) 