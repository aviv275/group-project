#!/usr/bin/env python3
"""
Complete ESG Fraud Detection Pipeline Runner

This script runs the entire pipeline from data quality analysis to model training to testing.
"""

import os
import sys
import subprocess
import time
from pathlib import Path
import argparse
import logging

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"🔄 {description}")
    print(f"{'='*60}")
    print(f"Running: {command}")
    
    start_time = time.time()
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        end_time = time.time()
        print(f"✅ {description} completed successfully in {end_time - start_time:.2f} seconds")
        if result.stdout:
            print("Output:")
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed with error code {e.returncode}")
        if e.stdout:
            print("STDOUT:")
            print(e.stdout)
        if e.stderr:
            print("STDERR:")
            print(e.stderr)
        return False

def main():
    """Run the complete ESG fraud detection pipeline."""
    
    print("🚀 ESG FRAUD DETECTION PIPELINE")
    print("=" * 60)
    
    # Check if we're in the right directory
    if not os.path.exists('data/Synthetic_ESG_Greenwashing_Dataset_500_flag80_v3.csv'):
        print("❌ Error: ESG dataset not found. Please run this script from the project root directory.")
        return False
    
    # Create necessary directories
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('reports', exist_ok=True)
    os.makedirs('reports/figures', exist_ok=True)
    os.makedirs('metrics', exist_ok=True)
    
    parser = argparse.ArgumentParser(description="ESG Fraud Detection Full Pipeline")
    parser.add_argument("--google_key", help="Google API Key for Gemini models.")
    parser.add_argument("--fast", action="store_true", help="Fast mode - skip hyperparameter tuning")
    args = parser.parse_args()
    
    # Add fast mode flag to training command if requested
    training_command = 'python3 src/run_sentence_transformer_model.py'
    if args.fast:
        training_command += ' --fast'
    
    pipeline_steps = [
        {
            'command': 'python3 run_data_quality.py',
            'description': 'Data Quality Analysis and Cleaning'
        },
        {
            'command': training_command,
            'description': 'Model Training (Sentence Transformers + SMOTE)'
        },
        {
            'command': 'python3 tests/test_sentence_transformer_results.py',
            'description': 'Model Validation and Testing'
        }
    ]
    
    # Run each step
    success_count = 0
    for i, step in enumerate(pipeline_steps, 1):
        print(f"\n📋 Step {i}/{len(pipeline_steps)}")
        if run_command(step['command'], step['description']):
            success_count += 1
        else:
            print(f"\n⚠️ Pipeline stopped at step {i}. Check the error above.")
            break
    
    # Summary
    print(f"\n{'='*60}")
    print("📊 PIPELINE SUMMARY")
    print(f"{'='*60}")
    print(f"Steps completed: {success_count}/{len(pipeline_steps)}")
    
    if success_count == len(pipeline_steps):
        print("🎉 All pipeline steps completed successfully!")
        print("\n📁 Generated Files:")
        print("  - data/clean_claims.parquet (cleaned data)")
        print("  - metrics/svm_gridsearch_results.json (SVM model results)")
        print("  - models/ (trained models)")
        print("  - reports/data_quality_metrics.json (quality metrics)")
        print("  - reports/figures/ (visualizations)")
        print("\n🚀 Next Steps:")
        print("  - Run 'cd app && streamlit run streamlit_app.py' to launch the web interface")
        print("  - Use 'python3 final_agent_test.py' to test the AI agent")
        print("  - Open notebooks/ for detailed analysis")
        print("\n📊 Model Performance:")
        print("  - Sentence Transformer embeddings (all-MiniLM-L6-v2)")
        print("  - SMOTE for class balancing")
        print("  - SVM with hyperparameter tuning")
        print("  - Expected ROC-AUC: ~0.569")
    else:
        print(f"⚠️ Pipeline completed with {len(pipeline_steps) - success_count} failures")
        print("Check the error messages above and fix any issues.")
    
    return success_count == len(pipeline_steps)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 