#!/usr/bin/env python3
"""
Final test script for the ESG Agent.

This script initializes the ESGAgent and runs a series of test claims
through it to validate the end-to-end analysis pipeline.
"""

import sys
import os
import json
import argparse

# Add src to path for module imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from agent_runner import ESGAgent

def print_analysis_results(results: dict):
    """Prints the analysis results in a structured format."""
    
    assessment = results.get("overall_assessment", {})
    risk_level = assessment.get("risk_level", "N/A")
    risk_score = assessment.get("overall_risk_score", 0.0)
    alert = assessment.get("fraud_alert", "N/A")
    
    print(f"  Risk Level: {risk_level} (Score: {risk_score:.3f})")
    print(f"  Fraud Alert: {alert}")
    
    print("\n  Breakdown:")
    model_preds = results.get("model_predictions", {})
    print(f"    - Claim Category: {model_preds.get('claim_category', 'N/A')}")
    print(f"    - Greenwashing Risk: {model_preds.get('greenwashing_risk', 0.0):.3f}")
    
    rag_analysis = results.get("rag_analysis", {})
    print(f"    - RAG System Risk: {rag_analysis.get('rag_risk_score', 0.0):.3f}")
    print(f"    - RAG Analysis: {rag_analysis.get('analysis', 'N/A')}")
    
    text_analysis = results.get("text_analysis", {})
    print(f"    - Text-based Risk: {text_analysis.get('text_risk_score', 0.0):.3f}")
    
    print("\n  Recommendations:")
    for rec in results.get("recommendations", ["No recommendations."]):
        print(f"    - {rec}")

def main():
    """Main function to run the agent test."""
    
    parser = argparse.ArgumentParser(description="ESG Claim Analysis Agent Test")
    parser.add_argument("--google_key", help="Google API Key for Gemini models.")
    args = parser.parse_args()

    print("ESG CLAIM ANALYSIS SYSTEM")
    print("=" * 60)

    # --- Initialize the Agent ---
    # The agent will load the models and set up all necessary components.
    google_api_key = args.google_key or os.getenv("GOOGLE_API_KEY") # Check for arg or env var
    
    try:
        agent = ESGAgent(
            category_model_path="models/category_classifier.pkl",
            greenwash_model_path="models/greenwashing_classifier.pkl",
            google_api_key=google_api_key
        )
        if not google_api_key:
            print("\nWARNING: GOOGLE_API_KEY not found. RAG analysis will be disabled.")
            
    except Exception as e:
        print(f"FATAL: Could not initialize ESGAgent. Error: {e}")
        print("Please ensure models exist by running the training pipeline.")
        sys.exit(1)
        
    # --- Test Claims ---
    test_claims = [
        "Our company has achieved 100% carbon neutrality through innovative renewable energy solutions.",
        "We are committed to sustainable practices and environmental stewardship.",
        "Our ESG initiatives have resulted in a 50% reduction in emissions while maintaining profitability.",
        "We pledge to achieve net-zero emissions by 2050 through comprehensive sustainability measures.",
        "Our company maintains the highest standards of corporate governance and ethical business practices."
    ]

    # --- Run Analysis ---
    for i, claim in enumerate(test_claims):
        print(f"\n\n{'=' * 80}")
        print(f"CLAIM {i+1} OF {len(test_claims)}")
        print(f"{'=' * 80}")
        print(f"\nANALYZING CLAIM: {claim}")
        print("-" * 80)
        
        try:
            # The agent's analyze_claim method runs the full pipeline
            results = agent.analyze_claim(claim)
            print_analysis_results(results)
        except Exception as e:
            print(f"\n--- ANALYSIS FAILED for claim {i+1} ---")
            print(f"Error: {e}")
            
    print(f"\n\n{'=' * 80}")
    print("AGENT TEST COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main() 