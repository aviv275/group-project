#!/usr/bin/env python3
"""
AI Agent Runner for ESG Fraud Detection.

This module implements a LangChain-based agent that orchestrates
ESG claim analysis using classification, RAG, and fraud detection.
"""

import argparse
import json
import sys
import os
from typing import Dict, Any, List, Optional
import logging
import pandas as pd
import numpy as np

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_prep import load_data, clean_data, engineer_features, engineer_features_sentence_transformer
from model_utils import load_model
from rag_utils import create_rag_system

# LangChain imports
from langchain_google_genai import ChatGoogleGenerativeAI

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

class ESGClaimClassifier:
    """Classifier for ESG claims."""
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize classifier.
        
        Args:
            model_path: Path to trained model
        """
        self.model = None
        if model_path and os.path.exists(model_path):
            try:
                self.model = load_model(model_path)
                logger.info(f"Loaded classifier from {model_path}")
            except Exception as e:
                logger.warning(f"Failed to load classifier: {e}")
    
    def classify_claim(self, claim_text: str) -> Dict[str, Any]:
        """
        Classify an ESG claim.
        
        Args:
            claim_text: Text of the ESG claim
            
        Returns:
            Classification results
        """
        if self.model is None:
            return {
                "category": "Unknown",
                "confidence": 0.0,
                "error": "No model loaded"
            }
        
        try:
            # Create sample data for prediction
            df = pd.DataFrame([{'esg_claim_text': claim_text}])
            df_features = engineer_features(df)
            
            # Make prediction
            if hasattr(self.model, 'predict'):
                prediction = self.model.predict(df_features)
                probabilities = self.model.predict_proba(df_features)
                
                categories = self.model.classes_ if hasattr(self.model, 'classes_') else ['Environmental', 'Social', 'Governance', 'Other']
                category = categories[prediction[0]]
                confidence = max(probabilities[0])
                
                return {
                    "category": category,
                    "confidence": float(confidence),
                    "probabilities": dict(zip(categories, probabilities[0].tolist()))
                }
            else:
                return {
                    "category": "Unknown",
                    "confidence": 0.0,
                    "error": "Model does not support prediction"
                }
                
        except Exception as e:
            logger.error(f"Classification failed: {e}")
            return {
                "category": "Unknown",
                "confidence": 0.0,
                "error": str(e)
            }

class ESGGreenwashDetector:
    """Greenwashing detection model."""
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize greenwashing detector.
        
        Args:
            model_path: Path to trained model
        """
        self.model = None
        if model_path and os.path.exists(model_path):
            try:
                self.model = load_model(model_path)
                logger.info(f"Loaded greenwashing detector from {model_path}")
            except Exception as e:
                logger.warning(f"Failed to load greenwashing detector: {e}")
    
    def detect_greenwashing(self, claim_text: str) -> Dict[str, Any]:
        """
        Detect potential greenwashing in a claim.
        
        Args:
            claim_text: Text of the ESG claim
            
        Returns:
            Greenwashing detection results
        """
        if self.model is None:
            return {
                "is_greenwashing": False,
                "confidence": 0.0,
                "error": "No model loaded"
            }
        
        try:
            # Create sample data for prediction
            df = pd.DataFrame([{'esg_claim_text': claim_text}])
            df_features = engineer_features(df)
            
            # Make prediction
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(df_features)
                greenwashing_prob = probabilities[0][1]  # Probability of greenwashing
                
                return {
                    "is_greenwashing": greenwashing_prob > 0.5,
                    "confidence": float(greenwashing_prob),
                    "probability": float(greenwashing_prob)
                }
            else:
                return {
                    "is_greenwashing": False,
                    "confidence": 0.0,
                    "error": "Model does not support probability prediction"
                }
                
        except Exception as e:
            logger.error(f"Greenwashing detection failed: {e}")
            return {
                "is_greenwashing": False,
                "confidence": 0.0,
                "error": str(e)
            }

class ESGAgent:
    """Main ESG fraud detection agent."""
    
    def __init__(self, 
                 category_model_path: str,
                 greenwash_gb_model_path: str,
                 greenwash_lr_model_path: str,
                 google_api_key: Optional[str] = None,
                 use_chroma: bool = False):
        """
        Initialize ESG agent.
        
        Args:
            category_model_path: Path to category classification model
            greenwash_gb_model_path: Path to Gradient Boosting greenwashing detection model
            greenwash_lr_model_path: Path to Logistic Regression greenwashing detection model
            google_api_key: Google API key for Gemini
            use_chroma: Whether to use ChromaDB for vector store
        """
        self.analyzer = ESGClaimAnalyzer(
            category_model_path=category_model_path,
            greenwash_gb_model_path=greenwash_gb_model_path,
            greenwash_lr_model_path=greenwash_lr_model_path
        )
        self.rag_analyzer = create_rag_system(use_chroma=use_chroma, google_api_key=google_api_key)
        
        # Initialize LLM if API key provided
        if google_api_key:
            os.environ["GOOGLE_API_KEY"] = google_api_key
            self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-preview-04-17", temperature=0, convert_system_message_to_human=True)
        else:
            self.llm = None
    
    def analyze_claim(self, claim_text: str, user_features: dict = None, use_rag: bool = True) -> Dict[str, Any]:
        """
        Orchestrates the analysis of an ESG claim.
        Optionally accepts user_features for LR model.
        If use_rag is False, RAG risk is set to 0.0 and not included in the risk calculation.
        """
        # --- 1. Text-based Analysis ---
        text_analysis_results = self._analyze_text_features(claim_text)
        
        # --- 2. Model-based Analysis ---
        model_results = self.analyzer.analyze(claim_text, user_features=user_features)

        # --- 3. RAG Analysis ---
        if use_rag:
            rag_results = self.rag_analyzer.analyze_claim(claim_text)
            rag_risk = rag_results.get("rag_risk_score", 0.0)
        else:
            rag_results = {}
            rag_risk = 0.0
        
        # --- 4. Combine and Assess ---
        assessment = self._combine_results(text_analysis_results, model_results, rag_results, rag_risk, use_rag)
        assessment_with_alert = self._determine_fraud_alert(assessment)
        
        # --- 5. Generate Recommendations ---
        recommendations = self._generate_recommendations(assessment_with_alert)
        
        final_result = {
            "text_analysis": text_analysis_results,
            "model_predictions": model_results,
            "rag_analysis": rag_results,
            "overall_assessment": assessment_with_alert,
            "recommendations": recommendations
        }
        
        return final_result

    def _analyze_text_features(self, text: str) -> Dict[str, Any]:
        """Performs basic text analysis to identify red flags."""
        text_lower = text.lower()
        risk_indicators = []
        
        # Indicators
        if "100%" in text or "fully" in text_lower: risk_indicators.append("Extreme claims (100%)")
        if "net zero" in text_lower or "net-zero" in text_lower: risk_indicators.append("Zero claims")
        if "sustainable" in text_lower and "committed" in text_lower: risk_indicators.append("Sustainability claim")
        if "carbon neutral" in text_lower: risk_indicators.append("Carbon neutrality claim")
        if "renewable" in text_lower: risk_indicators.append("Renewable energy mention")
        if not any(char.isdigit() for char in text): risk_indicators.append("Vague language")

        score = len(risk_indicators) * 0.1 + ("100%" in text) * 0.2
        return {
            "text_length": len(text),
            "word_count": len(text.split()),
            "avg_word_length": len(text.replace(' ','')) / len(text.split()) if len(text.split()) > 0 else 0,
            "risk_indicators": risk_indicators,
            "text_risk_score": min(score, 1.0)
        }

    def _combine_results(self, text: Dict, model: Dict, rag: Dict, rag_risk: float, use_rag: bool) -> Dict[str, Any]:
        """Combine analysis results into a final assessment."""
        gb_risk = model.get("greenwashing_risk_gb", 0.0)
        lr_risk = model.get("greenwashing_risk_lr", 0.0)
        text_risk = text.get("text_risk_score", 0.0)
        # If not using RAG, set rag_risk to 0.0 and adjust weights
        if use_rag:
            weights = {"greenwashing": 0.5, "text": 0.25, "rag": 0.25}
        else:
            weights = {"greenwashing": 0.67, "text": 0.33, "rag": 0.0}
        greenwashing_risk = max(gb_risk, lr_risk)
        overall_risk_score = (
            greenwashing_risk * weights["greenwashing"] +
            text_risk * weights["text"] +
            rag_risk * weights["rag"]
        )
        return {
            "overall_risk_score": min(overall_risk_score, 1.0),
        }

    def _determine_fraud_alert(self, assessment: Dict) -> Dict[str, Any]:
        """Determine fraud alert level and message."""
        score = assessment["overall_risk_score"]
        if score > 0.6:
            assessment["risk_level"] = "HIGH"
            assessment["fraud_alert"] = "HIGH RISK - Immediate review required"
        elif score > 0.3:
            assessment["risk_level"] = "MEDIUM"
            assessment["fraud_alert"] = "POTENTIAL RISK - Further investigation recommended"
        else:
            assessment["risk_level"] = "LOW"
            assessment["fraud_alert"] = "CLEAR"
        return assessment

    def _generate_recommendations(self, assessment: Dict) -> List[str]:
        """Generates actionable recommendations based on the assessment."""
        recs = []
        if assessment["risk_level"] == "HIGH":
            recs.append("Urgently verify data with independent third-party sources.")
            recs.append("Cross-reference claims with financial reports and operational data.")
        if assessment["risk_level"] in ["MEDIUM", "HIGH"]:
            recs.append("Request more specific data and clarification from the reporting entity.")
        if "Vague language" in assessment.get("text_analysis", {}).get("risk_indicators", []):
             recs.append("Advise for more quantifiable and specific claims.")
        if "Extreme claims" in assessment.get("text_analysis", {}).get("risk_indicators", []):
             recs.append("Review language for extreme or vague claims.")
        
        if not recs:
            recs.append("Claim appears to be well-structured and transparent.")
            
        return recs


class ESGClaimAnalyzer:
    """Tool for analyzing a single ESG claim using models."""

    def __init__(self, category_model_path: str, greenwash_gb_model_path: str, greenwash_lr_model_path: str):
        self.category_model = load_model(category_model_path) if os.path.exists(category_model_path) else None
        self.greenwash_gb_model = load_model(greenwash_gb_model_path) if os.path.exists(greenwash_gb_model_path) else None
        self.greenwash_lr_model = load_model(greenwash_lr_model_path) if os.path.exists(greenwash_lr_model_path) else None

    def analyze(self, claim_text: str, user_features: dict = None) -> Dict[str, Any]:
        """Analyzes a single ESG claim. Optionally uses user_features for LR model."""
        if not self.greenwash_gb_model and not self.greenwash_lr_model:
            return {"error": "No greenwashing models are loaded."}
        
        df = pd.DataFrame([{'esg_claim_text': claim_text}])
        results = {}
        # Get Gradient Boosting predictions
        if self.greenwash_gb_model:
            try:
                features = engineer_features_sentence_transformer(df)
                gb_prob = self.greenwash_gb_model.predict_proba(features)[0][1]
                results["greenwashing_risk_gb"] = float(gb_prob)
            except Exception as e:
                logger.error(f"Gradient Boosting prediction failed: {e}")
                results["greenwashing_risk_gb"] = 0.0
                results["gb_error"] = str(e)
        else:
            results["greenwashing_risk_gb"] = 0.0
            results["gb_error"] = "Model not loaded"
        # Get Logistic Regression predictions
        if self.greenwash_lr_model:
            try:
                # Use the same sentence transformer features as GB model
                features = engineer_features_sentence_transformer(df)
                lr_prob = self.greenwash_lr_model.predict_proba(features)[0][1]
                results["greenwashing_risk_lr"] = float(lr_prob)
            except Exception as e:
                logger.error(f"Logistic Regression prediction failed: {e}")
                results["greenwashing_risk_lr"] = 0.0
                results["lr_error"] = str(e)
        else:
            results["greenwashing_risk_lr"] = 0.0
            results["lr_error"] = "Model not loaded"
        gb_risk = results.get("greenwashing_risk_gb", 0.0)
        lr_risk = results.get("greenwashing_risk_lr", 0.0)
        results["greenwashing_risk"] = (gb_risk + lr_risk) / 2.0
        return results


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="ESG Claim Analysis Agent")
    parser.add_argument("claim_text", type=str, help="The ESG claim text to analyze.")
    parser.add_argument("--cat_model", type=str, default="models/category_classifier.pkl", help="Path to category classifier model.")
    parser.add_argument("--gb_model", type=str, default="models/tuned_gradient_boosting_sentence_embeddings.pkl", help="Path to Gradient Boosting greenwashing model.")
    parser.add_argument("--lr_model", type=str, default="models/logistic_regression_sentence_embeddings.pkl", help="Path to Logistic Regression greenwashing model.")
    parser.add_argument("--google_key", type=str, default=None, help="Google API key for Gemini.")
    args = parser.parse_args()

    agent = ESGAgent(
        category_model_path=args.cat_model,
        greenwash_gb_model_path=args.gb_model,
        greenwash_lr_model_path=args.lr_model,
        google_api_key=args.google_key
    )
    
    result = agent.analyze_claim(args.claim_text)
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main() 