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

from data_prep import load_data, clean_data, engineer_features
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
                 greenwash_model_path: str,
                 google_api_key: Optional[str] = None,
                 use_chroma: bool = False):
        """
        Initialize ESG agent.
        
        Args:
            category_model_path: Path to category classification model
            greenwash_model_path: Path to greenwashing detection model
            google_api_key: Google API key for Gemini
            use_chroma: Whether to use ChromaDB for vector store
        """
        self.analyzer = ESGClaimAnalyzer(
            category_model_path=category_model_path,
            greenwash_model_path=greenwash_model_path
        )
        self.rag_analyzer = create_rag_system(use_chroma=use_chroma, google_api_key=google_api_key)
        
        # Initialize LLM if API key provided
        if google_api_key:
            os.environ["GOOGLE_API_KEY"] = google_api_key
            self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-preview-04-17", temperature=0, convert_system_message_to_human=True)
        else:
            self.llm = None
    
    def analyze_claim(self, claim_text: str) -> Dict[str, Any]:
        """
        Orchestrates the analysis of an ESG claim.
        
        This method combines text analysis, model predictions, and RAG analysis
        to create a comprehensive assessment of the claim.
        """
        # --- 1. Text-based Analysis ---
        text_analysis_results = self._analyze_text_features(claim_text)
        
        # --- 2. Model-based Analysis ---
        model_results = self.analyzer.analyze(claim_text)

        # --- 3. RAG Analysis ---
        rag_results = self.rag_analyzer.analyze_claim(claim_text)
        
        # --- 4. Combine and Assess ---
        assessment = self._combine_results(text_analysis_results, model_results, rag_results)
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

    def _combine_results(self, text: Dict, model: Dict, rag: Dict) -> Dict[str, Any]:
        """Combine analysis results into a final assessment."""
        
        # Risk score calculation
        greenwashing_risk = model.get("greenwashing_risk", 0.0)
        text_risk = text.get("text_risk_score", 0.0)
        rag_risk = rag.get("rag_risk_score", 0.0)
        
        # Combine risks - weighted average
        # Assign weights to each component
        weights = {"greenwashing": 0.5, "text": 0.25, "rag": 0.25}
        overall_risk_score = (greenwashing_risk * weights["greenwashing"]) + \
                             (text_risk * weights["text"]) + \
                             (rag_risk * weights["rag"])
                             
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

    def __init__(self, category_model_path: str, greenwash_model_path: str):
        self.category_model = load_model(category_model_path)
        self.greenwash_model = load_model(greenwash_model_path)

    def analyze(self, claim_text: str) -> Dict[str, Any]:
        """Analyzes a single ESG claim."""
        
        if not self.category_model or not self.greenwash_model:
            return {"error": "One or more models are not loaded."}
            
        # Create a single-column DataFrame for feature engineering
        df = pd.DataFrame([{'esg_claim_text': claim_text}])
        
        # Engineer features
        try:
            features = engineer_features(df)
        except Exception as e:
            logger.error(f"Feature engineering failed: {e}")
            return {"error": f"Feature engineering failed: {e}"}

        # Get predictions
        try:
            category_pred = self.category_model.predict(features)[0]
            category_label = self.category_model.classes_[category_pred] if hasattr(self.category_model, 'classes_') else str(category_pred)
            
            greenwash_prob = self.greenwash_model.predict_proba(features)[0][1]
        except Exception as e:
            logger.error(f"Model prediction failed: {e}")
            return {"error": f"Model prediction failed: {e}"}
            
        return {
            "claim_category": category_label,
            "greenwashing_risk": greenwash_prob
        }


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="ESG Claim Analysis Agent")
    parser.add_argument("claim_text", type=str, help="The ESG claim text to analyze.")
    parser.add_argument("--cat_model", type=str, default="models/category_classifier.pkl", help="Path to category classifier model.")
    parser.add_argument("--gw_model", type=str, default="models/greenwashing_classifier.pkl", help="Path to greenwashing classifier model.")
    parser.add_argument("--google_key", type=str, default=None, help="Google API key for Gemini.")
    args = parser.parse_args()

    agent = ESGAgent(
        category_model_path=args.cat_model,
        greenwash_model_path=args.gw_model,
        google_api_key=args.google_key
    )
    
    result = agent.analyze_claim(args.claim_text)
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main() 