"""
Enhanced Streamlit app for ESG claim analysis with RAG and AI agent.

This app provides an interactive interface for analyzing ESG claims using
machine learning, RAG analysis, and AI agent capabilities.
"""

import streamlit as st
import pandas as pd
import os
import sys
import pickle
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.agent_runner import ESGAgent
from src.data_prep import engineer_features

@st.cache_resource
def load_esg_agent(google_api_key=None):
    """Load the ESG agent with optional API key."""
    try:
        agent = ESGAgent(
            category_model_path="models/category_classifier.pkl",
            greenwash_model_path="models/greenwashing_classifier.pkl",
            google_api_key=google_api_key
        )
        return agent
    except Exception as e:
        st.error(f"Failed to load ESG Agent: {e}")
        return None

def get_risk_color(risk_level):
    """Return color based on risk level."""
    if risk_level == "HIGH":
        return "#FF4B4B"  # Red
    if risk_level == "MEDIUM":
        return "#FFA500"  # Orange
    if risk_level == "LOW":
        return "#28A745"  # Green
    return "#FFFFFF"    # Default

def main():
    st.set_page_config(page_title="ESG Fraud Detection", layout="wide")
    st.title("🌱 ESG Fraud Detection Platform")

    # --- Sidebar for Configuration ---
    st.sidebar.header("Configuration")
    
    # API Key Configuration
    api_key_option = st.sidebar.selectbox(
        "API Key Configuration",
        ["No API Key (Basic Analysis)", "Use Google API Key (Full Analysis)"]
    )
    
    google_api_key = None
    if api_key_option == "Use Google API Key (Full Analysis)":
        google_api_key = st.sidebar.text_input(
            "Google API Key",
            type="password",
            help="Enter your Google API key for Gemini LLM analysis"
        )
        if not google_api_key:
            st.sidebar.warning("Please enter a Google API key for full analysis")
            google_api_key = None
    
    # Load ESG Agent
    agent = load_esg_agent(google_api_key)
    if agent is None:
        st.error("Failed to initialize ESG Agent. Please check your configuration.")
        return

    # Analysis Mode Selection
    analysis_mode = st.sidebar.selectbox(
        "Choose Analysis Mode",
        ["Single Claim Analysis", "Batch File Analysis"]
    )

    # --- Main Content Area ---
    if analysis_mode == "Single Claim Analysis":
        st.header("Single Claim Analysis")
        
        # Show API status
        if google_api_key:
            st.success("✅ Full Analysis Mode: Using Google Gemini for detailed analysis")
        else:
            st.info("ℹ️ Basic Analysis Mode: Using ML models only (no LLM analysis)")
        
        claim_text = st.text_area(
            "Enter ESG Claim Text:", 
            height=150, 
            placeholder="e.g., We achieved 100% carbon neutrality through innovative renewable energy solutions..."
        )
        
        if st.button("Analyze Claim"):
            if claim_text:
                with st.spinner("Analyzing claim..."):
                    try:
                        results = agent.analyze_claim(claim_text)
                        
                        # Display results
                        st.subheader("Analysis Results")
                        
                        # Overall assessment
                        assessment = results.get("overall_assessment", {})
                        risk_level = assessment.get("risk_level", "N/A")
                        risk_score = assessment.get("overall_risk_score", 0.0)
                        fraud_alert = assessment.get("fraud_alert", "N/A")
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            color = get_risk_color(risk_level)
                            st.markdown(f'**Risk Level**')
                            st.markdown(f'<p style="color:{color}; font-size: 2em; font-weight: bold; margin-bottom: -10px;">{risk_level}</p>', unsafe_allow_html=True)
                            st.markdown(f'<p style="color:{color};">Score: {risk_score:.3f}</p>', unsafe_allow_html=True)

                        with col2:
                            color = get_risk_color(risk_level)
                            st.markdown('**Fraud Alert**')
                            st.markdown(f'<p style="color:{color}; font-size: 1.2em; font-weight: bold; white-space: normal;">{fraud_alert}</p>', unsafe_allow_html=True)

                        with col3:
                            model_preds = results.get("model_predictions", {})
                            category = model_preds.get("claim_category", "N/A")
                            st.metric("Claim Category", category)
                        
                        # Detailed breakdown
                        st.subheader("Detailed Breakdown")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("**Model Predictions:**")
                            greenwash_risk = results.get("model_predictions", {}).get("greenwashing_risk", 0.0)
                            st.write(f"- Greenwashing Risk: {greenwash_risk:.3f}")
                            
                            text_analysis = results.get("text_analysis", {})
                            text_risk = text_analysis.get("text_risk_score", 0.0)
                            st.write(f"- Text-based Risk: {text_risk:.3f}")
                        
                        with col2:
                            st.write("**RAG Analysis:**")
                            rag_analysis = results.get("rag_analysis", {})
                            rag_risk = rag_analysis.get("rag_risk_score", 0.0)
                            st.write(f"- RAG System Risk: {rag_risk:.3f}")
                            
                            if google_api_key:
                                rag_text = rag_analysis.get("analysis", "No analysis available")
                                with st.expander("**RAG Analysis Details**", expanded=True):
                                    st.markdown(rag_text)
                            else:
                                st.info("RAG analysis disabled (no API key). Enable by providing a Google API key.")
                        
                        # Recommendations
                        st.subheader("Recommendations")
                        recommendations = results.get("recommendations", ["No recommendations available"])
                        for rec in recommendations:
                            st.write(f"• {rec}")
                            
                    except Exception as e:
                        st.error(f"Analysis failed: {e}")

    elif analysis_mode == "Batch File Analysis":
        st.header("Batch File Analysis")
        
        # Show API status
        if google_api_key:
            st.success("✅ Full Analysis Mode: Using Google Gemini for detailed analysis")
        else:
            st.info("ℹ️ Basic Analysis Mode: Using ML models only (no LLM analysis)")
        
        uploaded_file = st.file_uploader("Upload a CSV file", type="csv")
        
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            st.write("Preview of uploaded data:")
            st.dataframe(df.head())

            if st.button("Run Batch Analysis"):
                with st.spinner("Analyzing batch file..."):
                    try:
                        results_list = []
                        progress_bar = st.progress(0)
                        
                        for idx, row in df.iterrows():
                            claim_text = row.get('esg_claim_text', '')
                            if claim_text:
                                result = agent.analyze_claim(claim_text)
                                results_list.append({
                                    'claim_text': claim_text,
                                    'risk_level': result.get('overall_assessment', {}).get('risk_level', 'N/A'),
                                    'risk_score': result.get('overall_assessment', {}).get('overall_risk_score', 0.0),
                                    'fraud_alert': result.get('overall_assessment', {}).get('fraud_alert', 'N/A'),
                                    'category': result.get('model_predictions', {}).get('claim_category', 'N/A'),
                                    'greenwashing_risk': result.get('model_predictions', {}).get('greenwashing_risk', 0.0),
                                    'rag_risk': result.get('rag_analysis', {}).get('rag_risk_score', 0.0)
                                })
                            
                            current_progress = float(idx + 1) / float(len(df))
                            progress_bar.progress(current_progress)
                        
                        results_df = pd.DataFrame(results_list)
                        st.subheader("Batch Analysis Results")
                        st.dataframe(results_df)
                        
                        # Summary statistics
                        st.subheader("Summary Statistics")
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            avg_risk = results_df['risk_score'].mean()
                            st.metric("Average Risk Score", f"{avg_risk:.3f}")
                        
                        with col2:
                            high_risk_count = len(results_df[results_df['risk_score'] > 0.5])
                            st.metric("High Risk Claims", high_risk_count)
                        
                        with col3:
                            fraud_alerts = len(results_df[results_df['fraud_alert'] == 'HIGH'])
                            st.metric("Fraud Alerts", fraud_alerts)
                            
                    except Exception as e:
                        st.error(f"Batch analysis failed: {e}")

if __name__ == "__main__":
    main() 