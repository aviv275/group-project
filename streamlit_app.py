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
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add src to path - This is no longer needed as the app is in the root
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.agent_runner import ESGAgent
from src.data_prep import engineer_features

# Global variable to track model type
MODEL_TYPE = "advanced"

@st.cache_resource
def load_esg_agent(google_api_key=None, model_type="advanced"):
    """Load the ESG agent with optional API key and model type."""
    global MODEL_TYPE
    MODEL_TYPE = model_type
    
    try:
        if model_type == "advanced":
            # For advanced model, we'll use a wrapper approach
            # Use the existing agent structure but with advanced model path
            advanced_model_path = "models/greenwashing_lr_advanced.pkl"
            category_model_path = "models/tuned_gradient_boosting_sentence_embeddings.pkl"
            
            # Create a wrapper that uses the advanced model
            agent = ESGAgent(
                category_model_path=category_model_path,
                greenwash_gb_model_path=advanced_model_path,  # Use advanced model as GB
                greenwash_lr_model_path=advanced_model_path,  # Use advanced model as LR
                google_api_key=google_api_key
            )
        else:
            # Use both Gradient Boosting and Logistic Regression models (legacy)
            gb_model_path = "models/tuned_gradient_boosting_sentence_embeddings.pkl"
            lr_model_path = "models/logistic_regression_sentence_embeddings.pkl"
            category_model_path = "models/tuned_gradient_boosting_sentence_embeddings.pkl"
            
            agent = ESGAgent(
                category_model_path=category_model_path,
                greenwash_gb_model_path=gb_model_path,
                greenwash_lr_model_path=lr_model_path,
                google_api_key=google_api_key
            )
        return agent
    except Exception as e:
        st.error(f"Failed to load ESG Agent: {e}")
        st.error("This might be due to model compatibility issues. Please ensure all required models are available.")
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
    
    # Model Selection
    model_type = st.sidebar.selectbox(
        "Model Type",
        ["Advanced Logistic Regression (Recommended)", "Legacy Models (GB + LR)"],
        help="Advanced model includes probability calibration and threshold optimization"
    )
    
    # Convert selection to internal format
    if "Advanced" in model_type:
        model_type_internal = "advanced"
        st.sidebar.success("✅ Using Advanced Model: Calibrated probabilities, optimized threshold")
    else:
        model_type_internal = "legacy"
        st.sidebar.info("ℹ️ Using Legacy Models: Gradient Boosting + Logistic Regression")
    
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
    agent = load_esg_agent(google_api_key, model_type_internal)
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
        
        # Show model and API status
        if model_type_internal == "advanced":
            st.success("🚀 Advanced Model Active: Probability calibration, threshold optimization, enhanced features")
        else:
            st.info("📊 Legacy Models Active: Gradient Boosting + Logistic Regression")
            
        if google_api_key:
            st.success("✅ Full Analysis Mode: Using Google Gemini for detailed analysis")
        else:
            st.info("ℹ️ Basic Analysis Mode: Using ML models only (no LLM analysis)")
        
        # --- User Inputs for ALL LR Features ---
        claim_text = st.text_area(
            "Enter ESG Claim Text:", 
            height=150, 
            placeholder="e.g., We achieved 100% carbon neutrality through innovative renewable energy solutions..."
        )
        claim_category = st.selectbox("Claim Category", ["Environmental", "Social", "Governance", "Other"])
        project_location = st.text_input("Project Location", value="USA")
        claimed_metric_type = st.text_input("Claimed Metric Type", value="carbon_offset")
        claimed_value = st.number_input("Claimed Value", value=0.0)
        actual_measured_value = st.number_input("Actual Measured Value", value=0.0, help="Required for advanced model features")
        report_year = st.number_input("Report Year", value=2024, step=1)
        claim_length = len(claim_text.split())
        st.markdown(f"**Claim Length:** {claim_length} words")
        
        if st.button("Analyze Claim"):
            if claim_text:
                with st.spinner("Analyzing claim..."):
                    try:
                        # For advanced model, call it directly
                        if model_type_internal == "advanced":
                            # Import the advanced model
                            from src.model_utils import AdvancedLogisticRegression
                            import pickle
                            from src.rag_utils import create_rag_system
                            
                            # Load the advanced model
                            with open("models/greenwashing_lr_advanced.pkl", "rb") as f:
                                advanced_model = pickle.load(f)
                            
                            # Initialize RAG system once
                            rag_system = create_rag_system(google_api_key=google_api_key)
                            
                            # Create DataFrame with the expected features
                            df_input = pd.DataFrame([{
                                "esg_claim_text": claim_text,
                                "claim_category": claim_category,
                                "project_location": project_location,
                                "claimed_metric_type": claimed_metric_type,
                                "claimed_value": claimed_value,
                                "actual_measured_value": actual_measured_value,
                                "report_year": report_year
                            }])
                            
                            # Add the engineered features that the advanced model expects
                            df_input["abs_value_deviation"] = abs(claimed_value - actual_measured_value)
                            df_input["rel_value_deviation"] = df_input["abs_value_deviation"] / (abs(claimed_value) + 1e-6)
                            
                            # Get prediction
                            risk_score = advanced_model.predict_proba(df_input)[0, 1]
                            threshold = advanced_model.get_threshold()
                            
                            # --- RAG/LLM Analysis ---
                            rag_results = rag_system.analyze_claim(claim_text)
                            
                            # Create results structure
                            results = {
                                "model_predictions": {
                                    "greenwashing_risk_gb": risk_score,
                                    "greenwashing_risk_lr": risk_score,
                                    "greenwashing_risk": risk_score
                                },
                                "overall_assessment": {
                                    "risk_level": "HIGH" if risk_score >= threshold else "LOW",
                                    "overall_risk_score": risk_score,
                                    "fraud_alert": "HIGH" if risk_score >= threshold else "CLEAR"
                                },
                                "rag_analysis": rag_results,
                                "recommendations": [
                                    "Review claim accuracy and supporting evidence.",
                                    "Consider independent verification of metrics."
                                ] if risk_score >= threshold else [
                                    "Claim appears to be well-structured and transparent."
                                ]
                            }
                        else:
                            # Use the agent for legacy models
                            user_features = {
                                "esg_claim_text": claim_text,
                                "claim_category": claim_category,
                                "project_location": project_location,
                                "claimed_metric_type": claimed_metric_type,
                                "claimed_value": claimed_value,
                                "actual_measured_value": actual_measured_value,
                                "report_year": report_year,
                                "claim_length": claim_length
                            }
                            results = agent.analyze_claim(claim_text, user_features=user_features)
                        
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
                            st.metric("Claim Category", claim_category)
                        
                        # Detailed breakdown
                        st.subheader("Detailed Breakdown")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("**Model Predictions:**")
                            model_preds = results.get("model_predictions", {})
                            if model_type_internal == "advanced":
                                advanced_risk = model_preds.get("greenwashing_risk_gb", 0.0)
                                threshold = results.get("overall_assessment", {}).get("threshold", 0.263)  # fallback to 0.263
                                if "threshold" not in results.get("overall_assessment", {}):
                                    # Try to get from model if not present
                                    try:
                                        from src.model_utils import AdvancedLogisticRegression
                                        import pickle
                                        with open("models/greenwashing_lr_advanced.pkl", "rb") as f:
                                            advanced_model = pickle.load(f)
                                        threshold = advanced_model.get_threshold()
                                    except Exception:
                                        threshold = 0.263
                                st.write(f"- **Advanced Model Risk:** {advanced_risk:.3f}")
                                st.write(f"- **Optimal Threshold:** {threshold:.3f}")
                                st.write(f"- **Prediction:** {'GREEN-WASH' if advanced_risk >= threshold else 'Clean'}")
                            else:
                                gb_risk = model_preds.get("greenwashing_risk_gb", 0.0)
                                lr_risk = model_preds.get("greenwashing_risk_lr", 0.0)
                                st.write(f"- **Gradient Boosting Risk:** {gb_risk:.3f}")
                                st.write(f"- **Logistic Regression Risk:** {lr_risk:.3f}")
                        
                        with col2:
                            st.write("**RAG Analysis:**")
                            rag_analysis = results.get("rag_analysis", {})
                            rag_risk = rag_analysis.get("rag_risk_score", 0.0)
                            st.write(f"- **RAG System Risk:** {rag_risk:.3f}")
                            
                            if google_api_key:
                                rag_text = rag_analysis.get("analysis", "No analysis available")
                                with st.expander("**RAG Analysis Details**", expanded=True):
                                    st.markdown(rag_text)
                            else:
                                st.info("RAG analysis disabled (no API key). Enable by providing a Google API key.")
                        
                        # Enhanced features (for advanced model)
                        if model_type_internal == "advanced":
                            st.subheader("Advanced Model Features")
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                abs_deviation = abs(claimed_value - actual_measured_value)
                                rel_deviation = abs_deviation / (abs(claimed_value) + 1e-6)
                                st.metric("Absolute Value Deviation", f"{abs_deviation:.2f}")
                                st.metric("Relative Value Deviation", f"{rel_deviation:.3f}")
                            
                            with col2:
                                st.write("**Feature Importance:**")
                                st.write("• Enhanced discrepancy detection")
                                st.write("• Calibrated probability scores")
                                st.write("• Optimized decision threshold")
                        
                        # Recommendations
                        st.subheader("Recommendations")
                        recommendations = results.get("recommendations", ["No recommendations available"])
                        for rec in recommendations:
                            st.write(f"• {rec}")
                            
                    except Exception as e:
                        st.error(f"Analysis failed: {e}")

    elif analysis_mode == "Batch File Analysis":
        st.header("Batch File Analysis")
        
        # Show model and API status
        if model_type_internal == "advanced":
            st.success("🚀 Advanced Model Active: Probability calibration, threshold optimization, enhanced features")
        else:
            st.info("📊 Legacy Models Active: Gradient Boosting + Logistic Regression")
            
        if google_api_key:
            st.success("✅ Full Analysis Mode: Using Google Gemini for detailed analysis")
        else:
            st.info("ℹ️ Basic Analysis Mode: Using ML models only (no LLM analysis)")
        
        # Sample file download
        st.subheader("📋 Sample Batch File")
        st.write("Download the sample file below to test batch analysis:")
        
        # Create sample data for download
        sample_data = pd.DataFrame({
            'esg_claim_text': [
                "Our company has achieved 100% carbon neutrality through innovative renewable energy solutions.",
                "We are committed to sustainable practices and environmental stewardship.",
                "Our ESG initiatives have resulted in a 50% reduction in emissions while maintaining profitability.",
                "We pledge to achieve net-zero emissions by 2050 through comprehensive sustainability measures.",
                "Our company maintains the highest standards of corporate governance and ethical business practices."
            ],
            'claim_category': ['Environmental', 'Environmental', 'Environmental', 'Environmental', 'Governance'],
            'project_location': ['USA', 'Europe', 'Global', 'Global', 'Global'],
            'claimed_metric_type': ['carbon_offset', 'sustainability', 'emissions_reduction', 'net_zero', 'governance'],
            'claimed_value': [100.0, 0.0, 50.0, 0.0, 0.0],
            'actual_measured_value': [95.0, 0.0, 55.0, 0.0, 0.0],
            'report_year': [2024, 2024, 2024, 2024, 2024]
        })
        
        csv = sample_data.to_csv(index=False)
        st.download_button(
            label="📥 Download Sample CSV",
            data=csv,
            file_name="sample_esg_claims.csv",
            mime="text/csv"
        )
        
        st.write("**Expected CSV Format:**")
        if model_type_internal == "advanced":
            st.code("esg_claim_text,claim_category,project_location,claimed_metric_type,claimed_value,actual_measured_value,report_year\n'Your ESG claim here','Environmental','USA','carbon_offset',100.0,95.0,2024\n'Another ESG claim here','Social','Europe','sustainability',0.0,0.0,2024", language="csv")
        else:
            st.code("esg_claim_text,claim_category,project_location,claimed_metric_type,claimed_value,report_year\n'Your ESG claim here','Environmental','USA','carbon_offset',100.0,2024\n'Another ESG claim here','Social','Europe','sustainability',0.0,2024", language="csv")
        
        # File upload
        st.subheader("📤 Upload Your Batch File")
        uploaded_file = st.file_uploader("Upload a CSV file", type="csv")
        
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                st.write("✅ File uploaded successfully!")
                st.write(f"📊 **File Preview** ({len(df)} rows):")
                st.dataframe(df.head())

                # Check if required column exists
                if 'esg_claim_text' not in df.columns:
                    st.error("❌ Error: CSV file must contain a column named 'esg_claim_text'")
                    st.write("**Available columns:**", list(df.columns))
                else:
                    st.success(f"✅ Found 'esg_claim_text' column with {len(df)} claims")
                    
                    # Analysis options
                    st.subheader("⚙️ Analysis Options")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        max_claims = st.number_input(
                            "Maximum claims to analyze (0 = all)",
                            min_value=0,
                            max_value=len(df),
                            value=min(10, len(df)),
                            help="Limit the number of claims to analyze for testing purposes"
                        )
                    
                    with col2:
                        include_rag = st.checkbox(
                            "Include RAG Analysis",
                            value=bool(google_api_key),
                            disabled=not google_api_key,
                            help="Include detailed RAG analysis (requires Google API key)"
                        )
                    
                    if st.button("🚀 Run Batch Analysis", type="primary"):
                        with st.spinner("Analyzing batch file..."):
                            try:
                                results_list = []
                                progress_bar = st.progress(0)
                                status_text = st.empty()
                                
                                # Limit claims if specified
                                if max_claims > 0:
                                    df_to_analyze = df.head(max_claims)
                                else:
                                    df_to_analyze = df
                                
                                total_claims = len(df_to_analyze)
                        
                                if model_type_internal == "advanced":
                                    from src.model_utils import AdvancedLogisticRegression
                                    import pickle
                                    from src.rag_utils import create_rag_system
                                    # Load the advanced model once
                                    with open("models/greenwashing_lr_advanced.pkl", "rb") as f:
                                        advanced_model = pickle.load(f)
                                    # Initialize RAG system once
                                    rag_system = create_rag_system(google_api_key=google_api_key)
                                
                                for idx, row in df_to_analyze.iterrows():
                                    claim_text = row.get('esg_claim_text', '')
                                    if claim_text:
                                        status_text.text(f"Analyzing claim {idx + 1} of {total_claims}...")
                                        
                                        try:
                                            if model_type_internal == "advanced":
                                                # Prepare input for advanced model
                                                df_input = pd.DataFrame([{
                                                    "esg_claim_text": claim_text,
                                                    "claim_category": row.get('claim_category', 'Environmental'),
                                                    "project_location": row.get('project_location', 'Unknown'),
                                                    "claimed_metric_type": row.get('claimed_metric_type', 'Unknown'),
                                                    "claimed_value": float(row.get('claimed_value', 0.0) or 0.0),
                                                    "actual_measured_value": float(row.get('actual_measured_value', 0.0) or 0.0),
                                                    "report_year": int(row.get('report_year', 2024) or 2024)
                                                }])
                                                df_input["abs_value_deviation"] = abs(df_input["claimed_value"].iloc[0] - df_input["actual_measured_value"].iloc[0])
                                                df_input["rel_value_deviation"] = df_input["abs_value_deviation"] / (abs(df_input["claimed_value"].iloc[0]) + 1e-6)
                                                risk_score = advanced_model.predict_proba(df_input)[0, 1]
                                                threshold = advanced_model.get_threshold()
                                                rag_results = rag_system.analyze_claim(claim_text) if include_rag else {"rag_risk_score": 0.0, "analysis": "RAG not run"}
                                                risk_level = "HIGH" if risk_score >= threshold else "LOW"
                                                fraud_alert = "HIGH" if risk_score >= threshold else "CLEAR"
                                                results_list.append({
                                                    'claim_id': idx + 1,
                                                    'claim_text': claim_text[:100] + "..." if len(claim_text) > 100 else claim_text,
                                                    'risk_level': risk_level,
                                                    'risk_score': risk_score,
                                                    'fraud_alert': fraud_alert,
                                                    'category': row.get('claim_category', 'Environmental'),
                                                    'advanced_risk': risk_score,
                                                    'threshold': threshold,
                                                    'rag_risk': rag_results.get('rag_risk_score', 0.0),
                                                    'rag_analysis': rag_results.get('analysis', ''),
                                                    'analysis_status': 'Success'
                                                })
                                            else:
                                                # Legacy model path (unchanged)
                                                user_features = {
                                                    'esg_claim_text': claim_text,
                                                    'claim_category': row.get('claim_category', 'Environmental'),
                                                    'project_location': row.get('project_location', 'Unknown'),
                                                    'claimed_metric_type': row.get('claimed_metric_type', 'Unknown'),
                                                    'claimed_value': float(row.get('claimed_value', 0.0) or 0.0),
                                                    'actual_measured_value': float(row.get('actual_measured_value', 0.0) or 0.0),
                                                    'report_year': int(row.get('report_year', 2024) or 2024),
                                                    'claim_length': len(claim_text.split())
                                                }
                                                result = agent.analyze_claim(claim_text, user_features=user_features)
                                                model_preds = result.get('model_predictions', {})
                                                gb_risk = model_preds.get('greenwashing_risk_gb', 0.0)
                                                lr_risk = model_preds.get('greenwashing_risk_lr', 0.0)
                                                advanced_risk = 0.0
                                                threshold = 0.0
                                                results_list.append({
                                                    'claim_id': idx + 1,
                                                    'claim_text': claim_text[:100] + "..." if len(claim_text) > 100 else claim_text,
                                                    'risk_level': result.get('overall_assessment', {}).get('risk_level', 'N/A'),
                                                    'risk_score': result.get('overall_assessment', {}).get('overall_risk_score', 0.0),
                                                    'fraud_alert': result.get('overall_assessment', {}).get('fraud_alert', 'N/A'),
                                                    'category': user_features['claim_category'],
                                                    'advanced_risk': advanced_risk,
                                                    'threshold': threshold,
                                                    'rag_risk': result.get('rag_analysis', {}).get('rag_risk_score', 0.0),
                                                    'rag_analysis': result.get('rag_analysis', {}).get('analysis', ''),
                                                    'analysis_status': 'Success'
                                                })
                                        except Exception as e:
                                            if idx == 0:
                                                import traceback
                                                print('Batch analysis error:', traceback.format_exc())
                                            results_list.append({
                                                'claim_id': idx + 1,
                                                'claim_text': claim_text[:100] + "..." if len(claim_text) > 100 else claim_text,
                                                'risk_level': 'ERROR',
                                                'risk_score': 0.0,
                                                'fraud_alert': 'Analysis Failed',
                                                'category': row.get('claim_category', 'N/A'),
                                                'advanced_risk': 0.0,
                                                'threshold': 0.0,
                                                'rag_risk': 0.0,
                                                'rag_analysis': '',
                                                'analysis_status': f'Error: {str(e)}'
                                            })
                                    
                                    current_progress = float(idx + 1) / float(total_claims)
                                    progress_bar.progress(current_progress)
                        
                                status_text.text("✅ Analysis complete!")
                                
                                # Create results dataframe
                                results_df = pd.DataFrame(results_list)
                                
                                # Drop gb_risk and lr_risk columns if advanced model is used
                                if model_type_internal == "advanced":
                                    results_df = results_df.drop(columns=["gb_risk", "lr_risk"], errors="ignore")
                                
                                # Display results
                                st.subheader("📊 Batch Analysis Results")
                        
                                # Summary statistics
                                st.write("**📈 Summary Statistics:**")
                                col1, col2, col3, col4 = st.columns(4)
                        
                                with col1:
                                    avg_risk = results_df['risk_score'].mean()
                                    st.metric("Average Risk Score", f"{avg_risk:.3f}")
                        
                                with col2:
                                    high_risk_count = len(results_df[results_df['risk_score'] > 0.5])
                                    st.metric("High Risk Claims", high_risk_count)
                        
                                with col3:
                                    fraud_alerts = len(results_df[results_df['fraud_alert'] == 'HIGH'])
                                    st.metric("Fraud Alerts", fraud_alerts)
                                
                                with col4:
                                    success_rate = len(results_df[results_df['analysis_status'] == 'Success']) / len(results_df) * 100
                                    st.metric("Success Rate", f"{success_rate:.1f}%")
                                
                                # Risk level distribution
                                st.write("**🎯 Risk Level Distribution:**")
                                risk_dist = results_df['risk_level'].value_counts()
                                st.bar_chart(risk_dist)
                                
                                # Detailed results table
                                st.write("**📋 Detailed Results:**")
                                st.dataframe(results_df, use_container_width=True)
                                
                                # Export results
                                st.subheader("💾 Export Results")
                                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                
                                # CSV export
                                csv_results = results_df.to_csv(index=False)
                                st.download_button(
                                    label="📥 Download Results as CSV",
                                    data=csv_results,
                                    file_name=f"esg_batch_analysis_{timestamp}.csv",
                                    mime="text/csv"
                                )
                                
                                # Excel export
                                try:
                                    import io
                                    buffer = io.BytesIO()
                                    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                                        results_df.to_excel(writer, sheet_name='Analysis_Results', index=False)
                                        
                                        # Add summary sheet
                                        summary_data = {
                                            'Metric': ['Total Claims', 'Average Risk Score', 'High Risk Claims', 'Fraud Alerts', 'Success Rate'],
                                            'Value': [
                                                str(len(results_df)),
                                                f"{avg_risk:.3f}",
                                                str(high_risk_count),
                                                str(fraud_alerts),
                                                f"{success_rate:.1f}%"
                                            ]
                                        }
                                        summary_df = pd.DataFrame(summary_data)
                                        summary_df.to_excel(writer, sheet_name='Summary', index=False)
                                    
                                    buffer.seek(0)
                                    st.download_button(
                                        label="📥 Download Results as Excel",
                                        data=buffer.getvalue(),
                                        file_name=f"esg_batch_analysis_{timestamp}.xlsx",
                                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                                    )
                                except ImportError:
                                    st.info("📝 Install openpyxl for Excel export: `pip install openpyxl`")
                            
                            except Exception as e:
                                st.error(f"❌ Batch analysis failed: {e}")
                                st.write("**Troubleshooting:**")
                                st.write("• Check that your CSV file has the correct format")
                                st.write("• Ensure the 'esg_claim_text' column exists")
                                st.write("• Verify that your models are properly loaded")
                                
            except Exception as e:
                st.error(f"❌ Error reading CSV file: {e}")
                st.write("**Please ensure your file is a valid CSV format.**")

if __name__ == "__main__":
    main() 