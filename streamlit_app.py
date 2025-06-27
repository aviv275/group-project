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

@st.cache_resource
def load_esg_agent(google_api_key=None):
    """Load the ESG agent with optional API key."""
    try:
        # Use both Gradient Boosting and Logistic Regression models
        gb_model_path = "models/tuned_gradient_boosting_sentence_embeddings.pkl"
        lr_model_path = "models/logistic_regression_sentence_embeddings.pkl"
        category_model_path = "models/tuned_gradient_boosting_sentence_embeddings.pkl"  # Use GB for category too
        
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
        report_year = st.number_input("Report Year", value=2024, step=1)
        claim_length = len(claim_text.split())
        st.markdown(f"**Claim Length:** {claim_length} words")
        
        if st.button("Analyze Claim"):
            if claim_text:
                with st.spinner("Analyzing claim..."):
                    try:
                        # Pass all user inputs to the agent
                        user_features = {
                            "esg_claim_text": claim_text,
                            "claim_category": claim_category,
                            "project_location": project_location,
                            "claimed_metric_type": claimed_metric_type,
                            "claimed_value": claimed_value,
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
                            
                            # Show both model scores
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
                        
                                for idx, row in df_to_analyze.iterrows():
                                    claim_text = row.get('esg_claim_text', '')
                                    if claim_text:
                                        status_text.text(f"Analyzing claim {idx + 1} of {total_claims}...")
                                        
                                        try:
                                            # Extract user features from CSV row
                                            user_features = {
                                                'esg_claim_text': claim_text,
                                                'claim_category': row.get('claim_category', 'Environmental'),
                                                'project_location': row.get('project_location', 'Unknown'),
                                                'claimed_metric_type': row.get('claimed_metric_type', 'Unknown'),
                                                'claimed_value': float(row.get('claimed_value', 0.0) or 0.0),
                                                'report_year': int(row.get('report_year', 2024) or 2024),
                                                'claim_length': len(claim_text.split())
                                            }
                                            
                                            result = agent.analyze_claim(claim_text, user_features=user_features)
                                            model_preds = result.get('model_predictions', {})
                                            results_list.append({
                                                'claim_id': idx + 1,
                                                'claim_text': claim_text[:100] + "..." if len(claim_text) > 100 else claim_text,
                                                'risk_level': result.get('overall_assessment', {}).get('risk_level', 'N/A'),
                                                'risk_score': result.get('overall_assessment', {}).get('overall_risk_score', 0.0),
                                                'fraud_alert': result.get('overall_assessment', {}).get('fraud_alert', 'N/A'),
                                                'category': user_features['claim_category'],
                                                'gb_risk': model_preds.get('greenwashing_risk_gb', 0.0),
                                                'lr_risk': model_preds.get('greenwashing_risk_lr', 0.0),
                                                'rag_risk': result.get('rag_analysis', {}).get('rag_risk_score', 0.0),
                                                'analysis_status': 'Success'
                                            })
                                        except Exception as e:
                                            results_list.append({
                                                'claim_id': idx + 1,
                                                'claim_text': claim_text[:100] + "..." if len(claim_text) > 100 else claim_text,
                                                'risk_level': 'ERROR',
                                                'risk_score': 0.0,
                                                'fraud_alert': 'Analysis Failed',
                                                'category': 'N/A',
                                                'gb_risk': 0.0,
                                                'lr_risk': 0.0,
                                                'rag_risk': 0.0,
                                                'analysis_status': f'Error: {str(e)[:50]}'
                                            })
                                    
                                    current_progress = float(idx + 1) / float(total_claims)
                                    progress_bar.progress(current_progress)
                        
                                status_text.text("✅ Analysis complete!")
                                
                                # Create results dataframe
                                results_df = pd.DataFrame(results_list)
                                
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
                                                len(results_df),
                                                f"{avg_risk:.3f}",
                                                high_risk_count,
                                                fraud_alerts,
                                                f"{success_rate:.1f}%"
                                            ]
                                        }
                                        summary_df = pd.DataFrame(summary_data)
                                        summary_df.to_excel(writer, sheet_name='Summary', index=False)
                                    
                                    buffer.seek(0)
                                    st.download_button(
                                        label="📥 Download Results as Excel",
                                        data=buffer,
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