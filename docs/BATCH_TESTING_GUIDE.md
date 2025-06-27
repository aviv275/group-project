# 🧪 Batch Testing Guide for ESG Fraud Detection

This guide will help you test the batch analysis functionality in your Streamlit app.

## 🚀 Quick Start

### 1. **Launch the Streamlit App**
```bash
streamlit run streamlit_app.py
```

### 2. **Access Batch Analysis**
- Open the app in your browser
- In the sidebar, select **"Batch File Analysis"** mode
- The app will show you a sample file to download

## 📋 Testing Options

### **Option 1: Use the Built-in Sample File**
1. Click **"📥 Download Sample CSV"** in the app
2. This downloads a file with 5 example ESG claims
3. Upload this file back to test the batch analysis

### **Option 2: Use the Provided Sample File**
1. Use the `sample_batch_claims.csv` file in your project
2. This contains 15 diverse ESG claims for comprehensive testing

### **Option 3: Create Your Own Test File**
Create a CSV file with this exact format:
```csv
esg_claim_text
"Your first ESG claim here"
"Your second ESG claim here"
"Another ESG claim for testing"
```

## 🎯 Test Scenarios

### **Basic Testing (No API Key)**
- ✅ Tests ML models only
- ✅ Faster processing
- ✅ No external API dependencies
- ❌ No RAG analysis

### **Full Testing (With Google API Key)**
- ✅ Tests complete pipeline
- ✅ Includes RAG analysis
- ✅ More detailed results
- ❌ Requires API key
- ❌ Slower processing

## 📊 Expected Results

### **Sample Claims Analysis**
The sample file includes claims that should produce various risk levels:

| Claim Type | Expected Risk Level | Notes |
|------------|-------------------|-------|
| Carbon neutrality claims | MEDIUM-HIGH | Often flagged for verification |
| Generic commitments | LOW-MEDIUM | Vague language |
| Specific metrics | LOW-MEDIUM | More credible |
| Governance claims | LOW | Usually legitimate |

### **Output Columns**
Your results will include:
- `claim_id`: Sequential identifier
- `claim_text`: Truncated claim text
- `risk_level`: HIGH/MEDIUM/LOW
- `risk_score`: 0.0-1.0 numerical score
- `fraud_alert`: Risk assessment
- `category`: Claim category
- `greenwashing_risk`: ML model score
- `rag_risk`: RAG analysis score
- `text_risk`: Text analysis score
- `analysis_status`: Success/Error status

## 🔧 Testing Features

### **Analysis Options**
- **Maximum claims**: Limit analysis for testing
- **Include RAG**: Toggle RAG analysis (requires API key)

### **Progress Tracking**
- Real-time progress bar
- Status updates for each claim
- Error handling for failed analyses

### **Results Export**
- **CSV Export**: Download results as CSV
- **Excel Export**: Download with summary sheet (requires openpyxl)

## 🐛 Troubleshooting

### **Common Issues**

#### **"Failed to load ESG Agent"**
- Check that model files exist in `models/` directory
- Ensure all required dependencies are installed

#### **"CSV file must contain 'esg_claim_text' column"**
- Verify your CSV has the exact column name
- Check for extra spaces or special characters

#### **"Analysis failed" for individual claims**
- Check claim text length (very long claims may cause issues)
- Verify claim text is not empty or corrupted

#### **Slow processing**
- Reduce the number of claims for testing
- Use basic mode (no API key) for faster processing

### **Performance Tips**
- Start with 5-10 claims for initial testing
- Use basic mode for quick validation
- Enable RAG only for final testing

## 📈 Interpreting Results

### **Risk Levels**
- **LOW (0.0-0.3)**: Likely legitimate claims
- **MEDIUM (0.3-0.7)**: Requires verification
- **HIGH (0.7-1.0)**: Potential greenwashing

### **Success Metrics**
- **Success Rate**: Should be 100% for valid claims
- **Average Risk Score**: Overall risk assessment
- **High Risk Claims**: Number requiring attention
- **Fraud Alerts**: Critical issues detected

## 🎉 Success Criteria

Your batch testing is successful when:
- ✅ All claims are processed without errors
- ✅ Results show varying risk levels
- ✅ Export functions work correctly
- ✅ Progress tracking is smooth
- ✅ Summary statistics are accurate

## 📝 Next Steps

After successful testing:
1. **Scale up**: Test with larger datasets
2. **Customize**: Modify analysis parameters
3. **Integrate**: Use in production workflows
4. **Monitor**: Track performance over time

---

**Happy Testing! 🚀** 