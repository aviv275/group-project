# ESG Fraud Detection Platform

A comprehensive AI-powered platform for detecting ESG fraud, greenwashing, and non-execution risk in Transition-Finance projects using Machine Learning, RAG (Retrieval-Augmented Generation), and AI agents.

## 🌟 Features

- **Multi-Model Analysis**: Advanced Logistic Regression (calibrated, threshold-optimized), Baseline (TF-IDF + Logistic Regression), and Transformer (Sentence Transformers) models
- **RAG System**: Real-time regulatory compliance checking against ESG frameworks
- **AI Agent**: LangChain-based agent orchestrating classification, detection, and RAG analysis
- **Interactive Demo**: Streamlit app with real-time analysis and fraud alerts
- **Business Intelligence**: Auto-generated pitch deck and grant opportunities
- **Comprehensive EDA**: Detailed exploratory data analysis and visualizations

## 🆕 New Advanced Model

The platform now includes an **Advanced Logistic Regression model** with significant improvements:

- **Probability Calibration**: Platt scaling for reliable confidence scores
- **Threshold Optimization**: Automatic threshold selection for target recall (default: 80%)
- **Enhanced Feature Engineering**: Absolute and relative value deviation features
- **Group-Balanced Cross-Validation**: Prevents data leakage between related claims
- **Improved Hyperparameter Tuning**: Extended grid search with L1/L2 regularization

See [Advanced Model Guide](docs/ADVANCED_MODEL_GUIDE.md) for detailed documentation.

## 📁 Project Structure

```
esg-fraud-detection/
├── data/                           # Data files
│   ├── Synthetic_ESG_Greenwashing_Dataset_200_v2.csv
│   └── clean_claims.parquet
├── src/                           # Source code
│   ├── data_prep.py              # Data cleaning and feature engineering
│   ├── model_utils.py            # Model training and evaluation (includes AdvancedLogisticRegression)
│   ├── rag_utils.py              # RAG system implementation
│   ├── agent_runner.py           # AI agent orchestrator
│   ├── train_pipeline.py         # Training pipeline CLI
│   ├── train_advanced_logistic_regression.py  # Advanced model training
│   └── run_advanced_model.py     # Advanced model runner
├── notebooks/                     # Jupyter notebooks
│   ├── 01_data_quality.ipynb     # Data audit and cleaning
│   ├── 02_eda.ipynb              # Exploratory data analysis
│   ├── 03_baselines.ipynb        # Baseline model training
│   ├── 04_model_tuning.ipynb     # Transformer model optimization
│   ├── 05_rag_agent.ipynb        # RAG system implementation
│   └── 06_business_plan.ipynb    # Business plan and pitch deck
├── app/                          # Streamlit application
│   └── streamlit_app.py          # Interactive demo
├── models/                       # Trained models
├── metrics/                      # Model performance metrics
├── reports/                      # Analysis reports and figures
├── docs/                         # Documentation
│   └── ADVANCED_MODEL_GUIDE.md   # Advanced model documentation
├── business/                     # Business documents
│   ├── pitch_deck.pptx           # Auto-generated pitch deck
│   └── grant_scan.md             # Grant opportunities
└── requirements.txt              # Python dependencies
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd esg-fraud-detection

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Network Connectivity & Fallback Mechanism

The platform includes a robust fallback mechanism for network connectivity issues:

- **Automatic Fallback**: If sentence transformer models can't be downloaded due to network issues, the system automatically falls back to TF-IDF features
- **Graceful Degradation**: The pipeline continues to work even without internet connectivity
- **Feature Compatibility**: All models work with both sentence embeddings and TF-IDF features

To test the fallback mechanism:
```bash
# Test fallback functionality
python3 test_fallback_mechanism.py
```

**Note**: The system will automatically detect network connectivity and choose the best available feature extraction method.

### 3. Data Preparation

```bash
# Run data quality notebook
jupyter notebook notebooks/01_data_quality.ipynb
# Execute all cells to clean and prepare the data
```

### 4. Train Models

```bash
# Train advanced logistic regression model (recommended)
python3 run_advanced_pipeline.py

# Train all models (advanced + legacy)
python3 run_pipeline.py

# Train only advanced model
python3 run_pipeline.py --advanced-only

# Train only legacy models
python3 run_pipeline.py --legacy-only

# Train with custom parameters
python3 run_advanced_pipeline.py --target-recall 0.85 --skip-data-quality

# Train baseline models (legacy)
python3 src/train_pipeline.py --task both --model baseline

# Train transformer models (requires sentence-transformers)
python3 src/train_pipeline.py --task both --model transformer

# Train both model types
python3 src/train_pipeline.py --task both --model both

# Setup RAG system
python3 src/train_pipeline.py --task rag
```

### 4. Launch Demo

```bash
# Start Streamlit app
streamlit run app/streamlit_app.py
```

## 📊 Complete Workflow

### Step 1: Data Analysis
```bash
# Run all analysis notebooks
jupyter notebook notebooks/01_data_quality.ipynb
jupyter notebook notebooks/02_eda.ipynb
```

### Step 2: Model Training (Choose One)
```bash
# Option A: Advanced Model (Recommended)
python3 run_advanced_pipeline.py

# Option B: All Models (Advanced + Legacy)
python3 run_pipeline.py

# Option C: Legacy Models Only
python3 run_pipeline.py --legacy-only
```

### Step 3: RAG System Setup
```bash
# Setup RAG with ESG corpora
python3 src/train_pipeline.py --task rag
```

### Step 4: AI Agent Testing
```bash
# Test AI agent on sample claims
python3 src/agent_runner.py --input "We will achieve net-zero emissions by 2050"
```

### Step 5: Business Plan Generation
```bash
# Generate pitch deck and business analysis
jupyter notebook notebooks/06_business_plan.ipynb
```

## 🔧 Advanced Usage

### Command Line Interface

#### Training Pipeline
```bash
# Train specific models
python src/train_pipeline.py --task category --model baseline
python src/train_pipeline.py --task greenwash --model transformer

# Custom data and output paths
python src/train_pipeline.py --data path/to/data.csv --output models/ --metrics metrics/
```

#### AI Agent
```bash
# Analyze single claim
python src/agent_runner.py --input "Your ESG claim text here"

# With OpenAI API key for enhanced analysis
python src/agent_runner.py --input "Claim text" --openai-key "your-api-key"

# Save results to file
python src/agent_runner.py --input "Claim text" --output results.json
```

### Streamlit App Features

1. **Single Claim Analysis**: Enter text and get comprehensive analysis
2. **Batch Analysis**: Upload CSV with multiple claims
3. **Model Comparison**: Compare baseline vs transformer models
4. **RAG Analysis**: Regulatory compliance checking
5. **AI Agent**: Full fraud detection with reasoning
6. **Visualizations**: Risk gauges, confusion matrices, performance metrics

### Jupyter Notebooks

#### 01_data_quality.ipynb
- Data schema validation
- Missing value analysis
- Duplicate detection
- Text preprocessing
- Feature engineering
- Data quality metrics

#### 02_eda.ipynb
- Comprehensive data exploration
- Category distribution analysis
- Sentiment analysis
- Temporal trends
- Correlation analysis
- Key insights summary

#### 03_baselines.ipynb
- TF-IDF + Logistic Regression models
- Cross-validation analysis
- ROC curves and confusion matrices
- Feature importance analysis
- Model comparison

#### 04_model_tuning.ipynb
- Sentence Transformer models
- Hyperparameter optimization
- Performance comparison
- Advanced feature engineering
- Model selection

#### 05_rag_agent.ipynb
- RAG system implementation
- Vector database setup
- Regulatory document processing
- Performance evaluation
- Agent integration testing

#### 06_business_plan.ipynb
- Market analysis
- Competitive landscape
- Financial projections
- Go-to-market strategy
- Auto-generated pitch deck

## 📈 Model Performance

### Advanced Logistic Regression Model (Recommended)
- **Greenwashing Detection**: ~82% AUC, calibrated probabilities, optimized threshold
- **Feature Engineering**: Enhanced with value deviation features
- **Probability Calibration**: Platt scaling for reliable confidence scores
- **Threshold Optimization**: Automatic selection for target recall (80%)

### Baseline Models
- **Category Classification**: ~85% accuracy, F1: 0.82
- **Greenwashing Detection**: ~78% accuracy, F1: 0.75

### Transformer Models
- **Category Classification**: ~88% accuracy, F1: 0.85
- **Greenwashing Detection**: ~82% accuracy, F1: 0.79

### RAG System
- **Regulatory Compliance**: Real-time analysis against ESG frameworks
- **Risk Scoring**: 0-1 scale with detailed rationale
- **KPI Extraction**: Automatic identification of relevant metrics

## 🎯 Key Capabilities

### ESG Fraud Detection
- **Greenwashing Detection**: Identify misleading environmental claims
- **Category Classification**: Environmental, Social, Governance, Other
- **Risk Scoring**: Quantified fraud risk assessment
- **Regulatory Compliance**: Check against EU Taxonomy, GRI Standards

### AI Agent Features
- **Multi-Step Analysis**: Classification → Detection → RAG → Fraud Alert
- **Explainable AI**: Detailed reasoning and recommendations
- **Real-Time Processing**: Instant analysis of ESG claims
- **Batch Processing**: Handle multiple claims efficiently

### Business Intelligence
- **Market Analysis**: $40B ESG market with 15% CAGR
- **Competitive Landscape**: Analysis of 8 major competitors
- **Financial Projections**: 5-year revenue and growth forecasts
- **Grant Opportunities**: 10+ relevant funding sources

## 🔑 API Keys (Optional)

For enhanced RAG and AI agent capabilities:

```bash
# Set OpenAI API key
export OPENAI_API_KEY="your-api-key-here"

# Or use in Streamlit app
# Enter API key in the sidebar when using RAG or AI Agent modes
```

## 📋 Requirements

### Core Dependencies
- Python 3.10+
- pandas, numpy, scikit-learn
- matplotlib, seaborn, plotly
- jupyter, streamlit

### ML & AI Dependencies
- sentence-transformers
- transformers, torch
- langchain, langchain-openai
- faiss-cpu, chromadb

### Business Tools
- python-pptx (for pitch deck generation)
- openpyxl (for Excel support)

## 🚨 Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
   ```bash
   pip install -r requirements.txt
   ```

2. **Model Training Fails**: Check data format and paths
   ```bash
   python src/train_pipeline.py --data data/clean_claims.parquet
   ```

3. **Network Connectivity Issues**: The system automatically falls back to TF-IDF features
   ```bash
   # Test fallback mechanism
   python3 test_fallback_mechanism.py
   
   # If sentence transformers fail to download, TF-IDF will be used automatically
   # No manual intervention required
   ```

4. **RAG System Issues**: Verify sentence-transformers installation
   ```bash
   pip install sentence-transformers
   ```

5. **Streamlit App Errors**: Check model files exist
   ```bash
   ls models/
   ```

### Performance Optimization

- **Large Datasets**: Use batch processing in notebooks
- **Memory Issues**: Reduce batch sizes in training
- **GPU Acceleration**: Install torch with CUDA support

## 📞 Support

For issues and questions:
- Check the troubleshooting section above
- Review the error logs in `reports/error_log.md`
- Ensure all dependencies are correctly installed

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

---

**Built with ❤️ for ESG transparency and fraud prevention** 