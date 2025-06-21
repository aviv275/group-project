# ESG Fraud Detection Pipeline Makefile

.PHONY: help install data-quality train test agent web clean all

help:  ## Show this help message
	@echo "ESG Fraud Detection Pipeline Commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install required dependencies
	pip install -r requirements.txt

data-quality:  ## Run data quality analysis and cleaning
	python3 run_data_quality.py

train:  ## Train baseline models
	python3 src/train_pipeline.py --data_path data/clean_claims.parquet --output_dir models

test:  ## Test the AI agent with sample claims
	python3 final_agent_test.py

agent:  ## Run comprehensive agent testing
	python3 final_agent_test.py

web:  ## Launch the Streamlit web interface
	cd app && streamlit run streamlit_app.py

notebook:  ## Convert Python script to Jupyter notebook
	jupytext --to notebook notebooks/01_data_quality_fixed.py

clean:  ## Clean generated files (keep data and models)
	rm -rf reports/figures/*.png
	rm -f reports/data_quality_metrics.json
	rm -f data/clean_claims*.parquet

clean-all:  ## Clean all generated files
	rm -rf data/clean_claims*.parquet
	rm -rf models/*.pkl
	rm -rf reports/figures/*.png
	rm -f reports/data_quality_metrics.json
	rm -f metrics/*.json

all: data-quality train test  ## Run complete pipeline (data quality + training + testing)

pipeline:  ## Run the complete pipeline using the master script
	python3 run_pipeline.py

status:  ## Show project status and file information
	@echo "📊 Project Status:"
	@echo "Dataset: $(shell ls -la Synthetic_ESG_Greenwashing_Dataset_200_v2.csv 2>/dev/null | wc -l | tr -d ' ') files"
	@echo "Cleaned data: $(shell ls -la data/clean_claims*.parquet 2>/dev/null | wc -l | tr -d ' ') files"
	@echo "Trained models: $(shell ls -la models/*.pkl 2>/dev/null | wc -l | tr -d ' ') files"
	@echo "Reports: $(shell ls -la reports/figures/*.png 2>/dev/null | wc -l | tr -d ' ') files"
	@echo "Notebooks: $(shell ls -la notebooks/*.ipynb 2>/dev/null | wc -l | tr -d ' ') files" 