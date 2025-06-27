# %% [markdown]
# ## **Model Pick**

# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score

# Load dataset
df = pd.read_csv("Synthetic_ESG_Greenwashing_Dataset_500_flag80_v3.csv")

# Feature engineering
df["claim_length"] = df["esg_claim_text"].apply(lambda x: len(str(x).split()))
text_feature = "esg_claim_text"
categorical_features = ["claim_category", "project_location", "claimed_metric_type"]
numerical_features = ["claimed_value", "report_year",'report_sentiment_score']
target = "greenwashing_flag"

# Define input/output
X = df[[text_feature] + categorical_features + numerical_features]
y = df[target]

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

# Preprocessing pipeline
preprocessor = ColumnTransformer(transformers=[
    ("text", Pipeline([
        ("tfidf", TfidfVectorizer(max_features=1000, ngram_range=(1, 2), stop_words="english")),
        ("svd", TruncatedSVD(n_components=100, random_state=42))
    ]), "esg_claim_text"),
    
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ("num", StandardScaler(), numerical_features)
])

# Model configurations
models = {
    "LogisticRegression": LogisticRegression(max_iter=1000, class_weight='balanced'),
    "RandomForest": RandomForestClassifier(n_estimators=100, max_depth=6, min_samples_leaf=5, class_weight='balanced', random_state=42),
    "XGBoost": XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1,
                             reg_alpha=1.0, reg_lambda=5.0, subsample=0.7,
                             colsample_bytree=0.7, use_label_encoder=False,
                             eval_metric='logloss', scale_pos_weight=1, random_state=42),
    "LightGBM": LGBMClassifier(n_estimators=100, max_depth=4, learning_rate=0.1,
                               class_weight='balanced', reg_alpha=1.0, reg_lambda=5.0,
                               subsample=0.7, colsample_bytree=0.7, random_state=42)
}

# Train, predict, evaluate
results = {}
for name, model in models.items():
    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("classifier", model)
    ])
    pipeline.fit(X_train, y_train)
    train_proba = pipeline.predict_proba(X_train)[:, 1]
    test_proba = pipeline.predict_proba(X_test)[:, 1]
    train_auc = roc_auc_score(y_train, train_proba)
    test_auc = roc_auc_score(y_test, test_proba)
    results[name] = {
        "Train AUC": train_auc,
        "Train Gini": 2 * train_auc - 1,
        "Test AUC": test_auc,
        "Test Gini": 2 * test_auc - 1
    }

# Display results
for name, metrics in results.items():
    print(f"\n{name}")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")


# %% [markdown]
# ## **Best Model with HP Tuning**

# %%
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, make_scorer

# Load and prep data
df = pd.read_csv("Synthetic_ESG_Greenwashing_Dataset_500_flag80_v3.csv")
df["claim_length"] = df["esg_claim_text"].apply(lambda x: len(str(x).split()))

text_feature = "esg_claim_text"
categorical_features = ["claim_category", "project_location", "claimed_metric_type"]
numerical_features = ["claimed_value", "report_year",'report_sentiment_score']
target = "greenwashing_flag"

X = df[[text_feature] + categorical_features + numerical_features]
y = df[target]

# Preprocessing pipeline
preprocessor = ColumnTransformer(transformers=[
    ("text", Pipeline([
        ("tfidf", TfidfVectorizer(max_features=1000, ngram_range=(1, 2), stop_words="english")),
        ("svd", TruncatedSVD(n_components=100, random_state=42))
    ]), text_feature),
    
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ("num", StandardScaler(), numerical_features)
])

# Define full pipeline
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear"))
])

# Grid search hyperparameters
param_grid = {
    "classifier__C": [0.01, 0.1, 1, 10],
    "classifier__penalty": ["l1", "l2"]
}

# Stratified CV + AUC as scoring
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=cv,
    scoring="roc_auc",
    n_jobs=-1,
    verbose=1
)

# Run tuning
grid_search.fit(X, y)

# Report
best_model = grid_search.best_estimator_
best_auc = grid_search.best_score_
best_gini = 2 * best_auc - 1

print("Best Params:", grid_search.best_params_)
print(f"CV Mean AUC: {best_auc:.4f}")
print(f"CV Mean Gini: {best_gini:.4f}")


# %%
best_model = grid_search.best_estimator_

# %%
import joblib

# Save best model to file
joblib.dump(best_model, "greenwashing_model.pkl")
print("Model saved as greenwashing_model.pkl")


# %% [markdown]
# ### **Model Input Explanation – Greenwashing Detection Pipeline**
# 
# This document outlines the expected inputs for the `greenwashing_model.pkl` prediction pipeline and what each feature represents.
# 
# 
# ###  **Input Fields**
# 
# ### `esg_claim_text`
# - **Type**: Free-form text  
# - **Description**: The core ESG claim made by an organization.  
# - **Purpose**: Used as textual input for TF-IDF vectorization and dimensionality reduction.  
# - **Example**:  
#   `"In 2023, we reduced Scope 1 and Scope 2 greenhouse gas emissions by 18% compared to our 2020 baseline..."`
# 
# ### `claim_category`
# - **Type**: Categorical  
# - **Values**: `Environmental`, `Social`, `Governance`, `Other`  
# - **Description**: Specifies the ESG domain of the claim.  
# - **Purpose**: Encoded via one-hot encoding to provide context.  
# - **Example**: `"Environmental"`
# 
# ### `project_location`
# - **Type**: Categorical  
# - **Description**: The geographical location of the ESG project or claim.  
# - **Purpose**: May capture regulatory or regional differences in ESG practices.  
# - **Example**: `"USA"`
# 
# ### `claimed_metric_type`
# - **Type**: Categorical  
# - **Description**: The type of ESG metric being reported (e.g., carbon offset, renewable energy use).  
# - **Purpose**: Helps the model understand the metric being claimed.  
# - **Example**: `"carbon_offset"`
# 
# ### `claimed_value`
# - **Type**: Numeric (float)  
# - **Description**: The quantitative value of the claim.  
# - **Purpose**: Provides the magnitude of the reported ESG performance.  
# - **Example**: `18.0` (e.g., 18% reduction)
# 
# ### `report_year`
# - **Type**: Numeric (int)  
# - **Description**: The year the claim was reported.  
# - **Purpose**: May reflect changes over time in regulation, expectations, or context.  
# - **Example**: `2024`
# 
# ### `report_sentiment_score`
# - **Type**: Numeric (float, typically from -1 to 1)  
# - **Description**: Sentiment polarity of the surrounding report text (if available).  
# - **Purpose**: May indicate persuasive tone or overconfidence in ESG statements.  
# - **Example**: `0.5` (neutral)

# %% [markdown]
# ## **Fraud Statement Example**

# %%
import joblib
import pandas as pd

# Load the saved model
model = joblib.load("greenwashing_model.pkl")

# Example input ESG claim text
esg_claim = "We achieved almost net-zero carbon emissions in 2024 across all global operations, without the need for carbon offsets or any operational changes."

# Create input as DataFrame (same structure expected by the pipeline)
df_input = pd.DataFrame({"esg_claim_text": [esg_claim]})

# Add placeholder values for required categorical and numerical fields
df_input["claim_category"] = "Environmental"
df_input["project_location"] = "USA"
df_input["claimed_metric_type"] = "carbon_offset"
df_input["claimed_value"] = 80.0
df_input["report_year"] = 2025
df_input["report_sentiment_score"] = 0.1  # neutral placeholder

# Predict probability and label
proba = model.predict_proba(df_input)[0][1]
label = model.predict(df_input)[0]

# Output
print(f"Fraud probability: {proba:.2f}")
print("Prediction:", "Greenwashing (Fraudulent)" if label == 1 else "Legit")


# %% [markdown]
# ## **Non-Fraud Statement Example**

# %%
import joblib
import pandas as pd

# Load the saved model
model = joblib.load("greenwashing_model.pkl")

# Example input ESG claim text
esg_claim = "In 2023, we reduced Scope 1 and Scope 2 greenhouse gas emissions by 18% compared to our 2020 baseline, through facility energy retrofits and switching to certified renewable electricity"

# Create input as DataFrame (same structure expected by the pipeline)
df_input = pd.DataFrame({"esg_claim_text": [esg_claim]})

# Add placeholder values for required categorical and numerical fields
df_input["claim_category"] = "Environmental"
df_input["project_location"] = "USA"
df_input["claimed_metric_type"] = "carbon_offset"
df_input["claimed_value"] = 18.0
df_input["report_year"] = 2024
df_input["report_sentiment_score"] = 0.5  # neutral placeholder

# Predict probability and label
proba = model.predict_proba(df_input)[0][1]
label = model.predict(df_input)[0]

# Output
print(f"Fraud probability: {proba:.2f}")
print("Prediction:", "Greenwashing (Fraudulent)" if label == 1 else "Legit")



