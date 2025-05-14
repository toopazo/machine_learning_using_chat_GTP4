
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# Load data
df = pd.read_csv("AirQualityData.csv")

# Create season
df["Date"] = pd.to_datetime(df["Date"])
df["Season"] = df["Date"].dt.month.map({12: "Winter", 1: "Winter", 2: "Winter",
                                        3: "Spring", 4: "Spring", 5: "Spring",
                                        6: "Summer", 7: "Summer", 8: "Summer",
                                        9: "Fall", 10: "Fall", 11: "Fall"})

# Drop missing AQI
df = df.dropna(subset=["AirQualityIndex"])

# Features and target
X = df.drop(columns=["AirQualityIndex", "Date"])
y = df["AirQualityIndex"]

# Identify column types
categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

# Preprocessor
preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
], remainder="passthrough")

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Pipeline with XGBoost
xgb_model = xgb.XGBRegressor(objective="reg:squarederror", random_state=42)
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", xgb_model)
])
pipeline.fit(X_train, y_train)

# Extract processed data and trained model
X_processed = pipeline.named_steps["preprocessor"].transform(X_test)
model = pipeline.named_steps["regressor"]
feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()

# SHAP analysis
explainer = shap.Explainer(model, X_processed)
shap_values = explainer(X_processed)

# Summary plot
shap.summary_plot(shap_values, features=X_processed, feature_names=feature_names, show=False)
plt.tight_layout()
plt.savefig("shap_summary_plot.png")
