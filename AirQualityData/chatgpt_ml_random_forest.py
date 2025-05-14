import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv("AirQualityData.csv")

# Parse dates and create additional time-based features
df["Date"] = pd.to_datetime(df["Date"])
df["Season"] = df["Date"].dt.month.map(
    {
        12: "Winter",
        1: "Winter",
        2: "Winter",
        3: "Spring",
        4: "Spring",
        5: "Spring",
        6: "Summer",
        7: "Summer",
        8: "Summer",
        9: "Fall",
        10: "Fall",
        11: "Fall",
    }
)

# Drop rows with missing target
df = df.dropna(subset=["AirQualityIndex"])

# Features and target
X = df.drop(columns=["AirQualityIndex", "Date"])
y = df["AirQualityIndex"]

# Identify categorical and numerical columns
categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

# Build preprocessing pipeline
preprocessor = ColumnTransformer(
    [("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)],
    remainder="passthrough",
)

# Full pipeline with model
model = Pipeline(
    steps=[
        ("preprocess", preprocessor),
        ("regressor", RandomForestRegressor(n_estimators=100, random_state=42)),
    ]
)

# Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"RMSE: {rmse:.2f}")
print(f"R² Score: {r2:.2f}")

# --- Feature Importance Plot ---
# Get feature names after preprocessing
encoder = model.named_steps["preprocess"].named_transformers_["cat"]
encoded_cat_features = encoder.get_feature_names_out(categorical_cols)
all_features = np.concatenate([encoded_cat_features, numerical_cols])

# Get importances from model
importances = model.named_steps["regressor"].feature_importances_
feat_imp = pd.Series(importances, index=all_features).sort_values(ascending=False)

# Plot
plt.figure(figsize=(10, 6))
sns.barplot(x=feat_imp.values[:15], y=feat_imp.index[:15])
plt.title("Top 15 Feature Importances for AQI Prediction")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig("feature_importances.png")
plt.show()
