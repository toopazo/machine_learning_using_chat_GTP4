import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb

# Load data
df = pd.read_csv("AirQualityData.csv")

# Process date and season
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

# Drop rows with missing AQI
df = df.dropna(subset=["AirQualityIndex"])

# Features and target
X = df.drop(columns=["AirQualityIndex", "Date"])
y = df["AirQualityIndex"]

# Identify categorical and numerical columns
categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

# Preprocessor
preprocessor = ColumnTransformer(
    [("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)],
    remainder="passthrough",
)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ----------------------------
# 1. Random Forest with GridSearch
# ----------------------------
rf_pipeline = Pipeline(
    [
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(random_state=42)),
    ]
)

rf_param_grid = {
    "regressor__n_estimators": [100, 200],
    "regressor__max_depth": [None, 10, 20],
}

rf_grid = GridSearchCV(
    rf_pipeline, rf_param_grid, cv=5, scoring="neg_root_mean_squared_error", n_jobs=-1
)
rf_grid.fit(X_train, y_train)

# Best RF model
rf_best = rf_grid.best_estimator_

# ----------------------------
# 2. XGBoost Regressor
# ----------------------------
xgb_pipeline = Pipeline(
    [
        ("preprocessor", preprocessor),
        ("regressor", xgb.XGBRegressor(objective="reg:squarederror", random_state=42)),
    ]
)

xgb_pipeline.fit(X_train, y_train)

# ----------------------------
# 3. Linear Regression
# ----------------------------
lr_pipeline = Pipeline(
    [("preprocessor", preprocessor), ("regressor", LinearRegression())]
)

lr_pipeline.fit(X_train, y_train)

# ----------------------------
# Evaluation on Test Set
# ----------------------------
models = {
    "Random Forest (Best)": rf_best,
    "XGBoost": xgb_pipeline,
    "Linear Regression": lr_pipeline,
}

results = []
for name, model in models.items():
    y_pred = model.predict(X_test)
    # rmse = mean_squared_error(y_test, y_pred, squared=False)
    rmse = root_mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    scores = cross_val_score(
        model, X_train, y_train, scoring="neg_root_mean_squared_error", cv=5
    )
    results.append(
        {
            "Model": name,
            "Test RMSE": rmse,
            "Test R2": r2,
            "CV RMSE (mean)": -scores.mean(),
            "CV RMSE (std)": scores.std(),
        }
    )

results_df = pd.DataFrame(results)
results_df.to_csv("model_comparison.csv", index=False)
print(results_df)

# ----------------------------
# Save Predictions from Best Model
# ----------------------------
y_pred_best = rf_best.predict(X_test)
pred_df = X_test.copy()
pred_df["Actual_AQI"] = y_test
pred_df["Predicted_AQI"] = y_pred_best
pred_df.to_csv("aqi_predictions.csv", index=False)

# ----------------------------
# Feature Importances
# ----------------------------
encoder = rf_best.named_steps["preprocessor"].named_transformers_["cat"]
encoded_cat_features = encoder.get_feature_names_out(categorical_cols)
all_features = np.concatenate([encoded_cat_features, numerical_cols])
importances = rf_best.named_steps["regressor"].feature_importances_
feat_imp = pd.Series(importances, index=all_features).sort_values(ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x=feat_imp.values[:15], y=feat_imp.index[:15])
plt.title("Top 15 Feature Importances for AQI Prediction (Random Forest)")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig("feature_importances.png")
plt.show()
