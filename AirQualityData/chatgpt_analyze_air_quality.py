
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Load the dataset
df = pd.read_csv("AirQualityData.csv")

# Create an output directory for plots
os.makedirs("plots", exist_ok=True)

# Summary of column types
print("=== Column Data Types ===")
print(df.dtypes)
print("\n")

# Identify categorical and numerical columns
categorical_cols = df.select_dtypes(include=['object', 'category', 'int64']).columns.tolist()
numerical_cols = df.select_dtypes(include=['float64']).columns.tolist()

print("Categorical columns:", categorical_cols)
print("Numerical columns:", numerical_cols)
print("\n")

# Summary statistics
print("=== Summary Statistics ===")
print(df.describe(include='all'))
print("\n")

# Distribution of categorical variables
for col in categorical_cols:
    plt.figure(figsize=(8, 4))
    sns.countplot(x=col, data=df)
    plt.title(f"Distribution of {col}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"plots/distribution_{col}.png")
    plt.close()

# Distribution of numerical variables
for col in numerical_cols:
    plt.figure(figsize=(8, 4))
    sns.histplot(df[col], kde=True, bins=30)
    plt.title(f"Distribution of {col}")
    plt.tight_layout()
    plt.savefig(f"plots/histogram_{col}.png")
    plt.close()

# Correlation plots with AQI
target_col = "AirQualityIndex"
for col in df.columns:
    if col != target_col and df[col].dtype != 'object':
        plt.figure(figsize=(8, 4))
        sns.scatterplot(data=df, x=col, y=target_col)
        plt.title(f"{col} vs {target_col}")
        plt.tight_layout()
        plt.savefig(f"plots/scatter_{col}_vs_AQI.png")
        plt.close()
