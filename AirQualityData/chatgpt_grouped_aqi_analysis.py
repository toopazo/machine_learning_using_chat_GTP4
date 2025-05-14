
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Load dataset
df = pd.read_csv("AirQualityData.csv")

# Convert Date to datetime and extract season
df["Date"] = pd.to_datetime(df["Date"])
def get_season(month):
    if month in [12, 1, 2]:
        return "Winter"
    elif month in [3, 4, 5]:
        return "Spring"
    elif month in [6, 7, 8]:
        return "Summer"
    else:
        return "Fall"
df["Season"] = df["Date"].dt.month.apply(get_season)

# Filter for the hour with the worst AQI
df = df[df["Hour"] == 5]

# Create plot output directory
os.makedirs("grouped_plots", exist_ok=True)

# Group by DayOfWeek
for day in sorted(df["DayOfWeek"].unique()):
    df_day = df[df["DayOfWeek"] == day]
    for col in df_day.select_dtypes(include="float64").columns:
        if col != "AirQualityIndex":
            plt.figure(figsize=(8, 4))
            sns.scatterplot(data=df_day, x=col, y="AirQualityIndex")
            plt.title(f"{col} vs AQI on Day {day} (5 AM)")
            plt.tight_layout()
            plt.savefig(f"grouped_plots/scatter_{col}_vs_AQI_day_{day}.png")
            plt.close()

# Group by Season
for season in df["Season"].unique():
    df_season = df[df["Season"] == season]
    for col in df_season.select_dtypes(include="float64").columns:
        if col != "AirQualityIndex":
            plt.figure(figsize=(8, 4))
            sns.scatterplot(data=df_season, x=col, y="AirQualityIndex")
            plt.title(f"{col} vs AQI in {season} (5 AM)")
            plt.tight_layout()
            plt.savefig(f"grouped_plots/scatter_{col}_vs_AQI_season_{season}.png")
            plt.close()
