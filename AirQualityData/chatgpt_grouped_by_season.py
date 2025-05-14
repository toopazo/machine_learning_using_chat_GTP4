import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("AirQualityData.csv")

# Convert Date to datetime and create Season column
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

# Filter for the hour with worst average AQI (5 AM)
df_filtered = df[df["Hour"] == 5]

# Create bar plot using seaborn with hue for season
plt.figure(figsize=(10, 6))
sns.barplot(
    data=df_filtered, x="DayOfWeek", y="AirQualityIndex", hue="Season", palette="Set2"
)
plt.title("Average AQI by Day of Week Grouped by Season (5 AM)")
plt.xlabel("Day of Week")
plt.ylabel("Average AQI")
plt.legend(title="Season")
plt.tight_layout()

# Save the plot if needed
plt.savefig("aqi_by_day_grouped_by_season.png")
# plt.show()
