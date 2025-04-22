import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read the CSV file
df = pd.read_csv('data/monthly_weather_summary.csv')

# Filter for the specific coordinates
filtered_df = df[(df['latitude'] == -15) & (df['longitude'] == 47)]

# Create a proper datetime column from year and month
filtered_df['date'] = pd.to_datetime(filtered_df[['year', 'month']].assign(day=1))

# Sort by date
filtered_df = filtered_df.sort_values('date')

# Calculate total precipitation per year
yearly_precipitation = filtered_df.groupby('year')['tp'].sum()

# Print yearly precipitation totals
print("\nTotal Precipitation per Year:")
print(yearly_precipitation)

# Create the plot
plt.figure(figsize=(12, 6))
sns.lineplot(data=filtered_df, x='date', y='tp')
plt.title('Monthly Precipitation (tp) at Coordinates (-15, 47)')
plt.xlabel('Date')
plt.ylabel('Precipitation (tp)')
plt.xticks(rotation=45)
plt.tight_layout()

# Save the plot
plt.savefig('precipitation_plot.png')
plt.close() 