import pandas as pd

# Read the two CSV files
df1 = pd.read_csv('data/20192022_monthly_weather_summary.csv')
df2 = pd.read_csv('data/20232024_monthly_weather_summary.csv')

# Combine the dataframes
combined_df = pd.concat([df1, df2], ignore_index=True)


combined_df = combined_df.sort_values(['year', 'month', 'latitude', 'longitude'])

# Save the combined dataframe
combined_df.to_csv('data/monthly_weather_summary.csv', index=False)

print("Files combined successfully!") 