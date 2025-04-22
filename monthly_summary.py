import xarray as xr
import pandas as pd
import numpy as np

# Load GRIB using cfgrib engine with chunking
ds = xr.open_dataset("data/data.grib", engine="cfgrib", chunks={'time': 100})

# Initialize an empty DataFrame to store monthly summaries
monthly_summaries = []

# Process data in chunks
chunk_size = 100
total_times = len(ds.time)
print(f"Total time steps: {total_times}")

for start_idx in range(0, total_times, chunk_size):
    end_idx = min(start_idx + chunk_size, total_times)
    print(f"Processing chunk from {start_idx} to {end_idx}")
    
    # Get chunk and convert to dataframe
    ds_chunk = ds.isel(time=slice(start_idx, end_idx))
    df_chunk = ds_chunk.to_dataframe().reset_index()
    
    # Extract year and month from time column
    df_chunk['year'] = df_chunk['time'].dt.year
    df_chunk['month'] = df_chunk['time'].dt.month
    
    # Group by year, month, latitude, and longitude and calculate sums
    monthly_chunk = df_chunk.groupby(
        ['year', 'month', 'latitude', 'longitude']
    ).agg({
        'tp': 'sum',
        'lsrr': 'sum'
    }).reset_index()
    
    monthly_summaries.append(monthly_chunk)

# Combine all chunks
final_summary = pd.concat(monthly_summaries, ignore_index=True)

# Group again to combine any partial months that were split across chunks
final_summary = final_summary.groupby(
    ['year', 'month', 'latitude', 'longitude']
).agg({
    'tp': 'sum',
    'lsrr': 'sum'
}).reset_index()

# Sort by time and location
final_summary = final_summary.sort_values(['year', 'month', 'latitude', 'longitude'])

print("\nFirst few rows of monthly summary:")
print(final_summary.head())

# Save to CSV
output_file = "monthly_weather_summary.csv"
final_summary.to_csv(output_file, index=False)
print(f"\nSaved complete summary to {output_file}")

# Print some basic statistics
print("\nSummary Statistics:")
print(f"Total months covered: {len(final_summary['year'].unique() * 12)}")
print(f"Total locations: {len(final_summary[['latitude', 'longitude']].drop_duplicates())}")
print("\nPrecipitation (tp) Statistics:")
print(final_summary['tp'].describe())
print("\nSurface Solar Radiation (lsrr) Statistics:")
print(final_summary['lsrr'].describe()) 