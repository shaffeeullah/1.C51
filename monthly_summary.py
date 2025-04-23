import xarray as xr
import pandas as pd

# Load each variable separately to ensure we get all of them
print("Loading variables separately...")

# Load tp (total precipitation)
print("\nLoading tp...")
ds_tp = xr.open_dataset("data/data.grib", engine="cfgrib", 
                       backend_kwargs={
                           'filter_by_keys': {
                               'shortName': 'tp',
                               'typeOfLevel': 'surface'
                           }
                       },
                       chunks={'time': 100})

# Print time range for tp dataset
print(f"TP dataset time range: {ds_tp.time.values[0]} to {ds_tp.time.values[-1]}")

# Load sf (snowfall)
print("Loading sf...")
ds_sf = xr.open_dataset("data/data.grib", engine="cfgrib", 
                       backend_kwargs={
                           'filter_by_keys': {
                               'shortName': 'sf',
                               'typeOfLevel': 'surface'
                           }
                       },
                       chunks={'time': 100})

# Load other variables
print("Loading u10, t2m, tcc...")
ds_others = xr.open_dataset("data/data.grib", engine="cfgrib", 
                           backend_kwargs={
                               'filter_by_keys': {
                                   'typeOfLevel': 'surface'
                               }
                           },
                           chunks={'time': 100})

# Process data in chunks
chunk_size = 100
total_times = len(ds_tp.time)  # Use tp dataset as reference for time steps
print(f"Total time steps: {total_times}")

# Initialize an empty DataFrame to store monthly summaries
monthly_summaries = []

for start_idx in range(0, total_times, chunk_size):
    end_idx = min(start_idx + chunk_size, total_times)
    print(f"\nProcessing chunk from {start_idx} to {end_idx}")
    
    # Get chunks from each dataset
    ds_tp_chunk = ds_tp.isel(time=slice(start_idx, end_idx))
    ds_sf_chunk = ds_sf.isel(time=slice(start_idx, end_idx))
    ds_others_chunk = ds_others.isel(time=slice(start_idx, end_idx))
    
    # Convert to dataframes
    df_tp = ds_tp_chunk.to_dataframe().reset_index()
    df_sf = ds_sf_chunk.to_dataframe().reset_index()
    df_others = ds_others_chunk.to_dataframe().reset_index()
    
    # Convert time to year and month
    df_tp['year'] = df_tp['time'].dt.year
    df_tp['month'] = df_tp['time'].dt.month
    df_sf['year'] = df_sf['time'].dt.year
    df_sf['month'] = df_sf['time'].dt.month
    df_others['year'] = df_others['time'].dt.year
    df_others['month'] = df_others['time'].dt.month
    
    # Group by year, month, latitude, and longitude and calculate statistics
    monthly_tp = df_tp.groupby(['year', 'month', 'latitude', 'longitude'])['tp'].sum().reset_index()
    monthly_sf = df_sf.groupby(['year', 'month', 'latitude', 'longitude'])['sf'].sum().reset_index()
    monthly_others = df_others.groupby(['year', 'month', 'latitude', 'longitude']).agg({
        'u10': 'mean',
        't2m': 'mean',
        'tcc': 'mean'
    }).reset_index()
    
    # First merge tp and sf
    monthly_chunk = monthly_tp.merge(monthly_sf, on=['year', 'month', 'latitude', 'longitude'], how='outer')
    
    # Then merge with others
    monthly_chunk = monthly_chunk.merge(monthly_others, on=['year', 'month', 'latitude', 'longitude'], how='outer')
    
    # Fill NaN values with 0 for precipitation and snowfall, and with mean for other variables
    monthly_chunk['tp'] = monthly_chunk['tp'].fillna(0)
    monthly_chunk['sf'] = monthly_chunk['sf'].fillna(0)
    monthly_chunk['u10'] = monthly_chunk['u10'].fillna(monthly_chunk['u10'].mean())
    monthly_chunk['t2m'] = monthly_chunk['t2m'].fillna(monthly_chunk['t2m'].mean())
    monthly_chunk['tcc'] = monthly_chunk['tcc'].fillna(monthly_chunk['tcc'].mean())
    
    monthly_summaries.append(monthly_chunk)

# Combine all chunks
final_summary = pd.concat(monthly_summaries, ignore_index=True)
print(f"\nShape after initial concat: {final_summary.shape}")

# Group by year, month, latitude, and longitude to combine any partial months
final_summary = final_summary.groupby(
    ['year', 'month', 'latitude', 'longitude']
).agg({
    'tp': 'sum',
    'sf': 'sum',
    'u10': 'mean',
    't2m': 'mean',
    'tcc': 'mean'
}).reset_index()

print(f"Shape after final groupby: {final_summary.shape}")

# Sort by time and location
final_summary = final_summary.sort_values(['year', 'month', 'latitude', 'longitude'])

# Save to CSV
output_file = "monthly_weather_summary.csv"
final_summary.to_csv(output_file, index=False)
print(f"\nSaved complete summary to {output_file}")