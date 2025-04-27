import xarray as xr
import pandas as pd

# Load GRIB using cfgrib engine with chunking and specify time variable
ds = xr.open_dataset("data/data.grib", engine="cfgrib", 
                    backend_kwargs={'filter_by_keys': {'typeOfLevel': 'surface'}},
                    chunks={'time': 100})

print("Available variables in dataset:")
for var in ds.data_vars:
    print(f"- {var}: {ds[var].attrs}")

# Print the time range of the dataset
print("\nTime range in dataset:")
print(f"Start time: {ds.time[0].values}")
print(f"End time: {ds.time[-1].values}")
print(f"Total time steps: {len(ds.time)}")

# Initialize variables for chunk loading
chunk_size = 100
start_idx = 0
all_filtered_data = []

while True:
    # Calculate end index for current chunk
    end_idx = min(start_idx + chunk_size, len(ds.time))
    
    print(f"\nLoading time slice from {start_idx} to {end_idx}")
    
    # Select current chunk and compute it fully
    ds_subset = ds.isel(time=slice(start_idx, end_idx))
    
    # Convert to dataframe first to handle the data more easily
    df = ds_subset.to_dataframe().reset_index()
    
    # Filter out nulls and zeros
    df_filtered = df.dropna(subset=['tp'])
    df_filtered = df_filtered[df_filtered['tp'] != 0]
    
    if len(df_filtered) > 0:
        print(f"Found {len(df_filtered)} precipitation records in time slice {start_idx} to {end_idx}")
        all_filtered_data.append(df_filtered)
    
    # Move to next chunk
    start_idx = end_idx
    
    # Check if we've reached the end of the dataset
    if start_idx >= len(ds.time):
        break

# Combine all filtered data
if all_filtered_data:
    final_df = pd.concat(all_filtered_data, ignore_index=True)
    print("\nSummary of all precipitation data found:")
    print(f"Total records: {len(final_df)}")
    print("\nTime range of precipitation data:")
    print(f"Start: {final_df['time'].min()}")
    print(f"End: {final_df['time'].max()}")
    print("\nFirst few rows of filtered data:")
    print(final_df.head())
else:
    print("No precipitation data found in the entire dataset")