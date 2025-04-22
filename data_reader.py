import xarray as xr

# Load GRIB using cfgrib engine with chunking
ds = xr.open_dataset("data/data.grib", engine="cfgrib", chunks={'time': 100})

print("Available variables in dataset:")
for var in ds.data_vars:
    print(f"- {var}: {ds[var].attrs}")

# Initialize variables for chunk loading
chunk_size = 100
start_idx = 0
found_data = False

while not found_data:
    # Calculate end index for current chunk
    end_idx = start_idx + chunk_size
    
    print(f"\nLoading time slice from {start_idx} to {end_idx}")
    
    # Select current chunk and compute it fully
    ds_subset = ds.isel(time=slice(start_idx, end_idx))
    
    # Convert to dataframe first to handle the data more easily
    df = ds_subset.to_dataframe().reset_index()
    
    # Filter out nulls and zeros
    df_filtered = df.dropna(subset=['tp'])
    df_filtered = df_filtered[df_filtered['tp'] != 0]
    
    if len(df_filtered) > 0:
        found_data = True
        print(f"Found precipitation data in time slice {start_idx} to {end_idx}")
        print("\nFirst few rows of filtered data:")
        print(df_filtered.head())
    else:
        print("No precipitation data in this chunk, moving to next chunk")
        start_idx = end_idx
        
        # Check if we've reached the end of the dataset
        if start_idx >= len(ds.time):
            print("Reached end of dataset without finding precipitation data")
            break