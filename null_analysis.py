import xarray as xr
import pandas as pd
import numpy as np

# Load GRIB using cfgrib engine with chunking
ds = xr.open_dataset("data/data.grib", engine="cfgrib", chunks={'time': 100})

# Initialize counters for total rows and null counts
total_rows = 0
null_counts = {
    'tp': 0,
    'lsrr': 0
}
zero_counts = {
    'tp': 0,
    'lsrr': 0
}

# Process data in chunks
chunk_size = 100
total_times = len(ds.time)
print(f"Total time steps: {total_times}")

for start_idx in range(0, total_times, chunk_size):
    end_idx = min(start_idx + chunk_size, total_times)
    print(f"\nProcessing chunk from {start_idx} to {end_idx}")
    
    # Get chunk and convert to dataframe
    ds_chunk = ds.isel(time=slice(start_idx, end_idx))
    df_chunk = ds_chunk.to_dataframe().reset_index()
    
    # Update total rows
    chunk_rows = len(df_chunk)
    total_rows += chunk_rows
    
    # Count nulls in this chunk
    for var in ['tp', 'lsrr']:
        null_count = df_chunk[var].isna().sum()
        null_counts[var] += null_count
        
        # Count zeros (only for non-null values)
        zero_count = (df_chunk[var] == 0).sum()
        zero_counts[var] += zero_count
        
    # Print chunk statistics
    print(f"Chunk size: {chunk_rows} rows")
    print("\nNull values in this chunk:")
    for var in ['tp', 'lsrr']:
        print(f"{var}: {df_chunk[var].isna().sum()} nulls ({df_chunk[var].isna().mean()*100:.2f}%)")
    
    print("\nZero values in this chunk:")
    for var in ['tp', 'lsrr']:
        zeros = (df_chunk[var] == 0).sum()
        print(f"{var}: {zeros} zeros ({zeros/chunk_rows*100:.2f}% of total)")

# Print final summary
print("\n=== FINAL SUMMARY ===")
print(f"Total rows processed: {total_rows}")

print("\nNull values summary:")
for var in ['tp', 'lsrr']:
    null_percent = (null_counts[var] / total_rows) * 100
    print(f"{var}:")
    print(f"  - Null count: {null_counts[var]:,}")
    print(f"  - Null percentage: {null_percent:.2f}%")
    
print("\nZero values summary:")
for var in ['tp', 'lsrr']:
    zero_percent = (zero_counts[var] / total_rows) * 100
    print(f"{var}:")
    print(f"  - Zero count: {zero_counts[var]:,}")
    print(f"  - Zero percentage: {zero_percent:.2f}%")

print("\nValid data summary:")
for var in ['tp', 'lsrr']:
    valid_count = total_rows - null_counts[var] - zero_counts[var]
    valid_percent = (valid_count / total_rows) * 100
    print(f"{var}:")
    print(f"  - Valid non-zero values: {valid_count:,}")
    print(f"  - Valid non-zero percentage: {valid_percent:.2f}%") 