import xarray as xr

# Load GRIB using cfgrib engine with chunking
ds = xr.open_dataset("data/data.grib", engine="cfgrib", chunks={'time': 100})
# Select only the first few time steps
ds_subset = ds.isel(time=slice(0, 2))
print(ds_subset)
df = ds_subset.to_dataframe().reset_index()
print(df.head())