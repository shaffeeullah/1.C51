import xarray as xr

# Load GRIB using cfgrib engine
ds = xr.open_dataset("data/data.grib", engine="cfgrib")
print(ds)
df = ds.to_dataframe().reset_index()
print(df.head())