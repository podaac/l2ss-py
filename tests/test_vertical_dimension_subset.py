import numpy as np
import xarray as xr
import pytest
from unittest.mock import MagicMock
from podaac.subsetter import subset
from podaac.subsetter.subset_harmony import L2SubsetterService
import types

def test_vertical_dimension_subset(tmp_path):
    # Create a dummy dataset with a vertical dimension (e.g., 'depth')
    lat = np.linspace(-90, 90, 4)
    lon = np.linspace(-180, 180, 6)
    depth = np.array([0, 10, 20, 30, 40])
    data = np.broadcast_to(depth, (len(lat), len(lon), len(depth)))  # shape: (lat, lon, depth)

    ds = xr.Dataset(
        {
            "temperature": (("lat", "lon", "depth"), data),
        },
        coords={
            "lat": lat,
            "lon": lon,
            "depth": depth,
        }
    )

    nc_path = tmp_path / "vertical_test.nc"
    ds.to_netcdf(nc_path)
    output_path = tmp_path / "vertical_subset.nc"

    # Subset to keep only depth between 10 and 30 (inclusive)
    bbox = np.array([[-180, 180], [-90, 90]])
    subset.subset(
        file_to_subset=str(nc_path),
        bbox=bbox,
        output_file=str(output_path),
        lat_var_names=["lat"],
        lon_var_names=["lon"],
        vertical_var="temperature",
        vertical_min=10,
        vertical_max=30,
        cut=False
    )

    ds_out = xr.open_dataset(output_path)
    # Check that only the correct temperature values remain
    arr = ds_out['temperature'].values
    assert np.any(np.isnan(arr)), "There should be NaN values present"
    assert np.any((arr >= 10) & (arr <= 30)), "There should be values between 10 and 30"
    ds_out.close()
