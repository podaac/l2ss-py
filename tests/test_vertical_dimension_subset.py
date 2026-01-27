import numpy as np
import xarray as xr
import pytest
from podaac.subsetter import subset

# Helper to create and write the test dataset
def create_vertical_test_dataset(tmp_path):
    lat = np.linspace(-90, 90, 4)
    lon = np.linspace(-180, 180, 6)
    depth = np.array([0, 10, 20, 30, 40])
    data = np.broadcast_to(depth, (len(lat), len(lon), len(depth)))
    ds = xr.Dataset(
        {"temperature": (("lat", "lon", "depth"), data)},
        coords={"lat": lat, "lon": lon, "depth": depth},
    )
    nc_path = tmp_path / "vertical_test.nc"
    ds.to_netcdf(nc_path)
    return nc_path

@pytest.mark.parametrize(
    "vertical_var,cut,check_nan",
    [
        ("temperature", False, True),
        ("temperature", True, False),
        ("depth", False, False),
        ("depth", True, False),
    ],
)
def test_vertical_dimension_subset(tmp_path, vertical_var, cut, check_nan):
    nc_path = create_vertical_test_dataset(tmp_path)
    output_path = tmp_path / "vertical_subset.nc"
    bbox = np.array([[-180, 180], [-90, 90]])
    subset.subset(
        file_to_subset=str(nc_path),
        bbox=bbox,
        output_file=str(output_path),
        lat_var_names=["lat"],
        lon_var_names=["lon"],
        vertical_var=vertical_var,
        vertical_min=10,
        vertical_max=30,
        cut=cut,
    )
    if vertical_var == "temperature" or not cut:
        ds_out = xr.open_dataset(output_path)
        arr = ds_out["temperature"].values
        if check_nan:
            assert np.any(np.isnan(arr)), "There should be NaN values present"
        else:
            assert not np.any(np.isnan(arr)), "There should be no NaN values present"
        assert np.any((arr >= 10) & (arr <= 30)), "There should be values between 10 and 30"
        ds_out.close()
