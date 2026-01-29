import numpy as np
import xarray as xr
import pytest
from podaac.subsetter import subset

def create_vertical_test_datatree(tmp_path):
    lat = np.linspace(-90, 90, 4)
    lon = np.linspace(-180, 180, 6)
    depth = np.array([0, 10, 20, 30, 40])

    # Root dataset
    data = np.broadcast_to(depth, (len(lat), len(lon), len(depth)))
    ds_root = xr.Dataset(
        {"temperature": (("lat", "lon", "depth"), data)},
        coords={"lat": lat, "lon": lon},
    )
    tree = xr.DataTree(ds_root, name="root")

    # Add children, each with its own dataset
    child1_data = np.broadcast_to(depth * 2, (len(lat), len(lon), len(depth)))
    ds_child1 = xr.Dataset(
        {"temperature": (("lat", "lon", "depth"), child1_data)},
        coords={"lat": lat, "lon": lon},
    )
    tree["child1"] = xr.DataTree(ds_child1, name="child1")

    child2_data = np.broadcast_to(depth + 5, (len(lat), len(lon), len(depth)))
    ds_child2 = xr.Dataset(
        {"temperature": (("lat", "lon", "depth"), child2_data)},
        coords={"lat": lat, "lon": lon},
    )
    tree["child2"] = xr.DataTree(ds_child2, name="child2")

    nc_path = tmp_path / "vertical_test.nc"
    tree.to_netcdf(nc_path)
    return nc_path

@pytest.mark.parametrize(
    "vertical_var,cut,check_nan,vertical_min,vertical_max",
    [
        ("temperature", False, True, 10, 30),
        ("temperature", True, False, 10, 30),
        ("depth", False, True, 1, 3),
        ("depth", True, False, 1, 3),
    ],
)
def test_vertical_dimension_subset(tmp_path, vertical_var, cut, check_nan, vertical_min, vertical_max):
    nc_path = create_vertical_test_datatree(tmp_path)
    output_path = tmp_path / "vertical_subset.nc"
    bbox = np.array([[-180, 180], [-90, 90]])
    subset.subset(
        file_to_subset=str(nc_path),
        bbox=bbox,
        output_file=str(output_path),
        lat_var_names=["lat"],
        lon_var_names=["lon"],
        vertical_var=vertical_var,
        vertical_min=vertical_min,
        vertical_max=vertical_max,
        cut=cut,
    )
    if vertical_var == "temperature" or not cut:
        ds_out = xr.open_dataset(output_path)
        arr = ds_out["temperature"].values
        if check_nan:
            assert np.any(np.isnan(arr)), "There should be NaN values present"
        else:
            assert not np.any(np.isnan(arr)), "There should be no NaN values present"
        assert np.any((arr >= 10) & (arr <= 30)), f"There should be values between {vertical_min} and {vertical_max}"
        ds_out.close()
