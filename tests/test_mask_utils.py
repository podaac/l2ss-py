# test_mask_utils.py
import pytest
import xarray as xr
import numpy as np
from podaac.subsetter.utils.mask_utils import align_time_to_lon_dim, align_dims_cond_only


class TestAlignTimeToLonDim:
    def test_rename_to_first_lon_dim(self):
        time = xr.DataArray(np.arange(5), dims="phony_dim")
        lon = xr.DataArray(np.zeros((5, 10)), dims=("x", "y"))
        cond = xr.DataArray(np.ones(5, dtype=bool), dims="phony_dim")

        result = align_time_to_lon_dim(time, lon, cond)

        assert "x" in result.dims
        assert "phony_dim" not in result.dims
        assert result.sizes["x"] == 5

    def test_rename_to_second_lon_dim(self):
        time = xr.DataArray(np.arange(10), dims="phony_dim")
        lon = xr.DataArray(np.zeros((5, 10)), dims=("x", "y"))
        cond = xr.DataArray(np.ones(10, dtype=bool), dims="phony_dim")

        result = align_time_to_lon_dim(time, lon, cond)

        assert "y" in result.dims
        assert "phony_dim" not in result.dims
        assert result.sizes["y"] == 10

    def test_no_rename_when_dim_already_matches(self):
        time = xr.DataArray(np.arange(5), dims="x")
        lon = xr.DataArray(np.zeros((5, 10)), dims=("x", "y"))
        cond = xr.DataArray(np.ones(5, dtype=bool), dims="x")

        result = align_time_to_lon_dim(time, lon, cond)

        assert result is cond
        assert "x" in result.dims


class TestAlignDimsCondOnly:
    def test_no_common_dims(self):
        ds = xr.Dataset({"a": ("x", np.arange(5))})
        cond = xr.DataArray(np.ones(5, dtype=bool), dims="y")

        result = align_dims_cond_only(ds, cond)

        # No matching sizes → unchanged
        assert result is cond
        assert "y" in result.dims

    def test_already_aligned(self):
        ds = xr.Dataset({"a": ("x", np.arange(5))})
        cond = xr.DataArray(np.ones(5, dtype=bool), dims="x")

        result = align_dims_cond_only(ds, cond)

        # Already aligned → unchanged
        assert result is cond
        assert "x" in result.dims

    def test_rename_dim(self):
        ds = xr.Dataset({"a": ("x", np.arange(5))})
        cond = xr.DataArray(np.ones(5, dtype=bool), dims="phony")

        result = align_dims_cond_only(ds, cond)

        # Size matches → should rename
        assert "x" not in result.dims
        assert "phony" in result.dims
