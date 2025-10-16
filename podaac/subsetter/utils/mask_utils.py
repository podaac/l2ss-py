"""
mask_utils.py
-------------

Utilities for creating and applying masks for subsetting operations.
Place all mask creation, manipulation, and application functions here.
"""
import xarray as xr


def align_time_to_lon_dim(time_data, lon_data, temporal_cond):
    """
    Aligns a 1D time_data variable to one of the dimensions of a 2D lon_data array,
    renaming time_data's dimension if it matches the size of one of lon_data's dims.

    This happens because combining a 2D x 2D x 1D bitwise mask with mismatched dimensions
    results in a 3D mask, which significantly increases memory usage. In this case, one
    of the dimensions is a "phony" dimension, so we need to align the time variable with
    the correct dimension to produce a proper 2D bitwise mask.

    Parameters:
        time_data (xr.DataArray): 1D array of time values.
        lon_data (xr.DataArray): 2D array with two dimensions.
        temporal_cond (xr.DataArray): 1D boolean mask along the time dimension.

    Returns:
        xr.DataArray: temporal_cond, potentially renamed to match lon_data's dims.
    """

    time_dim = time_data.dims[0]
    if time_dim not in lon_data.dims:
        time_dim_size = time_data.sizes.get(time_dim)

        lon_dim_1, lon_dim_2 = lon_data.dims
        lon_dim_1_size = lon_data.sizes.get(lon_dim_1)
        lon_dim_2_size = lon_data.sizes.get(lon_dim_2)

        if time_dim_size == lon_dim_1_size:
            return temporal_cond.rename({time_dim: lon_dim_1})
        if time_dim_size == lon_dim_2_size:
            return temporal_cond.rename({time_dim: lon_dim_2})

    return temporal_cond  # Return unchanged if no renaming needed


def align_dims_cond_only(dataset: xr.Dataset, cond: xr.Dataset) -> xr.Dataset:
    """
    Align dims in `cond` to match `dataset` only if they are unaligned but sizes match.
    Works for both Dataset and DataArray.
    """
    # Helper to get dim sizes
    def get_sizes(obj):
        if isinstance(obj, xr.Dataset):
            return dict(obj.dims)
        return dict(obj.sizes)

    dataset_sizes = get_sizes(dataset)
    cond_sizes = get_sizes(cond)

    dataset_dims = set(dataset_sizes)
    cond_dims = set(cond_sizes)
    matches = dataset_dims & cond_dims

    # Return early if none or all dims match
    if len(matches) == 0 or matches == dataset_dims:
        return cond

    # Build mapping for dims that need alignment based on size
    dim_map = {}
    for cdim, csize in cond_sizes.items():
        if cdim in dataset_sizes and dataset_sizes[cdim] == csize:
            continue  # already aligned
        for ddim, dsize in dataset_sizes.items():
            if dsize == csize and ddim not in cond_dims and ddim not in dim_map.values():
                dim_map[cdim] = ddim
                break

    if dim_map:
        cond = cond.rename(dim_map)

    return cond
