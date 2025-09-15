"""
mask_utils.py
-------------

Utilities for creating and applying masks for subsetting operations.
Place all mask creation, manipulation, and application functions here.
"""


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
