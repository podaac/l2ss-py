# pylint: disable=too-many-branches
"""
vertical_subset.py

Contains vertical subsetting logic for NetCDF/xarray DataTree datasets.
"""
from typing import List, Optional, Union
import numpy as np
import xarray as xr


def vertical_subset(
    dataset: Union[xr.Dataset, xr.DataTree],
    return_dataset: Union[xr.Dataset, xr.DataTree],
    lat_var_names: List[str],
    lon_var_names: List[str],
    vertical_var: Optional[str] = None,
    vertical_min: Optional[float] = None,
    vertical_max: Optional[float] = None,
    cut: bool = True
) -> Union[xr.Dataset, xr.DataTree]:
    """
    Perform vertical subsetting on a DataTree-aware xarray dataset.

    Parameters
    ----------
    dataset : xarray.Dataset or DataTree
        The original dataset (before spatial/temporal subsetting)
    return_dataset : xarray.Dataset or DataTree
        The dataset after spatial/temporal subsetting, to apply vertical mask/slice to
    lat_var_names : list[str]
        List of latitude variable names
    lon_var_names : list[str]
        List of longitude variable names
    vertical_var : str, optional
        Name of the vertical variable or dimension
    vertical_min : float, optional
        Minimum value for vertical subsetting (inclusive)
    vertical_max : float, optional
        Maximum value for vertical subsetting (inclusive)
    cut : bool, optional
        Whether to slice output to first/last valid vertical indices

    Returns
    -------
    xarray.Dataset or DataTree
        The vertically subsetted dataset
    """
    dimensional_subsetting = None
    vert_mask = None
    vertical_data = None
    vert_dim = None

    if vertical_var is not None:
        try:
            vertical_data = dataset[vertical_var]
            dimensional_subsetting = "variable"
        except KeyError:
            dimensional_subsetting = "dimension"

        if dimensional_subsetting == "variable":
            # Find the vertical dimension name (not in lat/lon dims)
            lat_dims = set(dataset[lat_var_names[0]].dims)
            lon_dims = set(dataset[lon_var_names[0]].dims)
            for dim in vertical_data.dims:
                if dim not in lat_dims and dim not in lon_dims:
                    vert_dim = dim
                    break
            if vert_dim is None:
                # Could not determine vertical dimension
                return return_dataset

            # Find axis index for vertical dimension
            vert_axis = vertical_data.dims.index(vert_dim)

            # Create mask for vertical range
            vert_mask = np.ones(vertical_data.shape, dtype=bool)
            if vertical_min is not None:
                vert_mask &= vertical_data.values >= vertical_min
            if vertical_max is not None:
                vert_mask &= vertical_data.values <= vertical_max

            # Handle NaN and fill values
            fill_value = vertical_data.attrs.get('_FillValue', None)
            missing_value = vertical_data.attrs.get('missing_value', None)
            mask_nan = ~np.isnan(vertical_data.values)
            mask_fill = np.ones(vertical_data.shape, dtype=bool)
            if fill_value is not None:
                mask_fill &= vertical_data.values != fill_value
            if missing_value is not None:
                mask_fill &= vertical_data.values != missing_value
            vert_mask_combined = vert_mask & mask_nan & mask_fill

            # Find first and last valid indices along the vertical axis
            valid_indices = [i for i in range(vertical_data.shape[vert_axis]) if np.any(np.take(vert_mask_combined, i, axis=vert_axis))]
            if valid_indices:
                first_valid = valid_indices[0]
                last_valid = valid_indices[-1]
            else:
                first_valid = last_valid = None

            # Use vert_mask_combined for masking
            def apply_vert_mask(ds):
                """
                Apply the vertical mask to all variables in the dataset whose dimensions match the mask.

                Parameters
                ----------
                ds : xarray.Dataset
                    Dataset to apply the mask to.

                Returns
                -------
                xarray.Dataset
                    Masked dataset.
                """
                masked = ds.copy()
                mask_dims = getattr(vertical_data, 'dims', None)
                for var in ds.data_vars:
                    da = ds[var]
                    # Only apply mask if dims match
                    if mask_dims is not None and tuple(da.dims) == tuple(mask_dims):
                        masked[var] = da.where(vert_mask)
                    else:
                        masked[var] = da
                return masked
            new_tree = xr.map_over_datasets(apply_vert_mask, return_dataset)

            # Slice new_tree by everything between first_valid and last_valid along the vertical axis
            if cut and first_valid is not None and last_valid is not None:
                dim_size = vertical_data.sizes.get(vert_dim, None)
                if dim_size is not None:
                    new_tree = new_tree.isel({vert_dim: slice(first_valid, last_valid + 1)})

            return new_tree

        if dimensional_subsetting == "dimension":
            vert_dim = vertical_var.lstrip('/')
            vert_values = get_vert_values(dataset, vert_dim)
            vert_mask = np.ones(vert_values.shape, dtype=bool)

            if vertical_min is not None:
                vert_mask &= vert_values >= vertical_min
            if vertical_max is not None:
                vert_mask &= vert_values <= vertical_max

            if cut:
                # Slice to only valid layers
                new_tree = return_dataset.isel({vert_dim: vert_mask})
            else:
                # Mask out invalid layers (set to NaN) but keep all layers
                def mask_layers(ds):
                    masked = ds.copy()
                    if vert_dim in ds.dims:
                        for var in ds.data_vars:
                            da = ds[var]
                            if vert_dim in da.dims:
                                axis = da.dims.index(vert_dim)
                                # Broadcast mask to variable shape
                                shape = [1]*da.ndim
                                shape[axis] = -1
                                expanded_mask = vert_mask.reshape(shape)
                                masked[var] = da.where(expanded_mask)
                            else:
                                masked[var] = da
                    return masked
                new_tree = xr.map_over_datasets(mask_layers, return_dataset)
            return new_tree

    return return_dataset


def get_vert_values(tree: xr.DataTree, vert_dim: str) -> np.ndarray:
    """
    Get vertical values for a given dimension in a DataTree.

    Parameters
    ----------
    tree : xarray.DataTree
        The DataTree to search for the vertical dimension.
    vert_dim : str
        Name of the vertical dimension.

    Returns
    -------
    numpy.ndarray
        Array of vertical values. If a coordinate exists, use its values. If index-only, returns np.arange(N) from the first node where the dimension exists.

    Raises
    ------
    KeyError
        If the dimension does not exist in any node.
    """
    for node in tree.values():
        ds = node.ds
        if ds is None:
            continue

        # Coordinate exists
        if vert_dim in ds.coords:
            return ds.coords[vert_dim].values

        # Variable acting as coordinate
        if vert_dim in ds.variables and ds[vert_dim].dims == (vert_dim,):
            return ds[vert_dim].values

        # Dimension exists but no coordinate → index-only
        if vert_dim in ds.dims:
            return np.arange(ds.dims[vert_dim])

    # Dimension not found anywhere
    raise KeyError(f"Vertical dimension '{vert_dim}' not found in DataTree")


def find_nodes_by_dim(tree: xr.DataTree, dim_name: str) -> list:
    """
    Return nodes that contain a dimension `dim_name`.

    Parameters
    ----------
    tree : xarray.DataTree
        The DataTree to search.
    dim_name : str
        Name of the dimension to look for.

    Returns
    -------
    list
        List of nodes containing the dimension.
    """
    result = []
    for node in tree.values():
        if node.ds is not None and dim_name in node.ds.dims:
            result.append(node)
    return result
