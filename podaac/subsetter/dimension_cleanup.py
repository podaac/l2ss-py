# This software may be subject to U.S. export control laws. By accepting
# this software, the user agrees to comply with all applicable U.S. export
# laws and regulations. User has the responsibility to obtain export
# licenses, or other export authority as may be required before exporting
# such information to foreign countries or providing access to foreign
# persons.

"""
======================
dimension_cleanup.py
======================
Functions which improve upon existing netCDF4 library existing functions
"""
import xarray as xr


def sync_dims_inplace(original_dataset: xr.Dataset, new_dataset: xr.Dataset) -> None:
    """
    Synchronize dimensions of variables in the new dataset with the original dataset.

    Parameters:
    original_dataset (xr.Dataset): The original dataset.
    new_dataset (xr.Dataset): The new dataset with possibly additional dimensions.
    """

    for variable_name in new_dataset.variables:
        if variable_name in original_dataset.variables:
            original_variable_dims = original_dataset[variable_name].dims
            new_variable_dims = new_dataset[variable_name].dims

            for new_dim in new_variable_dims:
                if new_dim not in original_variable_dims:
                    new_dataset[variable_name] = new_dataset[variable_name].isel({new_dim: 0})


def remove_duplicate_dims_xarray(dataset: xr.Dataset) -> xr.Dataset:
    """
    Handle datasets with duplicate dimensions in xarray while preserving
    encodings and handling multiple duplicate dimensions.
    """
    # Work with a copy
    ds = dataset.copy(deep=True)

    # Store original encodings and chunking
    original_encodings = {var: ds[var].encoding.copy() for var in ds.variables}

    # Process each variable that has duplicate dimensions
    for var_name, var in ds.variables.items():  # pylint: disable=too-many-nested-blocks
        dim_list = list(var.dims)

        # Skip if no duplicates in this variable
        if not any(dim_list.count(dim) > 1 for dim in dim_list):
            continue

        # Create new dimension names for duplicates
        new_dims = list(dim_list)
        dims_renamed = set()

        for i, dim in enumerate(dim_list):
            if dim_list.count(dim) > 1 and dim not in dims_renamed:
                # Find all occurrences after the first one
                for j in range(i + 1, len(dim_list)):
                    if dim_list[j] == dim:
                        new_dim_name = f"{dim}_{j}"
                        new_dims[j] = new_dim_name

                        # Create new dimension if it doesn't exist
                        if new_dim_name not in ds.dims:
                            ds = ds.assign_coords({
                                new_dim_name: ds[dim].copy() if dim in ds.coords else range(ds.dims[dim])
                            })
                dims_renamed.add(dim)

        # Create new variable with renamed dimensions
        data = var.values
        attrs = var.attrs.copy()
        encoding = original_encodings[var_name].copy()

        # Remove problematic encoding keys
        for key in ['dimensions', 'source', 'original_shape']:
            encoding.pop(key, None)

        # Create new variable
        ds[var_name] = xr.Variable(
            dims=new_dims,
            data=data,
            attrs=attrs
        )

        # Restore encoding
        ds[var_name].encoding.update(encoding)

    return ds
