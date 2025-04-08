# Copyright 2019, by the California Institute of Technology.
# ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.
# Any commercial use must be negotiated with the Office of Technology
# Transfer at the California Institute of Technology.
#
# This software may be subject to U.S. export control laws. By accepting
# this software, the user agrees to comply with all applicable U.S. export
# laws and regulations. User has the responsibility to obtain export
# licenses, or other export authority as may be required before exporting
# such information to foreign countries or providing access to foreign
# persons.

"""
======================
xarray_enhancements.py
======================

Functions which improve upon existing xarray functionality, optimized
for this specific use-case.
"""

import logging
from typing import Union

import numpy as np
import xarray as xr
from podaac.subsetter import dimension_cleanup as dc


def get_indexers_from_1d(cond: xr.Dataset) -> dict:
    """
    Get indexers from a dataset with 1 dimension.

    Parameters
    ----------
    cond : xarray.Dataset
        Contains the result of the initial lat lon condition.

    Returns
    -------
    dict
        Indexer dictionary for the provided condition.
    """
    cols = cond.values

    if not cols.any():
        logging.info("No data within the given bounding box.")

    indexers = {
        cond.dims[0]: np.where(cols)[0]
    }
    return indexers


def get_indexers_from_nd(cond: xr.Dataset, cut: bool) -> dict:
    """
    Get indexers from a dataset with more than one dimension.

    Parameters
    ----------
    cond : xarray.Dataset
        Contains the result of the initial lat lon condition.
    cut : bool
        True if the scanline should be cut.

    Returns
    -------
    dict
        Indexer dictionary for the provided condition.
    """
    # check if the lat/lon coordinate numpy array has 2 or more dimensions
    transpose = dim_grid = False
    ndim = cond.values.squeeze().ndim

    # Determine axes and flags
    if ndim == 2:
        x_axis, y_axis = 1, 0
    else:
        if 'xtrack' in cond.dims and 'atrack' in cond.dims:
            x_axis, y_axis = cond.dims.index('xtrack'), cond.dims.index('atrack')
            transpose = True
        elif 'xdim_grid' in cond.dims and 'ydim_grid' in cond.dims:
            x_axis, y_axis = cond.dims.index('xdim_grid'), cond.dims.index('ydim_grid')
            dim_grid = x_axis == 1 and y_axis == 0
        else:
            x_axis, y_axis = 2, 1

    # Compute rows and columns
    squeezed_values = cond.values.squeeze()
    rows = np.any(squeezed_values, axis=x_axis)
    cols = np.any(squeezed_values, axis=y_axis) if cut else np.ones(len(squeezed_values[0]))

    # Log information about subsetted area
    if np.all(rows) and np.all(cols):
        logging.info("Subsetted area equal to the original granule.")
    if not np.any(rows) or not np.any(cols):
        logging.info("No data within the given bounding box.")

    # Determine dimensions and clean them up
    cond_dims = list(cond.dims)
    cond_shape = list(cond.shape)
    cond_dims = [dim for dim, size in zip(cond_dims, cond_shape) if size > 1]

    # Adjust for 3D data
    if rows.ndim > 1:
        if transpose:
            rows, cols = rows.transpose()[0], cols.transpose()[0]
        elif not dim_grid:
            rows, cols = rows[0], cols[0]

    # Generate indexers
    indexers = {
        cond_dims[y_axis]: np.where(rows)[0],
        cond_dims[x_axis]: np.where(cols)[0]
    }

    return indexers


def copy_empty_dataset(dataset: xr.Dataset) -> xr.Dataset:
    """
    Copy a dataset into a new, empty dataset. This dataset should:
        * Contain the same structure as the input dataset (only include
          requested variables, if variable subset)
        * Contain the same global metadata as the input dataset
        * Contain a history field which describes this subset operation.

    Parameters
    ----------
    dataset: xarray.Dataset
        The dataset to copy into a empty dataset.

    Returns
    -------
    xarray.Dataset
        The new dataset which has no data.
    """
    # Create a dict object where each key is a variable in the dataset and the value is an
    # array initialized to the fill value for that variable or NaN if there is no fill value
    # attribute for the variable

    empty_data = {k: np.full(v.shape, dataset.variables[k].attrs.get('_FillValue', np.nan), dtype=v.dtype) for k, v in dataset.items()}

    # Create a copy of the dataset filled with the empty data. Then select the first index along each
    # dimension and return the result
    return dataset.copy(data=empty_data).isel({dim: slice(0, 1, 1) for dim in dataset.dims})


def cast_type(var: xr.DataArray, var_type: str) -> xr.DataArray:
    """
    Type cast a variable into a var type.

    Parameters
    ----------
    var: xr.DataArray
        The dataarray to be type casted.
    var_type: string
        New type the variable will be type casted to.
    Returns
    -------
    xr.DataArray
        The newly type casted variable.
    """

    return var.astype(var_type)


def get_variables_with_indexers(dataset, indexers):
    """
    returns a list of variables with bounding box dimensions and variables that
    don't have bounding box dimensions
    """
    index_list = list(indexers.keys())
    subset_vars = []
    no_subset_vars = []
    for i in list(dataset.variables.keys()):
        variable_dims = list(dataset[i].dims)
        if any(item in index_list for item in variable_dims):
            subset_vars.append(i)
        else:
            no_subset_vars.append(i)

    return subset_vars, no_subset_vars


def where(dataset: xr.Dataset, cond: Union[xr.Dataset, xr.DataArray], cut: bool, pixel_subset=False) -> xr.Dataset:
    """
    Return a dataset which meets the given condition.

    This is a modification of the existing xarray 'where' function.
    https://github.com/pydata/xarray/blob/master/xarray/core/common.py#L999

    Parameters
    ----------
    dataset : xarray.Dataset
        The dataset to filter and return.
    cond : DataArray or Dataset with boolean dtype
        Locations at which to preserve this object's values.
    cut : boolean
        True if the scanline should be cut, False if the scanline should
        not be cut.
    pixel_subset : boolean
        Cut the lon lat based on the rows and columns within the bounding box,
        but could result with lon lats that are outside the bounding box
    Returns
    -------
    xarray.Dataset
        The filtered Dataset

    Notes
    -----
    The `cond` variable contains a boolean mask of valid data indices.
    However in that mask, True represents valid data and False
    represents invalid data.
    """

    if cond.values.ndim == 1:
        indexers = get_indexers_from_1d(cond)
    else:
        indexers = get_indexers_from_nd(cond, cut)
    # If any of the indexer dimensions are empty, return an empty dataset
    if not all(len(value) > 0 for value in indexers.values()):
        return copy_empty_dataset(dataset)

    # This will be true if any variables in the dataset have a partial
    # overlap with the coordinate dims. If so, the cond should be
    # applied per-variable rather than to the entire dataset.
    partial_dim_in_in_vars = np.any(
        [len(set(indexers.keys()).intersection(var.dims)) > 0 and len(
            indexers.keys() - var.dims) > 0 for _, var in dataset.variables.items()]
    )

    indexed_cond = cond.isel(**indexers)
    indexed_ds = dataset.isel(**indexers)

    if pixel_subset:
        # Directly assign indexed_ds to new_dataset when pixel_subset is True
        new_dataset = indexed_ds
    else:
        # Get variables that should and shouldn't be subsetted
        subset_vars, non_subset_vars = get_variables_with_indexers(dataset, indexers)

        # Subset the indexed dataset based on the condition
        subsetted_data = indexed_ds[subset_vars].where(indexed_cond)

        # Extract data for variables that shouldn't be subsetted
        non_subsetted_data = indexed_ds[non_subset_vars]

        # Merge the subsetted and non-subsetted datasets
        new_dataset = xr.merge([non_subsetted_data, subsetted_data])

        process_dataset_variables(
            new_dataset=new_dataset,
            indexed_ds=indexed_ds,
            dataset=dataset,
            indexers=indexers,
            partial_dim_in_in_vars=partial_dim_in_in_vars,
            cond=cond,
            pixel_subset=pixel_subset
        )

    dc.sync_dims_inplace(dataset, new_dataset)
    return new_dataset


def process_dataset_variables(new_dataset, indexed_ds, dataset, indexers, partial_dim_in_in_vars, cond, pixel_subset=False):
    """
    Process dataset variables by handling type casting, fill values, and dimension indexing.

    Parameters:
    -----------
    new_dataset : xarray.Dataset
        The target dataset to be modified
    indexed_ds : xarray.Dataset
        The indexed source dataset
    dataset : xarray.Dataset
        The original dataset
    indexers : dict
        Dictionary of dimension indices
    partial_dim_in_in_vars : bool
        Flag indicating if partial dimensions are in variables
    cond : xarray.DataArray
        Condition array for filtering
    pixel_subset : bool, optional
        Flag for pixel subsetting, defaults to False

    Returns:
    --------
    None (modifies new_dataset in place)
    """

    # Cast all variables to their original type
    for variable_name, variable in new_dataset.data_vars.items():
        original_type = indexed_ds[variable_name].dtype
        new_type = variable.dtype
        indexed_var = indexed_ds[variable_name]

        if partial_dim_in_in_vars and (indexers.keys() - dataset[variable_name].dims) and set(
                indexers.keys()).intersection(dataset[variable_name].dims):

            missing_dim = (indexers.keys() - dataset[variable_name].dims).pop()  # Assume only 1
            var_indexers = {
                dim_name: dim_value for dim_name, dim_value in indexers.items()
                if dim_name in dataset[variable_name].dims
            }

            var_cond = cond.any(axis=cond.dims.index(missing_dim)).isel(**var_indexers)
            indexed_var = dataset[variable_name].isel(**var_indexers)
            new_dataset[variable_name] = indexed_var.where(var_cond)
            variable = new_dataset[variable_name]
        elif partial_dim_in_in_vars and (indexers.keys() - dataset[variable_name].dims) and set(
                indexers.keys()).intersection(new_dataset[variable_name].dims):
            new_dataset[variable_name] = indexed_var
            new_dataset[variable_name].attrs = indexed_var.attrs
            variable.attrs = indexed_var.attrs

        # Check if variable has no _FillValue. If so, use original data
        if not pixel_subset:
            if '_FillValue' not in variable.attrs or len(indexed_var.shape) == 0:
                if original_type != new_type:
                    new_dataset[variable_name] = xr.apply_ufunc(cast_type, variable,
                                                                str(original_type), dask='allowed',
                                                                keep_attrs=True)

                # Replace nans with values from original dataset. If the
                # variable has more than one dimension, copy the entire
                # variable over, otherwise use a NaN mask to copy over the
                # relevant values.
                new_dataset[variable_name] = indexed_var
                new_dataset[variable_name].attrs = indexed_var.attrs
                variable.attrs = indexed_var.attrs
                new_dataset[variable_name].encoding['_FillValue'] = None
                variable.encoding['_FillValue'] = None

            else:
                # Manually replace nans with FillValue
                # If variable represents time, cast _FillValue to datetime
                fill_value = new_dataset[variable_name].attrs.get('_FillValue')
                if np.issubdtype(new_dataset[variable_name].dtype, np.dtype(np.datetime64)):
                    fill_value = np.datetime64('nat')
                if np.issubdtype(new_dataset[variable_name].dtype, np.dtype(np.timedelta64)):
                    fill_value = np.timedelta64('nat')
                new_dataset[variable_name] = new_dataset[variable_name].fillna(fill_value)
                if original_type != new_type:
                    new_dataset[variable_name] = xr.apply_ufunc(cast_type, new_dataset[variable_name],
                                                                str(original_type), dask='allowed',
                                                                keep_attrs=True)
