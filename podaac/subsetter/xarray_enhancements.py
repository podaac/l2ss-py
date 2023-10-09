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

    rows = np.any(cond.values.squeeze(), axis=1)
    if cut:
        cols = np.any(cond.values.squeeze(), axis=0)
    else:
        cols = np.ones(len(cond.values[0]))

    # If the subsetted area is equal to the original area
    if np.all(rows) & np.all(cols):
        logging.info("Subsetted area equal to the original granule.")

    # If the subsetted area is empty
    if not np.any(rows) | np.any(cols):
        logging.info("No data within the given bounding box.")

    cond_shape_list = list(cond.shape)
    cond_list = list(cond.dims)
    output = [idx for idx, element in enumerate(cond_shape_list) if element == 1]
    for i in output:
        cond_list.pop(i)

    indexers = {
        cond_list[0]: np.where(rows)[0],
        cond_list[1]: np.where(cols)[0]
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
    empty_data = {k: np.full(v.shape, dataset.variables[k].attrs.get('_FillValue', np.nan)) for k, v in
                  dataset.items()}

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


def where(dataset: xr.Dataset, cond: Union[xr.Dataset, xr.DataArray], cut: bool) -> xr.Dataset:
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
    subset_vars, non_subset_vars = get_variables_with_indexers(dataset, indexers)

    # dataset with variables that need to be subsetted
    new_dataset_sub = indexed_ds[subset_vars].where(indexed_cond)
    # data with variables that shouldn't be subsetted
    new_dataset_non_sub = indexed_ds[non_subset_vars]

    # merge the datasets
    new_dataset = xr.merge([new_dataset_non_sub, new_dataset_sub])

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
            var_cond = cond.sel({missing_dim: 1}).isel(**var_indexers)
            indexed_var = dataset[variable_name].isel(**var_indexers)
            new_dataset[variable_name] = indexed_var.where(var_cond)
            variable = new_dataset[variable_name]
        elif partial_dim_in_in_vars and (indexers.keys() - dataset[variable_name].dims) and set(
                indexers.keys()).intersection(new_dataset[variable_name].dims):
            new_dataset[variable_name] = indexed_var

            new_dataset[variable_name].attrs = indexed_var.attrs
            variable.attrs = indexed_var.attrs
        # Check if variable has no _FillValue. If so, use original data
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

    dc.sync_dims_inplace(dataset, new_dataset)
    return new_dataset
