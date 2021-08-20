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
import numpy as np
import xarray as xr


def get_indexers_from_1d(cond):
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


def get_indexers_from_nd(cond, cut):
    """
    Get indexers from a dataset with more than 1 dimensions.

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
    rows = np.any(cond.values, axis=1)

    if cut:
        cols = np.any(cond.values, axis=0)
    else:
        cols = np.ones(len(cond.values[0]))

    # If the subsetted area is equal to the original area
    if np.all(rows) & np.all(cols):
        logging.info("Subsetted area equal to the original granule.")

    # If the subsetted area is empty
    if not np.any(rows) | np.any(cols):
        logging.info("No data within the given bounding box.")

    indexers = {
        cond.dims[0]: np.where(rows)[0],
        cond.dims[1]: np.where(cols)[0]
    }

    return indexers


def copy_empty_dataset(dataset):
    """
    Copy an dataset into a new, empty dataset. This dataset should:
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
    empty_dataset = xr.Dataset()
    for variable_name, variable in dataset.data_vars.items():
        empty_dataset[variable_name] = []
        empty_dataset[variable_name].attrs = variable.attrs
    # Copy global metadata
    empty_dataset.attrs = dataset.attrs
    return empty_dataset


def cast_type(var, var_type):
    """
    Type cast a variable into a var type.

    Parameters
    ----------
    var: xarray.core.dataarray.DataArray
        The dataarray to be type casted.
    var_type: string
        New type the variable will be type casted to.
    Returns
    -------
    xarray.core.dataarray.DataArray
        The newly type casted variable.
    """

    return var.astype(var_type)


def where(dataset, cond, cut):
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

    indexed_cond = cond.isel(**indexers)
    indexed_ds = dataset.isel(**indexers)
    new_dataset = indexed_ds.where(indexed_cond)

    # Cast all variables to their original type
    for variable_name, variable in new_dataset.data_vars.items():
        original_type = indexed_ds[variable_name].dtype
        new_type = variable.dtype

        # Check if variable has no _FillValue. If so, use original data
        if '_FillValue' not in variable.attrs:

            if original_type != new_type:
                new_dataset[variable_name] = xr.apply_ufunc(cast_type, variable,
                                                            str(original_type), dask='allowed',
                                                            keep_attrs=True)

            # Replace nans with values from original dataset. If the
            # variable has more than one dimension, copy the entire
            # variable over, otherwise use a NaN mask to copy over the
            # relevant values.
            if len(variable.shape) > 1:
                new_dataset[variable_name] = indexed_ds[variable_name]
            else:
                nan_mask = np.isnan(variable.data)
                if nan_mask.any():
                    variable.data[nan_mask] = indexed_ds[variable_name][nan_mask]

            new_dataset[variable_name].attrs = indexed_ds[variable_name].attrs
            variable.attrs = indexed_ds[variable_name].attrs
            new_dataset[variable_name].encoding['_FillValue'] = None
            variable.encoding['_FillValue'] = None

        else:
            # Manually replace nans with FillValue
            variable.data[np.isnan(variable.data)] = variable.attrs.get("_FillValue")

            if original_type != new_type:
                new_dataset[variable_name] = xr.apply_ufunc(cast_type, variable,
                                                            str(original_type), dask='allowed',
                                                            keep_attrs=True)

    return new_dataset
