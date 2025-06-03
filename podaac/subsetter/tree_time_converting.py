"""
Converts the time variable to datetime if xarray doesn't decode times

Parameters
----------
dataset : xr.Dataset
time_vars : list
instrument_type: string

Returns
-------
xr.Dataset
datetime.datetime
"""

import datetime
from typing import Tuple, Union
import xarray as xr
import numpy as np


def compute_utc_name(group: xr.Dataset) -> Union[str, None]:
    """
    Get the name of the utc variable if it is there to determine origin time
    """
    for var_name in list(group.data_vars.keys()):
        if 'utc' in var_name.lower() and 'time' in var_name.lower():
            return var_name

    return None


def get_start_date(instrument_type: str) -> Union[datetime.datetime, None]:
    """
    returns the start date based on the instrument type
    """
    if instrument_type in ['OMI', 'MLS']:
        start_date = datetime.datetime.strptime("1993-01-01T00:00:00.00", "%Y-%m-%dT%H:%M:%S.%f")
    elif instrument_type in ['GPM']:
        start_date = datetime.datetime.strptime("1980-01-06T00:00:00.00", "%Y-%m-%dT%H:%M:%S.%f")
    else:
        return None

    return start_date


def get_group_by_path(data_tree: xr.Dataset, path: str) -> xr.Dataset:
    """
    Navigate through the data tree to get the group containing the variable

    Parameters
    ----------
    data_tree : xr.Dataset
        The root dataset containing nested groups
    path : str
        Path to the variable in format '/group1/group2/var'

    Returns
    -------
    xr.Dataset
        The group containing the target variable
    str
        The variable name
    """
    parts = path.strip('/').split('/')
    current_group = data_tree

    # Navigate through all parts except the last (which is the variable name)
    for group_name in parts[:-1]:
        current_group = current_group[group_name]

    return current_group, parts[-1]


def update_coord_everywhere(node, coord_name, new_values):
    """
    Update the coordinate in this node, all parents, and all descendants
    to ensure alignment in the datatree.
    """
    # Update this node if it has the coordinate
    if coord_name in node.ds.coords:
        dims = node.ds[coord_name].dims
        node.ds = node.ds.assign_coords({coord_name: (dims, new_values)})

    # Update all parents
    parent = node.parent
    while parent is not None:
        if coord_name in parent.ds.coords:
            dims = parent.ds[coord_name].dims
            parent.ds = parent.ds.assign_coords({coord_name: (dims, new_values)})
        parent = parent.parent

    # Update all descendants recursively
    for child in node.children.values():
        update_coord_everywhere(child, coord_name, new_values)


def convert_to_datetime(data_tree: xr.Dataset, time_vars: list, instrument_type: str) -> Tuple[xr.Dataset, datetime.datetime]:
    """
    Convert time variables in a data tree from seconds since the start date to datetime format.
    Handles nested group structures with time variables specified as '/group1/group2/var'.

    Parameters
    ----------
    data_tree : xr.Dataset
        The root dataset containing nested groups
    time_vars : list
        List of paths to time variables in format '/group1/group2/var'
    instrument_type : str
        Type of instrument to determine the start date

    Returns
    -------
    xr.Dataset
        Modified data tree with converted datetime values
    datetime.datetime
        Start date used for conversion
    """
    start_date = get_start_date(instrument_type)
    for var_path in time_vars:
        group, var_name = get_group_by_path(data_tree, var_path)

        if np.issubdtype(group[var_name].dtype, np.dtype(float)) or np.issubdtype(group[var_name].dtype, np.float32):

            # adjust the time values from the start date
            if start_date:
                # create array of the start time in datetime format
                date_time_array = np.full(group[var_name].shape, start_date)
                # add seconds since the start time to the start time to get the time at the data point
                new_values = date_time_array.astype("datetime64[ns]") + group[var_name].astype('timedelta64[s]').values
                try:
                    group[var_name].values = date_time_array.astype("datetime64[ns]") + group[var_name].astype('timedelta64[s]').values
                except ValueError:
                    pass
                update_coord_everywhere(data_tree, var_name, new_values)
                continue

            # if there isn't a start_date, get it from the UTC variable
            utc_var_name = compute_utc_name(group)
            if utc_var_name:
                start_seconds = group[var_name].values[0]
                new_values = [datetime.datetime(i[0], i[1], i[2], hour=i[3], minute=i[4], second=i[5])
                              for i in group[utc_var_name].values]
                start_date = new_values[0] - np.timedelta64(int(start_seconds), 's')
                update_coord_everywhere(data_tree, var_name, new_values)
                return data_tree, start_date
        else:
            pass

    return data_tree, start_date
