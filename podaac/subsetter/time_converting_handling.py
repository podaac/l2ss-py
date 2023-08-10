"""
Converts the time variable to datetime if xarray doesn't decode times

Parameters
----------
dataset : xr.Dataset
time_vars : list

Returns
-------
xr.Dataset
datetime.datetime
"""

import datetime
import xarray as xr
from typing import Tuple
import numpy as np
from podaac.subsetter import subset

def get_start_date(file_type):

    if file_type in ['OMI', 'MLS']:
        start_date = datetime.datetime.strptime("1993-01-01T00:00:00.00", "%Y-%m-%dT%H:%M:%S.%f")
    elif file_type in ['GPM']:
        start_date = datetime.datetime.strptime("1980-01-06T00:00:00.00", "%Y-%m-%dT%H:%M:%S.%f")
    else:
        return None

    return start_date

def convert_to_datetime(dataset: xr.Dataset, time_vars: list, file_type) -> Tuple[xr.Dataset, datetime.datetime]:

    start_date = get_start_date(file_type)

    for var in time_vars:

        if np.issubdtype(dataset[var].dtype, np.dtype(float)) or np.issubdtype(dataset[var].dtype, np.float32):
            # adjust the time values from the start date
            if start_date:
                dataset[var].values = [start_date + datetime.timedelta(seconds=i) for i in dataset[var].values]
                continue

            utc_var_name = subset.compute_utc_name(dataset)
            if utc_var_name:
                start_seconds = dataset[var].values[0]
                dataset[var].values = [datetime.datetime(i[0], i[1], i[2], hour=i[3], minute=i[4], second=i[5]) for i in dataset[utc_var_name].values]
                start_date = dataset[var].values[0] - np.timedelta64(int(start_seconds), 's')
                return dataset, start_date

        else:
            pass

    return dataset, start_date
