"""
===============
time_utils.py
===============

Utility functions for time and temporal condition handling.
"""

import datetime
import operator
import functools
import re
import pandas as pd
import numpy as np
from dateutil import parser
import julian
import xarray as xr


def _translate_timestamp(str_timestamp: str) -> datetime.datetime:
    """
    Translate timestamp to datetime object
    """
    allowed_ts_formats = [
        '%Y-%m-%dT%H:%M:%SZ',
        '%Y-%m-%dT%H:%M:%S%Z',
        '%Y-%m-%dT%H:%M:%S.%fZ',
        '%Y-%m-%dT%H:%M:%S.%f%Z',
        '%Y-%m-%d %H:%M:%S',
    ]
    for timestamp_format in allowed_ts_formats:
        try:
            return datetime.datetime.strptime(str_timestamp, timestamp_format)
        except ValueError:
            pass
    return datetime.datetime.fromisoformat(str_timestamp)


def _datetime_from_mjd(dataset: xr.Dataset, time_var_name: str):
    """
    Translate the modified julian date from the long name in the time attribute.
    """
    time_var = dataset[time_var_name]
    if 'long_name' in time_var.attrs:
        mdj_string = time_var.attrs['long_name']
        mjd = mdj_string[mdj_string.find("(") + 1:mdj_string.find(")")].split("= ")[1]
        try:
            mjd_float = float(mjd)
        except ValueError:
            return None
        mjd_datetime = julian.from_jd(mjd_float, fmt='mjd')
        return mjd_datetime
    return None


def _extract_epoch(description: str) -> str:
    """
    Extracts the ISO 8601 epoch from a description string.
    Example: "seconds since 1 January 1990" → "1990-01-01T00:00:00"
    """
    match = re.search(r'seconds since (.+)', description, re.IGNORECASE)
    if not match:
        raise ValueError("Epoch not found in description.")
    date_str = match.group(1).strip(" )")
    parsed_date = parser.parse(date_str)
    return parsed_date.isoformat()


def _convert_time_from_description(seconds_since, description: str):
    """
    Convert time array from seconds-since format using the description.
    """
    epoch = _extract_epoch(description)
    epoch_dt64 = np.datetime64(epoch, 's')
    delta = np.array(seconds_since) * np.timedelta64(1, 's')
    result = epoch_dt64 + delta
    if isinstance(seconds_since, xr.DataArray):
        return xr.DataArray(result, coords=seconds_since.coords, dims=seconds_since.dims, attrs=seconds_since.attrs)
    return result


def build_temporal_cond(min_time: str, max_time: str, dataset: xr.Dataset, time_var_name: str):
    """
    Build the temporal condition used in the xarray 'where' call which
    drops data not in the given bounds.
    """
    def build_cond(str_timestamp, compare):
        timestamp = pd.to_datetime(_translate_timestamp(str_timestamp)).to_datetime64()
        time_data = dataset[time_var_name]
        dtype = time_data.dtype
        if np.issubdtype(dtype, np.datetime64):
            pass
        elif np.issubdtype(dtype, np.timedelta64):
            if _is_time_mjd(dataset, time_var_name):
                mjd_datetime = _datetime_from_mjd(dataset, time_var_name)
                if mjd_datetime is None:
                    raise ValueError('Unable to get datetime from dataset to calculate time delta')
                timestamp -= np.datetime64(mjd_datetime)
            else:
                epoch_var = _get_time_epoch_var(dataset, time_var_name)
                epoch_datetime = dataset[epoch_var].values[0]
                timestamp -= epoch_datetime
        elif np.issubdtype(dtype, np.floating):
            description = time_data.attrs.get('description')
            long_name = time_data.attrs.get('long_name')
            if description:
                epoch_datetime = _extract_epoch(description)
                time_data = _convert_time_from_description(time_data, description)
            elif long_name == "Approximate observation time for each row":
                start_time = dataset.attrs.get('REV_START_TIME')
                date_str = start_time.split("T")[0]
                start_date = pd.to_datetime(date_str, format="%Y-%j")
                time_data = np.datetime64(start_date) + time_data.values.astype('timedelta64[s]')
        if getattr(time_data, 'long_name', None) == "reference time of sst file":
            base_time = dataset['time'].astype('datetime64[s]')
            offset = dataset['sst_dtime'].astype('timedelta64[s]')
            time_data = base_time + offset
        return compare(time_data, timestamp)
    temporal_conds = []
    if min_time:
        temporal_conds.append(build_cond(min_time, operator.ge))
    if max_time:
        temporal_conds.append(build_cond(max_time, operator.le))
    return functools.reduce(operator.and_, temporal_conds, True)


def _get_time_epoch_var(tree, time_var_name: str) -> str:
    """
    Get the name of the epoch time var. This is only needed in the case
    where there is a single time var (of size 1) that contains the time
    epoch used by the actual time var.
    """
    path_parts = time_var_name.split('/')
    group_path = '/'.join(path_parts[:-1])
    var_name = path_parts[-1]
    dataset = tree[group_path].ds if group_path else tree.ds
    time_var = dataset[var_name]
    if 'comment' in time_var.attrs:
        epoch_var_name = time_var.attrs['comment'].split('plus')[0].strip()
    elif 'time' in dataset.variables.keys() and var_name != 'time':
        epoch_var_name = f"{group_path}/time" if group_path else "time"
    elif any('time' in s for s in list(dataset.variables.keys())) and var_name != 'time':
        for i in list(dataset.variables.keys()):
            if i.endswith('time'):
                epoch_var_name = f"{group_path}/{i}" if group_path else i
                break
        else:
            raise ValueError('Unable to determine time variables')
        return epoch_var_name
    else:
        raise ValueError('Unable to determine time variables')
    return epoch_var_name


def _is_time_mjd(dataset: xr.Dataset, time_var_name: str) -> bool:
    """
    Check to see if the time format is a time delta from a modified julian date.
    """
    time_var = dataset[time_var_name]
    if 'comment' in time_var.attrs:
        if 'Modified Julian Day' in time_var.attrs['comment']:
            return True
    return False
