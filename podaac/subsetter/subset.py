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
=========
subset.py
=========

Functions related to subsetting a NetCDF file.
"""

import datetime
import functools
import json
import operator
import os
import re
from itertools import zip_longest
from typing import Any, List, Optional, Tuple, Union
import dateutil
from dateutil import parser

import cf_xarray as cfxr
import cftime
import geopandas as gpd
import importlib_metadata
import julian
import netCDF4 as nc
import numpy as np
import pandas as pd
import xarray as xr
import xarray.coding.times
from shapely.geometry import Point, Polygon, MultiPolygon
from shapely.ops import transform

from podaac.subsetter import gpm_cleanup as gc
from podaac.subsetter import time_converting as tc
from podaac.subsetter import dimension_cleanup as dc
from podaac.subsetter import xarray_enhancements as xre
from podaac.subsetter.group_handling import GROUP_DELIM, transform_grouped_dataset, recombine_grouped_datasets, \
    h5file_transform

SERVICE_NAME = 'l2ss-py'


def apply_scale_offset(scale: float, offset: float, value: float) -> float:
    """Apply scale and offset to the given value"""
    return (value + offset) / scale


def remove_scale_offset(value: float, scale: float, offset: float) -> float:
    """Remove scale and offset from the given value"""
    return (value * scale) - offset


def convert_bound(bound: np.ndarray, coord_max: int, coord_var: xr.DataArray) -> np.ndarray:
    """
    This function will return a converted bound, which matches the
    range of the given input file.

    Parameters
    ----------
    bound : np.array
        1-dimensional 2-element numpy array which represent the lower
        and upper bounding box on this coordinate, respectively.
    coord_max : integer
        The max value which is possible given this coordinate. For
        example, the max for longitude is 360.
    coord_var : xarray.DataArray
        The xarray variable for some coordinate.

    Returns
    -------
    np.array
        1-dimensional 2-element number array which represents the lower
        and upper bounding box on this coordinate and has been converted
        based on the valid bounds coordinate range of the dataset.

    Notes
    -----
    Assumption that 0 is always on the prime meridian/equator.
    """

    scale = coord_var.attrs.get('scale_factor', 1.0)
    offset = coord_var.attrs.get('add_offset', 0.0)
    valid_min = coord_var.attrs.get('valid_min', None)

    if valid_min is None or valid_min > 0:
        # If coord var doesn't contain valid min, attempt to find
        # manually. Note: Given the perfect storm, this could still fail
        # to find the actual bounds.

        # Filter out _FillValue from data before calculating min and max
        fill_value = coord_var.attrs.get('_FillValue', None)
        var_values = coord_var.values
        if fill_value:
            var_values = np.where(var_values != fill_value, var_values, np.nan)
        var_min = np.nanmin(var_values)
        var_max = np.nanmax(var_values)

        if 0 <= var_min <= var_max <= (coord_max / scale):
            valid_min = 0

    # If the file coords are 0 --> max
    if valid_min == 0:
        bound = (bound + coord_max) % coord_max

        # If the right/top bound is 0, set to max.
        if bound[1] == 0:
            bound[1] = coord_max

        # If edges are the same, assume it wraps and return all
        if bound[0] == bound[1]:
            bound = np.array([0, coord_max])

    # If the file longitude is -coord_max/2 --> coord_max/2
    if valid_min != 0:
        # If edges are the same, assume it wraps and return all
        if bound[0] == bound[1]:
            bound = np.array([-(coord_max / 2), coord_max / 2])

    # Calculate scale and offset so the bounds match the coord data
    return apply_scale_offset(scale, offset, bound)


def convert_bbox(bbox: np.ndarray, dataset: xr.Dataset, lat_var_name: str, lon_var_name: str) -> np.ndarray:
    """
    This function will return a converted bbox which matches the range
    of the given input file. This will convert both the latitude and
    longitude range. For example, an input dataset can have a valid
    longitude range of -180 --> 180 or of 0 --> 360.

    Parameters
    ----------
    bbox : np.array
        The bounding box
    dataset : xarray.Dataset
        The dataset which is being subset.
    lat_var_name : str
        Name of the lat variable in the given dataset
    lon_var_name : str
        Name of the lon variable in the given dataset

    Returns
    -------
    bbox : np.array
        The new bbox which matches latitude and longitude ranges of the
        input file.

    Notes
    -----
    Assumption that the provided bounding box is always between
    -180 --> 180 for longitude and -90, 90 for latitude.
    """
    return np.array([convert_bound(bbox[0], 360, dataset[lon_var_name]),
                     convert_bound(bbox[1], 180, dataset[lat_var_name])])


def set_json_history(dataset: xr.Dataset, cut: bool, file_to_subset: str,
                     bbox: np.ndarray = None, shapefile: str = None, origin_source=None) -> None:
    """
    Set the 'json_history' metadata header of the granule to reflect the
    current version of the subsetter, as well as the parameters used
    to call the subsetter. This will append an json array to the json_history of
    the following format:

    Parameters
    ----------
    dataset : xarray.Dataset
        The dataset to change the header of
    cut : boolean
        True to cut the scanline
    file_to_subset : string
        The filepath of the file which was used to subset
    bbox : np.ndarray
        The requested bounding box
    shapefile : str
        Name of the shapefile to include in the version history
    TODO: add docstring and type hint for `origin_source` parameter.
    """

    params = f'cut={cut}'
    if bbox is not None:
        params = f'bbox={bbox.tolist()} {params}'
    elif shapefile is not None:
        params = f'shapefile={shapefile} {params}'

    history_json = dataset.attrs.get('history_json', [])
    if history_json:
        history_json = json.loads(history_json)

    if origin_source:
        derived_from = origin_source
    else:
        derived_from = os.path.basename(file_to_subset)

    new_history_json = {
        "date_time": datetime.datetime.now(tz=datetime.timezone.utc).isoformat(),
        "derived_from": derived_from,
        "program": SERVICE_NAME,
        "version": importlib_metadata.distribution(SERVICE_NAME).version,
        "parameters": params,
        "program_ref": "https://cmr.earthdata.nasa.gov:443/search/concepts/S1962070864-POCLOUD",
        "$schema": "https://harmony.earthdata.nasa.gov/schemas/history/0.1.0/history-v0.1.0.json"
    }

    history_json.append(new_history_json)
    dataset.attrs['history_json'] = json.dumps(history_json)


def set_version_history(dataset: xr.Dataset, cut: bool, bbox: np.ndarray = None, shapefile: str = None) -> None:
    """
    Set the 'history' metadata header of the granule to reflect the
    current version of the subsetter, as well as the parameters used
    to call the subsetter. This will append a line to the history of
    the following format:

    TIMESTAMP podaac.subsetter VERSION (PARAMS)

    Parameters
    ----------
    dataset : xarray.Dataset
        The dataset to change the header of
    cut : boolean
        True to cut the scanline
    bbox : np.ndarray
        The requested bounding box
    shapefile : str
        Name of the shapefile to include in the version history

    """

    version = importlib_metadata.distribution(SERVICE_NAME).version
    history = dataset.attrs.get('history', "")
    timestamp = datetime.datetime.utcnow()
    params = f'cut={cut}'
    if bbox is not None:
        params = f'bbox={bbox.tolist()} {params}'
    elif shapefile is not None:
        params = f'shapefile={shapefile} {params}'

    history += f"\n{timestamp} {SERVICE_NAME} v{version} ({params})"
    dataset.attrs['history'] = history.strip()


def calculate_chunks(dataset: xr.Dataset) -> dict:
    """
    For the given dataset, calculate if the size on any dimension is
    worth chunking. Any dimension larger than 4000 will be chunked. This
    is done to ensure that the variable can fit in memory.

    Parameters
    ----------
    dataset : xarray.Dataset
        The dataset to calculate chunks for.

    Returns
    -------
    dict
        The chunk dictionary, where the key is the dimension and the
        value is 4000 or 500 depending on how many dimensions.
    """
    if len(dataset.dims) <= 3:
        chunk = {dim: 4000 for dim in dataset.dims
                 if dataset.sizes[dim] > 4000
                 and len(dataset.dims) > 1}
    else:
        chunk = {dim: 500 for dim in dataset.dims
                 if dataset.sizes[dim] > 500}

    return chunk


def find_matching_coords(dataset: xr.Dataset, match_list: List[str]) -> List[str]:
    """
    As a backup for finding a coordinate var, look at the 'coordinates'
    metadata attribute of all data vars in the granule. Return any
    coordinate vars that have name matches with the provided
    'match_list'

    Parameters
    ----------
    dataset : xr.Dataset
        Dataset to search data variable coordinate metadata attribute
    match_list : list (str)
        List of possible matches to search for. For example,
        ['lat', 'latitude'] would search for variables in the
        'coordinates' metadata attribute containing either 'lat'
        or 'latitude'

    Returns
    -------
    list (str)
        List of matching coordinate variables names
    """
    coord_attrs = [
        var.attrs['coordinates'] for var_name, var in dataset.data_vars.items()
        if 'coordinates' in var.attrs
    ]
    coord_attrs = list(set(coord_attrs))
    match_coord_vars = []
    for coord_attr in coord_attrs:
        coords = coord_attr.split(' ')
        match_vars = [
            coord for coord in coords
            if any(coord_cand in coord for coord_cand in match_list)
        ]
        if match_vars and match_vars[0] in dataset:
            # Check if the var actually exists in the dataset
            match_coord_vars.append(match_vars[0])
    return match_coord_vars


def compute_coordinate_variable_names(dataset: xr.Dataset) -> Tuple[Union[List[str], str], Union[List[str], str]]:
    """
    Given a dataset, determine the coordinate variable from a list
    of options

    Parameters
    ----------
    dataset: xr.Dataset
        The dataset to find the coordinate variables for

    Returns
    -------
    tuple, str
        Tuple of strings (or list of strings), where the first element is the lat coordinate
        name and the second element is the lon coordinate name
    """

    dataset = xr.decode_cf(dataset, decode_times=False)

    # look for lon and lat using standard name in coordinates and axes
    custom_criteria = {
        "latitude": {
            "standard_name": "latitude|projection_y_coordinate",
        },
        "longitude": {
            "standard_name": "longitude|projection_x_coordinate",
        }
    }

    possible_lat_coord_names = ['lat', 'latitude', 'y']
    possible_lon_coord_names = ['lon', 'longitude', 'x']

    def var_is_coord(var_name, possible_coord_names):
        var_name = var_name.strip(GROUP_DELIM).split(GROUP_DELIM)[-1]
        return var_name.lower() in possible_coord_names

    lat_coord_names = list(filter(
        lambda var_name: var_is_coord(var_name, possible_lat_coord_names), dataset.variables))
    lon_coord_names = list(filter(
        lambda var_name: var_is_coord(var_name, possible_lon_coord_names), dataset.variables))

    if len(lat_coord_names) < 1 or len(lon_coord_names) < 1:
        lat_coord_names = find_matching_coords(dataset, possible_lat_coord_names)
        lon_coord_names = find_matching_coords(dataset, possible_lon_coord_names)

    # Couldn't find lon lat in data variables look in coordinates
    if len(lat_coord_names) < 1 or len(lon_coord_names) < 1:
        with cfxr.set_options(custom_criteria=custom_criteria):
            lat_coord_names = dataset.cf.coordinates.get('latitude', [])
            lon_coord_names = dataset.cf.coordinates.get('longitude', [])

        if len(lat_coord_names) < 1 or len(lon_coord_names) < 1:
            try:
                lat_coord_names = [dataset.cf["latitude"].name]
                lon_coord_names = [dataset.cf["longitude"].name]
            except KeyError:
                pass

    if len(lat_coord_names) < 1 or len(lon_coord_names) < 1:
        raise ValueError('Could not determine coordinate variables')

    return lat_coord_names, lon_coord_names


def is_360(lon_var: xr.DataArray, scale: float, offset: float) -> bool:
    """
    Determine if given dataset is a '360' dataset or not.

    Parameters
    ----------
    lon_var : xr.DataArray
        The lon variable from the xarray Dataset
    scale : float
        Used to remove scale and offset for easier calculation
    offset : float
        Used to remove scale and offset for easier calculation

    Returns
    -------
    bool
        True if dataset is 360, False if not. Defaults to False.
    """
    valid_min = lon_var.attrs.get('valid_min', None)

    if valid_min is None or valid_min > 0:
        var_min = remove_scale_offset(np.amin(lon_var.values), scale, offset)
        var_max = remove_scale_offset(np.amax(lon_var.values), scale, offset)

        if var_min < 0:
            return False
        if var_max > 180:
            return True

    if valid_min == 0:
        return True
    if valid_min < 0:
        return False

    return False


def get_spatial_bounds(dataset: xr.Dataset, lat_var_names: str, lon_var_names: str) -> Union[np.ndarray, None]:
    """
    Get the spatial bounds for this dataset. These values are masked
    and scaled.

    Parameters
    ----------
    dataset : xr.Dataset
        Dataset to retrieve spatial bounds for
    lat_var_names : str
        Name of the lat variable
    lon_var_names : str
        Name of the lon variable

    Returns
    -------
    np.array
        [[lon min, lon max], [lat min, lat max]]
    """

    lat_var_name = lat_var_names[0] if len(lat_var_names) == 1 else [
        lat_name for lat_name in lat_var_names if lat_name in dataset.data_vars.keys()
    ][0]
    lon_var_name = lon_var_names[0] if len(lon_var_names) == 1 else [
        lon_name for lon_name in lon_var_names if lon_name in dataset.data_vars.keys()
    ][0]

    # Get scale from coordinate variable metadata attributes
    lat_scale = dataset[lat_var_name].attrs.get('scale_factor', 1.0)
    lon_scale = dataset[lon_var_name].attrs.get('scale_factor', 1.0)
    lat_offset = dataset[lat_var_name].attrs.get('add_offset', 0.0)
    lon_offset = dataset[lon_var_name].attrs.get('add_offset', 0.0)
    lon_valid_min = dataset[lon_var_name].attrs.get('valid_min', None)
    lat_fill_value = dataset[lat_var_name].attrs.get('_FillValue', None)
    lon_fill_value = dataset[lon_var_name].attrs.get('_FillValue', None)

    # Apply mask and scale to min/max coordinate variables to get
    # spatial bounds

    # Remove fill value. Might cause errors when getting min and max
    lats = dataset[lat_var_name].values.flatten()
    lons = dataset[lon_var_name].values.flatten()

    if lat_fill_value:
        lats = list(filter(lambda a: not a == lat_fill_value, lats))
    if lon_fill_value:
        lons = list(filter(lambda a: not a == lon_fill_value, lons))

    if len(lats) == 0 or len(lons) == 0:
        return None

    original_min_lat = remove_scale_offset(np.nanmin(lats), lat_scale, lat_offset)
    original_max_lat = remove_scale_offset(np.nanmax(lats), lat_scale, lat_offset)
    original_min_lon = remove_scale_offset(np.nanmin(lons), lon_scale, lon_offset)
    original_max_lon = remove_scale_offset(np.nanmax(lons), lon_scale, lon_offset)

    min_lat = round(original_min_lat, 5)
    max_lat = round(original_max_lat, 5)
    min_lon = round(original_min_lon, 1)
    max_lon = round(original_max_lon, 1)

    # Convert longitude to [-180,180] format
    if lon_valid_min == 0 or 0 <= min_lon <= max_lon <= 360:
        min_lon = (min_lon - 360) if min_lon > 180 else min_lon
        max_lon = (max_lon - 360) if max_lon > 180 else max_lon
        if min_lon == max_lon:
            min_lon, max_lon = -180, 180

    # After rounding to 1 if not at the edges then round to 5
    if min_lon != -180:
        min_lon = round((original_min_lon - 360) if original_min_lon > 180 else original_min_lon, 5)

    if max_lon != 180:
        max_lon = round((original_max_lon - 360) if original_max_lon > 180 else original_max_lon, 5)

    return np.array([[min_lon, max_lon], [min_lat, max_lat]])


def compute_time_variable_name(dataset: xr.Dataset, lat_var: xr.Variable, total_time_vars: list) -> str:
    """
    Try to determine the name of the 'time' variable. This is done as
    follows:

    - The variable name contains 'time'
    - The variable dimensions match the dimensions of the given lat var
    - The variable that hasn't already been found

    Parameters
    ----------
    dataset : xr.Dataset
        xarray dataset to find time variable from
    lat_var : xr.Variable
        Lat variable for this dataset

    Returns
    -------
    str
        The name of the variable

    Raises
    ------
    ValueError
        If the time variable could not be determined
    """

    time_vars = find_matching_coords(dataset, ['time'])
    if time_vars:
        # There should only be one time var match (this is called once
        # per lat var)
        return time_vars[0]

    # Filter variables with 'time' in the name to avoid extra work
    time_vars = list(filter(lambda var_name: 'time' in var_name, dataset.sizes.keys()))

    for var_name in time_vars:
        if var_name not in total_time_vars and "time" in var_name and dataset[var_name].squeeze().dims == lat_var.squeeze().dims:
            return var_name

    # first check if any variables are named 'time'
    for var_name in list(dataset.data_vars.keys()):
        var_name_time = var_name.strip(GROUP_DELIM).split(GROUP_DELIM)[-1]
        if len(dataset[var_name].squeeze().dims) == 0:
            continue
        if var_name not in total_time_vars and ('time' == var_name_time.lower() or 'timeMidScan' == var_name_time) and dataset[var_name].squeeze().dims[0] in lat_var.squeeze().dims:
            return var_name

    time_units_pattern = re.compile(r"(days|d|hours|hr|h|minutes|min|m|seconds|sec|s) since \d{4}-\d{2}-\d{2}( \d{2}:\d{2}:\d{2})?")
    # Check variables for common time variable indicators
    for var_name, var in dataset.variables.items():
        # pylint: disable=too-many-boolean-expressions
        if ((('standard_name' in var.attrs and var.attrs['standard_name'] == 'time') or
           ('axis' in var.attrs and var.attrs['axis'] == 'T') or
           ('units' in var.attrs and time_units_pattern.match(var.attrs['units'])))) and var_name not in total_time_vars:
            return var_name

    # then check if any variables have 'time' in the string if the above loop doesn't return anything
    for var_name in list(dataset.data_vars.keys()):
        var_name_time = var_name.strip(GROUP_DELIM).split(GROUP_DELIM)[-1]
        if len(dataset[var_name].squeeze().dims) == 0:
            continue
        if var_name not in total_time_vars and 'time' in var_name_time.lower() and dataset[var_name].squeeze().dims[0] in lat_var.squeeze().dims:
            return var_name

    return None


def compute_utc_name(dataset: xr.Dataset) -> Union[str, None]:
    """
    Get the name of the utc variable if it is there to determine origine time
    """
    for var_name in list(dataset.data_vars.keys()):
        if 'utc' in var_name.lower() and 'time' in var_name.lower():
            return var_name

    return None


def translate_longitude(geometry):
    """
    Translates the longitude values of a Shapely geometry from the range [-180, 180) to [0, 360).

    Parameters
    ----------
    geometry : shapely.geometry.base.BaseGeometry
        The input shape geometry to be translated

    Returns
    -------
    geometry
        The translated shape geometry
    """

    def translate_point(point):
        # Translate the point's x-coordinate (longitude) by adding 360
        return Point((point.x + 360) % 360, point.y)

    def translate_polygon(polygon):
        def translate_coordinates(coords):
            if len(coords[0]) == 2:
                return [((x + 360) % 360, y) for x, y in coords]
            if len(coords[0]) == 3:
                return [((x + 360) % 360, y, z) for x, y, z in coords]
            return coords

        exterior = translate_coordinates(polygon.exterior.coords)

        interiors = [
            translate_coordinates(ring.coords)
            for ring in polygon.interiors
        ]

        return Polygon(exterior, interiors)

    if isinstance(geometry, (Point, Polygon)):  # pylint: disable=no-else-return
        return translate_point(geometry) if isinstance(geometry, Point) else translate_polygon(geometry)
    elif isinstance(geometry, MultiPolygon):
        # Translate each polygon in the MultiPolygon
        translated_polygons = [translate_longitude(subgeometry) for subgeometry in geometry.geoms]
        return MultiPolygon(translated_polygons)
    else:
        # Handle other geometry types as needed
        return geometry


def get_time_epoch_var(dataset: xr.Dataset, time_var_name: str) -> str:
    """
    Get the name of the epoch time var. This is only needed in the case
    where there is a single time var (of size 1) that contains the time
    epoch used by the actual time var.

    Parameters
    ----------
    dataset : xr.Dataset
        Dataset that contains time var
    time_var_name : str
        The name of the actual time var (with matching dims to the
        coord vars)

    Returns
    -------
    str
        The name of the epoch time variable
    """
    time_var = dataset[time_var_name]

    if 'comment' in time_var.attrs:
        epoch_var_name = time_var.attrs['comment'].split('plus')[0].strip()
    elif 'time' in dataset.variables.keys() and time_var_name != 'time':
        epoch_var_name = 'time'
    elif any('time' in s for s in list(dataset.variables.keys())) and time_var_name != 'time':
        for i in list(dataset.variables.keys()):
            group_list = i.split(GROUP_DELIM)
            if group_list[-1] == 'time':
                epoch_var_name = i
                break
        return epoch_var_name
    else:
        raise ValueError('Unable to determine time variables')

    return epoch_var_name


def is_time_mjd(dataset: xr.Dataset, time_var_name: str) -> bool:
    """
    Check to see if the time format is a time delta from a modified julian date.

    Parameters
    ----------
    dataset : xr.Dataset
        Dataset that contains time var
    time_var_name : str
        The name of the actual time var (with matching dims to the
        coord vars)

    Returns
    -------
    boolean
        is time delta format in modified julian date
    """
    time_var = dataset[time_var_name]
    if 'comment' in time_var.attrs:
        if 'Modified Julian Day' in time_var.attrs['comment']:
            return True
    return False


def translate_timestamp(str_timestamp: str) -> datetime.datetime:
    """
    Translate timestamp to datetime object

    Parameters
    ----------
    str_timestamp : str
        Timestamp string. ISO or RFC

    Returns
    -------
    datetime
        Constructed Datetime object
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


def datetime_from_mjd(dataset: xr.Dataset, time_var_name: str) -> Union[datetime.datetime, None]:
    """
    Translate the modified julian date from the long name in the time attribute.

    Parameters
    ----------
    dataset : xr.Dataset
        Dataset that contains time var
    time_var_name : str
        The name of the actual time var (with matching dims to the
        coord vars)

    Returns
    -------
    datetime
        the datetime of the modified julian date
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


def build_temporal_cond(min_time: str, max_time: str, dataset: xr.Dataset, time_var_name: str) -> Union[np.ndarray, bool]:
    """
    Build the temporal condition used in the xarray 'where' call which
    drops data not in the given bounds. If the data in the time var is
    of type 'datetime', assume this is a normal case where the time var
    uses the epoch from the 'units' metadata attribute to get epoch. If
    the data in the time var is of type 'timedelta', the epoch var is
    needed to calculate the datetime.

    Parameters
    ----------
    min_time : str
        ISO timestamp representing the lower temporal bound
    max_time : str
        ISO timestamp representing the upper temporal bound
    dataset : xr.Dataset
        Dataset to build the condition off of
    time_var_name : str
        Name of the time variable

    Returns
    -------
    np.ndarray or bool
        If temporally subsetted, returns a boolean ND-array the shape
        of which matches the dimensions of the coordinate vars. 'True'
        is essentially a noop.
    """

    def build_cond(str_timestamp, compare):
        timestamp = translate_timestamp(str_timestamp)
        time_data = dataset[time_var_name]

        if np.issubdtype(time_data.dtype, np.datetime64):
            timestamp = pd.to_datetime(timestamp).to_datetime64()
        elif np.issubdtype(time_data.dtype, np.timedelta64):
            if is_time_mjd(dataset, time_var_name):
                mjd_datetime = datetime_from_mjd(dataset, time_var_name)
                if mjd_datetime is None:
                    raise ValueError('Unable to get datetime from dataset to calculate time delta')
                timestamp = np.datetime64(timestamp) - np.datetime64(mjd_datetime)
            else:
                epoch_time_var_name = get_time_epoch_var(dataset, time_var_name)
                epoch_datetime = dataset[epoch_time_var_name].values[0]
                timestamp = np.datetime64(timestamp) - epoch_datetime

        if getattr(time_data, 'long_name', None) == "reference time of sst file":
            timedelta_seconds = dataset['sst_dtime'].astype('timedelta64[s]')
            time_data = time_data + timedelta_seconds

        return compare(time_data, timestamp)

    temporal_conds = []
    if min_time:
        temporal_conds.append(build_cond(min_time, operator.ge))
    if max_time:
        temporal_conds.append(build_cond(max_time, operator.le))

    if temporal_conds:
        temporal_cond = functools.reduce(lambda cond_a, cond_b: cond_a & cond_b, temporal_conds)
    else:
        temporal_cond = True

    return temporal_cond


def get_base_group_names(lats: List[str]) -> Tuple[List[str], List[Union[int, str]]]:  # pylint: disable=too-many-branches
    """Latitude groups may be at different depths. This function gets the level
    number that makes each latitude group unique from the other latitude names"""
    unique_groups = []
    group_list = [lat.strip(GROUP_DELIM).split(GROUP_DELIM) for lat in lats]

    # make all lists of group levels the same length
    group_list = list(zip(*zip_longest(*group_list, fillvalue='')))

    # put the groups in the same levels in the same list
    group_list_transpose = np.array(group_list).T.tolist()

    diff_count = ['' for _ in range(len(group_list))]
    group_count = 0
    # loop through each group level
    for my_list in group_list_transpose:
        for i in range(len(my_list)):  # pylint: disable=consider-using-enumerate
            count = 0
            for j in range(len(my_list)):  # pylint: disable=consider-using-enumerate
                # go through each lat name and compare the level names
                if my_list[i] == my_list[j] and not isinstance(diff_count[j], int):
                    count += 1
            # if the lat names is equivalent to only itself then insert the level number
            if count == 1:
                if isinstance(diff_count[i], int):
                    continue
                if 'lat' in my_list[i].lower():  # if we get to the end of the list, go to the previous level
                    diff_count[i] = group_count - 1
                    continue

                diff_count[i] = group_count

        group_count += 1

    # go back and re-put together the unique groups
    for lat in enumerate(lats):
        unique_groups.append(f'{GROUP_DELIM}{GROUP_DELIM.join(lat[1].strip(GROUP_DELIM).split(GROUP_DELIM)[:(diff_count[lat[0]]+1)])}')
    return unique_groups, diff_count


def subset_with_bbox(dataset: xr.Dataset,  # pylint: disable=too-many-branches
                     lat_var_names: list,
                     lon_var_names: list,
                     time_var_names: list,
                     variables: Optional[List[str]] = None,
                     bbox: np.ndarray = None,
                     cut: bool = True,
                     min_time: str = None,
                     max_time: str = None,
                     pixel_subset: bool = False) -> np.ndarray:
    """
    Subset an xarray Dataset using a spatial bounding box.

    Parameters
    ----------
    dataset : xr.Dataset
        Dataset to subset
    lat_var_names : list
        Name of the latitude variables in the given dataset
    lon_var_names : list
        Name of the longitude variables in the given dataset
    time_var_names : list
        Name of the time variables in the given dataset
    variables : list[str]
        List of variables to include in the result
    bbox : np.array
        Spatial bounding box to subset Dataset with.
    cut : bool
        True if scanline should be cut.
    min_time : str
        ISO timestamp of min temporal bound
    max_time : str
        ISO timestamp of max temporal bound
    pixel_subset : boolean
        Cut the lon lat based on the rows and columns within the bounding box,
        but could result with lon lats that are outside the bounding box
    TODO: add docstring and type hint for `variables` parameter.

    Returns
    -------
    np.array
        Spatial bounds of Dataset after subset operation
    TODO - fix this docstring type and the type hint to match code (currently returning a list[xr.Dataset])
    """
    lon_bounds, lat_bounds = convert_bbox(bbox, dataset, lat_var_names[0], lon_var_names[0])
    # condition should be 'or' instead of 'and' when bbox lon_min > lon_max
    oper = operator.and_

    if lon_bounds[0] > lon_bounds[1]:
        oper = operator.or_

    # get unique group names for latitude coordinates
    diff_count = [-1]
    if len(lat_var_names) > 1:
        unique_groups, diff_count = get_base_group_names(lat_var_names)
    else:
        unique_groups = [f'{GROUP_DELIM}{GROUP_DELIM.join(x.strip(GROUP_DELIM).split(GROUP_DELIM)[:-1])}' for x in lat_var_names]

    datasets = []
    total_list = []  # don't include repeated variables

    for lat_var_name, lon_var_name, time_var_name, diffs in zip_longest(lat_var_names, lon_var_names, time_var_names, diff_count, fillvalue=None):

        # If there is an extra time variable with no lon lat zip to it
        if lon_var_name is None and lat_var_name is None:
            continue

        if GROUP_DELIM in lat_var_name:
            lat_var_prefix = GROUP_DELIM.join(lat_var_name.strip(GROUP_DELIM).split(GROUP_DELIM)[:(diffs+1)])

            if diffs == -1:  # if the lat name is in the root group: take only the root group vars
                group_vars = list(dataset.data_vars.keys())
                # include the coordinate variables if user asks for
                group_vars.extend([
                        var for var in list(dataset.coords.keys())
                        if var in variables and var not in group_vars
                    ])
            else:
                group_vars = [
                    var for var in dataset.data_vars.keys()
                    if GROUP_DELIM.join(var.strip(GROUP_DELIM).split(GROUP_DELIM)[:(diffs+1)]) == lat_var_prefix
                ]
                # include variables that aren't in a latitude group
                if variables:
                    group_vars.extend([
                        var for var in dataset.variables.keys()
                        if (var in variables and
                            var not in group_vars and
                            var not in total_list and
                            not var.startswith(tuple(unique_groups))
                            )
                    ])
                else:
                    group_vars.extend([
                        var for var in dataset.data_vars.keys()
                        if (var not in group_vars and
                            var not in total_list and
                            not var.startswith(tuple(unique_groups))
                            )
                        ])

                # group dimensions do not get carried over if unused by data variables (MLS nTotalTimes var)
                # get all dimensions from data variables
                dim_list = []
                for var in group_vars:
                    dim_list.extend(list(list(dataset[var].dims)))
                # get all group dimensions
                group_dims = [
                    dim for dim in list(dataset.coords.keys())
                    if GROUP_DELIM.join(dim.strip(GROUP_DELIM).split(GROUP_DELIM)[:(diffs+1)]) == lat_var_prefix
                ]
                # include any group dimensions that aren't accounted for in variable dimensions
                var_included = list(set(group_dims) - set(dim_list))
                group_vars.extend(var_included)

        else:
            group_vars = list(dataset.keys())

        group_dataset = dataset[group_vars]

        # Calculate temporal conditions
        temporal_cond = build_temporal_cond(min_time, max_time, group_dataset, time_var_name)
        group_dataset = xre.where(
            group_dataset,
            oper(
                (group_dataset[lon_var_name] >= lon_bounds[0]),
                (group_dataset[lon_var_name] <= lon_bounds[1])
            ) &
            (group_dataset[lat_var_name] >= lat_bounds[0]) &
            (group_dataset[lat_var_name] <= lat_bounds[1]) &
            temporal_cond,
            cut,
            pixel_subset=pixel_subset
        )

        datasets.append(group_dataset)
        total_list.extend(group_vars)
        if diffs == -1:
            return datasets

    dim_cleaned_datasets = dc.recreate_pixcore_dimensions(datasets)
    return dim_cleaned_datasets


def subset_with_shapefile(dataset: xr.Dataset,
                          lat_var_name: str,
                          lon_var_name: str,
                          shapefile: str,
                          cut: bool,
                          chunks) -> np.ndarray:
    """
    Subset an xarray Dataset using a shapefile

    Parameters
    ----------
    dataset : xr.Dataset
        Dataset to subset
    lat_var_name : str
        Name of the latitude variable in the given dataset
    lon_var_name : str
        Name of the longitude variable in the given dataset
    shapefile : str
        Absolute path to the shapefile used to subset the given dataset
    cut : bool
        True if scanline should be cut.
    TODO: add docstring and type hint for `chunks` parameter.

    Returns
    -------
    np.array
        Spatial bounds of Dataset after shapefile subset operation
    TODO - fix this docstring type and the type hint to match code (currently returning a xr.Dataset)
    """
    shapefile_df = gpd.read_file(shapefile)

    lat_scale = dataset[lat_var_name].attrs.get('scale_factor', 1.0)
    lon_scale = dataset[lon_var_name].attrs.get('scale_factor', 1.0)
    lat_offset = dataset[lat_var_name].attrs.get('add_offset', 0.0)
    lon_offset = dataset[lon_var_name].attrs.get('add_offset', 0.0)

    # If data is '360', convert shapefile to '360' as well. There is an
    # assumption that the shapefile is -180,180.
    if is_360(dataset[lon_var_name], lon_scale, lon_offset):
        # Transform
        shapefile_df.geometry = shapefile_df['geometry'].apply(translate_longitude)

    # Mask and scale shapefile
    def scale(lon, lat, extra=None):  # pylint: disable=unused-argument
        lon = tuple(map(functools.partial(apply_scale_offset, lon_scale, lon_offset), lon))
        lat = tuple(map(functools.partial(apply_scale_offset, lat_scale, lat_offset), lat))
        return lon, lat

    geometries = [transform(scale, geometry) for geometry in shapefile_df.geometry]
    shapefile_df.geometry = geometries

    def in_shape(lon, lat):
        point = Point(lon, lat)
        point_in_shapefile = shapefile_df.contains(point)
        return point_in_shapefile.array[0]

    dask = "forbidden"
    if chunks:
        dask = "allowed"

    in_shape_vec = np.vectorize(in_shape)
    boolean_mask = xr.apply_ufunc(in_shape_vec, dataset[lon_var_name], dataset[lat_var_name], dask=dask)
    return xre.where(dataset, boolean_mask, cut)


def get_coordinate_variable_names(dataset: xr.Dataset,
                                  lat_var_names: list = None,
                                  lon_var_names: list = None,
                                  time_var_names: list = None):
    """
    Retrieve coordinate variables for this dataset. If coordinate
    variables are provided, use those, Otherwise, attempt to determine
    coordinate variables manually.

    Parameters
    ----------
    dataset : xr.Dataset
        xarray Dataset used to compute coordinate variables manually.
        Only used if lat, lon, or time vars are not provided.
    lat_var_names : list
        List of latitude coordinate variables.
    lon_var_names : list
        List of longitude coordinate variables.
    time_var_names : list
        List of time coordinate variables.

    Returns
    -------
    TODO: add return type docstring and type hint.
    """

    if not lat_var_names or not lon_var_names:
        lat_var_names, lon_var_names = compute_coordinate_variable_names(dataset)
    if not time_var_names:
        time_var_names = []
        for lat_var_name in lat_var_names:
            time_var_names.append(compute_time_variable_name(dataset,
                                                             dataset[lat_var_name],
                                                             time_var_names))

        time_var_names.append(compute_utc_name(dataset))
        time_var_names = [x for x in time_var_names if x is not None]  # remove Nones and any duplicates

    return lat_var_names, lon_var_names, time_var_names


def open_as_nc_dataset(filepath: str) -> Tuple[nc.Dataset, bool]:
    """Open netcdf file, and flatten groups if they exist."""
    hdf_type = None
    # Open dataset with netCDF4 first, so we can get group info
    try:
        nc_dataset = nc.Dataset(filepath, mode='r')
        has_groups = bool(nc_dataset.groups)
        # If dataset has groups, transform to work with xarray
        if has_groups:
            nc_dataset = transform_grouped_dataset(nc_dataset, filepath)

    except OSError:
        nc_dataset, has_groups, hdf_type = h5file_transform(filepath)

    nc_dataset = dc.remove_duplicate_dims(nc_dataset)

    return nc_dataset, has_groups, hdf_type


def override_decode_cf_datetime() -> None:
    """
    WARNING !!! REMOVE AT EARLIEST XARRAY FIX, this is a override to xarray override_decode_cf_datetime function.
    xarray has problems decoding time units with format `seconds since 2000-1-1 0:0:0 0`, this solves by testing
    the unit to see if its parsable, if it is use original function, if not format unit into a parsable format.

    https://github.com/pydata/xarray/issues/7210
    """

    orig_decode_cf_datetime = xarray.coding.times.decode_cf_datetime

    def decode_cf_datetime(num_dates, units, calendar=None, use_cftime=None):
        try:
            parser.parse(units.split('since')[-1])
            return orig_decode_cf_datetime(num_dates, units, calendar, use_cftime)
        except dateutil.parser.ParserError:
            reference_time = cftime.num2date(0, units, calendar)
            units = f"{units.split('since')[0]} since {reference_time}"
            return orig_decode_cf_datetime(num_dates, units, calendar, use_cftime)

    xarray.coding.times.decode_cf_datetime = decode_cf_datetime


def test_access_sst_dtime_values(datafile):
    """
    Test accessing values of 'sst_dtime' variable in a NetCDF file.

    Parameters
    ----------
    datafile (str): Path to the NetCDF file.

    Returns
    -------
    access_successful (bool): True if 'sst_dtime' values are accessible, False otherwise.
    """

    nc_dataset, _, _ = open_as_nc_dataset(datafile)
    args = {
        'decode_coords': False,
        'mask_and_scale': True,
        'decode_times': True
    }
    try:
        with xr.open_dataset(
                xr.backends.NetCDF4DataStore(nc_dataset),
                **args
        ) as dataset:
            # pylint: disable=pointless-statement
            for var_name in dataset.variables:
                dataset[var_name].values
    except (TypeError, ValueError, KeyError):
        return False
    return True


def subset(file_to_subset: str, bbox: np.ndarray, output_file: str,
           variables: Union[List[str], str, None] = (),
           # pylint: disable=too-many-branches, disable=too-many-statements
           cut: bool = True, shapefile: str = None, min_time: str = None, max_time: str = None,
           origin_source: str = None,
           lat_var_names: List[str] = (), lon_var_names: List[str] = (), time_var_names: List[str] = (),
           pixel_subset: bool = False, stage_file_name_subsetted_true: str = None, stage_file_name_subsetted_false: str = None
           ) -> Union[np.ndarray, None]:
    """
    Subset a given NetCDF file given a bounding box

    Parameters
    ----------
    file_to_subset : string
        The location of the file which will be subset
    bbox : np.ndarray
        The chosen bounding box. This is a tuple of tuples formatted
        as such: ((west, east), (south, north)). The assumption is that
        the valid range is ((-180, 180), (-90, 90)). This will be
        transformed as appropriate if the actual longitude range is
        0-360.
    output_file : string
        The file path for the output of the subsetting operation.
    variables : list, str, optional
        List of variables to include in the resulting data file.
        NOTE: This will remove ALL variables which are not included
        in this list, including coordinate variables!
    cut : boolean
        True if the scanline should be cut, False if the scanline should
        not be cut. Defaults to True.
    shapefile : str
        Name of local shapefile used to subset given file.
    min_time : str
        ISO timestamp representing the lower bound of the temporal
        subset to be performed. If this value is not provided, the
        granule will not be subset temporally on the lower bound.
    max_time : str
        ISO timestamp representing the upper bound of the temporal
        subset to be performed. If this value is not provided, the
        granule will not be subset temporally on the upper bound.
    origin_source : str
        Original location or filename of data to be used in "derived from"
        history element.
    lat_var_names : list
        List of variables that represent the latitude coordinate
        variables for this granule. This list will only contain more
        than one value in the case where there are multiple groups and
        different coordinate variables for each group.
    lon_var_names : list
        List of variables that represent the longitude coordinate
        variables for this granule. This list will only contain more
        than one value in the case where there are multiple groups and
        different coordinate variables for each group.
    time_var_names : list
        List of variables that represent the time coordinate
        variables for this granule. This list will only contain more
        than one value in the case where there are multiple groups and
        different coordinate variables for each group.
    pixel_subset : boolean
        Cut the lon lat based on the rows and columns within the bounding box,
        but could result with lon lats that are outside the bounding box
    stage_file_name_subsetted_true: str
        stage file name if subsetting is true name depends on result of subset
    stage_file_name_subsetted_false: str
        stage file name if subsetting is false name depends on result of subset
    """
    file_extension = os.path.splitext(file_to_subset)[1]
    nc_dataset, has_groups, hdf_type = open_as_nc_dataset(file_to_subset)

    override_decode_cf_datetime()

    if has_groups:
        # Make sure all variables start with '/'
        if variables:
            variables = ['/' + var if not var.startswith('/') else var for var in variables]
        lat_var_names = ['/' + var if not var.startswith('/') else var for var in lat_var_names]
        lon_var_names = ['/' + var if not var.startswith('/') else var for var in lon_var_names]
        time_var_names = ['/' + var if not var.startswith('/') else var for var in time_var_names]
        # Replace all '/' with GROUP_DELIM
        if variables:
            variables = [var.replace('/', GROUP_DELIM) for var in variables]
        lat_var_names = [var.replace('/', GROUP_DELIM) for var in lat_var_names]
        lon_var_names = [var.replace('/', GROUP_DELIM) for var in lon_var_names]
        time_var_names = [var.replace('/', GROUP_DELIM) for var in time_var_names]

    if '.HDF5' == file_extension:
        # GPM files will have a ScanTime group
        if 'ScanTime' in [var.split('__')[-2] for var in list(nc_dataset.variables.keys())]:
            gc.change_var_dims(nc_dataset, variables)
            hdf_type = 'GPM'
    args = {
        'decode_coords': False,
        'mask_and_scale': False,
        'decode_times': False
    }
    # clean up time variable in SNDR before decode_times
    # SNDR.AQUA files have ascending node time blank
    if any('__asc_node_tai93' in i for i in list(nc_dataset.variables)):
        asc_time_var = nc_dataset.variables['__asc_node_tai93']
        if not asc_time_var[:] > 0:
            del nc_dataset.variables['__asc_node_tai93']

    if min_time or max_time:
        args['decode_times'] = True
        float_dtypes = ['float64', 'float32']
        fill_value_f8 = nc.default_fillvals.get('f8')

        for time_variable in (v for v in nc_dataset.variables.keys() if 'time' in v):
            time_var = nc_dataset[time_variable]

            if (getattr(time_var, '_FillValue', None) == fill_value_f8 and time_var.dtype in float_dtypes) or \
               (getattr(time_var, 'long_name', None) == "reference time of sst file"):
                args['mask_and_scale'] = True
                if getattr(time_var, 'long_name', None) == "reference time of sst file":
                    args['mask_and_scale'] = test_access_sst_dtime_values(file_to_subset)
                break

    if hdf_type == 'GPM':
        args['decode_times'] = False

    with xr.open_dataset(
            xr.backends.NetCDF4DataStore(nc_dataset),
            **args
    ) as dataset:

        original_dataset = dataset

        lat_var_names, lon_var_names, time_var_names = get_coordinate_variable_names(
            dataset=dataset,
            lat_var_names=lat_var_names,
            lon_var_names=lon_var_names,
            time_var_names=time_var_names
        )

        if not time_var_names and (min_time or max_time):
            raise ValueError('Could not determine time variable')

        start_date = None
        if hdf_type and (min_time or max_time):
            dataset, start_date = tc.convert_to_datetime(dataset, time_var_names, hdf_type)

        chunks = calculate_chunks(dataset)
        if chunks:
            dataset = dataset.chunk(chunks)
        if variables:
            # Drop variables that aren't explicitly requested, except lat_var_name and
            # lon_var_name which are needed for subsetting
            variables_upper = [variable.upper() for variable in variables]
            vars_to_drop = [
                var_name for var_name, var in dataset.data_vars.items()
                if var_name.upper() not in variables_upper
                and var_name not in lat_var_names
                and var_name not in lon_var_names
                and var_name not in time_var_names
            ]

            dataset = dataset.drop_vars(vars_to_drop)
        if shapefile:
            datasets = [
                subset_with_shapefile(dataset, lat_var_names[0], lon_var_names[0], shapefile, cut, chunks)
            ]
        elif bbox is not None:
            datasets = subset_with_bbox(
                dataset=dataset,
                lat_var_names=lat_var_names,
                lon_var_names=lon_var_names,
                time_var_names=time_var_names,
                variables=variables,
                bbox=bbox,
                cut=cut,
                min_time=min_time,
                max_time=max_time,
                pixel_subset=pixel_subset
            )
        else:
            raise ValueError('Either bbox or shapefile must be provided')

        spatial_bounds = []
        for dataset in datasets:
            # Set version and JSON history efficiently
            set_version_history(dataset, cut, bbox, shapefile)
            set_json_history(dataset, cut, file_to_subset, bbox, shapefile, origin_source)

            if has_groups:
                spatial_bounds.append(get_spatial_bounds(
                    dataset=dataset,
                    lat_var_names=lat_var_names,
                    lon_var_names=lon_var_names
                ))
            else:
                for var, data_var in dataset.data_vars.items():
                    # Optimize string handling and decoding
                    fill_value = data_var.attrs.get('_FillValue')
                    if data_var.dtype == 'S1' and isinstance(fill_value, bytes):
                        data_var.attrs['_FillValue'] = fill_value.decode('UTF-8')

                    # Efficiently prepare encoding options
                    var_encoding = {
                        "zlib": True,
                        "complevel": 5,
                        "_FillValue": original_dataset[var].encoding.get('_FillValue')
                    }

                    # Load, write, and release memory efficiently
                    data_var.load()
                    data_var.to_netcdf(output_file, 'a', encoding={var: var_encoding})
                    del data_var  # Explicitly free memory

                # Set global attributes efficiently
                with nc.Dataset(output_file, 'a') as dataset_attr:
                    dataset_attr.setncatts(dataset.attrs)

        if has_groups:
            recombine_grouped_datasets(datasets, output_file, start_date, time_var_names)
            # Check if the spatial bounds are all 'None'. This means the
            # subset result is empty.

            if any(bound is None for bound in spatial_bounds):
                return None
            min_lon = min(bound[0][0] for bound in spatial_bounds)
            max_lon = max(bound[0][1] for bound in spatial_bounds)
            min_lat = min(bound[1][0] for bound in spatial_bounds)
            max_lat = max(bound[1][1] for bound in spatial_bounds)
            spatial_bounds_array = np.array([[min_lon, max_lon], [min_lat, max_lat]])
        else:
            spatial_bounds_array = get_spatial_bounds(
                dataset=datasets[0],
                lat_var_names=lat_var_names,
                lon_var_names=lon_var_names
            )

        update_netcdf_attrs(output_file,
                            datasets,
                            lon_var_names,
                            lat_var_names,
                            spatial_bounds_array,
                            stage_file_name_subsetted_true,
                            stage_file_name_subsetted_false)

        return spatial_bounds_array


def update_netcdf_attrs(output_file: str,
                        datasets: List[xr.Dataset],
                        lon_var_names: List[str],
                        lat_var_names: List[str],
                        spatial_bounds_array: Optional[list] = None,
                        stage_file_name_subsetted_true: Optional[str] = None,
                        stage_file_name_subsetted_false: Optional[str] = None) -> None:
    """
    Update NetCDF file attributes with spatial bounds and product name information.

    Args:
        output_file (str): Path to the NetCDF file to be updated
        datasets (list): List of datasets to extract longitude and latitude information
        lon_var_names (list): List of possible longitude variable names
        lat_var_names (list): List of possible latitude variable names
        spatial_bounds_array (list, optional): Nested list containing spatial bounds in format:
            [[lon_min, lon_max], [lat_min, lat_max]]
        stage_file_name_subsetted_true (str, optional): Product name when subset is True
        stage_file_name_subsetted_false (str, optional): Product name when subset is False

    Notes:
        - Updates various geospatial attributes in the NetCDF file
        - Removes deprecated center coordinate attributes
        - Sets the product name based on provided parameters
        - Preserves original attribute types when setting new values

    Example:
        >>> spatial_bounds = [[120.5, 130.5], [-10.5, 10.5]]
        >>> update_netcdf_attrs("output.nc", datasets, ["lon"], spatial_bounds, "subset_true.nc", "subset_false.nc")
    """

    lons_easternmost = []
    lons_westernmost = []
    final_eastmost = None
    final_westmost = None

    for dataset in datasets:

        lon_var_name = None
        if len(lon_var_names) == 1:
            lon_var_name = lon_var_names[0]
        else:
            matched_names = [lon_name for lon_name in lon_var_names if lon_name in dataset.data_vars]
            if matched_names:
                lon_var_name = matched_names[0]

        eastmost, westmost = get_east_west_lon(dataset, lon_var_name)

        if eastmost and westmost:
            lons_easternmost.append(eastmost)
            lons_westernmost.append(westmost)

    with nc.Dataset(output_file, 'a') as dataset_attr:
        original_attrs = dataset_attr.ncattrs()

        def set_attr_with_type(attr_name: str, value: Any) -> None:
            """Set attribute while preserving its original type if it exists."""
            if attr_name in original_attrs:
                original_type = type(dataset_attr.getncattr(attr_name))
                dataset_attr.setncattr(attr_name, original_type(value))

        if spatial_bounds_array is not None:
            # Define geographical bounds mapping
            bounds_mapping = {
                'geospatial_lat_max': (1, 1),
                'geospatial_lat_min': (1, 0)
            }

            # Set all geographic bounds attributes
            for attr_name, (i, j) in bounds_mapping.items():
                set_attr_with_type(attr_name, spatial_bounds_array[i][j])

            # Remove deprecated center coordinates
            deprecated_keys = {
                "start_center_longitude",
                "start_center_latitude",
                "end_center_longitude",
                "end_center_latitude",
                "northernmost_latitude",
                "southernmost_latitude",
                "easternmost_longitude",
                "westernmost_longitude"
            }

            # Safely remove deprecated attributes if they exist
            for key in deprecated_keys & set(dataset_attr.ncattrs()):
                dataset_attr.delncattr(key)

            # Set CRS and bounds
            if lons_westernmost:
                final_westmost = min(lons_westernmost, key=lambda lon: lon if lon >= 0 else lon + 360)
                set_attr_with_type("geospatial_lon_min", final_westmost)
            if lons_easternmost:
                final_eastmost = max(lons_easternmost, key=lambda lon: lon if lon >= 0 else lon + 360)
                set_attr_with_type("geospatial_lon_max", final_eastmost)
            dataset_attr.setncattr("geospatial_bounds_crs", "EPSG:4326")
            geospatial_bounds = (
                create_geospatial_bounds(datasets, lon_var_names, lat_var_names)
                or create_geospatial_bounding_box(spatial_bounds_array, final_eastmost, final_westmost)
            )
            dataset_attr.setncattr("geospatial_bounds", geospatial_bounds)

        # Set product name based on conditions
        has_spatial_bounds = spatial_bounds_array is not None and spatial_bounds_array.size > 0
        product_name = (stage_file_name_subsetted_true if has_spatial_bounds and stage_file_name_subsetted_true
                        else stage_file_name_subsetted_false if stage_file_name_subsetted_false
                        else output_file)

        set_attr_with_type('product_name', product_name)


def create_geospatial_bounding_box(spatial_bounds_array, east, west):
    """
    Generate a Well-Known Text (WKT) POLYGON string representing the geospatial bounds.

    The polygon is defined using the min/max longitude and latitude values and follows
    the format: "POLYGON ((lon_min lat_min, lon_max lat_min, lon_max lat_max,
                           lon_min lat_max, lon_min lat_min))"

    This ensures the polygon forms a closed loop.

    Parameters:
    -----------
    spatial_bounds_array : list of lists
        A 2D list where:
        - spatial_bounds_array[0] contains [lon_min, lon_max]
        - spatial_bounds_array[1] contains [lat_min, lat_max]
    east: float or None
        - longitude spacial bound east
    west: float or None
        - longitude spacial bound west
    Returns:
    --------
    str
        A WKT POLYGON string representing the bounding box.

    Example:
    --------
    >>> spatial_bounds = [[81.489693, 85.129562], [-78.832314, 49.646988]]
    >>> create_geospatial_bounds(spatial_bounds)
    'POLYGON ((81.489693 -78.832314, 85.129562 -78.832314, 85.129562 49.646988, 81.489693 49.646988, 81.489693 -78.832314))'
    """
    lon_min, lon_max = spatial_bounds_array[0]
    lat_min, lat_max = spatial_bounds_array[1]

    if east:
        lon_min = east
    if west:
        lon_max = west

    # Construct the WKT polygon string (ensuring the loop closes)
    wkt_polygon = (
        f"POLYGON (({lon_min:.5f} {lat_min:.5f}, {lon_max:.5f} {lat_min:.5f}, "
        f"{lon_max:.5f} {lat_max:.5f}, {lon_min:.5f} {lat_max:.5f}, {lon_min:.5f} {lat_min:.5f}))"
    )

    return wkt_polygon


def create_geospatial_bounds(datasets, lon_var_names, lat_var_names):
    """Create geospatial bounds from 4 corners of 2d array"""

    for dataset in datasets:
        lon_var_name = None
        lat_var_name = None

        if len(lon_var_names) == 1:
            lon_var_name = lon_var_names[0]
        else:
            matched_names = [lon_name for lon_name in lon_var_names if lon_name in dataset.data_vars]
            if matched_names:
                lon_var_name = matched_names[0]
            else:
                continue

        if len(lat_var_names) == 1:
            lat_var_name = lat_var_names[0]
        else:
            matched_names = [lat_name for lat_name in lat_var_names if lat_name in dataset.data_vars]
            if matched_names:
                lat_var_name = matched_names[0]
            else:
                continue

        lon = dataset[lon_var_name].values
        lat = dataset[lat_var_name].values

        lon_fill_value = dataset[lon_var_name].attrs.get('_FillValue', None)
        lat_fill_value = dataset[lon_var_name].attrs.get('_FillValue', None)

        break

    # Check if the variables are 2D arrays
    if lon.ndim != 2 or lat.ndim != 2:
        return None

    # Get the shape of the arrays (assuming they are the same shape)
    nrows, ncols = lon.shape

    points = [
        (lon[0, 0], lat[0, 0]),        # Top-left
        (lon[nrows - 1, 0], lat[nrows - 1, 0]),  # Bottom-left
        (lon[nrows - 1, ncols - 1], lat[nrows - 1, ncols - 1]),  # Bottom-right
        (lon[0, ncols - 1], lat[0, ncols - 1])   # Top-right
    ]

    # Check for NaN or fill values in corner points
    if any(np.isnan(point[0]) or np.isnan(point[1]) or point[0] == lon_fill_value or point[1] == lat_fill_value for point in points):
        return None

    # Sort points in counter-clockwise order
    sorted_points = ensure_counter_clockwise(points)

    # Create a counter-clockwise WKT polygon with precision 5
    wkt_polygon = (
        f"POLYGON(({sorted_points[0][0]:.5f} {sorted_points[0][1]:.5f}, "
        f"{sorted_points[1][0]:.5f} {sorted_points[1][1]:.5f}, "
        f"{sorted_points[2][0]:.5f} {sorted_points[2][1]:.5f}, "
        f"{sorted_points[3][0]:.5f} {sorted_points[3][1]:.5f}, "
        f"{sorted_points[0][0]:.5f} {sorted_points[0][1]:.5f}))"
    )

    return wkt_polygon


def shoelace_area(points):
    """Computes the signed area of a polygon.
       Negative area → Counterclockwise
       Positive area → Clockwise (needs reversing)
    """
    x, y = np.array(points)[:, 0], np.array(points)[:, 1]
    return 0.5 * np.sum(x[:-1] * y[1:] - x[1:] * y[:-1])


def ensure_counter_clockwise(points):
    """Ensures the points are ordered counterclockwise."""
    area = shoelace_area(points)
    if area > 0:  # Clockwise → Reverse order
        return points[::-1]
    return points


def get_east_west_lon(dataset, lon_var_name):
    """
    Determines the easternmost and westernmost longitudes from a dataset,
    correctly handling cases where the data crosses the antimeridian.

    Parameters:
        dataset: xarray.Dataset or similar
            The dataset containing longitude values.
        lon_var_name: str
            The name of the longitude variable in the dataset.

    Returns:
        tuple: (westmost, eastmost)
            The westernmost and easternmost longitudes in [-180, 180] range.
    """
    lon_2d = dataset[lon_var_name].values
    fill_value = dataset[lon_var_name].attrs.get('_FillValue', None)

    lon_flat = lon_2d.flatten()
    if fill_value is not None:
        lon_flat = lon_flat[lon_flat != fill_value]
    lon_flat = lon_flat[~np.isnan(lon_flat)]
    if lon_flat.size == 0:
        return None, None  # No valid longitude data

    crosses_antimeridian = np.any((lon_flat[:-1] > 150) & (lon_flat[1:] < -150))

    # Convert longitudes to [0, 360] range
    lon_360 = np.where(lon_flat < 0, lon_flat + 360, lon_flat)

    # Sort longitudes
    lon_sorted = np.sort(lon_360)

    # Compute gaps
    gaps = np.diff(lon_sorted)
    wrap_gap = lon_sorted[0] + 360 - lon_sorted[-1]
    gaps = np.append(gaps, wrap_gap)

    # Find the largest gap
    max_gap_index = np.argmax(gaps)

    if crosses_antimeridian:
        eastmost_360 = lon_sorted[max_gap_index]
        westmost_360 = lon_sorted[(max_gap_index + 1) % len(lon_sorted)]
    else:
        eastmost_360 = np.max(lon_flat)
        westmost_360 = np.min(lon_flat)

    def convert_to_standard(lon):
        return lon - 360 if lon > 180 else lon

    eastmost = round(convert_to_standard(eastmost_360), 5)
    westmost = round(convert_to_standard(westmost_360), 5)

    return eastmost, westmost
