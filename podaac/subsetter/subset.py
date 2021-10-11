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

import geopandas as gpd
import importlib_metadata
import julian
import numpy as np
import xarray as xr
import netCDF4 as nc
import pandas as pd
from shapely.geometry import Point
from shapely.ops import transform

from podaac.subsetter import xarray_enhancements as xre


GROUP_DELIM = '__'
SERVICE_NAME = 'l2ss-py'


def apply_scale_offset(scale, offset, value):
    """Apply scale and offset to the given value"""
    return (value + offset) / scale


def remove_scale_offset(value, scale, offset):
    """Remove scale and offset from the given value"""
    return (value * scale) - offset


def convert_bound(bound, coord_max, coord_var):
    """
    This function will return a converted bound which which matches the
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


def convert_bbox(bbox, dataset, lat_var_name, lon_var_name):
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


def set_json_history(dataset, cut, file_to_subset, bbox=None, shapefile=None,
                     origin_source=None):
    """
    Set the 'json_history' metadata header of the granule to reflect the
    current version of the subsetter, as well as the parameters used
    to call the subsetter. This will append an json array to the json_history of
    the following format:

    Parameters
    ----------
    dataset : xarray.Dataset
        The dataset to change the header of
    bbox : np.ndarray
        The requested bounding box
    file_to_subset : string
        The filepath of the file which was used to subset
    cut : boolean
        True to cut the scanline
    shapefile : str
        Name of the shapefile to include in the version history

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


def set_version_history(dataset, cut, bbox=None, shapefile=None):
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
    bbox : np.ndarray
        The requested bounding box
    cut : boolean
        True to cut the scanline
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


def calculate_chunks(dataset):
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
        value is 4000.
    """
    chunk_dict = {dim: 4000 for dim in dataset.dims
                  if dataset.dims[dim] > 4000
                  and len(dataset.dims) > 1}
    return chunk_dict


def find_matching_coords(dataset, match_list):
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


def get_coord_variable_names(dataset):
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
        Tuple of strings, where the first element is the lat coordinate
        name and the second element is the lon coordinate name
    """
    possible_lat_coord_names = ['lat', 'latitude', 'y']
    possible_lon_coord_names = ['lon', 'longitude', 'x']

    def var_is_coord(var_name, possible_coord_names):
        var_name = var_name.strip(GROUP_DELIM).split(GROUP_DELIM)[-1]
        return var_name in possible_coord_names

    lat_coord_names = list(filter(
        lambda var_name: var_is_coord(var_name, possible_lat_coord_names), dataset.variables))
    lon_coord_names = list(filter(
        lambda var_name: var_is_coord(var_name, possible_lon_coord_names), dataset.variables))

    if len(lat_coord_names) < 1 or len(lon_coord_names) < 1:
        lat_coord_names = find_matching_coords(dataset, possible_lat_coord_names)
        lon_coord_names = find_matching_coords(dataset, possible_lon_coord_names)

    if len(lat_coord_names) < 1 or len(lon_coord_names) < 1:
        raise ValueError('Could not determine coordinate variables')

    return lat_coord_names, lon_coord_names


def is_360(lon_var, scale, offset):
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


def get_spatial_bounds(dataset, lat_var_names, lon_var_names):
    """
    Get the spatial bounds for this dataset. These values are masked
    and scaled.

    Parameters
    ----------
    dataset : xr.Dataset
        Dataset to retrieve spatial bounds for
    lat_var_name : str
        Name of the lat variable
    lon_var_name : str
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

    min_lat = remove_scale_offset(np.nanmin(lats), lat_scale, lat_offset)
    max_lat = remove_scale_offset(np.nanmax(lats), lat_scale, lat_offset)
    min_lon = remove_scale_offset(np.nanmin(lons), lon_scale, lon_offset)
    max_lon = remove_scale_offset(np.nanmax(lons), lon_scale, lon_offset)

    min_lat = round(min_lat, 1)
    max_lat = round(max_lat, 1)
    min_lon = round(min_lon, 1)
    max_lon = round(max_lon, 1)

    # Convert longitude to [-180,180] format
    if lon_valid_min == 0 or 0 <= min_lon <= max_lon <= 360:
        if min_lon > 180:
            min_lon -= 360
        if max_lon > 180:
            max_lon -= 360
        if min_lon == max_lon:
            min_lon = -180
            max_lon = 180

    return np.array([[min_lon, max_lon], [min_lat, max_lat]])


def get_time_variable_name(dataset, lat_var):
    """
    Try to determine the name of the 'time' variable. This is done as
    follows:

    - The variable name contains 'time'
    - The variable dimensions match the dimensions of the given lat var

    Parameters
    ----------
    dataset : xr.Dataset:
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

    for var_name in list(dataset.dims.keys()):
        if "time" in var_name and dataset[var_name].squeeze().dims == lat_var.dims:
            return var_name
    for var_name in list(dataset.data_vars.keys()):
        if "time" in var_name and dataset[var_name].squeeze().dims == lat_var.dims:
            return var_name
    raise ValueError('Unable to determine time variable')


def get_time_epoch_var(dataset, time_var_name):
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
    else:
        raise ValueError('Unable to determine time variables')

    return epoch_var_name


def is_time_mjd(dataset, time_var_name):
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


def translate_timestamp(str_timestamp):
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
        '%Y-%m-%dT%H:%M:%S.%f%Z'
    ]

    for timestamp_format in allowed_ts_formats:
        try:
            return datetime.datetime.strptime(str_timestamp, timestamp_format)
        except ValueError:
            pass
    return datetime.datetime.fromisoformat(str_timestamp)


def datetime_from_mjd(dataset, time_var_name):
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
        mjd = mdj_string[mdj_string.find("(")+1:mdj_string.find(")")].split("= ")[1]
        try:
            mjd_float = float(mjd)
        except ValueError:
            return None
        mjd_datetime = julian.from_jd(mjd_float, fmt='mjd')
        return mjd_datetime

    return None


def build_temporal_cond(min_time, max_time, dataset, time_var_name):
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
    np.array or boolean
        If temporally subsetted, returns a boolean ND-array the shape
        of which matches the dimensions of the coordinate vars. 'True'
        is essentially a noop.
    """

    def build_cond(str_timestamp, compare):
        timestamp = translate_timestamp(str_timestamp)

        if np.issubdtype(dataset[time_var_name].dtype, np.dtype(np.datetime64)):
            timestamp = pd.to_datetime(timestamp)
        if np.issubdtype(dataset[time_var_name].dtype, np.dtype(np.timedelta64)):

            if is_time_mjd(dataset, time_var_name):
                # mjd when timedelta based on
                mjd_datetime = datetime_from_mjd(dataset, time_var_name)
                if mjd_datetime is None:
                    raise ValueError('Unable to get datetime from dataset to calculate time delta')

                # timedelta between timestamp and mjd
                timestamp = np.datetime64(timestamp) - np.datetime64(mjd_datetime)
            else:
                epoch_time_var_name = get_time_epoch_var(dataset, time_var_name)
                epoch_datetime = dataset[epoch_time_var_name].values[0]
                timestamp = np.datetime64(timestamp) - epoch_datetime

        return compare(dataset[time_var_name], timestamp)

    temporal_conds = []
    if min_time:
        comparison_op = operator.ge
        temporal_conds.append(build_cond(min_time, comparison_op))
    if max_time:
        comparison_op = operator.le
        temporal_conds.append(build_cond(max_time, comparison_op))
    temporal_cond = True
    if min_time or max_time:
        temporal_cond = functools.reduce(lambda cond_a, cond_b: cond_a & cond_b, temporal_conds)
    return temporal_cond


def subset_with_bbox(dataset, lat_var_names, lon_var_names, time_var_names, bbox=None, cut=True,
                     min_time=None, max_time=None):
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
    bbox : np.array
        Spatial bounding box to subset Dataset with.
    cut : bool
        True if scanline should be cut.
    min_time : str
        ISO timestamp of min temporal bound
    max_time : str
        ISO timestamp of max temporal bound

    Returns
    -------
    np.array
        Spatial bounds of Dataset after subset operation
    """
    lon_bounds, lat_bounds = convert_bbox(bbox, dataset, lat_var_names[0], lon_var_names[0])
    # condition should be 'or' instead of 'and' when bbox lon_min > lon_max
    oper = operator.and_

    if lon_bounds[0] > lon_bounds[1]:
        oper = operator.or_

    datasets = []
    for lat_var_name, lon_var_name, time_var_name in zip(
            lat_var_names, lon_var_names, time_var_names
    ):
        if GROUP_DELIM in lat_var_name:
            var_prefix = GROUP_DELIM.join(lat_var_name.strip(GROUP_DELIM).split(GROUP_DELIM)[:-1])
            group_vars = [
                var for var in dataset.data_vars.keys()
                if var.startswith(f'{GROUP_DELIM}{var_prefix}')
            ]
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
            cut
        )
        datasets.append(group_dataset)

    return datasets


def subset_with_shapefile(dataset, lat_var_name, lon_var_name, shapefile, cut):
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
    shapefile : np.array
        Absolute path to the shapefile used to subset the given dataset
    cut : bool
        True if scanline should be cut.
    Returns
    -------
    np.array
        Spatial bounds of Dataset after shapefile subset operation
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
        def convert_180_to_360(lon, lat):
            return tuple(map(lambda value: value + 360 if value < 0 else value, lon)), lat

        geometries = [transform(convert_180_to_360, geometry) for geometry in
                      shapefile_df.geometry]
        shapefile_df.geometry = geometries

    # Mask and scale shapefile
    def scale(lon, lat):
        lon = tuple(map(functools.partial(apply_scale_offset, lon_scale, lon_offset), lon))
        lat = tuple(map(functools.partial(apply_scale_offset, lat_scale, lat_offset), lat))
        return lon, lat

    geometries = [transform(scale, geometry) for geometry in shapefile_df.geometry]
    shapefile_df.geometry = geometries

    def in_shape(lon, lat):
        point = Point(lon, lat)
        point_in_shapefile = shapefile_df.contains(point)
        return point_in_shapefile.array[0]

    in_shape_vec = np.vectorize(in_shape)
    boolean_mask = xr.apply_ufunc(in_shape_vec, dataset[lon_var_name], dataset[lat_var_name])
    return xre.where(dataset, boolean_mask, cut)


def transform_grouped_dataset(nc_dataset, file_to_subset):
    """
    Transform a netCDF4 Dataset that has groups to an xarray compatible
    dataset. xarray does not work with groups, so this transformation
    will flatten the variables in the dataset and use the group path as
    the new variable name. For example, data_01 > km > sst would become
    'data_01__km__sst', where GROUP_DELIM is __.

    This same pattern is applied to dimensions, which are located under
    the appropriate group. They are renamed and placed in the root
    group.

    Parameters
    ----------
    nc_dataset : nc.Dataset
        netCDF4 Dataset that contains groups

    Returns
    -------
    nc.Dataset
        netCDF4 Dataset that does not contain groups and that has been
        flattened.
    """

    # Close the existing read-only dataset and reopen in append mode
    nc_dataset.close()
    nc_dataset = nc.Dataset(file_to_subset, 'r+')

    dimensions = {}

    def walk(group_node, path):
        for key, item in group_node.items():
            group_path = f'{path}{GROUP_DELIM}{key}'

            # If there are variables in this group, copy to root group
            # and then delete from current group
            if item.variables:
                # Copy variables to root group with new name
                for var_name, var in item.variables.items():
                    var_group_name = f'{group_path}{GROUP_DELIM}{var_name}'
                    nc_dataset.variables[var_group_name] = var
                # Delete variables
                var_names = list(item.variables.keys())
                for var_name in var_names:
                    del item.variables[var_name]

            if item.dimensions:
                dims = list(item.dimensions.keys())
                for dim_name in dims:
                    new_dim_name = f'{group_path.replace("/", GROUP_DELIM)}{GROUP_DELIM}{dim_name}'
                    item.dimensions[new_dim_name] = item.dimensions[dim_name]
                    dimensions[new_dim_name] = item.dimensions[dim_name]
                    item.renameDimension(dim_name, new_dim_name)

            # If there are subgroups in this group, call this function
            # again on that group.
            if item.groups:
                walk(item.groups, group_path)

        # Delete non-root groups
        group_names = list(group_node.keys())
        for group_name in group_names:
            del group_node[group_name]

    walk(nc_dataset.groups, '')

    # Update the dimensions of the dataset in the root group
    nc_dataset.dimensions.update(dimensions)

    return nc_dataset


def recombine_grouped_datasets(datasets, output_file):
    """
    Given a list of xarray datasets, combine those datasets into a
    single netCDF4 Dataset and write to the disk. Each dataset has been
    transformed using its group path and needs to be un-transformed and
    placed in the appropriate group.

    Parameters
    ----------
    datasets : list (xr.Dataset)
        List of xarray datasets to be combined
    output_file : str
        Name of the output file to write the resulting NetCDF file to.
    """

    def get_nested_group(dataset, group_path):
        nested_group = dataset
        for group in group_path.strip(GROUP_DELIM).split(GROUP_DELIM)[:-1]:
            nested_group = nested_group.groups[group]
        return nested_group

    base_dataset = nc.Dataset(output_file, mode='w')

    for dataset in datasets:
        groups = set(
            '/'.join(var_name.split(GROUP_DELIM)[:-1]) for var_name in dataset.variables.keys()
        )
        for group in groups:
            base_dataset.createGroup(group)

        for dim_name in list(dataset.dims.keys()):
            new_dim_name = dim_name.split(GROUP_DELIM)[-1]
            dim_group = get_nested_group(base_dataset, dim_name)
            dim_group.createDimension(new_dim_name, dataset.dims[dim_name])

        # Rename variables
        for var_name in list(dataset.variables.keys()):
            new_var_name = var_name.split(GROUP_DELIM)[-1]
            var_group = get_nested_group(base_dataset, var_name)
            var_dims = list(var_group.dimensions.keys())
            variable = dataset.variables[var_name]
            if not var_dims:
                var_group_parent = var_group
                # This group doesn't contain dimensions. Look at parent group to find dimensions.
                while not var_dims:
                    var_group_parent = var_group_parent.parent
                    var_dims = list(var_group_parent.dimensions.keys())

            if np.issubdtype(
                    dataset.variables[var_name].dtype, np.dtype(np.datetime64)
            ) or np.issubdtype(
                dataset.variables[var_name].dtype, np.dtype(np.timedelta64)
            ):
                # Use xarray datetime encoder
                cf_dt_coder = xr.coding.times.CFDatetimeCoder()
                encoded_var = cf_dt_coder.encode(dataset.variables[var_name])
                variable = encoded_var

            var_group.createVariable(new_var_name, variable.dtype, var_dims)

            # Copy attributes
            var_attrs = variable.attrs
            var_group.variables[new_var_name].setncatts(var_attrs)

            # Copy data
            var_group.variables[new_var_name].set_auto_maskandscale(False)
            var_group.variables[new_var_name][:] = variable.data

    # Remove group vars from base dataset
    for var_name in list(base_dataset.variables.keys()):
        if GROUP_DELIM in var_name:
            del base_dataset.variables[var_name]

    # Remove group dims from base dataset
    for dim_name in list(base_dataset.dimensions.keys()):
        if GROUP_DELIM in dim_name:
            del base_dataset.dimensions[dim_name]

    # Copy global attributes
    base_dataset.setncatts(datasets[0].attrs)
    # Write and close
    base_dataset.close()


def subset(file_to_subset, bbox, output_file, variables=None,  # pylint: disable=too-many-branches
           cut=True, shapefile=None, min_time=None, max_time=None, origin_source=None):
    """
    Subset a given NetCDF file given a bounding box

    Parameters
    ----------
    file_to_subset : string
        The location of the file which will be subset
    output_file : string
        The file path for the output of the subsetting operation.
    bbox : np.ndarray
        The chosen bounding box. This is a tuple of tuples formatted
        as such: ((west, east), (south, north)). The assumption is that
        the valid range is ((-180, 180), (-90, 90)). This will be
        transformed as appropriate if the actual longitude range is
        0-360.
    shapefile : str
        Name of local shapefile used to subset given file.
    variables : list, str, optional
        List of variables to include in the resulting data file.
        NOTE: This will remove ALL variables which are not included
        in this list, including coordinate variables!
    cut : boolean
        True if the scanline should be cut, False if the scanline should
        not be cut. Defaults to True.
    min_time : str
        ISO timestamp representing the lower bound of the temporal
        subset to be performed. If this value is not provided, the
        granule will not be subset temporally on the lower bound.
    max_time : str
        ISO timestamp representing the upper bound of the temporal
        subset to be performed. If this value is not provided, the
        granule will not be subset temporally on the upper bound.
    """

    # Open dataset with netCDF4 first, so we can get group info
    nc_dataset = nc.Dataset(file_to_subset, mode='r')

    has_groups = bool(nc_dataset.groups)

    # If dataset has groups, transform to work with xarray
    if has_groups:
        nc_dataset = transform_grouped_dataset(nc_dataset, file_to_subset)

    args = {
        'decode_coords': False,
        'mask_and_scale': False,
        'decode_times': False
    }

    if min_time or max_time:
        args['decode_times'] = True

    with xr.open_dataset(
            xr.backends.NetCDF4DataStore(nc_dataset),
            **args
    ) as dataset:

        lat_var_names, lon_var_names = get_coord_variable_names(dataset)
        time_var_names = [
            get_time_variable_name(
                dataset, dataset[lat_var_name]
            ) for lat_var_name in lat_var_names
        ]
        chunks_dict = calculate_chunks(dataset)

        if chunks_dict:
            dataset = dataset.chunk(chunks_dict)

        if variables:
            # Drop variables that aren't explicitly requested, except lat_var_name and
            # lon_var_name which are needed for subsetting
            variables = [variable.upper() for variable in variables]
            vars_to_drop = [var_name for var_name, var in dataset.data_vars.items() if
                            var_name.upper() not in variables and var_name not in lat_var_names and
                            var_name not in lon_var_names]
            dataset = dataset.drop_vars(vars_to_drop)

        if bbox is not None:
            datasets = subset_with_bbox(
                dataset=dataset,
                lat_var_names=lat_var_names,
                lon_var_names=lon_var_names,
                time_var_names=time_var_names,
                bbox=bbox,
                cut=cut,
                min_time=min_time,
                max_time=max_time
            )
        elif shapefile:
            datasets = [
                subset_with_shapefile(dataset, lat_var_names[0], lon_var_names[0], shapefile, cut)
            ]
        else:
            raise ValueError('Either bbox or shapefile must be provided')

        spatial_bounds = []
        for dataset in datasets:
            set_version_history(dataset, cut, bbox, shapefile)
            set_json_history(dataset, cut, file_to_subset, bbox, shapefile, origin_source)

            if has_groups:
                spatial_bounds.append(get_spatial_bounds(
                    dataset=dataset,
                    lat_var_names=lat_var_names,
                    lon_var_names=lon_var_names
                ))
            else:
                encoding = {}
                compression = dict(zlib=True, complevel=5, _FillValue=None)

                if (min_time or max_time) and any(dataset.dims.values()):
                    encoding = {
                        var_name: {
                            'units': nc_dataset.variables[var_name].__dict__['units'],
                            'zlib': True,
                            "complevel": 5,
                            "_FillValue": None
                        } for var_name in time_var_names
                        if 'units' in nc_dataset.variables[var_name].__dict__
                    }

                for var in dataset.data_vars:
                    if var not in encoding:
                        encoding[var] = compression

                dataset.load().to_netcdf(output_file, 'w', encoding=encoding)

        if has_groups:
            recombine_grouped_datasets(datasets, output_file)
            return np.array([[
                min(lon[0][0][0] for lon in zip(spatial_bounds)),
                max(lon[0][0][1] for lon in zip(spatial_bounds))
            ], [
                min(lat[0][1][0] for lat in zip(spatial_bounds)),
                max(lat[0][1][1] for lat in zip(spatial_bounds))
            ]])

        return get_spatial_bounds(
            dataset=dataset,
            lat_var_names=lat_var_names,
            lon_var_names=lon_var_names
        )
