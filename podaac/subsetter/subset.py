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
from typing import Any, Dict, List, Optional, Tuple, Union

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
from shapely.geometry import MultiPolygon, Point, Polygon
from shapely.ops import transform

from xarray import DataTree

from podaac.subsetter import (
    dimension_cleanup as dc,
    datatree_subset,
    tree_time_converting as tree_time_converting
)
from podaac.subsetter.group_handling import (
    GROUP_DELIM,
    h5file_transform,
    transform_grouped_dataset,
)


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

    lon_data = dataset[lon_var_name]
    lat_data = dataset[lat_var_name]

    return np.array([convert_bound(bbox[0], 360, lon_data),
                     convert_bound(bbox[1], 180, lat_data)])


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

    dataset = xr.decode_cf(dataset)

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
        lat_coord_names = datatree_subset.find_matching_coords(dataset, possible_lat_coord_names)
        lon_coord_names = datatree_subset.find_matching_coords(dataset, possible_lon_coord_names)

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


def get_time_epoch_var(tree: DataTree, time_var_name: str) -> str:
    """
    Get the name of the epoch time var. This is only needed in the case
    where there is a single time var (of size 1) that contains the time
    epoch used by the actual time var.

    Parameters
    ----------
    tree : DataTree
        DataTree that contains time var
    time_var_name : str
        The name of the actual time var (with matching dims to the
        coord vars)

    Returns
    -------
    str
        The name of the epoch time variable
    """
    # Split the time_var_name path to get the group and variable name
    path_parts = time_var_name.split('/')
    group_path = '/'.join(path_parts[:-1])
    var_name = path_parts[-1]

    # Get the dataset at the correct group level
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


def get_variable_data(dtree, var_path):
    """
    Retrieve data from a DataTree object given a variable path.

    Parameters:
    - dtree: DataTree object
    - var_path: str, path to the variable (e.g., "group/time")

    Returns:
    - The data of the variable if found, else None.
    """
    parts = var_path.split("/")  # Split path into group and variable names
    group_name, var_name = "/".join(parts[:-1]), parts[-1]  # Extract group and variable

    try:
        group = dtree[group_name] if group_name else dtree  # Get group or root
        return group.ds[var_name]  # Extract variable values
    except KeyError:
        return None


def new_build_temporal_cond(min_time: str, max_time: str, dataset: xr.Dataset, time_var_name: str
                            ) -> Union[np.ndarray, bool]:
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
            timestamp = pd.to_datetime(timestamp).to_datetime64()
            time_data = dataset['time']
            time_data = time_data.astype('datetime64[s]')
            timedelta_seconds = dataset['sst_dtime'].astype('timedelta64[s]')
            time_data = time_data + timedelta_seconds

        return compare(time_data, timestamp)

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


def get_path(s):
    """Extracts the path by removing the last part after the final '/'."""
    path = s.rsplit('/', 1)[0] if '/' in s else s
    return path if path.startswith('/') else f'/{path}'


def subset_with_shapefile_multi(dataset: xr.Dataset,
                                lat_var_names: List[str],
                                lon_var_names: List[str],
                                shapefile: str,
                                cut: bool,
                                chunks,
                                pixel_subset: bool) -> Dict[str, np.ndarray]:
    """
    Subset an xarray Dataset using a shapefile for multiple latitude and longitude variable pairs

    Parameters
    ----------
    dataset : xr.Dataset
        Dataset to subset
    lat_var_names : List[str]
        List of latitude variable names in the given dataset
    lon_var_names : List[str]
        List of longitude variable names in the given dataset
    shapefile : str
        Absolute path to the shapefile used to subset the given dataset
    cut : bool
        True if scanline should be cut
    chunks : Union[Dict, None]
        Chunking specification for dask arrays

    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary mapping variable names to their respective boolean masks
        Keys are formatted as "{lat_var_name}_{lon_var_name}"
    """
    if len(lat_var_names) != len(lon_var_names):
        raise ValueError("Number of latitude variables must match number of longitude variables")

    shapefile_df = gpd.read_file(shapefile)
    masks = {}

    # Mask and scale shapefile
    def scale(lon, lat, extra=None):  # pylint: disable=unused-argument
        lon = tuple(map(functools.partial(apply_scale_offset, lon_scale, lon_offset), lon))
        lat = tuple(map(functools.partial(apply_scale_offset, lat_scale, lat_offset), lat))
        return lon, lat

    def in_shape(data_lon, data_lat):
        point = Point(data_lon, data_lat)
        point_in_shapefile = current_shapefile_df.contains(point)
        return point_in_shapefile.array[0]

    for lat_var_name, lon_var_name in zip(lat_var_names, lon_var_names):
        # Get scaling factors and offsets for this pair
        lat_scale = dataset[lat_var_name].attrs.get('scale_factor', 1.0)
        lon_scale = dataset[lon_var_name].attrs.get('scale_factor', 1.0)
        lat_offset = dataset[lat_var_name].attrs.get('add_offset', 0.0)
        lon_offset = dataset[lon_var_name].attrs.get('add_offset', 0.0)

        # Create a copy of shapefile_df for this pair
        current_shapefile_df = shapefile_df.copy()

        # If data is '360', convert shapefile to '360' as well
        if is_360(dataset[lon_var_name], lon_scale, lon_offset):
            current_shapefile_df.geometry = current_shapefile_df['geometry'].apply(translate_longitude)

        geometries = [transform(scale, geometry) for geometry in current_shapefile_df.geometry]
        current_shapefile_df.geometry = geometries

        dask = "forbidden"
        if chunks:
            dask = "allowed"

        in_shape_vec = np.vectorize(in_shape)
        boolean_mask = xr.apply_ufunc(
            in_shape_vec,
            dataset[lon_var_name],
            dataset[lat_var_name],
            dask=dask
        )

        # Store mask with a key that combines both variable names
        lat_path = get_path(lat_var_name)
        masks[lat_path] = boolean_mask

    return_dataset = datatree_subset.where_tree(dataset, masks, cut, pixel_subset)
    return return_dataset


def subset_with_bbox(dataset: xr.Dataset,  # pylint: disable=too-many-branches
                     lat_var_names: list,
                     lon_var_names: list,
                     time_var_names: list,
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
    import numpy as np

    oper = np.logical_and

    if lon_bounds[0] > lon_bounds[1]:
        oper = np.logical_or

    subset_dictionary = {}

    for lat_var_name, lon_var_name, time_var_name in zip(lat_var_names, lon_var_names, time_var_names):

        lat_path = get_path(lat_var_name)
        lon_path = get_path(lon_var_name)
        time_path = get_path(time_var_name)

        lon_data = dataset[lon_var_name]
        lat_data = dataset[lat_var_name]
        time_data = dataset[time_var_name]

        temporal_cond = new_build_temporal_cond(min_time, max_time, dataset, time_var_name)
        if time_data.ndim == 1 and lon_data.ndim == 2:
            temporal_cond = align_time_to_lon_dim(time_data, lon_data, temporal_cond)

        operation = (
            oper((lon_data >= lon_bounds[0]), (lon_data <= lon_bounds[1])) &
            (lat_data >= lat_bounds[0]) &
            (lat_data <= lat_bounds[1]) &
            temporal_cond
        )

        # We want the lon lat time path to be the same
        # timeMidScan_datetime is a time made for ges disc collection in a ScanTime group
        if lat_path == lon_path == time_path or 'timeMidScan_datetime' in time_var_name:
            subset_dictionary[lat_path] = operation

    return_dataset = datatree_subset.where_tree(dataset, subset_dictionary, cut, pixel_subset)
    return return_dataset


def align_time_to_lon_dim(time_data, lon_data, temporal_cond):
    """
    Aligns a 1D time_data variable to one of the dimensions of a 2D lon_data array,
    renaming time_data's dimension if it matches the size of one of lon_data's dims.

    This happens because combining a 2D x 2D x 1D bitwise mask with mismatched dimensions 
    results in a 3D mask, which significantly increases memory usage. In this case, one 
    of the dimensions is a "phony" dimension, so we need to align the time variable with 
    the correct dimension to produce a proper 2D bitwise mask.

    Parameters:
        time_data (xr.DataArray): 1D array of time values.
        lon_data (xr.DataArray): 2D array with two dimensions.
        temporal_cond (xr.DataArray): 1D boolean mask along the time dimension.

    Returns:
        xr.DataArray: temporal_cond, potentially renamed to match lon_data's dims.
    """

    time_dim = time_data.dims[0]
    if time_dim not in lon_data.dims:
        time_dim_size = time_data.sizes.get(time_dim)

        lon_dim_1, lon_dim_2 = lon_data.dims
        lon_dim_1_size = lon_data.sizes.get(lon_dim_1)
        lon_dim_2_size = lon_data.sizes.get(lon_dim_2)

        if time_dim_size == lon_dim_1_size:
            return temporal_cond.rename({time_dim: lon_dim_1})
        elif time_dim_size == lon_dim_2_size:
            return temporal_cond.rename({time_dim: lon_dim_2})

    return temporal_cond  # Return unchanged if no renaming needed


def normalize_paths(paths):
    """
    Convert paths with __ notation to normal group paths, removing leading /
    and converting __ to /

    Parameters
    ----------
    paths : list of str
        List of paths with __ notation

    Returns
    -------
    list of str
        Normalized paths

    Examples
    --------
    >>> paths = ['/__geolocation__latitude', '/__geolocation__longitude']
    >>> normalize_paths(paths)
    ['/geolocation/latitude', '/geolocation/longitude']
    """
    normalized = []
    for path in paths:
        # Replace double underscore with slash
        path = path.replace('__', '/')
        # Remove any double slashes
        while '//' in path:
            path = path.replace('//', '/')
        # Ensure path starts with /
        if not path.startswith('/'):
            path = '/' + path
        normalized.append(path)
    return normalized


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

    if isinstance(dataset, xr.Dataset):
        tree = DataTree(dataset=dataset)
    else:
        tree = dataset

    dataset = tree

    if not lat_var_names or not lon_var_names:
        lon_var_names, lat_var_names = datatree_subset.compute_coordinate_variable_names_from_tree(dataset)
    if not time_var_names:
        time_var_names = []
        for lat_var_name in lat_var_names:

            parent_path = '/'.join(lat_var_name.split('/')[:-1])  # gives "data_20/c"

            subtree = dataset[parent_path]  # Gets the subtree at data_20/c
            variable = dataset[lat_var_name]  # Gets the latitude variable
            time_name = datatree_subset.compute_time_variable_name_tree(subtree,
                                                                        variable,
                                                                        time_var_names)
            time_var = f"{parent_path}{time_name}"
            time_var_names.append(time_var)

        if not time_var_names:
            time_var_names.append(compute_utc_name(dataset))

        seen = set()
        time_var_names = [x for x in time_var_names if x is not None and not (x in seen or seen.add(x))]

    lat_var_names = normalize_paths(lat_var_names)
    lon_var_names = normalize_paths(lon_var_names)
    time_var_names = normalize_paths(time_var_names)
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


def test_access_sst_dtime_values(nc_dataset):
    """
    Test accessing values of 'sst_dtime' variable in a NetCDF file.

    Parameters
    ----------
    nc_dataset (netCDF4.Dataset): An open NetCDF dataset.

    Returns
    -------
    access_successful (bool): True if 'sst_dtime' values are accessible, False otherwise.
    """
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
            for var_name in dataset.variables:
                dataset[var_name].values  # pylint: disable=pointless-statement
    except (TypeError, ValueError, KeyError):
        return False
    return True


def get_hdf_type(tree: xr.DataTree) -> Optional[str]:
    """
    Determine the HDF type (OMI or MLS) from a DataTree object.

    Parameters
    ----------
    tree : DataTree
        DataTree object containing the HDF data

    Returns
    -------
    Optional[str]
        'OMI', 'MLS', or None if type cannot be determined
    """
    try:
        # Try to get instrument information from FILE_ATTRIBUTES
        additional_attrs = tree['/HDFEOS/ADDITIONAL/FILE_ATTRIBUTES']
        if additional_attrs is not None and 'InstrumentName' in additional_attrs.attrs:
            instrument = additional_attrs.attrs['InstrumentName']
            if isinstance(instrument, bytes):
                instrument = instrument.decode("utf-8")
        else:
            return None

        # Determine HDF type based on instrument name
        if 'OMI' in instrument:
            return 'OMI'
        if 'MLS' in instrument:
            return 'MLS'

    except (KeyError, AttributeError):
        pass

    return None


def subset(file_to_subset: str, bbox: np.ndarray, output_file: str,
           variables: Union[List[str], str, None] = (),
           # pylint: disable=too-many-branches, disable=too-many-statements
           cut: bool = True, shapefile: str = None, min_time: str = None, max_time: str = None,
           origin_source: str = None,
           lat_var_names: List[str] = (), lon_var_names: List[str] = (), time_var_names: List[str] = (),
           pixel_subset: bool = False, stage_file_name_subsetted_true: str = None,
           stage_file_name_subsetted_false: str = None
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

    # clean up time variable in SNDR before decode_times
    # SNDR.AQUA files have ascending node time blank
    if any('__asc_node_tai93' in i for i in list(nc_dataset.variables)):
        asc_time_var = nc_dataset.variables['__asc_node_tai93']
        if not asc_time_var[:] > 0:
            del nc_dataset.variables['__asc_node_tai93']

    """

    file_extension = os.path.splitext(file_to_subset)[1]
    override_decode_cf_datetime()

    hdf_type = False

    args = {
        'decode_coords': False,
        'mask_and_scale': False,
        'decode_times': False
    }

    with xr.open_datatree(file_to_subset, **args) as dataset:
        if '.HDF5' == file_extension:
            for group in dataset.groups:
                if "ScanTime" in group:
                    hdf_type = 'GPM'

    if min_time or max_time:
        fill_value_f8 = nc.default_fillvals.get('f8')
        float_dtypes = ['float64', 'float32']
        args['decode_times'] = True
        # try to open file to see if we can access the time variable
        try:
            with nc.Dataset(file_to_subset, 'r') as nc_dataset:
                for time_variable in (v for v in nc_dataset.variables.keys() if 'time' in v):
                    time_var = nc_dataset[time_variable]
                    if (getattr(time_var, '_FillValue', None) == fill_value_f8 and time_var.dtype in float_dtypes) or \
                       (getattr(time_var, 'long_name', None) == "reference time of sst file"):
                        args['mask_and_scale'] = True
                        if getattr(time_var, 'long_name', None) == "reference time of sst file":
                            args['mask_and_scale'] = test_access_sst_dtime_values(nc_dataset)
                        break
        except Exception:  # pylint: disable=broad-exception-caught
            pass

    if hdf_type == 'GPM':
        args['decode_times'] = False

    time_encoding = {}
    time_calendar_attributes = {}

    if args['decode_times']:
        # Get time encoding
        with xr.open_datatree(file_to_subset, decode_times=False) as dataset:

            lat_var_names, lon_var_names, time_var_names = get_coordinate_variable_names(
                dataset=dataset,
                lat_var_names=lat_var_names,
                lon_var_names=lon_var_names,
                time_var_names=time_var_names
            )
            for time in time_var_names:

                time_var = dataset[time]
                var_name = os.path.basename(time)
                group_path = os.path.dirname(time)

                units = time_var.attrs.get('units')
                dtype = time_var.dtype
                calendar = time_var.attrs.get('calendar')

                if group_path not in time_encoding:
                    time_encoding[group_path] = {}

                time_encoding[group_path][var_name] = {}

                if calendar:
                    time_encoding[group_path][var_name]['calendar'] = calendar
                if units:
                    time_encoding[group_path][var_name]['units'] = units
                time_encoding[group_path][var_name]['dtype'] = dtype

                if calendar:
                    time_calendar_attributes[time] = calendar

    with xr.open_datatree(file_to_subset, **args) as dataset:

        if hdf_type is False:
            hdf_type = get_hdf_type(dataset)

        lat_var_names, lon_var_names, time_var_names = get_coordinate_variable_names(
            dataset=dataset,
            lat_var_names=lat_var_names,
            lon_var_names=lon_var_names,
            time_var_names=time_var_names
        )

        if not time_var_names and (min_time or max_time):
            raise ValueError('Could not determine time variable')

        if '.HDF5' == file_extension:
            time_var_names = []
            for group in dataset.groups:
                if "ScanTime" in group:
                    group_dataset = dataset[group].ds
                    dataset[group].ds = datatree_subset.update_dataset_with_time(group_dataset, group_path=group)
                    time_var_names.append(group + '/timeMidScan_datetime')

        if hdf_type and (min_time or max_time):
            dataset, _ = tree_time_converting.convert_to_datetime(dataset, time_var_names, hdf_type)

        chunks = calculate_chunks(dataset)
        if chunks:
            dataset = dataset.chunk(chunks)
        if variables:
            # Drop variables that aren't explicitly requested, except lat_var_name and
            # lon_var_name which are needed for subsetting
            normalized_variables = [f"/{s.replace('__', '/').lstrip('/')}".upper() for s in variables]

            keep_variables = normalized_variables + lon_var_names + lat_var_names + time_var_names

            all_data_variables = datatree_subset.get_vars_with_paths(dataset)
            drop_variables = [
                var for var in all_data_variables
                if var not in keep_variables and var.upper() not in keep_variables
            ]

            dataset = datatree_subset.drop_vars_by_path(dataset, drop_variables)

        if shapefile:
            subsetted_dataset = subset_with_shapefile_multi(
                dataset,
                lat_var_names,
                lon_var_names,
                shapefile,
                cut,
                chunks,
                pixel_subset
            )
        elif bbox is not None:
            subsetted_dataset = subset_with_bbox(
                dataset=dataset,
                lat_var_names=lat_var_names,
                lon_var_names=lon_var_names,
                time_var_names=time_var_names,
                bbox=bbox,
                cut=cut,
                min_time=min_time,
                max_time=max_time,
                pixel_subset=pixel_subset
            )
        else:
            raise ValueError('Either bbox or shapefile must be provided')

        set_version_history(subsetted_dataset, cut, bbox, shapefile)
        set_json_history(subsetted_dataset, cut, file_to_subset, bbox, shapefile, origin_source)

        if time_calendar_attributes:
            for time_var, calendar in time_calendar_attributes.items():
                if 'calendar' in subsetted_dataset[time_var].attrs:
                    subsetted_dataset[time_var].attrs['calendar'] = calendar
                    # if we set the calendar attribute remove calendar encoding
                    var_name = os.path.basename(time_var)
                    group_path = os.path.dirname(time_var)
                    # Safely remove calendar from encoding if it exists
                    if group_path in time_encoding and var_name in time_encoding[group_path]:
                        time_encoding[group_path][var_name].pop('calendar', None)

        subsetted_dataset = datatree_subset.clean_inherited_coords(subsetted_dataset)

        encoding = datatree_subset.prepare_basic_encoding(subsetted_dataset, time_encoding)

        spatial_bounds_array = datatree_subset.tree_get_spatial_bounds(
            subsetted_dataset,
            lat_var_names,
            lon_var_names
        )

        update_netcdf_attrs(output_file,
                            subsetted_dataset,
                            lon_var_names,
                            lat_var_names,
                            spatial_bounds_array,
                            stage_file_name_subsetted_true,
                            stage_file_name_subsetted_false)

        subsetted_dataset.to_netcdf(output_file, encoding=encoding)

        return spatial_bounds_array


def update_netcdf_attrs(output_file: str,
                        dataset: xr.DataTree,
                        lon_var_names: List[str],
                        lat_var_names: List[str],
                        spatial_bounds_array: Optional[list] = None,
                        stage_file_name_subsetted_true: Optional[str] = None,
                        stage_file_name_subsetted_false: Optional[str] = None) -> None:
    """
    Update NetCDF file attributes with spatial bounds and product name information.

    Args:
        output_file (str): Path to the NetCDF file to be updated
        dataset (xr.DataTree): xarray data tree
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

    for lon_var_name in lon_var_names:

        eastmost, westmost = get_east_west_lon(dataset, lon_var_name)

        if eastmost and westmost:
            lons_easternmost.append(eastmost)
            lons_westernmost.append(westmost)

    def set_attr_with_type(tree: DataTree, attr_name: str, value: Any) -> None:
        """Set attribute on a DataTree node while preserving its original type, if it exists."""
        original_attrs = tree.attrs
        if attr_name in original_attrs:
            original_type = type(original_attrs[attr_name])
            tree.attrs[attr_name] = original_type(value)

    if spatial_bounds_array is not None:
        # Define geographical bounds mapping
        bounds_mapping = {
            'geospatial_lat_max': (1, 1),
            'geospatial_lat_min': (1, 0)
        }

        for attr_name, (i, j) in bounds_mapping.items():
            set_attr_with_type(dataset, attr_name, spatial_bounds_array[i][j])

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

        for key in deprecated_keys & dataset.attrs.keys():
            del dataset.attrs[key]

        # Set CRS and bounds
        if lons_westernmost:
            final_westmost = min(lons_westernmost, key=lambda lon: lon if lon >= 0 else lon + 360)
            set_attr_with_type(dataset, "geospatial_lon_min", final_westmost)
        if lons_easternmost:
            final_eastmost = max(lons_easternmost, key=lambda lon: lon if lon >= 0 else lon + 360)
            set_attr_with_type(dataset, "geospatial_lon_max", final_eastmost)
        dataset.attrs["geospatial_bounds_crs"] = "EPSG:4326"
        geospatial_bounds = (
            create_geospatial_bounds(dataset, lon_var_names, lat_var_names)
            or create_geospatial_bounding_box(spatial_bounds_array, final_eastmost, final_westmost)
        )
        dataset.attrs["geospatial_bounds"] = geospatial_bounds

    # Set product name based on conditions
    has_spatial_bounds = spatial_bounds_array is not None and spatial_bounds_array.size > 0
    product_name = (stage_file_name_subsetted_true if has_spatial_bounds and stage_file_name_subsetted_true
                    else stage_file_name_subsetted_false if stage_file_name_subsetted_false
                    else output_file)

    set_attr_with_type(dataset, "product_name", product_name)


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


def create_geospatial_bounds(dataset, lon_var_names, lat_var_names):
    """Create geospatial bounds from 4 corners of 2d array"""

    for lon_var_name, lat_var_name in zip(lon_var_names, lat_var_names):

        lon = dataset[lon_var_name]
        lat = dataset[lat_var_name]

        lon_fill_value = lon.attrs.get('_FillValue', None)
        lat_fill_value = lat.attrs.get('_FillValue', None)

        break

    lon_scale = lon.attrs.get('scale_factor', 1.0)
    lon_offset = lon.attrs.get('add_offset', 0.0)

    lat_scale = lat.attrs.get('scale_factor', 1.0)
    lat_offset = lat.attrs.get('add_offset', 0.0)

    # Check if the variables are 2D arrays
    if lon.ndim != 2 or lat.ndim != 2:
        return None

    # Get the shape of the arrays (assuming they are the same shape)
    nrows, ncols = lon.shape

    points = [
        (
            float(remove_scale_offset(lon[0, 0], lon_scale, lon_offset)),
            float(remove_scale_offset(lat[0, 0], lat_scale, lat_offset))
        ),
        (
            float(remove_scale_offset(lon[nrows - 1, 0], lon_scale, lon_offset)),
            float(remove_scale_offset(lat[nrows - 1, 0], lat_scale, lat_offset))
        ),
        (
            float(remove_scale_offset(lon[nrows - 1, ncols - 1], lon_scale, lon_offset)),
            float(remove_scale_offset(lat[nrows - 1, ncols - 1], lat_scale, lat_offset))
        ),
        (
            float(remove_scale_offset(lon[0, ncols - 1], lon_scale, lon_offset)),
            float(remove_scale_offset(lat[0, ncols - 1], lat_scale, lat_offset))
        )
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

    lon_2d = dataset[lon_var_name]

    if lon_2d is None:
        return None, None

    fill_value = lon_2d.attrs.get('_FillValue', None)

    lon_flat = lon_2d.values.flatten()
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
