"""
===============
coordinate_utils.py
===============

Utility functions for coordinate operations and transformations.
"""

from typing import Tuple, Union
import numpy as np
import xarray as xr

from podaac.subsetter import datatree_subset


def _apply_scale_offset(scale: float, offset: float, value: float) -> float:
    """Apply scale and offset to the given value"""
    return (value + offset) / scale


def remove_scale_offset(value: float, scale: float, offset: float) -> float:
    """Remove scale and offset from the given value"""
    return (value * scale) - offset


def _convert_bound(bound: np.ndarray, coord_max: int, coord_var: xr.DataArray) -> np.ndarray:
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
    return _apply_scale_offset(scale, offset, bound)


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

    return np.array([_convert_bound(bbox[0], 360, lon_data),
                     _convert_bound(bbox[1], 180, lat_data)])


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


def get_coordinate_variable_names(dataset: xr.Dataset,
                                  lat_var_names: list = None,
                                  lon_var_names: list = None,
                                  time_var_names: list = None) -> Tuple[list[str], list[str], list[str]]:
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
    tuple[list[str], list[str], list[str]]
        A tuple containing three lists:
        - lat_var_names
            Normalized latitude coordinate variable names/paths.
        - lon_var_names
            Normalized longitude coordinate variable names/paths.
        - time_var_names
            Normalized time coordinate variable names/paths.
    """

    if isinstance(dataset, xr.Dataset):
        tree = xr.DataTree(dataset=dataset)
    else:
        tree = dataset

    if not lat_var_names or not lon_var_names:
        lon_var_names, lat_var_names = datatree_subset.compute_coordinate_variable_names_from_tree(tree)
    if not time_var_names:
        time_var_names = []
        for lat_var_name in lat_var_names:

            parent_path = '/'.join(lat_var_name.split('/')[:-1])  # gives "data_20/c"
            subtree = tree[parent_path]  # Gets the subtree at data_20/c
            variable = tree[lat_var_name]  # Gets the latitude variable
            time_name = datatree_subset.compute_time_variable_name_tree(subtree,
                                                                        variable,
                                                                        time_var_names)
            if time_name:
                time_name = time_name.strip('/')
                time_var = f"{parent_path}/{time_name}"
                time_var_names.append(time_var)

        if not time_var_names:
            time_var_names.append(_compute_utc_name(tree))

        if time_name is None:
            global_time_name = datatree_subset.compute_time_variable_name_tree(tree,
                                                                               variable,
                                                                               time_var_names)
            if global_time_name:
                time_var_names.append(global_time_name)

        seen = set()
        time_var_names = [x for x in time_var_names if x is not None and not (x in seen or seen.add(x))]

    lat_var_names = _normalize_paths(lat_var_names)
    lon_var_names = _normalize_paths(lon_var_names)
    time_var_names = _normalize_paths(time_var_names)
    return lat_var_names, lon_var_names, time_var_names


def _compute_utc_name(dataset: xr.Dataset) -> Union[str, None]:
    """
    Get the name of the utc variable if it is there to determine origine time
    """
    for var_name in list(dataset.data_vars.keys()):
        if 'utc' in var_name.lower() and 'time' in var_name.lower():
            return var_name

    return None


def _normalize_paths(paths):
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
    >>> _normalize_paths(paths)
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
