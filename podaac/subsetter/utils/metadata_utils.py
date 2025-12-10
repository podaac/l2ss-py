"""
===============
metadata_utils.py
===============

Utility functions for metadata operations and history management.
"""

import datetime
import json
import os
import re
from typing import Any, List, Optional

import importlib_metadata
import netCDF4 as nc
import numpy as np
import xarray as xr
from xarray import DataTree

from podaac.subsetter.utils import spatial_utils


SERVICE_NAME = 'l2ss-py'


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


def _get_group(ds, group_path):
    """Traverse the group path and return the final group."""
    if group_path in ('', '/'):
        return ds
    group = ds
    for part in group_path.strip('/').split('/'):
        group = group.groups[part]
    return group


def ensure_time_units(nc_file, time_encoding):
    """
    Update the units attribute for time variables in specified netCDF groups.

    Args:
        nc_file (str): Path to the netCDF file.
        time_encoding (dict): Dictionary structure {group: {var_name: {attr: value}}}
    """
    with nc.Dataset(nc_file, 'r+') as ds:
        for group_name, vars_dict in time_encoding.items():
            try:
                group = _get_group(ds, group_name)
            except KeyError:
                continue

            for var_name, attr_dict in vars_dict.items():
                if var_name not in group.variables:
                    continue
                var = group.variables[var_name]
                if 'units' in attr_dict:
                    current_units = getattr(var, 'units', None)
                    correct_units = attr_dict['units']
                    if current_units != correct_units:
                        var.units = correct_units
                if 'calendar' in attr_dict:
                    current_calendar = getattr(var, 'calendar', None)
                    correct_calendar = attr_dict['calendar']
                    if current_calendar != correct_calendar:
                        var.calendar = correct_calendar


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

        eastmost, westmost = spatial_utils.get_east_west_lon(dataset, lon_var_name)

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
            spatial_utils.create_geospatial_bounds(dataset, lon_var_names, lat_var_names)
            or spatial_utils.create_geospatial_bounding_box(spatial_bounds_array, final_eastmost, final_westmost)
        )
        dataset.attrs["geospatial_bounds"] = geospatial_bounds

    # Set product name based on conditions
    has_spatial_bounds = spatial_bounds_array is not None and spatial_bounds_array.size > 0
    product_name = (stage_file_name_subsetted_true if has_spatial_bounds and stage_file_name_subsetted_true
                    else stage_file_name_subsetted_false if stage_file_name_subsetted_false
                    else output_file)

    set_attr_with_type(dataset, "product_name", product_name)


def legalize_attr_name(attr):
    """
    make sure attr name is legal
    """
    # Only allow ASCII letters, digits, underscore, hyphen; no spaces, no leading digit
    if not isinstance(attr, str):
        return attr
    name = attr.replace(" ", "_")
    name = re.sub(r'[^A-Za-z0-9_\-]', '', name)
    if re.match(r'^\d', name):
        name = '_' + name
    return name


def check_illegal_datatree_attrs(dt):
    """
    Checks for illegal attribute names in a DataTree and returns True if any are found.
    Does not modify anything.
    """
    def is_illegal(name):
        if not isinstance(name, str):
            return True
        if re.match(r'^\d', name):
            return True
        if re.search(r'[^A-Za-z0-9_\-]', name):
            return True
        return False

    found_illegal = False
    for node in dt.subtree:
        ds = node.ds
        if ds is not None:
            # Check global attributes
            for k in ds.attrs:
                if is_illegal(k):
                    found_illegal = True
                    break
            # Check variable attributes
            for var_name in ds.variables:
                var = ds[var_name]
                for k in var.attrs:
                    if is_illegal(k):
                        found_illegal = True
                        break
    return found_illegal


def fix_illegal_datatree_attrs(dt):
    """
    Fix illegal attribute names in a DataTree (in-place).
    """
    for node in dt.subtree:
        ds = node.ds
        if ds is None:
            continue

        # Fix global attrs
        new_attrs = {}
        for k, v in ds.attrs.items():
            new_k = legalize_attr_name(k)
            if new_k in new_attrs:
                raise ValueError(f"Collision after legalization: {k} -> {new_k}")
            new_attrs[new_k] = v
        ds.attrs.clear()
        ds.attrs.update(new_attrs)

        # Fix variable attrs
        for var_name, var in ds.variables.items():
            new_var_attrs = {}
            for k, v in var.attrs.items():
                new_k = legalize_attr_name(k)
                if new_k in new_var_attrs:
                    raise ValueError(f"Collision in {var_name} attrs: {k} -> {new_k}")
                new_var_attrs[new_k] = v
            var.attrs.clear()
            var.attrs.update(new_var_attrs)
