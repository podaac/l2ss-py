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

import os
from itertools import zip_longest
from typing import List, Union

import geopandas as gpd
import netCDF4 as nc
import numpy as np
import xarray as xr
from shapely.geometry import Point

from podaac.subsetter import (
    datatree_subset,
    tree_time_converting as tree_time_converting
)
from podaac.subsetter.utils import mask_utils
from podaac.subsetter.utils import coordinate_utils
from podaac.subsetter.utils import metadata_utils
from podaac.subsetter.utils import spatial_utils
from podaac.subsetter.utils import time_utils
from podaac.subsetter.utils import file_utils
from podaac.subsetter.utils import variables_utils

SERVICE_NAME = 'l2ss-py'


def subset_with_shapefile_multi(dataset: xr.Dataset,
                                lat_var_names: List[str],
                                lon_var_names: List[str],
                                shapefile: str,
                                cut: bool,
                                pixel_subset: bool) -> xr.Dataset:
    """
    Subset an xarray Dataset using a shapefile for multiple latitude and longitude variable pairs

    Returns
    -------
    xr.Dataset
        The subsetted dataset
    """
    if len(lat_var_names) != len(lon_var_names):
        raise ValueError("Number of latitude variables must match number of longitude variables")

    shapefile_df = gpd.read_file(shapefile).to_crs("EPSG:4326")
    masks = {}

    for lat_var_name, lon_var_name in zip(lat_var_names, lon_var_names):
        lat = dataset[lat_var_name]
        lon = dataset[lon_var_name]

        lat_scale = lat.attrs.get("scale_factor", 1.0)
        lon_scale = lon.attrs.get("scale_factor", 1.0)
        lat_offset = lat.attrs.get("add_offset", 0.0)
        lon_offset = lon.attrs.get("add_offset", 0.0)

        # Apply scale and offset
        lat_vals = lat.values * lat_scale + lat_offset
        lon_vals = lon.values * lon_scale + lon_offset

        # Handle 2D or 1D lat/lon
        if lat_vals.ndim == 1 and lon_vals.ndim == 1:
            lon2d, lat2d = np.meshgrid(lon_vals, lat_vals)
        else:
            lat2d = lat_vals
            lon2d = lon_vals

        # Convert shapefile to 0-360 if needed
        current_shapefile_df = shapefile_df.copy()
        if coordinate_utils.is_360(lon, lon_scale, lon_offset):
            current_shapefile_df["geometry"] = current_shapefile_df["geometry"].apply(spatial_utils.translate_longitude)

        # Flatten points and convert to GeoDataFrame
        flat_points = np.column_stack((lon2d.ravel(), lat2d.ravel()))
        point_gdf = gpd.GeoDataFrame(
            geometry=[Point(xy) for xy in flat_points],
            crs="EPSG:4326"
        )

        # Spatial join to find points inside shapefile
        joined = gpd.sjoin(point_gdf, current_shapefile_df, how="left", predicate="intersects")
        inside_mask_flat = ~joined.index_right.isna().to_numpy()
        inside_mask = inside_mask_flat.reshape(lat2d.shape)

        # Create DataArray aligned with original dims
        mask_da = xr.DataArray(inside_mask, dims=lat.dims, coords=lat.coords)

        lat_path = file_utils.get_path(lat_var_name)
        masks[lat_path] = mask_da

    # Apply your datatree-aware masking logic
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
    lon_bounds, lat_bounds = coordinate_utils.convert_bbox(bbox, dataset, lat_var_names[0], lon_var_names[0])

    # condition should be 'or' instead of 'and' when bbox lon_min > lon_max
    oper = np.logical_and

    if lon_bounds[0] > lon_bounds[1]:
        oper = np.logical_or

    subset_dictionary = {}

    if not time_var_names:  # time_var_names == [] or evaluates to False
        iterator = zip_longest(lat_var_names, lon_var_names, [])
    else:
        iterator = zip(lat_var_names, lon_var_names, time_var_names)

    for lat_var_name, lon_var_name, time_var_name in iterator:

        lat_path = file_utils.get_path(lat_var_name)
        lon_path = file_utils.get_path(lon_var_name)

        lon_data = dataset[lon_var_name]
        lat_data = dataset[lat_var_name]

        temporal_cond = time_utils.build_temporal_cond(min_time, max_time, dataset, time_var_name)
        time_path = None
        if time_var_name:
            time_path = file_utils.get_path(time_var_name)
            time_data = dataset[time_var_name]

            if time_data.ndim == 1 and lon_data.ndim == 2 and temporal_cond is not True:
                temporal_cond = mask_utils.align_time_to_lon_dim(time_data, lon_data, temporal_cond)

        operation = (
            oper((lon_data >= lon_bounds[0]), (lon_data <= lon_bounds[1])) &
            (lat_data >= lat_bounds[0]) &
            (lat_data <= lat_bounds[1]) &
            temporal_cond
        )

        # We want the lon lat time path to be the same
        # timeMidScan_datetime is a time made for ges disc collection in a ScanTime group
        if (
            lat_path == lon_path == time_path
            or (time_var_name is not None and 'timeMidScan_datetime' in time_var_name)
            or (lon_path == lat_path and time_var_name is None)
           ):
            subset_dictionary[lat_path] = operation
        elif lat_path == lon_path and len(time_var_names) == 1:
            subset_dictionary[lat_path] = operation

    return_dataset = datatree_subset.where_tree(dataset, subset_dictionary, cut, pixel_subset)
    return return_dataset


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
    file_utils.override_decode_cf_datetime()

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
                            args['mask_and_scale'] = file_utils.test_access_sst_dtime_values(nc_dataset)
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

            lat_var_names, lon_var_names, time_var_names = coordinate_utils.get_coordinate_variable_names(
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

        hdf_type = file_utils.get_hdf_type(dataset)

        lat_var_names, lon_var_names, time_var_names = coordinate_utils.get_coordinate_variable_names(
            dataset=dataset,
            lat_var_names=lat_var_names,
            lon_var_names=lon_var_names,
            time_var_names=time_var_names
        )

        if '.HDF5' == file_extension:
            new_time_var_names = []
            for group in dataset.groups:
                if "ScanTime" in group:
                    group_dataset = dataset[group].ds
                    dataset[group].ds = datatree_subset.update_dataset_with_time(group_dataset, group_path=group)
                    if 'timeMidScan_datetime' in dataset[group].ds:
                        new_time_var_names.append(group + '/timeMidScan_datetime')

            if new_time_var_names:
                time_var_names = new_time_var_names

        if not time_var_names and (min_time or max_time):
            raise ValueError('Could not determine time variable')

        if hdf_type and (min_time or max_time):
            dataset, _ = tree_time_converting.convert_to_datetime(dataset, time_var_names, hdf_type)

        chunks = file_utils.calculate_chunks(dataset)
        all_vars = variables_utils.get_all_variable_names_from_dtree(dataset)
        if chunks:
            dataset = dataset.chunk(chunks)
        if variables:
            # Drop variables that aren't explicitly requested, except lat_var_name and
            # lon_var_name which are needed for subsetting
            normalized_variables = [f"/{s.replace('__', '/').lstrip('/')}".upper() for s in variables]

            keep_variables = normalized_variables + lon_var_names + lat_var_names + time_var_names
            keep_variables = variables_utils.normalize_candidate_paths_against_dtree(keep_variables, all_vars)

            all_data_variables = datatree_subset.get_vars_with_paths(dataset)
            drop_variables = [
                var for var in all_data_variables
                if var not in keep_variables and var.upper() not in keep_variables
            ]

            dataset = datatree_subset.drop_vars_by_path(dataset, drop_variables)

        lon_var_names = variables_utils.normalize_candidate_paths_against_dtree(lon_var_names, all_vars)
        lat_var_names = variables_utils.normalize_candidate_paths_against_dtree(lat_var_names, all_vars)
        time_var_names = variables_utils.normalize_candidate_paths_against_dtree(time_var_names, all_vars)

        if shapefile:
            subsetted_dataset = subset_with_shapefile_multi(
                dataset,
                lat_var_names,
                lon_var_names,
                shapefile,
                cut,
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

        metadata_utils.set_version_history(subsetted_dataset, cut, bbox, shapefile)
        metadata_utils.set_json_history(subsetted_dataset, cut, file_to_subset, bbox, shapefile, origin_source)

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

        metadata_utils.update_netcdf_attrs(output_file,
                                           subsetted_dataset,
                                           lon_var_names,
                                           lat_var_names,
                                           spatial_bounds_array,
                                           stage_file_name_subsetted_true,
                                           stage_file_name_subsetted_false)

        try:
            subsetted_dataset.to_netcdf(output_file, encoding=encoding)
        except AttributeError as e:
            if "NetCDF: Name contains illegal characters" in str(e):
                metadata_utils.fix_illegal_datatree_attrs(subsetted_dataset)
                subsetted_dataset.to_netcdf(output_file, encoding=encoding)
            else:
                raise

        metadata_utils.ensure_time_units(output_file, time_encoding)

        # ensure all the dimensions are on the root node when we pixel subset
        if pixel_subset:
            def add_all_group_dims_to_root_inplace(nc_path):
                def collect_dims(group, dims):
                    for dimname, dim in group.dimensions.items():
                        if dimname not in dims:
                            dims[dimname] = len(dim) if not dim.isunlimited() else None
                    for subgrp in group.groups.values():
                        collect_dims(subgrp, dims)

                with nc.Dataset(nc_path, 'r+') as ds:
                    all_dims = {}
                    collect_dims(ds, all_dims)
                    for dimname, size in all_dims.items():
                        if dimname not in ds.dimensions:
                            ds.createDimension(dimname, size)
            add_all_group_dims_to_root_inplace(output_file)

        return spatial_bounds_array
