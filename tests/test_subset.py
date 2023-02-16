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
==============
test_subset.py
==============
Test the subsetter functionality.

Unit tests for the L2 subsetter. These tests are all related to the
subsetting functionality itself, and should provide coverage on the
following files:
    - podaac.subsetter.subset.py
    - podaac.subsetter.xarray_enhancements.py
"""
import json
import operator
import os
import shutil
import tempfile
import unittest
from os import listdir
from os.path import dirname, join, realpath, isfile, basename

import geopandas as gpd
import importlib_metadata
import netCDF4
import netCDF4 as nc
import h5py
import numpy as np
import pandas as pd
import pytest
import xarray as xr
import urllib.parse
from jsonschema import validate
from shapely.geometry import Point
from unittest import TestCase

from podaac.subsetter import subset
from podaac.subsetter.group_handling import GROUP_DELIM
from podaac.subsetter.subset import SERVICE_NAME
from podaac.subsetter import xarray_enhancements as xre
from podaac.subsetter import dimension_cleanup as dc


@pytest.fixture(scope='class')
def data_dir():
    test_dir = dirname(realpath(__file__))
    return join(test_dir, 'data')


@pytest.fixture(scope='class')
def subset_output_dir(data_dir):
    subset_output_dir = tempfile.mkdtemp(dir=data_dir)
    yield subset_output_dir
    shutil.rmtree(subset_output_dir)


@pytest.fixture(scope='class')
def history_json_schema():
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://harmony.earthdata.nasa.gov/history.schema.json",
        "title": "Data Processing History",
        "description": "A history record of processing that produced a given data file. For more information, see: https://wiki.earthdata.nasa.gov/display/TRT/In-File+Provenance+Metadata+-+TRT-42",
        "type": ["array", "object"],
        "items": {"$ref": "#/definitions/history_record"},

        "definitions": {
            "history_record": {
                "type": "object",
                "properties": {
                    "date_time": {
                        "description": "A Date/Time stamp in ISO-8601 format, including time-zone, GMT (or Z) preferred",
                        "type": "string",
                        "format": "date-time"
                    },
                    "derived_from": {
                        "description": "List of source data files used in the creation of this data file",
                        "type": ["array", "string"],
                        "items": {"type": "string"}
                    },
                    "program": {
                        "description": "The name of the program which generated this data file",
                        "type": "string"
                    },
                    "version": {
                        "description": "The version identification of the program which generated this data file",
                        "type": "string"
                    },
                    "parameters": {
                        "description": "The list of parameters to the program when generating this data file",
                        "type": ["array", "string"],
                        "items": {"type": "string"}
                    },
                    "program_ref": {
                        "description": "A URL reference that defines the program, e.g., a UMM-S reference URL",
                        "type": "string"
                    },
                    "$schema": {
                        "description": "The URL to this schema",
                        "type": "string"
                    }
                },
                "required": ["date_time", "program"],
                "additionalProperties": False
            }
        }
    }


def data_files():
    test_dir = dirname(realpath(__file__))
    test_data_dir = join(test_dir, 'data')
    return [f for f in listdir(test_data_dir) if isfile(join(test_data_dir, f)) and f.endswith(".nc")]


TEST_DATA_FILES = data_files()


@pytest.mark.parametrize("test_file", TEST_DATA_FILES)
def test_subset_variables(test_file, data_dir, subset_output_dir, request):
    """
    Test that all variables present in the original NetCDF file
    are present after the subset takes place, and with the same
    attributes.
    """

    bbox = np.array(((-180, 90), (-90, 90)))
    output_file = "{}_{}".format(request.node.name, test_file)
    subset.subset(
        file_to_subset=join(data_dir, test_file),
        bbox=bbox,
        output_file=join(subset_output_dir, output_file)
    )

    in_ds = xr.open_dataset(join(data_dir, test_file),
                            decode_times=False,
                            decode_coords=False)
    out_ds = xr.open_dataset(join(subset_output_dir, output_file),
                             decode_times=False,
                             decode_coords=False)

    for in_var, out_var in zip(in_ds.data_vars.items(), out_ds.data_vars.items()):
        # compare names
        assert in_var[0] == out_var[0]

        # compare attributes
        np.testing.assert_equal(in_var[1].attrs, out_var[1].attrs)

        # compare type and dimension names
        assert in_var[1].dtype == out_var[1].dtype
        assert in_var[1].dims == out_var[1].dims

    in_ds.close()
    out_ds.close()


@pytest.mark.parametrize("test_file", TEST_DATA_FILES)
def test_subset_bbox(test_file, data_dir, subset_output_dir, request):
    """
    Test that all data present is within the bounding box given,
    and that the correct bounding box is used. This test assumed
    that the scanline *is* being cut.
    """

    # pylint: disable=too-many-locals
    bbox = np.array(((-180, 90), (-90, 90)))
    output_file = "{}_{}".format(request.node.name, test_file)
    subset_output_file = join(subset_output_dir, output_file)
    subset.subset(
        file_to_subset=join(data_dir, test_file),
        bbox=bbox,
        output_file=subset_output_file
    )

    out_ds, rename_vars, _ = subset.open_as_nc_dataset(subset_output_file)
    out_ds = xr.open_dataset(xr.backends.NetCDF4DataStore(out_ds),
                             decode_times=False,
                             decode_coords=False,
                             mask_and_scale=False)

    lat_var_name, lon_var_name = subset.compute_coordinate_variable_names(out_ds)

    lat_var_name = lat_var_name[0]
    lon_var_name = lon_var_name[0]

    lon_bounds, lat_bounds = subset.convert_bbox(bbox, out_ds, lat_var_name, lon_var_name)

    lats = out_ds[lat_var_name].values
    lons = out_ds[lon_var_name].values

    np.warnings.filterwarnings('ignore')

    # Step 1: Get mask of values which aren't in the bounds.

    # For lon spatial condition, need to consider the
    # lon_min > lon_max case. If that's the case, should do
    # an 'or' instead.
    oper = operator.and_ if lon_bounds[0] < lon_bounds[1] else operator.or_

    # In these two masks, True == valid and False == invalid
    lat_truth = np.ma.masked_where((lats >= lat_bounds[0])
                                   & (lats <= lat_bounds[1]), lats).mask
    lon_truth = np.ma.masked_where(oper((lons >= lon_bounds[0]),
                                        (lons <= lon_bounds[1])), lons).mask

    # combine masks
    spatial_mask = np.bitwise_and(lat_truth, lon_truth)

    # Create a mask which represents the valid matrix bounds of
    # the spatial mask. This is used in the case where a var
    # has no _FillValue.
    if lon_truth.ndim == 1:
        bound_mask = spatial_mask
    else:
        rows = np.any(spatial_mask, axis=1)
        cols = np.any(spatial_mask, axis=0)
        bound_mask = np.array([[r & c for c in cols] for r in rows])

    # If all the lat/lon values are valid, the file is valid and
    # there is no need to check individual variables.
    if np.all(spatial_mask):
        return

    # Step 2: Get mask of values which are NaN or "_FillValue in
    # each variable.
    for var_name, var in out_ds.data_vars.items():
        # remove dimension of '1' if necessary
        vals = np.squeeze(var.values)

        # Get the Fill Value
        fill_value = var.attrs.get('_FillValue')

        # If _FillValue isn't provided, check that all values
        # are in the valid matrix bounds go to the next variable
        if fill_value is None:
            combined_mask = np.ma.mask_or(spatial_mask, bound_mask)
            np.testing.assert_equal(bound_mask, combined_mask)
            continue

        # If the shapes of this var doesn't match the mask,
        # reshape the var so the comparison can be made. Take
        # the first index of the unknown dims. This makes
        # assumptions about the ordering of the dimensions.
        if vals.shape != out_ds[lat_var_name].shape and vals.shape:
            slice_list = []
            for dim in var.dims:
                if dim in out_ds[lat_var_name].dims:
                    slice_list.append(slice(None))
                else:
                    slice_list.append(slice(0, 1))
            vals = np.squeeze(vals[tuple(slice_list)])

        # Skip for byte type.
        if vals.dtype == 'S1':
            continue

        # In this mask, False == NaN and True = valid
        var_mask = np.invert(np.ma.masked_invalid(vals).mask)
        fill_mask = np.invert(np.ma.masked_values(vals, fill_value).mask)

        var_mask = np.bitwise_and(var_mask, fill_mask)

        if var_mask.shape != spatial_mask.shape:
            # This may be a case where the time represents lines,
            # or some other case where the variable doesn't share
            # a shape with the coordinate variables.
            continue

        # Step 3: Combine the spatial and var mask with 'or'
        combined_mask = np.ma.mask_or(var_mask, spatial_mask)

        # Step 4: compare the newly combined mask and the
        # spatial mask created from the lat/lon masks. They
        # should be equal, because the 'or' of the two masks
        # where out-of-bounds values are 'False' will leave
        # those values assuming there are only NaN values
        # in the data at those locations.
        np.testing.assert_equal(spatial_mask, combined_mask)

    out_ds.close()


@pytest.mark.parametrize("test_file", TEST_DATA_FILES)
def test_subset_empty_bbox(test_file, data_dir, subset_output_dir, request):
    """
    Test that an empty file is returned when the bounding box
    contains no data.
    """

    bbox = np.array(((120, 125), (-90, -85)))
    output_file = "{}_{}".format(request.node.name, test_file)
    subset.subset(
        file_to_subset=join(data_dir, test_file),
        bbox=bbox,
        output_file=join(subset_output_dir, output_file)
    )
    test_input_dataset = xr.open_dataset(
        join(data_dir, test_file),
        decode_times=False,
        decode_coords=False,
        mask_and_scale=False
    )
    empty_dataset = xr.open_dataset(
        join(subset_output_dir, output_file),
        decode_times=False,
        decode_coords=False,
        mask_and_scale=False
    )

    # Ensure all variables are present but empty.
    for variable_name, variable in empty_dataset.data_vars.items():
        assert np.all(variable.data == variable.attrs.get('_FillValue', np.nan) or np.isnan(variable.data))

    assert test_input_dataset.dims.keys() == empty_dataset.dims.keys()


def test_bbox_conversion(data_dir):
    """
    Test that the bounding box conversion returns expected
    results. Expected results are hand-calculated.
    """

    ds_180 = xr.open_dataset(join(data_dir, "MODIS_A-JPL-L2P-v2014.0.nc"),
                             decode_times=False,
                             decode_coords=False)

    ds_360 = xr.open_dataset(join(
        data_dir,
        "ascat_20150702_084200_metopa_45145_eps_o_250_2300_ovw.l2.nc"),
        decode_times=False,
        decode_coords=False)

    # Elements in each tuple are:
    # ds type, lon_range, expected_result
    test_bboxes = [
        (ds_180, (-180, 180), (-180, 180)),
        (ds_360, (-180, 180), (0, 360)),
        (ds_180, (-180, 0), (-180, 0)),
        (ds_360, (-180, 0), (180, 360)),
        (ds_180, (-80, 80), (-80, 80)),
        (ds_360, (-80, 80), (280, 80)),
        (ds_180, (0, 180), (0, 180)),
        (ds_360, (0, 180), (0, 180)),
        (ds_180, (80, -80), (80, -80)),
        (ds_360, (80, -80), (80, 280)),
        (ds_180, (-80, -80), (-180, 180)),
        (ds_360, (-80, -80), (0, 360))
    ]

    lat_var = 'lat'
    lon_var = 'lon'

    for test_bbox in test_bboxes:
        dataset = test_bbox[0]
        lon_range = test_bbox[1]
        expected_result = test_bbox[2]
        actual_result, _ = subset.convert_bbox(np.array([lon_range, [0, 0]]),
                                               dataset, lat_var, lon_var)

        np.testing.assert_equal(actual_result, expected_result)


def compare_java(test_file, cut, data_dir, subset_output_dir, request):
    """
    Run the L2 subsetter and compare the result to the equivelant
    legacy (Java) subsetter result.
    Parameters
    ----------
    test_file : str
        path to test file.
    cut : boolean
        True if the subsetter should return compact.
    """
    bbox_map = [("ascat_20150702_084200", ((-180, 0), (-90, 0))),
                ("ascat_20150702_102400", ((-180, 0), (-90, 0))),
                ("MODIS_A-JPL", ((65.8, 86.35), (40.1, 50.15))),
                ("MODIS_T-JPL", ((-78.7, -60.7), (-54.8, -44))),
                ("VIIRS", ((-172.3, -126.95), (62.3, 70.65))),
                ("AMSR2-L2B_v08_r38622", ((-180, 0), (-90, 0)))]

    java_files_dir = join(data_dir, "java_results", "cut" if cut else "uncut")

    java_files = [join(java_files_dir, f) for f in listdir(java_files_dir) if
                  isfile(join(java_files_dir, f)) and f.endswith(".nc")]

    file, bbox = next(iter([b for b in bbox_map if b[0] in test_file]))
    java_file = next(iter([f for f in java_files if file in f]))

    output_file = "{}_{}".format(urllib.parse.quote_plus(request.node.name), test_file)
    subset.subset(
        file_to_subset=join(data_dir, test_file),
        bbox=np.array(bbox),
        output_file=join(subset_output_dir, output_file),
        cut=cut
    )

    j_ds = xr.open_dataset(join(data_dir, java_file),
                           decode_times=False,
                           decode_coords=False,
                           mask_and_scale=False)

    py_ds = xr.open_dataset(join(subset_output_dir, output_file),
                            decode_times=False,
                            decode_coords=False,
                            mask_and_scale=False)

    for var_name, var in j_ds.data_vars.items():
        # Compare shape
        np.testing.assert_equal(var.shape, py_ds[var_name].shape)

        # Compare meta
        np.testing.assert_equal(var.attrs, py_ds[var_name].attrs)

        # Compare data
        np.testing.assert_equal(var.values, py_ds[var_name].values)

    # Compare meta. History will always be different, so remove
    # from the headers for comparison.
    del j_ds.attrs['history']
    del py_ds.attrs['history']
    del py_ds.attrs['history_json']
    np.testing.assert_equal(j_ds.attrs, py_ds.attrs)


@pytest.mark.parametrize("test_file", [
    "ascat_20150702_084200_metopa_45145_eps_o_250_2300_ovw.l2.nc",
    "ascat_20150702_102400_metopa_45146_eps_o_250_2300_ovw.l2.nc",
    "MODIS_A-JPL-L2P-v2014.0.nc",
    "MODIS_T-JPL-L2P-v2014.0.nc",
    "VIIRS_NPP-NAVO-L2P-v3.0.nc",
    "AMSR2-L2B_v08_r38622-v02.0-fv01.0.nc"
])
def test_compare_java_compact(test_file, data_dir, subset_output_dir, request):
    """
    Tests that the results of the subsetting operation is
    equivalent to the Java subsetting result on the same bounding
    box. For simplicity the subsetted Java granules have been
    manually run and copied into this project. This test DOES
    cut the scanline.
    """

    compare_java(test_file, True, data_dir, subset_output_dir, request)


@pytest.mark.parametrize("test_file", [
    "ascat_20150702_084200_metopa_45145_eps_o_250_2300_ovw.l2.nc",
    "ascat_20150702_102400_metopa_45146_eps_o_250_2300_ovw.l2.nc",
    "MODIS_A-JPL-L2P-v2014.0.nc",
    "MODIS_T-JPL-L2P-v2014.0.nc",
    "VIIRS_NPP-NAVO-L2P-v3.0.nc",
    "AMSR2-L2B_v08_r38622-v02.0-fv01.0.nc"
])
def test_compare_java(test_file, data_dir, subset_output_dir, request):
    """
    Tests that the results of the subsetting operation is
    equivalent to the Java subsetting result on the same bounding
    box. For simplicity the subsetted Java granules have been
    manually run and copied into this project. This runs does NOT
    cut the scanline.
    """

    compare_java(test_file, False, data_dir, subset_output_dir, request)


def test_history_metadata_append(data_dir, subset_output_dir, request):
    """
    Tests that the history metadata header is appended to when it
    already exists.
    """
    test_file = next(filter(
        lambda f: '20180101005944-REMSS-L2P_GHRSST-SSTsubskin-AMSR2-L2B_rt_r29918-v02.0-fv01.0.nc' in f
        , TEST_DATA_FILES))
    output_file = "{}_{}".format(request.node.name, test_file)
    subset.subset(
        file_to_subset=join(data_dir, test_file),
        bbox=np.array(((-180, 180), (-90.0, 90))),
        output_file=join(subset_output_dir, output_file)
    )

    in_nc = xr.open_dataset(join(data_dir, test_file))
    out_nc = xr.open_dataset(join(subset_output_dir, output_file))

    # Assert that the original granule contains history
    assert in_nc.attrs.get('history') is not None

    # Assert that input and output files have different history
    assert in_nc.attrs['history'] != out_nc.attrs['history']

    # Assert that last line of history was created by this service
    assert SERVICE_NAME in out_nc.attrs['history'].split('\n')[-1]

    # Assert that the old history is still in the subsetted granule
    assert in_nc.attrs['history'] in out_nc.attrs['history']


def test_history_metadata_create(data_dir, subset_output_dir, request):
    """
    Tests that the history metadata header is created when it does
    not exist. All test granules contain this header already, so
    for this test the header will be removed manually from a granule.
    """
    test_file = next(filter(
        lambda f: '20180101005944-REMSS-L2P_GHRSST-SSTsubskin-AMSR2-L2B_rt_r29918-v02.0-fv01.0.nc' in f
        , TEST_DATA_FILES))
    output_file = "{}_{}".format(request.node.name, test_file)

    # Remove the 'history' metadata from the granule
    in_nc = xr.open_dataset(join(data_dir, test_file))
    del in_nc.attrs['history']
    in_nc.to_netcdf(join(subset_output_dir, 'int_{}'.format(output_file)), 'w')

    subset.subset(
        file_to_subset=join(subset_output_dir, "int_{}".format(output_file)),
        bbox=np.array(((-180, 180), (-90.0, 90))),
        output_file=join(subset_output_dir, output_file)
    )

    out_nc = xr.open_dataset(join(subset_output_dir, output_file))

    # Assert that the input granule contains no history
    assert in_nc.attrs.get('history') is None

    # Assert that the history was created by this service
    assert SERVICE_NAME in out_nc.attrs['history']

    # Assert that the history created by this service is the only
    # line present in the history.
    assert '\n' not in out_nc.attrs['history']


@pytest.mark.parametrize("test_file", TEST_DATA_FILES)
def test_specified_variables(test_file, data_dir, subset_output_dir, request):
    """
    Test that the variables which are specified when calling the subset
    operation are present in the resulting subsetted data file,
    and that the variables which are specified are not present.
    """
    bbox = np.array(((-180, 180), (-90, 90)))
    output_file = "{}_{}".format(request.node.name, test_file)

    in_ds, rename_vars, _ = subset.open_as_nc_dataset(join(data_dir, test_file))
    in_ds = xr.open_dataset(xr.backends.NetCDF4DataStore(in_ds),
                            decode_times=False,
                            decode_coords=False)
    # Non-data vars are by default included in the result
    non_data_vars = set(in_ds.variables.keys()) - set(in_ds.data_vars.keys())

    # Coordinate variables are always included in the result
    lat_var_names, lon_var_names, time_var_names = subset.get_coordinate_variable_names(in_ds)
    coordinate_variables = lat_var_names + lon_var_names + time_var_names

    # Pick some variable to include in the result
    included_variables = set([variable[0] for variable in in_ds.data_vars.items()][::2])
    included_variables = list(included_variables)

    # All other data variables should be dropped
    expected_excluded_variables = list(set(variable[0] for variable in in_ds.data_vars.items())
                                       - set(included_variables) - set(coordinate_variables))

    subset.subset(
        file_to_subset=join(data_dir, test_file),
        bbox=bbox,
        output_file=join(subset_output_dir, output_file),
        variables=[var.replace(GROUP_DELIM, '/') for var in included_variables]
    )

    out_ds, rename_vars, _ = subset.open_as_nc_dataset(join(subset_output_dir, output_file))
    out_ds = xr.open_dataset(xr.backends.NetCDF4DataStore(out_ds),
                             decode_times=False,
                             decode_coords=False)

    out_vars = [out_var for out_var in out_ds.variables.keys()]

    assert set(out_vars) == set(included_variables + coordinate_variables).union(non_data_vars)
    assert set(out_vars).isdisjoint(expected_excluded_variables)

    in_ds.close()
    out_ds.close()


def test_calculate_chunks():
    """
    Test that the calculate chunks function in the subset module
    correctly calculates and returns the chunks dims dictionary.
    """
    rs = np.random.RandomState(0)
    dataset = xr.DataArray(
        rs.randn(2, 4000, 4001),
        dims=['x', 'y', 'z']
    ).to_dataset(name='foo')

    chunk_dict = subset.calculate_chunks(dataset)

    assert chunk_dict.get('x') is None
    assert chunk_dict.get('y') is None
    assert chunk_dict.get('z') == 4000


def test_missing_coord_vars(data_dir, subset_output_dir):
    """
    As of right now, the subsetter expects the data to contain lat
    and lon variables. If not present, an error is thrown.
    """
    file = 'MODIS_T-JPL-L2P-v2014.0.nc'
    ds = xr.open_dataset(join(data_dir, file),
                         decode_times=False,
                         decode_coords=False,
                         mask_and_scale=False)

    # Manually remove var which will cause error when attempting
    # to subset.
    ds = ds.drop_vars(['lat'])

    output_file = '{}_{}'.format('missing_coords', file)
    ds.to_netcdf(join(subset_output_dir, output_file))

    bbox = np.array(((-180, 180), (-90, 90)))

    with pytest.raises(ValueError):
        subset.subset(
            file_to_subset=join(subset_output_dir, output_file),
            bbox=bbox,
            output_file=''
        )


def test_data_1D(data_dir, subset_output_dir, request):
    """
    Test that subsetting a 1-D granule does not result in failure.
    """
    merged_jason_filename = 'JA1_GPN_2PeP001_002_20020115_060706_20020115_070316.nc'
    output_file = "{}_{}".format(request.node.name, merged_jason_filename)

    subset.subset(
        file_to_subset=join(data_dir, merged_jason_filename),
        bbox=np.array(((-180, 0), (-90, 0))),
        output_file=join(subset_output_dir, output_file)
    )

    xr.open_dataset(join(subset_output_dir, output_file))


def test_get_coord_variable_names(data_dir):
    """
    Test that the expected coord variable names are returned
    """
    file = 'MODIS_T-JPL-L2P-v2014.0.nc'
    ds = xr.open_dataset(join(data_dir, file),
                         decode_times=False,
                         decode_coords=False,
                         mask_and_scale=False)

    old_lat_var_name = 'lat'
    old_lon_var_name = 'lon'

    lat_var_name, lon_var_name = subset.compute_coordinate_variable_names(ds)

    assert lat_var_name[0] == old_lat_var_name
    assert lon_var_name[0] == old_lon_var_name

    new_lat_var_name = 'latitude'
    new_lon_var_name = 'x'
    ds = ds.rename({old_lat_var_name: new_lat_var_name,
                    old_lon_var_name: new_lon_var_name})

    lat_var_name, lon_var_name = subset.compute_coordinate_variable_names(ds)

    assert lat_var_name[0] == new_lat_var_name
    assert lon_var_name[0] == new_lon_var_name


def test_cannot_get_coord_variable_names(data_dir):
    """
    Test that, when given a dataset with coord vars which are not
    expected, a ValueError is raised.
    """
    file = 'MODIS_T-JPL-L2P-v2014.0.nc'
    ds = xr.open_dataset(join(data_dir, file),
                         decode_times=False,
                         decode_coords=False,
                         mask_and_scale=False)

    old_lat_var_name = 'lat'
    new_lat_var_name = 'foo'

    ds = ds.rename({old_lat_var_name: new_lat_var_name})
    # Remove 'coordinates' attribute
    for var_name, var in ds.items():
        if 'coordinates' in var.attrs:
            del var.attrs['coordinates']

    with pytest.raises(ValueError) as e_info:
        subset.compute_coordinate_variable_names(ds)


def test_get_spatial_bounds(data_dir):
    """
    Test that the get_spatial_bounds function works as expected.
    The get_spatial_bounds function should return lat/lon min/max
    which is masked and scaled for both variables. The values
    should also be adjusted for -180,180/-90,90 coordinate types
    """
    ascat_filename = 'ascat_20150702_084200_metopa_45145_eps_o_250_2300_ovw.l2.nc'
    ghrsst_filename = '20190927000500-JPL-L2P_GHRSST-SSTskin-MODIS_A-D-v02.0-fv01.0.nc'

    ascat_dataset = xr.open_dataset(
        join(data_dir, ascat_filename),
        decode_times=False,
        decode_coords=False,
        mask_and_scale=False
    )
    ghrsst_dataset = xr.open_dataset(
        join(data_dir, ghrsst_filename),
        decode_times=False,
        decode_coords=False,
        mask_and_scale=False
    )

    # ascat1 longitude is -0 360, ghrsst modis A is -180 180
    # Both have metadata for valid_min

    # Manually calculated spatial bounds
    ascat_expected_lat_min = -89.4
    ascat_expected_lat_max = 89.2
    ascat_expected_lon_min = -180.0
    ascat_expected_lon_max = 180.0

    ghrsst_expected_lat_min = -77.2
    ghrsst_expected_lat_max = -53.6
    ghrsst_expected_lon_min = -170.5
    ghrsst_expected_lon_max = -101.7

    min_lon, max_lon, min_lat, max_lat = subset.get_spatial_bounds(
        dataset=ascat_dataset,
        lat_var_names=['lat'],
        lon_var_names=['lon']
    ).flatten()

    assert np.isclose(min_lat, ascat_expected_lat_min)
    assert np.isclose(max_lat, ascat_expected_lat_max)
    assert np.isclose(min_lon, ascat_expected_lon_min)
    assert np.isclose(max_lon, ascat_expected_lon_max)

    # Remove the label from the dataset coordinate variables indicating the valid_min.
    del ascat_dataset['lat'].attrs['valid_min']
    del ascat_dataset['lon'].attrs['valid_min']

    min_lon, max_lon, min_lat, max_lat = subset.get_spatial_bounds(
        dataset=ascat_dataset,
        lat_var_names=['lat'],
        lon_var_names=['lon']
    ).flatten()

    assert np.isclose(min_lat, ascat_expected_lat_min)
    assert np.isclose(max_lat, ascat_expected_lat_max)
    assert np.isclose(min_lon, ascat_expected_lon_min)
    assert np.isclose(max_lon, ascat_expected_lon_max)

    # Repeat test, but with GHRSST granule

    min_lon, max_lon, min_lat, max_lat = subset.get_spatial_bounds(
        dataset=ghrsst_dataset,
        lat_var_names=['lat'],
        lon_var_names=['lon']
    ).flatten()

    assert np.isclose(min_lat, ghrsst_expected_lat_min)
    assert np.isclose(max_lat, ghrsst_expected_lat_max)
    assert np.isclose(min_lon, ghrsst_expected_lon_min)
    assert np.isclose(max_lon, ghrsst_expected_lon_max)

    # Remove the label from the dataset coordinate variables indicating the valid_min.

    del ghrsst_dataset['lat'].attrs['valid_min']
    del ghrsst_dataset['lon'].attrs['valid_min']

    min_lon, max_lon, min_lat, max_lat = subset.get_spatial_bounds(
        dataset=ghrsst_dataset,
        lat_var_names=['lat'],
        lon_var_names=['lon']
    ).flatten()

    assert np.isclose(min_lat, ghrsst_expected_lat_min)
    assert np.isclose(max_lat, ghrsst_expected_lat_max)
    assert np.isclose(min_lon, ghrsst_expected_lon_min)
    assert np.isclose(max_lon, ghrsst_expected_lon_max)


def test_shapefile_subset(data_dir, subset_output_dir, request):
    """
    Test that using a shapefile to subset data instead of a bbox
    works as expected
    """
    shapefile = 'test.shp'
    ascat_filename = 'ascat_20150702_084200_metopa_45145_eps_o_250_2300_ovw.l2.nc'
    output_filename = f'{request.node.name}_{ascat_filename}'

    shapefile_file_path = join(data_dir, 'test_shapefile_subset', shapefile)
    ascat_file_path = join(data_dir, ascat_filename)
    output_file_path = join(subset_output_dir, output_filename)

    subset.subset(
        file_to_subset=ascat_file_path,
        bbox=None,
        output_file=output_file_path,
        shapefile=shapefile_file_path
    )

    # Check that each point of data is within the shapefile
    shapefile_df = gpd.read_file(shapefile_file_path)
    with xr.open_dataset(output_file_path) as result_dataset:
        def in_shape(lon, lat):
            if np.isnan(lon) or np.isnan(lat):
                return
            point = Point(lon, lat)
            point_in_shapefile = shapefile_df.contains(point)
            assert point_in_shapefile[0]

        in_shape_vec = np.vectorize(in_shape)
        in_shape_vec(result_dataset.lon, result_dataset.lat)


def test_variable_subset_oco2(data_dir, subset_output_dir):
    """
    variable subsets for groups and root group using a '/'
    """

    oco2_file_name = 'oco2_LtCO2_190201_B10206Ar_200729175909s.nc4'
    output_file_name = 'oco2_test_out.nc'
    shutil.copyfile(os.path.join(data_dir, 'OCO2', oco2_file_name),
                    os.path.join(subset_output_dir, oco2_file_name))
    bbox = np.array(((-180, 180), (-90.0, 90)))
    variables = ['/xco2', '/xco2_quality_flag', '/Retrieval/water_height', '/sounding_id']
    subset.subset(
        file_to_subset=join(data_dir, 'OCO2', oco2_file_name),
        bbox=bbox,
        variables=variables,
        output_file=join(subset_output_dir, output_file_name),
    )

    out_nc = nc.Dataset(join(subset_output_dir, output_file_name))
    var_listout = list(out_nc.groups['Retrieval'].variables.keys())
    assert ('water_height' in var_listout)


def test_variable_subset_s6(data_dir, subset_output_dir):
    """
    multiple variable subset of variables in different groups in oco3
    """

    s6_file_name = 'S6A_P4_2__LR_STD__ST_002_140_20201207T011501_20201207T013023_F00.nc'
    output_file_name = 's6_test_out.nc'
    shutil.copyfile(os.path.join(data_dir, 'sentinel_6', s6_file_name),
                    os.path.join(subset_output_dir, s6_file_name))
    bbox = np.array(((-180, 180), (-90.0, 90)))
    variables = ['/data_01/ku/range_ocean_mle3_rms', '/data_20/ku/range_ocean']
    subset.subset(
        file_to_subset=join(subset_output_dir, s6_file_name),
        bbox=bbox,
        variables=variables,
        output_file=join(subset_output_dir, output_file_name),
    )

    out_nc = nc.Dataset(join(subset_output_dir, output_file_name))
    var_listout = list(out_nc.groups['data_01'].groups['ku'].variables.keys())
    var_listout.extend(list(out_nc.groups['data_20'].groups['ku'].variables.keys()))
    assert ('range_ocean_mle3_rms' in var_listout)
    assert ('range_ocean' in var_listout)


def test_transform_grouped_dataset(data_dir, subset_output_dir):
    """
    Test that the transformation function results in a correctly
    formatted dataset.
    """
    s6_file_name = 'S6A_P4_2__LR_STD__ST_002_140_20201207T011501_20201207T013023_F00.nc'
    shutil.copyfile(os.path.join(data_dir, 'sentinel_6', s6_file_name),
                    os.path.join(subset_output_dir, s6_file_name))

    nc_ds = nc.Dataset(os.path.join(data_dir, 'sentinel_6', s6_file_name))
    nc_ds_transformed = subset.transform_grouped_dataset(
        nc.Dataset(os.path.join(subset_output_dir, s6_file_name), 'r'),
        os.path.join(subset_output_dir, s6_file_name)
    )

    # The original ds has groups
    assert nc_ds.groups

    # There should be no groups in the new ds
    assert not nc_ds_transformed.groups

    # The original ds has no variables in the root group
    assert not nc_ds.variables

    # The new ds has variables in the root group
    assert nc_ds_transformed.variables

    # Each var in the new ds should map to a variable in the old ds
    for var_name, var in nc_ds_transformed.variables.items():
        path = var_name.strip('__').split('__')

        group = nc_ds[path[0]]
        for g in path[1:-1]:
            group = group[g]
        assert var_name.strip('__').split('__')[-1] in group.variables.keys()


def test_group_subset(data_dir, subset_output_dir):
    """
    Ensure a subset function can be run on a granule that contains
    groups without errors, and that the subsetted data is within
    the given spatial bounds.
    """
    s6_file_name = 'S6A_P4_2__LR_STD__ST_002_140_20201207T011501_20201207T013023_F00.nc'
    s6_output_file_name = 'SS_S6A_P4_2__LR_STD__ST_002_140_20201207T011501_20201207T013023_F00.nc'
    # Copy S6 file to temp dir
    shutil.copyfile(
        os.path.join(data_dir, 'sentinel_6', s6_file_name),
        os.path.join(subset_output_dir, s6_file_name)
    )

    # Make sure it runs without errors
    bbox = np.array(((150, 180), (-90, -50)))
    bounds = subset.subset(
        file_to_subset=os.path.join(subset_output_dir, s6_file_name),
        bbox=bbox,
        output_file=os.path.join(subset_output_dir, s6_output_file_name)
    )

    # Check that bounds are within requested bbox
    assert bounds[0][0] >= bbox[0][0]
    assert bounds[0][1] <= bbox[0][1]
    assert bounds[1][0] >= bbox[1][0]
    assert bounds[1][1] <= bbox[1][1]


def test_json_history_metadata_append(history_json_schema, data_dir, subset_output_dir, request):
    """
    Tests that the json history metadata header is appended to when it
    already exists. First we create a fake json_history header for input file.
    """
    test_file = next(filter(
        lambda f: '20180101005944-REMSS-L2P_GHRSST-SSTsubskin-AMSR2-L2B_rt_r29918-v02.0-fv01.0.nc' in f
        , TEST_DATA_FILES))
    output_file = "{}_{}".format(request.node.name, test_file)
    input_file_subset = join(subset_output_dir, "int_{}".format(output_file))

    fake_history = [
        {
            "date_time": "2021-05-10T14:30:24.553263",
            "derived_from": basename(input_file_subset),
            "program": SERVICE_NAME,
            "version": importlib_metadata.distribution(SERVICE_NAME).version,
            "parameters": "bbox=[[-180.0, 180.0], [-90.0, 90.0]] cut=True",
            "program_ref": "https://cmr.earthdata.nasa.gov:443/search/concepts/S1962070864-POCLOUD",
            "$schema": "https://harmony.earthdata.nasa.gov/schemas/history/0.1.0/history-v0.1.0.json"
        }
    ]

    in_nc = xr.open_dataset(join(data_dir, test_file))
    in_nc.attrs['history_json'] = json.dumps(fake_history)
    in_nc.to_netcdf(join(subset_output_dir, 'int_{}'.format(output_file)), 'w')

    subset.subset(
        file_to_subset=input_file_subset,
        bbox=np.array(((-180, 180), (-90.0, 90))),
        output_file=join(subset_output_dir, output_file)
    )

    out_nc = xr.open_dataset(join(subset_output_dir, output_file))

    history_json = json.loads(out_nc.attrs['history_json'])
    assert len(history_json) == 2

    validate(instance=history_json, schema=history_json_schema)

    for history in history_json:
        assert "date_time" in history
        assert history.get('program') == SERVICE_NAME
        assert history.get('derived_from') == basename(input_file_subset)
        assert history.get('version') == importlib_metadata.distribution(SERVICE_NAME).version
        assert history.get('parameters') == 'bbox=[[-180.0, 180.0], [-90.0, 90.0]] cut=True'
        assert history.get(
            'program_ref') == "https://cmr.earthdata.nasa.gov:443/search/concepts/S1962070864-POCLOUD"
        assert history.get(
            '$schema') == "https://harmony.earthdata.nasa.gov/schemas/history/0.1.0/history-v0.1.0.json"


def test_json_history_metadata_create(history_json_schema, data_dir, subset_output_dir, request):
    """
    Tests that the json history metadata header is created when it does
    not exist. All test granules does not contain this header.
    """
    test_file = next(filter(
        lambda f: '20180101005944-REMSS-L2P_GHRSST-SSTsubskin-AMSR2-L2B_rt_r29918-v02.0-fv01.0.nc' in f
        , TEST_DATA_FILES))
    output_file = "{}_{}".format(request.node.name, test_file)

    # Remove the 'history' metadata from the granule
    in_nc = xr.open_dataset(join(data_dir, test_file))
    in_nc.to_netcdf(join(subset_output_dir, 'int_{}'.format(output_file)), 'w')

    input_file_subset = join(subset_output_dir, "int_{}".format(output_file))
    subset.subset(
        file_to_subset=input_file_subset,
        bbox=np.array(((-180, 180), (-90.0, 90))),
        output_file=join(subset_output_dir, output_file)
    )

    out_nc = xr.open_dataset(join(subset_output_dir, output_file))

    history_json = json.loads(out_nc.attrs['history_json'])
    assert len(history_json) == 1

    validate(instance=history_json, schema=history_json_schema)

    for history in history_json:
        assert "date_time" in history
        assert history.get('program') == SERVICE_NAME
        assert history.get('derived_from') == basename(input_file_subset)
        assert history.get('version') == importlib_metadata.distribution(SERVICE_NAME).version
        assert history.get('parameters') == 'bbox=[[-180.0, 180.0], [-90.0, 90.0]] cut=True'
        assert history.get(
            'program_ref') == "https://cmr.earthdata.nasa.gov:443/search/concepts/S1962070864-POCLOUD"
        assert history.get(
            '$schema') == "https://harmony.earthdata.nasa.gov/schemas/history/0.1.0/history-v0.1.0.json"


def test_json_history_metadata_create_origin_source(history_json_schema, data_dir, subset_output_dir, request):
    """
    Tests that the json history metadata header is created when it does
    not exist. All test granules does not contain this header.
    """
    test_file = next(filter(
        lambda f: '20180101005944-REMSS-L2P_GHRSST-SSTsubskin-AMSR2-L2B_rt_r29918-v02.0-fv01.0.nc' in f
        , TEST_DATA_FILES))
    output_file = "{}_{}".format(request.node.name, test_file)

    # Remove the 'history' metadata from the granule
    in_nc = xr.open_dataset(join(data_dir, test_file))
    in_nc.to_netcdf(join(subset_output_dir, 'int_{}'.format(output_file)), 'w')

    input_file_subset = join(subset_output_dir, "int_{}".format(output_file))
    subset.subset(
        file_to_subset=input_file_subset,
        bbox=np.array(((-180, 180), (-90.0, 90))),
        output_file=join(subset_output_dir, output_file),
        origin_source="fake_original_file.nc"
    )

    out_nc = xr.open_dataset(join(subset_output_dir, output_file))

    history_json = json.loads(out_nc.attrs['history_json'])
    assert len(history_json) == 1

    validate(instance=history_json, schema=history_json_schema)

    for history in history_json:
        assert "date_time" in history
        assert history.get('program') == SERVICE_NAME
        assert history.get('derived_from') == "fake_original_file.nc"
        assert history.get('version') == importlib_metadata.distribution(SERVICE_NAME).version
        assert history.get('parameters') == 'bbox=[[-180.0, 180.0], [-90.0, 90.0]] cut=True'
        assert history.get(
            'program_ref') == "https://cmr.earthdata.nasa.gov:443/search/concepts/S1962070864-POCLOUD"
        assert history.get(
            '$schema') == "https://harmony.earthdata.nasa.gov/schemas/history/0.1.0/history-v0.1.0.json"


def test_temporal_subset_ascat(data_dir, subset_output_dir, request):
    """
    Test that a temporal subset results in a granule that only
    contains times within the given bounds.
    """
    bbox = np.array(((-180, 180), (-90, 90)))
    file = 'ascat_20150702_084200_metopa_45145_eps_o_250_2300_ovw.l2.nc'
    output_file = "{}_{}".format(request.node.name, file)
    min_time = '2015-07-02T09:00:00'
    max_time = '2015-07-02T10:00:00'

    subset.subset(
        file_to_subset=join(data_dir, file),
        bbox=bbox,
        output_file=join(subset_output_dir, output_file),
        min_time=min_time,
        max_time=max_time
    )

    in_ds = xr.open_dataset(join(data_dir, file),
                            decode_times=False,
                            decode_coords=False)

    out_ds = xr.open_dataset(join(subset_output_dir, output_file),
                             decode_times=False,
                             decode_coords=False)

    # Check that 'time' types match
    assert in_ds.time.dtype == out_ds.time.dtype

    in_ds.close()
    out_ds.close()

    # Check that all times are within the given bounds. Open
    # dataset using 'decode_times=True' for auto-conversions to
    # datetime
    out_ds = xr.open_dataset(join(subset_output_dir, output_file),
                             decode_coords=False)

    start_dt = subset.translate_timestamp(min_time)
    end_dt = subset.translate_timestamp(max_time)

    # All dates should be within the given temporal bounds.
    assert (out_ds.time >= pd.to_datetime(start_dt)).all()
    assert (out_ds.time <= pd.to_datetime(end_dt)).all()


def test_temporal_subset_modis_a(data_dir, subset_output_dir, request):
    """
    Test that a temporal subset results in a granule that only
    contains times within the given bounds.
    """
    bbox = np.array(((-180, 180), (-90, 90)))
    file = 'MODIS_A-JPL-L2P-v2014.0.nc'
    output_file = "{}_{}".format(request.node.name, file)
    min_time = '2019-08-05T06:57:00'
    max_time = '2019-08-05T06:58:00'
    # Actual min is 2019-08-05T06:55:01.000000000
    # Actual max is 2019-08-05T06:59:57.000000000

    subset.subset(
        file_to_subset=join(data_dir, file),
        bbox=bbox,
        output_file=join(subset_output_dir, output_file),
        min_time=min_time,
        max_time=max_time
    )

    in_ds = xr.open_dataset(join(data_dir, file),
                            decode_times=False,
                            decode_coords=False)

    out_ds = xr.open_dataset(join(subset_output_dir, output_file),
                             decode_times=False,
                             decode_coords=False)

    # Check that 'time' types match
    assert in_ds.time.dtype == out_ds.time.dtype

    in_ds.close()
    out_ds.close()

    # Check that all times are within the given bounds. Open
    # dataset using 'decode_times=True' for auto-conversions to
    # datetime
    out_ds = xr.open_dataset(join(subset_output_dir, output_file),
                             decode_coords=False)

    start_dt = subset.translate_timestamp(min_time)
    end_dt = subset.translate_timestamp(max_time)

    epoch_dt = out_ds['time'].values[0]

    # All timedelta + epoch should be within the given temporal bounds.
    assert out_ds.sst_dtime.min() + epoch_dt >= np.datetime64(start_dt)
    assert out_ds.sst_dtime.min() + epoch_dt <= np.datetime64(end_dt)


def test_temporal_subset_s6(data_dir, subset_output_dir, request):
    """
    Test that a temporal subset results in a granule that only
    contains times within the given bounds.
    """
    bbox = np.array(((-180, 180), (-90, 90)))
    file = 'S6A_P4_2__LR_STD__ST_002_140_20201207T011501_20201207T013023_F00.nc'
    # Copy S6 file to temp dir
    shutil.copyfile(
        os.path.join(data_dir, 'sentinel_6', file),
        os.path.join(subset_output_dir, file)
    )
    output_file = "{}_{}".format(request.node.name, file)
    min_time = '2020-12-07T01:20:00'
    max_time = '2020-12-07T01:25:00'
    # Actual min is 2020-12-07T01:15:01.000000000
    # Actual max is 2020-12-07T01:30:23.000000000

    subset.subset(
        file_to_subset=join(subset_output_dir, file),
        bbox=bbox,
        output_file=join(subset_output_dir, output_file),
        min_time=min_time,
        max_time=max_time
    )

    # Check that all times are within the given bounds. Open
    # dataset using 'decode_times=True' for auto-conversions to
    # datetime
    out_ds = xr.open_dataset(
        join(subset_output_dir, output_file),
        decode_coords=False,
        group='data_01'
    )

    start_dt = subset.translate_timestamp(min_time)
    end_dt = subset.translate_timestamp(max_time)

    # All dates should be within the given temporal bounds.
    assert (out_ds.time >= pd.to_datetime(start_dt)).all()
    assert (out_ds.time <= pd.to_datetime(end_dt)).all()


@pytest.mark.parametrize('test_file', TEST_DATA_FILES)
def test_get_time_variable_name(test_file, data_dir, subset_output_dir):
    args = {
        'decode_coords': False,
        'mask_and_scale': False,
        'decode_times': True
    }
    ds, rename_vars, _ = subset.open_as_nc_dataset(os.path.join(data_dir, test_file))
    ds = xr.open_dataset(xr.backends.NetCDF4DataStore(ds), **args)

    lat_var_name = subset.compute_coordinate_variable_names(ds)[0][0]
    time_var_name = subset.compute_time_variable_name(ds, ds[lat_var_name])

    assert time_var_name is not None
    assert 'time' in time_var_name


def test_subset_jason(data_dir, subset_output_dir, request):
    bbox = np.array(((-180, 0), (-90, 90)))
    file = 'JA1_GPN_2PeP001_002_20020115_060706_20020115_070316.nc'
    output_file = "{}_{}".format(request.node.name, file)
    min_time = "2002-01-15T06:07:06Z"
    max_time = "2002-01-15T06:30:16Z"

    subset.subset(
        file_to_subset=os.path.join(data_dir, file),
        bbox=bbox,
        min_time=min_time,
        max_time=max_time,
        output_file=os.path.join(subset_output_dir, output_file)
    )


@pytest.mark.parametrize('test_file', TEST_DATA_FILES)
def test_subset_size(test_file, data_dir, subset_output_dir, request):
    bbox = np.array(((-180, 0), (-30, 90)))
    output_file = "{}_{}".format(request.node.name, test_file)
    input_file_path = os.path.join(data_dir, test_file)
    output_file_path = os.path.join(subset_output_dir, output_file)

    subset.subset(
        file_to_subset=input_file_path,
        bbox=bbox,
        output_file=output_file_path
    )

    original_file_size = os.path.getsize(input_file_path)
    subset_file_size = os.path.getsize(output_file_path)

    assert subset_file_size < original_file_size


def test_duplicate_dims_sndr(data_dir, subset_output_dir, request):
    """
    Check if SNDR Climcaps files run successfully even though
    these files have variables with duplicate dimensions
    """
    SNDR_dir = join(data_dir, 'SNDR')
    sndr_file = 'SNDR.J1.CRIMSS.20210224T0100.m06.g011.L2_CLIMCAPS_RET.std.v02_28.G.210331064430.nc'

    bbox = np.array(((-180, 90), (-90, 90)))
    output_file = "{}_{}".format(request.node.name, sndr_file)
    shutil.copyfile(
        os.path.join(SNDR_dir, sndr_file),
        os.path.join(subset_output_dir, sndr_file)
    )
    box_test = subset.subset(
        file_to_subset=join(subset_output_dir, sndr_file),
        bbox=bbox,
        output_file=join(subset_output_dir, output_file),
        min_time='2021-02-24T00:50:20Z',
        max_time='2021-02-24T01:09:55Z'
    )
    # check if the box_test is

    in_nc = nc.Dataset(join(SNDR_dir, sndr_file))
    out_nc = nc.Dataset(join(subset_output_dir, output_file))

    for var_name, variable in in_nc.variables.items():
        assert in_nc[var_name].shape == out_nc[var_name].shape


def test_duplicate_dims_tropomi(data_dir, subset_output_dir, request):
    """
    Check if SNDR Climcaps files run successfully even though
    these files have variables with duplicate dimensions
    """
    TROP_dir = join(data_dir, 'tropomi')
    trop_file = 'S5P_OFFL_L2__AER_LH_20210704T005246_20210704T023416_19290_02_020200_20210708T023111.nc'

    bbox = np.array(((-180, 180), (-90, 90)))
    output_file = "{}_{}".format(request.node.name, trop_file)
    shutil.copyfile(
        os.path.join(TROP_dir, trop_file),
        os.path.join(subset_output_dir, trop_file)
    )
    box_test = subset.subset(
        file_to_subset=join(subset_output_dir, trop_file),
        bbox=bbox,
        output_file=join(subset_output_dir, output_file)
    )
    # check if the box_test is

    in_nc = nc.Dataset(join(TROP_dir, trop_file))
    out_nc = nc.Dataset(join(subset_output_dir, output_file))

    for var_name, variable in in_nc.groups['PRODUCT'].groups['SUPPORT_DATA'].groups[
        'DETAILED_RESULTS'].variables.items():
        assert variable.shape == \
               out_nc.groups['PRODUCT'].groups['SUPPORT_DATA'].groups['DETAILED_RESULTS'].variables[var_name].shape


def test_omi_novars_subset(data_dir, subset_output_dir, request):
    """
    Check that the OMI variables are conserved when no variable are specified
    the data field and lat/lon are in different groups
    """
    omi_dir = join(data_dir, 'OMI')
    omi_file = 'OMI-Aura_L2-OMSO2_2020m0116t1207-o82471_v003-2020m0223t142939.he5'

    bbox = np.array(((-180, 90), (-90, 90)))
    output_file = "{}_{}".format(request.node.name, omi_file)
    shutil.copyfile(
        os.path.join(omi_dir, omi_file),
        os.path.join(subset_output_dir, omi_file)
    )
    box_test = subset.subset(
        file_to_subset=join(subset_output_dir, omi_file),
        bbox=bbox,
        output_file=join(subset_output_dir, output_file),
    )
    # check if the box_test is

    in_nc = nc.Dataset(join(omi_dir, omi_file))
    out_nc = nc.Dataset(join(subset_output_dir, output_file))

    for var_name, variable in in_nc.groups['HDFEOS'].groups['SWATHS'].groups['OMI Total Column Amount SO2'].groups[
        'Geolocation Fields'].variables.items():
        assert in_nc.groups['HDFEOS'].groups['SWATHS'].groups['OMI Total Column Amount SO2'].groups[
                   'Geolocation Fields'].variables[var_name].shape == \
               out_nc.groups['HDFEOS'].groups['SWATHS'].groups['OMI Total Column Amount SO2'].groups[
                   'Geolocation Fields'].variables[var_name].shape


def test_root_group(data_dir, subset_output_dir):
    """test that the GROUP_DELIM string, '__', is added to variables in the root group"""

    sndr_file_name = 'SNDR.SNPP.CRIMSS.20200118T0024.m06.g005.L2_CLIMCAPS_RET.std.v02_28.G.200314032326_subset.nc'
    shutil.copyfile(os.path.join(data_dir, 'SNDR', sndr_file_name),
                    os.path.join(subset_output_dir, sndr_file_name))

    nc_dataset = nc.Dataset(os.path.join(subset_output_dir, sndr_file_name))

    args = {
        'decode_coords': False,
        'mask_and_scale': False,
        'decode_times': False
    }
    nc_dataset = subset.transform_grouped_dataset(nc_dataset, os.path.join(subset_output_dir, sndr_file_name))
    with xr.open_dataset(
            xr.backends.NetCDF4DataStore(nc_dataset),
            **args
    ) as dataset:
        var_list = list(dataset.variables)
        assert (var_list[0][0:2] == subset.GROUP_DELIM)
        group_lst = []
        for var_name in dataset.variables.keys():  # need logic if there is data in the top level not in a group
            group_lst.append('/'.join(var_name.split(subset.GROUP_DELIM)[:-1]))
        group_lst = ['/' if group == '' else group for group in group_lst]
        groups = set(group_lst)
        expected_group = {'/mw', '/ave_kern', '/', '/mol_lay', '/aux'}
        assert (groups == expected_group)


def test_get_time_squeeze(data_dir, subset_output_dir):
    """test builtin squeeze method on the lat and time variables so
    when the two have the same shape with a time and delta time in
    the tropomi product granuales the get_time_variable_name returns delta time as well"""

    tropomi_file_name = 'S5P_OFFL_L2__SO2____20200713T002730_20200713T020900_14239_01_020103_20200721T191355_subset.nc4'
    shutil.copyfile(os.path.join(data_dir, 'tropomi', tropomi_file_name),
                    os.path.join(subset_output_dir, tropomi_file_name))

    nc_dataset = nc.Dataset(os.path.join(subset_output_dir, tropomi_file_name))

    args = {
        'decode_coords': False,
        'mask_and_scale': False,
        'decode_times': False
    }
    nc_dataset = subset.transform_grouped_dataset(nc_dataset,
                                                  os.path.join(subset_output_dir, tropomi_file_name))
    with xr.open_dataset(
            xr.backends.NetCDF4DataStore(nc_dataset),
            **args
    ) as dataset:
        lat_var_name = subset.compute_coordinate_variable_names(dataset)[0][0]
        time_var_name = subset.compute_time_variable_name(dataset, dataset[lat_var_name])
        lat_dims = dataset[lat_var_name].squeeze().dims
        time_dims = dataset[time_var_name].squeeze().dims
        assert (lat_dims == time_dims)


def test_get_indexers_nd(data_dir, subset_output_dir):
    """test that the time coordinate is not included in the indexers. Also test that the dimensions are the same for
       a global box subset"""
    tropomi_file_name = 'S5P_OFFL_L2__SO2____20200713T002730_20200713T020900_14239_01_020103_20200721T191355_subset.nc4'
    shutil.copyfile(os.path.join(data_dir, 'tropomi', tropomi_file_name),
                    os.path.join(subset_output_dir, tropomi_file_name))

    nc_dataset = nc.Dataset(os.path.join(subset_output_dir, tropomi_file_name))

    args = {
        'decode_coords': False,
        'mask_and_scale': False,
        'decode_times': False
    }
    nc_dataset = subset.transform_grouped_dataset(nc_dataset,
                                                  os.path.join(subset_output_dir, tropomi_file_name))
    with xr.open_dataset(
            xr.backends.NetCDF4DataStore(nc_dataset),
            **args
    ) as dataset:
        time_var_names = []
        lat_var_name = subset.compute_coordinate_variable_names(dataset)[0][0]
        lon_var_name = subset.compute_coordinate_variable_names(dataset)[1][0]
        time_var_name = subset.compute_time_variable_name(dataset, dataset[lat_var_name])
        oper = operator.and_

        cond = oper(
            (dataset[lon_var_name] >= -180),
            (dataset[lon_var_name] <= 180)
        ) & (dataset[lat_var_name] >= -90) & (dataset[lat_var_name] <= 90) & True

        indexers = xre.get_indexers_from_nd(cond, True)
        indexed_cond = cond.isel(**indexers)
        indexed_ds = dataset.isel(**indexers)
        new_dataset = indexed_ds.where(indexed_cond)

        assert ((time_var_name not in indexers.keys()) == True)  # time can't be in the index
        assert (new_dataset.dims == dataset.dims)


def test_variable_type_string_oco2(data_dir, subset_output_dir):
    """Code passes a ceating a variable that is type object in oco2 file"""

    oco2_file_name = 'oco2_LtCO2_190201_B10206Ar_200729175909s.nc4'
    output_file_name = 'oco2_test_out.nc'
    shutil.copyfile(os.path.join(data_dir, 'OCO2', oco2_file_name),
                    os.path.join(subset_output_dir, oco2_file_name))
    bbox = np.array(((-180, 180), (-90.0, 90)))

    subset.subset(
        file_to_subset=join(data_dir, 'OCO2', oco2_file_name),
        bbox=bbox,
        output_file=join(subset_output_dir, output_file_name),
    )

    in_nc = xr.open_dataset(join(data_dir, 'OCO2', oco2_file_name))
    out_nc = xr.open_dataset(join(subset_output_dir, output_file_name))
    assert (in_nc.variables['source_files'].dtype == out_nc.variables['source_files'].dtype)


def test_transform_h5py_dataset(data_dir, subset_output_dir):
    """
    Test that the transformation function results in a correctly
    formatted dataset for h5py files
    """
    OMI_file_name = 'OMI-Aura_L2-OMSO2_2020m0116t1207-o82471_v003-2020m0223t142939.he5'
    shutil.copyfile(os.path.join(data_dir, 'OMI', OMI_file_name),
                    os.path.join(subset_output_dir, OMI_file_name))

    h5_ds = h5py.File(os.path.join(data_dir, 'OMI', OMI_file_name), 'r')

    entry_lst = []
    # Get root level objects
    key_lst = list(h5_ds.keys())

    # Go through every level of the file to fill out the remaining objects
    for entry_str in key_lst:
        # If object is a group, add it to the loop list
        if (isinstance(h5_ds[entry_str], h5py.Group)):
            for group_keys in list(h5_ds[entry_str].keys()):
                if (isinstance(h5_ds[entry_str + "/" + group_keys], h5py.Dataset)):
                    entry_lst.append(entry_str + "/" + group_keys)
                key_lst.append(entry_str + "/" + group_keys)

    nc_dataset, has_groups = subset.h5file_transform(os.path.join(subset_output_dir, OMI_file_name))

    nc_vars_flattened = list(nc_dataset.variables.keys())
    for i in range(len(entry_lst)):  # go through all the datasets in h5py file
        input_variable = '__' + entry_lst[i].replace('/', '__')
        output_variable = nc_vars_flattened[i]
        assert (input_variable == output_variable)

    nc_dataset.close()
    h5_ds.close()


def test_variable_dims_matched_tropomi(data_dir, subset_output_dir):
    """
    Code must match the dimensions for each variable rather than
    assume all dimensions in a group are the same
    """

    tropomi_file_name = 'S5P_OFFL_L2__SO2____20200713T002730_20200713T020900_14239_01_020103_20200721T191355_subset.nc4'
    output_file_name = 'tropomi_test_out.nc'
    shutil.copyfile(os.path.join(data_dir, 'tropomi', tropomi_file_name),
                    os.path.join(subset_output_dir, tropomi_file_name))

    in_nc = nc.Dataset(os.path.join(subset_output_dir, tropomi_file_name))

    # Get variable dimensions from input dataset
    in_var_dims = {
        var_name: [dim.split(subset.GROUP_DELIM)[-1] for dim in var.dimensions]
        for var_name, var in in_nc.groups['PRODUCT'].variables.items()
    }

    # Get variables from METADATA group
    in_var_dims.update(
        {
            var_name: [dim.split(subset.GROUP_DELIM)[-1] for dim in var.dimensions]
            for var_name, var in in_nc.groups['METADATA'].groups['QA_STATISTICS'].variables.items()
        }
    )
    # Include PRODUCT>SUPPORT_DATA>GEOLOCATIONS location
    in_var_dims.update(
        {
            var_name: [dim.split(subset.GROUP_DELIM)[-1] for dim in var.dimensions]
            for var_name, var in
            in_nc.groups['PRODUCT'].groups['SUPPORT_DATA'].groups['GEOLOCATIONS'].variables.items()
        }
    )

    out_nc = subset.transform_grouped_dataset(
        in_nc, os.path.join(subset_output_dir, tropomi_file_name)
    )

    # Get variable dimensions from output dataset
    out_var_dims = {
        var_name.split(subset.GROUP_DELIM)[-1]: [dim.split(subset.GROUP_DELIM)[-1] for dim in var.dimensions]
        for var_name, var in out_nc.variables.items()
    }

    TestCase().assertDictEqual(in_var_dims, out_var_dims)


def test_temporal_merged_topex(data_dir, subset_output_dir, request):
    """
    Test that a temporal subset results in a granule that only
    contains times within the given bounds.
    """
    bbox = np.array(((-180, 180), (-90, 90)))
    file = 'Merged_TOPEX_Jason_OSTM_Jason-3_Cycle_002.V4_2.nc'
    # Copy S6 file to temp dir
    shutil.copyfile(
        os.path.join(data_dir, file),
        os.path.join(subset_output_dir, file)
    )
    output_file = "{}_{}".format(request.node.name, file)
    min_time = '1992-01-01T00:00:00'
    max_time = '1992-11-01T00:00:00'
    # Actual min is 2020-12-07T01:15:01.000000000
    # Actual max is 2020-12-07T01:30:23.000000000

    subset.subset(
        file_to_subset=join(subset_output_dir, file),
        bbox=bbox,
        output_file=join(subset_output_dir, output_file),
        min_time=min_time,
        max_time=max_time
    )

    # Check that all times are within the given bounds. Open
    # dataset using 'decode_times=True' for auto-conversions to
    # datetime
    out_ds = xr.open_dataset(
        join(subset_output_dir, output_file),
        decode_coords=False
    )

    start_dt = subset.translate_timestamp(min_time)
    end_dt = subset.translate_timestamp(max_time)

    # delta time from the MJD of this data collection
    mjd_dt = np.datetime64("1992-01-01")
    start_delta_dt = np.datetime64(start_dt) - mjd_dt
    end_delta_dt = np.datetime64(end_dt) - mjd_dt

    # All dates should be within the given temporal bounds.
    assert (out_ds.time.values >= start_delta_dt).all()
    assert (out_ds.time.values <= end_delta_dt).all()


def test_get_time_epoch_var(data_dir, subset_output_dir):
    """
    Test that get_time_epoch_var method returns the 'time' variable for the tropomi CH4 granule"
    """
    bbox = np.array(((-180, 180), (-90, 90)))
    tropomi_file = 'S5P_OFFL_L2__CH4____20190319T110835_20190319T125006_07407_01_010202_20190325T125810_subset.nc4'

    shutil.copyfile(os.path.join(data_dir, 'tropomi', tropomi_file),
                    os.path.join(subset_output_dir, tropomi_file))

    nc_dataset = nc.Dataset(os.path.join(subset_output_dir, tropomi_file), mode='r')

    nc_dataset = subset.transform_grouped_dataset(nc_dataset, os.path.join(subset_output_dir, tropomi_file))

    args = {
        'decode_coords': False,
        'mask_and_scale': False,
        'decode_times': False
    }

    with xr.open_dataset(
            xr.backends.NetCDF4DataStore(nc_dataset),
            **args
    ) as dataset:
        lat_var_names, lon_var_names = subset.compute_coordinate_variable_names(dataset)
        time_var_names = [
            subset.compute_time_variable_name(
                dataset, dataset[lat_var_name]
            ) for lat_var_name in lat_var_names
        ]
        epoch_time_var = subset.get_time_epoch_var(dataset, time_var_names[0])

        assert epoch_time_var.split('__')[-1] == 'time'


def test_temporal_variable_subset(data_dir, subset_output_dir, request):
    """
    Test that both a temporal and variable subset can be executed
    on a granule, and that all of the data within that granule is
    subsetted as expected.
    """
    bbox = np.array(((-180, 180), (-90, 90)))
    file = 'ascat_20150702_084200_metopa_45145_eps_o_250_2300_ovw.l2.nc'
    output_file = "{}_{}".format(request.node.name, file)
    min_time = '2015-07-02T09:00:00'
    max_time = '2015-07-02T10:00:00'
    variables = [
        'wind_speed',
        'wind_dir'
    ]

    subset.subset(
        file_to_subset=join(data_dir, file),
        bbox=bbox,
        output_file=join(subset_output_dir, output_file),
        min_time=min_time,
        max_time=max_time,
        variables=variables
    )

    in_ds = xr.open_dataset(join(data_dir, file),
                            decode_times=False,
                            decode_coords=False)

    out_ds = xr.open_dataset(join(subset_output_dir, output_file),
                             decode_times=False,
                             decode_coords=False)

    # Check that 'time' types match
    assert in_ds.time.dtype == out_ds.time.dtype

    in_ds.close()
    out_ds.close()

    # Check that all times are within the given bounds. Open
    # dataset using 'decode_times=True' for auto-conversions to
    # datetime
    out_ds = xr.open_dataset(join(subset_output_dir, output_file),
                             decode_coords=False)

    start_dt = subset.translate_timestamp(min_time)
    end_dt = subset.translate_timestamp(max_time)

    # All dates should be within the given temporal bounds.
    assert (out_ds.time >= pd.to_datetime(start_dt)).all()
    assert (out_ds.time <= pd.to_datetime(end_dt)).all()

    # Only coordinate variables and variables requested in variable
    # subset should be present.
    assert set(np.append(['lat', 'lon', 'time'], variables)) == set(out_ds.data_vars.keys())


def test_temporal_he5file_subset(data_dir, subset_output_dir):
    """
    Test that the time type changes to datetime for subsetting
    """

    OMI_file_names = ['OMI-Aura_L2-OMSO2_2020m0116t1207-o82471_v003-2020m0223t142939.he5',
                      'OMI-Aura_L2-OMBRO_2020m0116t1207-o82471_v003-2020m0116t182003.he5']
    OMI_copy_file = 'OMI_copy_testing_2.he5'
    for i in OMI_file_names:
        shutil.copyfile(os.path.join(data_dir, 'OMI', i),
                        os.path.join(subset_output_dir, OMI_copy_file))
        min_time = '2020-01-16T12:30:00Z'
        max_time = '2020-01-16T12:40:00Z'
        bbox = np.array(((-180, 180), (-90, 90)))
        nc_dataset, has_groups = subset.h5file_transform(os.path.join(subset_output_dir, OMI_copy_file))

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
            lat_var_names, lon_var_names, time_var_names = subset.get_coordinate_variable_names(
                dataset=dataset,
                lat_var_names=None,
                lon_var_names=None,
                time_var_names=None
            )
            if 'BRO' in i:
                assert any('utc' in x.lower() for x in time_var_names)

            dataset, start_date = subset.convert_to_datetime(dataset, time_var_names)
            assert dataset[time_var_names[0]].dtype == 'datetime64[ns]'


def test_he5_timeattrs_output(data_dir, subset_output_dir, request):
    """Test that the time attributes in the output match the attributes of the input for OMI test files"""

    omi_dir = join(data_dir, 'OMI')
    omi_file = 'OMI-Aura_L2-OMBRO_2020m0116t1207-o82471_v003-2020m0116t182003.he5'
    omi_file_input = 'input' + omi_file
    bbox = np.array(((-180, 90), (-90, 90)))
    output_file = "{}_{}".format(request.node.name, omi_file)
    shutil.copyfile(
        os.path.join(omi_dir, omi_file),
        os.path.join(subset_output_dir, omi_file)
    )
    shutil.copyfile(
        os.path.join(omi_dir, omi_file),
        os.path.join(subset_output_dir, omi_file_input)
    )

    min_time = '2020-01-16T12:30:00Z'
    max_time = '2020-01-16T12:40:00Z'
    bbox = np.array(((-180, 180), (-90, 90)))
    nc_dataset_input = nc.Dataset(os.path.join(subset_output_dir, omi_file_input))
    incut_set = nc_dataset_input.groups['HDFEOS'].groups['SWATHS'].groups['OMI Total Column Amount BrO'].groups[
        'Geolocation Fields']
    xr_dataset_input = xr.open_dataset(xr.backends.NetCDF4DataStore(incut_set))
    inattrs = xr_dataset_input['Time'].attrs

    subset.subset(
        file_to_subset=os.path.join(subset_output_dir, omi_file),
        bbox=bbox,
        output_file=os.path.join(subset_output_dir, output_file),
        min_time=min_time,
        max_time=max_time
    )

    output_ncdataset = nc.Dataset(os.path.join(subset_output_dir, output_file))
    outcut_set = output_ncdataset.groups['HDFEOS'].groups['SWATHS'].groups['OMI Total Column Amount BrO'].groups[
        'Geolocation Fields']
    xrout_dataset = xr.open_dataset(xr.backends.NetCDF4DataStore(outcut_set))
    outattrs = xrout_dataset['Time'].attrs

    for key in inattrs.keys():
        if isinstance(inattrs[key], np.ndarray):
            if np.array_equal(inattrs[key], outattrs[key]):
                pass
            else:
                raise AssertionError('Attributes for {} do not equal each other'.format(key))
        else:
            assert inattrs[key] == outattrs[key]


def test_temporal_subset_lines(data_dir, subset_output_dir, request):
    bbox = np.array(((-180, 180), (-90, 90)))
    file = 'SWOT_L2_LR_SSH_Expert_368_012_20121111T235910_20121112T005015_DG10_01.nc'
    output_file = "{}_{}".format(request.node.name, file)
    min_time = '2012-11-11T23:59:10'
    max_time = '2012-11-12T00:20:10'

    subset.subset(
        file_to_subset=join(data_dir, file),
        bbox=bbox,
        output_file=join(subset_output_dir, output_file),
        min_time=min_time,
        max_time=max_time
    )

    ds = xr.open_dataset(
        join(subset_output_dir, output_file),
        decode_times=False,
        decode_coords=False
    )

    assert ds.time.dims != ds.latitude.dims


def test_grouped_empty_subset(data_dir, subset_output_dir, request):
    """
    Test that an empty subset of a grouped dataset returns 'None'
    spatial bounds.
    """
    bbox = np.array(((-10, 10), (-10, 10)))
    file = 'S6A_P4_2__LR_STD__ST_002_140_20201207T011501_20201207T013023_F00.nc'
    output_file = "{}_{}".format(request.node.name, file)

    shutil.copyfile(os.path.join(data_dir, 'sentinel_6', file),
                    os.path.join(subset_output_dir, file))

    spatial_bounds = subset.subset(
        file_to_subset=join(subset_output_dir, file),
        bbox=bbox,
        output_file=join(subset_output_dir, output_file)
    )

    assert spatial_bounds is None


def test_get_time_OMI(data_dir, subset_output_dir):
    """
    Test that code get time variables for OMI .he5 files"
    """
    omi_file = 'OMI-Aura_L2-OMSO2_2020m0116t1207-o82471_v003-2020m0223t142939.he5'

    shutil.copyfile(os.path.join(data_dir, 'OMI', omi_file),
                    os.path.join(subset_output_dir, omi_file))

    nc_dataset, has_groups = subset.h5file_transform(os.path.join(subset_output_dir, omi_file))

    args = {
        'decode_coords': False,
        'mask_and_scale': False,
        'decode_times': False
    }

    with xr.open_dataset(
            xr.backends.NetCDF4DataStore(nc_dataset),
            **args
    ) as dataset:
        time_var_names = []
        lat_var_names, lon_var_names = subset.compute_coordinate_variable_names(dataset)
        time_var_names = [
            subset.compute_time_variable_name(
                dataset, dataset[lat_var_name]
            ) for lat_var_name in lat_var_names
        ]
        assert "Time" in time_var_names[0]
        assert "Latitude" in lat_var_names[0]


def test_empty_temporal_subset(data_dir, subset_output_dir, request):
    """
    Test the edge case where a subsetted empty granule
    (due to bbox) is temporally subset, which causes the encoding
    step to fail due to size '1' data for each dimension.
    """
    #  37.707:38.484
    bbox = np.array(((37.707, 38.484), (-13.265, -12.812)))
    file = '20190927000500-JPL-L2P_GHRSST-SSTskin-MODIS_A-D-v02.0-fv01.0.nc'
    output_file = "{}_{}".format(request.node.name, file)
    min_time = '2019-09-01'
    max_time = '2019-09-30'

    subset.subset(
        file_to_subset=join(data_dir, file),
        bbox=bbox,
        output_file=join(subset_output_dir, output_file),
        min_time=min_time,
        max_time=max_time
    )

    # Check that all times are within the given bounds. Open
    # dataset using 'decode_times=True' for auto-conversions to
    # datetime
    ds = xr.open_dataset(
        join(subset_output_dir, output_file),
        decode_coords=False
    )

    assert all(dim_size == 1 for dim_size in ds.dims.values())


def test_passed_coords(data_dir, subset_output_dir):
    """
    Ensure the coordinates passed in to the subsetter are
    utilized and not manually calculated.
    """
    file = 'ascat_20150702_084200_metopa_45145_eps_o_250_2300_ovw.l2.nc'

    dataset = xr.open_dataset(join(data_dir, file),
                              decode_times=False,
                              decode_coords=False)

    dummy_lats = ['dummy_lat']
    dummy_lons = ['dummy_lon']
    dummy_times = ['dummy_time']

    actual_lats = ['lat']
    actual_lons = ['lon']
    actual_times = ['time']

    # When none are passed in, variables are computed manually
    lats, lons, times = subset.get_coordinate_variable_names(
        dataset,
        lat_var_names=None,
        lon_var_names=None,
        time_var_names=None
    )

    assert lats == actual_lats
    assert lons == actual_lons
    assert times == actual_times

    # When lats or lons are passed in, only time is computed manually
    # This case is a bit different because the lat values are used to
    # compute the time variable so we can't pass in dummy values.

    lats, lons, times = subset.get_coordinate_variable_names(
        dataset,
        lat_var_names=actual_lats,
        lon_var_names=dummy_lons,
        time_var_names=None,
    )

    assert lats == actual_lats
    assert lons == dummy_lons
    assert times == actual_times
    # When only time is passed in, lats and lons are computed manually
    lats, lons, times = subset.get_coordinate_variable_names(
        dataset,
        lat_var_names=None,
        lon_var_names=None,
        time_var_names=dummy_times
    )
    assert lats == actual_lats
    assert lons == actual_lons
    assert times == dummy_times

    # When time, lats, and lons are passed in, nothing is computed manually
    lats, lons, times = subset.get_coordinate_variable_names(
        dataset,
        lat_var_names=dummy_lats,
        lon_var_names=dummy_lons,
        time_var_names=dummy_times
    )

    assert lats == dummy_lats
    assert lons == dummy_lons
    assert times == dummy_times


def test_var_subsetting_tropomi(data_dir, subset_output_dir, request):
    """
    Check that variable subsetting is the same if a leading slash is included
    """
    trop_dir = join(data_dir, 'tropomi')
    trop_file = 'S5P_OFFL_L2__CH4____20190319T110835_20190319T125006_07407_01_010202_20190325T125810_subset.nc4'
    variable_slash = ['/PRODUCT/methane_mixing_ratio']
    variable_noslash = ['PRODUCT/methane_mixing_ratio']
    bbox = np.array(((-180, 180), (-90, 90)))
    output_file_slash = "{}_{}".format(request.node.name, trop_file)
    output_file_noslash = "{}_noslash_{}".format(request.node.name, trop_file)
    shutil.copyfile(
        os.path.join(trop_dir, trop_file),
        os.path.join(subset_output_dir, trop_file)
    )
    shutil.copyfile(
        os.path.join(trop_dir, trop_file),
        os.path.join(subset_output_dir, 'slashtest' + trop_file)
    )
    subset.subset(
        file_to_subset=join(subset_output_dir, trop_file),
        bbox=bbox,
        output_file=join(subset_output_dir, output_file_slash),
        variables=variable_slash
    )
    subset.subset(
        file_to_subset=join(subset_output_dir, 'slashtest' + trop_file),
        bbox=bbox,
        output_file=join(subset_output_dir, output_file_noslash),
        variables=variable_noslash
    )

    slash_dataset = nc.Dataset(join(subset_output_dir, output_file_slash))
    noslash_dataset = nc.Dataset(join(subset_output_dir, output_file_noslash))

    assert list(slash_dataset.groups['PRODUCT'].variables) == list(noslash_dataset.groups['PRODUCT'].variables)

def test_tropomi_utc_time(data_dir, subset_output_dir, request):
    """Verify that the time UTC values are conserved in S5P files"""
    trop_dir = join(data_dir, 'tropomi')
    trop_file = 'S5P_OFFL_L2__CH4____20190319T110835_20190319T125006_07407_01_010202_20190325T125810_subset.nc4'
    variable = ['/PRODUCT/time_utc']
    bbox = np.array(((-180, 180), (-90, 90)))
    output_file = "{}_{}".format(request.node.name, trop_file)
    shutil.copyfile(
        os.path.join(trop_dir, trop_file),
        os.path.join(subset_output_dir, trop_file)
    )
    subset.subset(
        file_to_subset=join(subset_output_dir, trop_file),
        bbox=bbox,
        output_file=join(subset_output_dir, output_file),
        variables=variable
    )

    in_nc_dataset = nc.Dataset(join(trop_dir, trop_file))
    out_nc_dataset = nc.Dataset(join(subset_output_dir, output_file))

    assert in_nc_dataset.groups['PRODUCT'].variables['time_utc'][:].squeeze()[0] ==\
                    out_nc_dataset.groups['PRODUCT'].variables['time_utc'][:].squeeze()[0]



def test_bad_time_unit(subset_output_dir):
    fill_val = -99999.0
    time_vals = np.random.rand(10)
    time_vals[0] = fill_val
    time_vals[-1] = fill_val

    data_vars = {
        'foo': (['x'], np.random.rand(10)),
        'time': (
            ['x'],
            time_vals,
            {
                'units': 'seconds since 2000-1-1 0:0:0 0',
                '_FillValue': fill_val,
                'standard_name': 'time',
                'calendar': 'standard'
            }
        ),
    }

    ds = xr.Dataset(
        data_vars=data_vars,
        coords={'x': (['x'], np.arange(10))}
    )

    nc_out_location = join(subset_output_dir, "bad_time.nc")
    ds.to_netcdf(nc_out_location)

    subset.override_decode_cf_datetime()

    ds_test = xr.open_dataset(nc_out_location)
    ds_test.close()
