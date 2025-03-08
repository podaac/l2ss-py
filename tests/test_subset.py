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
# pylint: disable=redefined-outer-name
# pylint: disable=invalid-name
# pylint: disable=consider-using-f-string

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
import urllib.parse
from os import listdir
from os.path import dirname, join, realpath, isfile, basename
from pathlib import Path
from unittest import TestCase

import geopandas as gpd
import importlib_metadata
import netCDF4 as nc
import h5py
import numpy as np
import warnings
import pandas as pd
import pytest
import xarray as xr
from jsonschema import validate
from shapely.geometry import Point
from unittest.mock import patch

from podaac.subsetter import subset
from podaac.subsetter.group_handling import GROUP_DELIM
from podaac.subsetter.subset import SERVICE_NAME
from podaac.subsetter import xarray_enhancements as xre
from podaac.subsetter import gpm_cleanup as gc
import gc as garbage_collection
from podaac.subsetter import time_converting as tc
# from podaac.subsetter import dimension_cleanup as dc


@pytest.fixture(autouse=True)
def close_all_datasets():
    """Ensure all netCDF4 and xarray datasets are closed after each test"""
    
    yield
    
    # Force garbage collection
    garbage_collection.collect()
    
    # Close netCDF4 datasets
    for obj in garbage_collection.get_objects():
        if isinstance(obj, nc.Dataset):
            try:
                obj.close()
            except:
                pass
        elif isinstance(obj, xr.Dataset):
            try:
                obj.close()
            except:
                pass


@pytest.fixture(scope='class')
def data_dir():
    """Gets the directory containing data files used for tests."""
    test_dir = dirname(realpath(__file__))
    return join(test_dir, 'data')


@pytest.fixture(scope='class')
def subset_output_dir(data_dir):
    """Makes a new temporary directory to hold the subset results while tests are running."""
    subset_output_dir = tempfile.mkdtemp(dir=data_dir)
    yield subset_output_dir
    shutil.rmtree(subset_output_dir)


@pytest.fixture(scope='class')
def history_json_schema():
    """Creates a metadata header schema."""
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
    """Get all the netCDF files from the test data directory."""
    test_dir = dirname(realpath(__file__))
    test_data_dir = join(test_dir, 'data')
    return [f for f in listdir(test_data_dir) if isfile(join(test_data_dir, f)) and f.endswith(".nc")]


TEST_DATA_FILES = data_files()


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

    ds = xr.Dataset({
        'temperature': ([], 1.0),  # Example variable
        'pressure': ([], 1000.0),   # Another example variable
    })

    # Remove 'coordinates' attribute
    for _, var in ds.items():
        if 'coordinates' in var.attrs:
            del var.attrs['coordinates']

    with pytest.raises(ValueError):
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
    assert 'water_height' in var_listout


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
    assert 'range_ocean_mle3_rms' in var_listout
    assert 'range_ocean' in var_listout


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
    for var_name, _ in nc_ds_transformed.variables.items():
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
        max_time=max_time,
        time_var_names=['sst_dtime']
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


def test_subset_jason(data_dir, subset_output_dir, request):
    """TODO: Give description to this test function."""
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


def test_cf_decode_times_sndr(data_dir, subset_output_dir, request):
    """
    Check that SNDR ascending and descending granule types are able
    to go through xarray cf_decode_times
    """
    SNDR_dir = join(data_dir, 'SNDR')
    sndr_files = ['SNDR.J1.CRIMSS.20210224T0100.m06.g011.L2_CLIMCAPS_RET.std.v02_28.G.210331064430.nc',
                  'SNDR.AQUA.AIRS.20140110T0305.m06.g031.L2_CLIMCAPS_RET.std.v02_39.G.210131015806.nc',
                  'SNDR.SNPP.CRIMSS.20200118T0024.m06.g005.L2_CLIMCAPS_RET.std.v02_28.G.200314032326_subset.nc']
    # do a longitude subset on these files that doesn't alter the resulting shape
    sndr_spatial = [(-180,-150), (-15,180), (-180,30)]
    for sndr_file, box in zip(sndr_files, sndr_spatial):
        bbox = np.array(((box[0], box[1]), (-90, 90)))
        output_file = "{}_{}".format(request.node.name, sndr_file)
        shutil.copyfile(
            os.path.join(SNDR_dir, sndr_file),
            os.path.join(subset_output_dir, sndr_file)
        )

        box_test = subset.subset(
            file_to_subset=join(subset_output_dir, sndr_file),
            bbox=bbox,
            output_file=join(subset_output_dir, output_file),
            min_time='2014-01-10T00:50:20Z',
            max_time='2021-02-24T01:09:55Z'
        )

        out_ds = xr.open_dataset(join(subset_output_dir, output_file),
                                    decode_coords=False)
        in_ds = xr.open_dataset(join(SNDR_dir, sndr_file),
                                    decode_coords=False)

        # do a longitude subset that cuts down on the file but the shape should remain the same
        assert out_ds['lon'].shape == in_ds['lon'].shape

        if not isinstance(box_test, np.ndarray):
            raise ValueError('Subset for SNDR not returned properly')

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
    _ = subset.subset(
        file_to_subset=join(subset_output_dir, sndr_file),
        bbox=bbox,
        output_file=join(subset_output_dir, output_file),
        min_time='2021-02-24T00:50:20Z',
        max_time='2021-02-24T01:09:55Z'
    )
    # check if the box_test is

    in_nc = nc.Dataset(join(SNDR_dir, sndr_file))
    out_nc = nc.Dataset(join(subset_output_dir, output_file))

    for var_name, _ in in_nc.variables.items():
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
    _ = subset.subset(
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

def test_duplicate_dims_tempo_ozone(data_dir, subset_output_dir, request):
    """
    Check if TEMPO Ozone files run successfully even though
    these files have variables with duplicate dimensions
    """
    TEMPO_dir = join(data_dir, 'TEMPO')
    tempo_ozone_file = 'TEMPO_O3PROF-PROXY_L2_V01_20130831T222959Z_S014G06.nc'

    bbox = np.array(((-180, 180), (-90, 90)))
    output_file = "{}_{}".format(request.node.name, tempo_ozone_file)
    shutil.copyfile(
        os.path.join(TEMPO_dir, tempo_ozone_file),
        os.path.join(subset_output_dir, tempo_ozone_file)
    )
    _ = subset.subset(
        file_to_subset=join(subset_output_dir, tempo_ozone_file),
        bbox=bbox,
        output_file=join(subset_output_dir, output_file)
    )
    # check if the box_test is

    in_nc = nc.Dataset(join(TEMPO_dir, tempo_ozone_file))
    out_nc = nc.Dataset(join(subset_output_dir, output_file))

    for var_name, variable in in_nc.groups['support_data'].variables.items():
        assert variable.shape == \
               out_nc.groups['support_data'].variables[var_name].shape

def test_no_null_time_values_in_time_and_space_subset_for_tempo(data_dir, subset_output_dir, request):
    """
    Check if TEMPO time variable has no null values when subsetting by time and space simultaneously.
    """
    TEMPO_dir = join(data_dir, 'TEMPO')
    tempo_no2_file = 'TEMPO_NO2_L2_V01_20231206T131913Z_S002G04.nc'

    bbox = np.array(((-87, -83), (28.5, 33.7)))
    output_file = "{}_{}".format(request.node.name, tempo_no2_file)
    shutil.copyfile(
        os.path.join(TEMPO_dir, tempo_no2_file),
        os.path.join(subset_output_dir, tempo_no2_file)
    )
    _ = subset.subset(
        file_to_subset=join(subset_output_dir, tempo_no2_file),
        min_time="2023-12-06T13:00:00",
        max_time="2023-12-06T15:00:00",
        bbox=bbox,
        output_file=join(subset_output_dir, output_file)
    )
    # check if the box_test is

    in_nc = nc.Dataset(join(TEMPO_dir, tempo_no2_file))
    out_nc = nc.Dataset(join(subset_output_dir, output_file))

    assert np.isnan(out_nc.groups['geolocation']['time'][:]).any() == False


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
    _ = subset.subset(
        file_to_subset=join(subset_output_dir, omi_file),
        bbox=bbox,
        output_file=join(subset_output_dir, output_file),
    )
    # check if the box_test is

    in_nc = nc.Dataset(join(omi_dir, omi_file))
    out_nc = nc.Dataset(join(subset_output_dir, output_file))

    for var_name, _ in in_nc.groups['HDFEOS'].groups['SWATHS'].groups['OMI Total Column Amount SO2'].groups[
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
        assert var_list[0][0:2] == subset.GROUP_DELIM
        group_lst = []
        for var_name in dataset.variables.keys():  # need logic if there is data in the top level not in a group
            group_lst.append('/'.join(var_name.split(subset.GROUP_DELIM)[:-1]))
        group_lst = ['/' if group == '' else group for group in group_lst]
        groups = set(group_lst)
        expected_group = {'/mw', '/ave_kern', '/', '/mol_lay', '/aux'}
        assert groups == expected_group


def test_get_time_squeeze(data_dir, subset_output_dir):
    """test builtin squeeze method on the lat and time variables so
    when the two have the same shape with a time and delta time in
    the tropomi product granuales the get_time_variable_name returns delta time as well"""

    tropomi_file_name = 'S5P_OFFL_L2__SO2____20200713T002730_20200713T020900_14239_01_020103_20200721T191355_subset.nc4'
    shutil.copyfile(os.path.join(data_dir, 'tropomi', tropomi_file_name),
                    os.path.join(subset_output_dir, tropomi_file_name))

    nc_dataset = nc.Dataset(os.path.join(subset_output_dir, tropomi_file_name))
    total_time_vars = ['__PRODUCT__time']

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
        time_var_name = subset.compute_time_variable_name(dataset, dataset[lat_var_name], total_time_vars)
        print(time_var_name)
        lat_dims = dataset[lat_var_name].squeeze().dims
        time_dims = dataset[time_var_name].squeeze().dims
        assert lat_dims == time_dims


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
        lat_var_name = subset.compute_coordinate_variable_names(dataset)[0][0]
        lon_var_name = subset.compute_coordinate_variable_names(dataset)[1][0]
        time_var_name = subset.compute_time_variable_name(dataset, dataset[lat_var_name], [])
        oper = operator.and_

        cond = oper(
            (dataset[lon_var_name] >= -180),
            (dataset[lon_var_name] <= 180)
        ) & (dataset[lat_var_name] >= -90) & (dataset[lat_var_name] <= 90) & True

        indexers = xre.get_indexers_from_nd(cond, True)
        indexed_cond = cond.isel(**indexers)
        indexed_ds = dataset.isel(**indexers)
        new_dataset = indexed_ds.where(indexed_cond)

        assert (time_var_name not in indexers) is True  # time can't be in the index
        assert new_dataset.dims == dataset.dims


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
    assert in_nc.variables['source_files'].dtype == out_nc.variables['source_files'].dtype


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
        if isinstance(h5_ds[entry_str], h5py.Group):
            for group_keys in list(h5_ds[entry_str].keys()):
                if isinstance(h5_ds[entry_str + "/" + group_keys], h5py.Dataset):
                    entry_lst.append(entry_str + "/" + group_keys)
                key_lst.append(entry_str + "/" + group_keys)

    nc_dataset, has_groups, hdf_type = subset.h5file_transform(os.path.join(subset_output_dir, OMI_file_name))
    assert 'OMI' == hdf_type
    nc_vars_flattened = list(nc_dataset.variables.keys())
    for i, entry in enumerate(entry_lst):  # go through all the datasets in h5py file
        input_variable = '__' + entry.replace('/', '__')
        output_variable = nc_vars_flattened[i]
        assert input_variable == output_variable

    nc_dataset.close()
    h5_ds.close()


def test_variable_dims_matched_tropomi(data_dir, subset_output_dir):
    """
    Code must match the dimensions for each variable rather than
    assume all dimensions in a group are the same
    """
    tropomi_file_name = 'S5P_OFFL_L2__SO2____20200713T002730_20200713T020900_14239_01_020103_20200721T191355_subset.nc4'

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
        lat_var_names, _ = subset.compute_coordinate_variable_names(dataset)
        time_var_names = ['__PRODUCT__time']
        for lat_var_name in lat_var_names:
            time_var_names.append(subset.compute_time_variable_name(
                    dataset, dataset[lat_var_name], time_var_names
                ))
        epoch_time_var = subset.get_time_epoch_var(dataset, time_var_names[1])

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

    OMI_file_names = [('OMI','OMI-Aura_L2-OMSO2_2020m0116t1207-o82471_v003-2020m0223t142939.he5'),
                      ('OMI','OMI-Aura_L2-OMBRO_2020m0116t1207-o82471_v003-2020m0116t182003.he5'),
                      ('MLS','MLS-Aura_L2GP-CO_v05-01-c01_2021d043.he5')]
    OMI_copy_file = 'OMI_copy_testing_2.he5'
    for i in OMI_file_names:
        shutil.copyfile(os.path.join(data_dir, i[0], i[1]),
                        os.path.join(subset_output_dir, OMI_copy_file))
        min_time = '2020-01-16T12:30:00Z'
        max_time = '2020-01-16T12:40:00Z'

        nc_dataset, has_groups, hdf_type = subset.h5file_transform(os.path.join(subset_output_dir, OMI_copy_file))
        assert has_groups == True
        assert i[0] == hdf_type
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
            _, _, time_var_names = subset.get_coordinate_variable_names(
                dataset=dataset,
                lat_var_names=None,
                lon_var_names=None,
                time_var_names=None
            )
            if 'BRO' in i:
                assert any('utc' in x.lower() for x in time_var_names)
            
            dataset, _ = tc.convert_to_datetime(dataset, time_var_names, hdf_type)
            assert dataset[time_var_names[0]].dtype == 'datetime64[ns]'


def test_omi_pixcor(data_dir, subset_output_dir, request):
    """
    OMI PIX COR collection has the same shape across groups but covers a different domain
    group to group. Dimension names had to be changed in order for copying data back into
    netCDF files. L2S developers not this collection was particularly tricky
    """
    omi_dir = join(data_dir, 'OMI')
    omi_file = 'OMI-Aura_L2-OMPIXCOR_2020m0116t1207-o82471_v003-2020m0116t174929.he5'
    omi_file_input = 'input' + omi_file
    bbox = np.array(((-180, 180), (-30, 30)))
    output_file = "{}_{}".format(request.node.name, omi_file)

    shutil.copyfile(
        os.path.join(omi_dir, omi_file),
        os.path.join(subset_output_dir, omi_file)
    )

    _ = subset.subset(
        file_to_subset=os.path.join(subset_output_dir, omi_file),
        bbox=bbox,
        output_file=os.path.join(subset_output_dir, output_file)
    )

    out_nc = nc.Dataset(join(subset_output_dir, output_file))

    assert out_nc


def test_MLS_levels(data_dir, subset_output_dir, request):
    """
    Test that the unique groups are determined before bounding box
    subsetting
    """
    mls_dir = join(data_dir, 'MLS')
    mls_file = 'MLS-Aura_L2GP-CO_v05-01-c01_2021d043.he5'
    mls_file_input = 'input' + mls_file
    bbox = np.array(((-180, 180), (-90, 90)))
    output_file = "{}_{}".format(request.node.name, mls_file)

    shutil.copyfile(
        os.path.join(mls_dir, mls_file),
        os.path.join(subset_output_dir, mls_file)
    )

    subset.subset(
        file_to_subset=os.path.join(subset_output_dir, mls_file),
        bbox=bbox,
        output_file=os.path.join(subset_output_dir, output_file)
    )

    in_ds = h5py.File(os.path.join(mls_dir, mls_file), "r")
    out_ds = h5py.File(os.path.join(subset_output_dir, output_file), "r")

    # check that the variable shapes are conserved in MLS
    for i in list(in_ds['HDFEOS']['SWATHS']['CO']['Geolocation Fields']):
        var_in_shape = in_ds['HDFEOS']['SWATHS']['CO']['Geolocation Fields'][i].shape
        var_out_shape = out_ds['HDFEOS']['SWATHS']['CO']['Geolocation Fields'][i].shape
        assert var_in_shape == var_out_shape

    for i in list(in_ds['HDFEOS']['SWATHS']['CO']['Data Fields']):
        var_in_shape = in_ds['HDFEOS']['SWATHS']['CO']['Data Fields'][i].shape
        var_out_shape = out_ds['HDFEOS']['SWATHS']['CO']['Data Fields'][i].shape
        assert var_in_shape == var_out_shape


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
    """TODO: Give description to this test function."""
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

    nc_dataset, has_groups, hdf_type = subset.h5file_transform(os.path.join(subset_output_dir, omi_file))

    args = {
        'decode_coords': False,
        'mask_and_scale': False,
        'decode_times': False
    }

    with xr.open_dataset(
            xr.backends.NetCDF4DataStore(nc_dataset),
            **args
    ) as dataset:
        lat_var_names, _ = subset.compute_coordinate_variable_names(dataset)
        time_var_names = []
        for lat_var_name in lat_var_names:
            time_var_names.append(subset.compute_time_variable_name(
                    dataset, dataset[lat_var_name], time_var_names
                ))
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


def test_passed_coords(data_dir):
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
    variable = ['/PRODUCT/time_utc', '/PRODUCT/corner']
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

    assert out_nc_dataset.groups['PRODUCT'].variables['corner']

def test_bad_time_unit(subset_output_dir):
    """TODO: give this function a description
    """
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

def test_get_unique_groups():
    """Test lat_var_names return the expected unique groups"""

    input_lats_s6 = ['__data_01__latitude', '__data_20__c__latitude', '__data_20__ku__latitude']

    unique_groups_s6, diff_counts_s6 = subset.get_base_group_names(input_lats_s6)

    expected_groups_s6 = ['__data_01', '__data_20__c', '__data_20__ku']
    expected_diff_counts_s6 = [0, 1, 1]

    assert expected_groups_s6 == unique_groups_s6
    assert expected_diff_counts_s6 == diff_counts_s6

    input_lats_mls = ['__HDF__swaths__o3__geo__latitude',
                        '__HDF__swaths__o3 columns__geo__latitude',
                        '__HDF__swaths__o3-apiori__geo__latitude']

    unique_groups_mls, diff_counts_mls = subset.get_base_group_names(input_lats_mls)

    expected_groups_mls = ['__HDF__swaths__o3',
                            '__HDF__swaths__o3 columns',
                            '__HDF__swaths__o3-apiori']
    expected_diff_counts_mls = [2, 2, 2]

    assert expected_groups_mls == unique_groups_mls
    assert expected_diff_counts_mls == diff_counts_mls

    input_lats_single = ['__latitude', '__geolocation__latitude']

    unique_groups_single, diff_counts_single = subset.get_base_group_names(input_lats_single)

    expected_groups_single = ['__', '__geolocation']
    expected_diff_counts_single = [-1, 0]
    
    assert expected_groups_single == unique_groups_single
    assert expected_diff_counts_single == diff_counts_single


def test_gpm_compute_new_var_data(data_dir, subset_output_dir, request):
    """Test GPM files that have scantime variable to compute the time for seconds
    since 1980-01-06"""
    
    gpm_dir = join(data_dir, 'GPM')
    gpm_file = 'GPM_test_file_2.HDF5'
    shutil.copyfile(
        os.path.join(gpm_dir, gpm_file),
        os.path.join(subset_output_dir, gpm_file)
    )

    nc_dataset, has_groups, file_extension = subset.open_as_nc_dataset(join(subset_output_dir, gpm_file))

    nc_dataset_new = gc.change_var_dims(nc_dataset, variables=None, time_name='__test_time')
    assert int(nc_dataset_new.variables["__FS__ScanTime__test_time"][:][0]) == 1306403820

    for var_name, var in nc_dataset.variables.items():
        dims = list(var.dimensions)
        
        for dim in dims:
            assert 'phony' not in dim
