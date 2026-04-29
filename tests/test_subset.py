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
"""
import gc as garbage_collection
import json
import operator
import os
import shutil
import tempfile
from os import listdir
from os.path import basename, dirname, isfile, join, realpath
from pathlib import Path

import geopandas as gpd
import h5py
import importlib_metadata
import netCDF4 as nc
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from harmony_service_lib.exceptions import NoDataException
from jsonschema import validate
from shapely.geometry import Point

from podaac.subsetter import datatree_subset, subset
from podaac.subsetter.datatree_subset import get_indexers_from_nd
from podaac.subsetter.subset import SERVICE_NAME
from podaac.subsetter.utils import coordinate_utils, file_utils, time_utils

GROUP_DELIM = '__'

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
                        "type": [ "array", "string", "object" ],
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


def assert_phony_dims(dtree: xr.DataTree, *, present: bool) -> None:
    """Simple helper func to check for phony dimes. If present = True,
    then we validate that they are present. If False, then we validate
    that they are not present.
    """
    for node in dtree.subtree:
        ds = node.ds
        if isinstance(ds, xr.Dataset) and ds.dims:
            for dim in ds.dims:
                if present:
                    assert "phony" in dim, (
                        f"Expected phony dim before subset, got: {dim}"
                    )
                else:
                    assert "phony" not in dim, (
                        f"Unexpected 'phony' dimension found: {dim}"
                    )


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
        actual_result, _ = coordinate_utils.convert_bbox(np.array([lon_range, [0, 0]]),
                                               dataset, lat_var, lon_var)

        np.testing.assert_equal(actual_result, expected_result)

def test_history_metadata_append(data_dir, subset_output_dir, request):
    """
    Tests that the history metadata header is appended to when it
    already exists.
    """
    test_file = next(f for f in TEST_DATA_FILES if '20180101005944-REMSS-L2P_GHRSST-SSTsubskin-AMSR2-L2B_rt_r29918-v02.0-fv01.0.nc' in f)

    output_file = f"{request.node.name}_{test_file}"

    subset.subset(
        file_to_subset=join(data_dir, test_file),
        bbox=np.array([[-180, 180], [-90, 90]]),
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
    output_file = f"{request.node.name}_{test_file}"

    # Remove the 'history' metadata from the granule
    in_nc = xr.open_dataset(join(data_dir, test_file))
    del in_nc.attrs['history']
    in_nc.to_netcdf(join(subset_output_dir, f'int_{output_file}'), 'w')

    subset.subset(
        file_to_subset=join(subset_output_dir, f"int_{output_file}"),
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

    chunk_dict = file_utils.calculate_chunks(dataset)

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
    output_file = f"{request.node.name}_{merged_jason_filename}"

    subset.subset(
        file_to_subset=join(data_dir, merged_jason_filename),
        bbox=np.array(((-180, 0), (-90, 0))),
        output_file=join(subset_output_dir, output_file)
    )

    xr.open_dataset(join(subset_output_dir, output_file))

def rename_variables_in_dtree(dtree: xr.DataTree, rename_dict: dict) -> xr.DataTree:
    for node in dtree.subtree:
        if node.ds is not None:
            node.ds = node.ds.rename(rename_dict)
    return dtree

def test_get_coord_variable_names(data_dir):
    """
    Test that the expected coord variable names are returned
    """
    file = 'MODIS_T-JPL-L2P-v2014.0.nc'
    ds = xr.open_datatree(join(data_dir, file),
                         decode_times=False,
                         decode_coords=False,
                         mask_and_scale=False)

    old_lat_var_name = 'lat'
    old_lon_var_name = 'lon'

    lon_var_name, lat_var_name = datatree_subset.compute_coordinate_variable_names_from_tree(ds)

    assert lat_var_name[0].strip('/') == old_lat_var_name
    assert lon_var_name[0].strip('/') == old_lon_var_name

    new_lat_var_name = 'latitude'
    new_lon_var_name = 'x'

    # Rename in all nodes
    rename_dict = {'lat': 'latitude', 'lon': 'x'}
    ds = rename_variables_in_dtree(ds, rename_dict)
    lon_var_name, lat_var_name = datatree_subset.compute_coordinate_variable_names_from_tree(ds)

    assert lat_var_name[0].strip('/') == new_lat_var_name
    assert lon_var_name[0].strip('/') == new_lon_var_name

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

    dtree = xr.DataTree(dataset=ds)

    with pytest.raises(ValueError):
        datatree_subset.compute_coordinate_variable_names_from_tree(dtree)


def test_get_spatial_bounds(data_dir):
    """
    Test that the get_spatial_bounds function works as expected.
    The get_spatial_bounds function should return lat/lon min/max
    which is masked and scaled for both variables. The values
    should also be adjusted for -180,180/-90,90 coordinate types
    """
    ascat_filename = 'ascat_20150702_084200_metopa_45145_eps_o_250_2300_ovw.l2.nc'
    ghrsst_filename = '20190927000500-JPL-L2P_GHRSST-SSTskin-MODIS_A-D-v02.0-fv01.0.nc'

    ascat_dataset = xr.open_datatree(
        join(data_dir, ascat_filename),
        decode_times=False,
        decode_coords=False,
        mask_and_scale=False
    )
    ghrsst_dataset = xr.open_datatree(
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

    min_lon, max_lon, min_lat, max_lat = datatree_subset.tree_get_spatial_bounds(
        datatree=ascat_dataset,
        lat_var_names=['lat'],
        lon_var_names=['lon']
    ).flatten()

    assert np.isclose(min_lat, ascat_expected_lat_min, atol=.1)
    assert np.isclose(max_lat, ascat_expected_lat_max, atol=.1)
    assert np.isclose(min_lon, ascat_expected_lon_min, atol=.1)
    assert np.isclose(max_lon, ascat_expected_lon_max, atol=.1)

    # Remove the label from the dataset coordinate variables indicating the valid_min.
    del ascat_dataset['lat'].attrs['valid_min']
    del ascat_dataset['lon'].attrs['valid_min']

    min_lon, max_lon, min_lat, max_lat = datatree_subset.tree_get_spatial_bounds(
        datatree=ascat_dataset,
        lat_var_names=['lat'],
        lon_var_names=['lon']
    ).flatten()

    assert np.isclose(min_lat, ascat_expected_lat_min, atol=.1)
    assert np.isclose(max_lat, ascat_expected_lat_max, atol=.1)
    assert np.isclose(min_lon, ascat_expected_lon_min, atol=.1)
    assert np.isclose(max_lon, ascat_expected_lon_max, atol=.1)

    # Repeat test, but with GHRSST granule

    min_lon, max_lon, min_lat, max_lat = datatree_subset.tree_get_spatial_bounds(
        datatree=ghrsst_dataset,
        lat_var_names=['lat'],
        lon_var_names=['lon']
    ).flatten()

    assert np.isclose(min_lat, ghrsst_expected_lat_min, atol=.1)
    assert np.isclose(max_lat, ghrsst_expected_lat_max, atol=.1)
    assert np.isclose(min_lon, ghrsst_expected_lon_min, atol=.1)
    assert np.isclose(max_lon, ghrsst_expected_lon_max, atol=.1)

    # Remove the label from the dataset coordinate variables indicating the valid_min.

    del ghrsst_dataset['lat'].attrs['valid_min']
    del ghrsst_dataset['lon'].attrs['valid_min']

    min_lon, max_lon, min_lat, max_lat = datatree_subset.tree_get_spatial_bounds(
        datatree=ghrsst_dataset,
        lat_var_names=['lat'],
        lon_var_names=['lon']
    ).flatten()

    assert np.isclose(min_lat, ghrsst_expected_lat_min, atol=.1)
    assert np.isclose(max_lat, ghrsst_expected_lat_max, atol=.1)
    assert np.isclose(min_lon, ghrsst_expected_lon_min, atol=.1)
    assert np.isclose(max_lon, ghrsst_expected_lon_max, atol=.1)


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

def test_temporal_subset_ascat(data_dir, subset_output_dir, request):
    """
    Test that a temporal subset results in a granule that only
    contains times within the given bounds.
    """
    bbox = np.array(((-180, 180), (-90, 90)))
    file = 'ascat_20150702_084200_metopa_45145_eps_o_250_2300_ovw.l2.nc'
    output_file = f"{request.node.name}_{file}"
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

    start_dt = time_utils._translate_timestamp(min_time)
    end_dt = time_utils._translate_timestamp(max_time)

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
    output_file = f"{request.node.name}_{file}"
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

    start_dt = time_utils._translate_timestamp(min_time)
    end_dt = time_utils._translate_timestamp(max_time)

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
    output_file = f"{request.node.name}_{file}"
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

    start_dt = time_utils._translate_timestamp(min_time)
    end_dt = time_utils._translate_timestamp(max_time)

    # All dates should be within the given temporal bounds.
    assert (out_ds.time >= pd.to_datetime(start_dt)).all()
    assert (out_ds.time <= pd.to_datetime(end_dt)).all()


def test_subset_jason(data_dir, subset_output_dir, request):
    """TODO: Give description to this test function."""
    bbox = np.array(((-180, 0), (-90, 90)))
    file = 'JA1_GPN_2PeP001_002_20020115_060706_20020115_070316.nc'
    output_file = f"{request.node.name}_{file}"
    min_time = "2002-01-15T06:07:06Z"
    max_time = "2002-01-15T06:30:16Z"

    subset.subset(
        file_to_subset=os.path.join(data_dir, file),
        bbox=bbox,
        min_time=min_time,
        max_time=max_time,
        output_file=os.path.join(subset_output_dir, output_file)
    )


@pytest.mark.skip(reason="Unable to open SNDR files can not delete variables or preprocess")
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
        output_file = f"{request.node.name}_{sndr_file}"
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
    output_file = f"{request.node.name}_{sndr_file}"
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
    output_file = f"{request.node.name}_{trop_file}"
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
    output_file = f"{request.node.name}_{tempo_ozone_file}"
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
    output_file = f"{request.node.name}_{tempo_no2_file}"
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


@pytest.mark.skip(reason='We no longer flatten groups but do we want to have same function copying the dims')
def test_get_time_squeeze(data_dir, subset_output_dir):
    """test builtin squeeze method on the lat and time variables so
    when the two have the same shape with a time and delta time in
    the tropomi product granuales the get_time_variable_name returns delta time as well"""

    tropomi_file_name = 'S5P_OFFL_L2__SO2____20200713T002730_20200713T020900_14239_01_020103_20200721T191355_subset.nc4'
    shutil.copyfile(os.path.join(data_dir, 'tropomi', tropomi_file_name),
                    os.path.join(subset_output_dir, tropomi_file_name))

    file = os.path.join(subset_output_dir, tropomi_file_name)
    
    args = {
        'decode_coords': False,
        'mask_and_scale': False,
        'decode_times': False
    }

    with xr.open_datatree(file, **args) as tree:
        
        lat_var_names = []
        lon_var_names = []
        time_var_names = ['/PRODUCT/time']
        lat_var_names, lon_var_names, time_var_names = coordinate_utils.get_coordinate_variable_names(
            dataset=tree,
            lat_var_names=lat_var_names,
            lon_var_names=lon_var_names,
            time_var_names=time_var_names
        )
        lat_var_name = lat_var_names[0]
        time_var_name = time_var_names[0]

        lat_dims = tree[lat_var_name].squeeze().dims
        time_dims = tree[time_var_name].squeeze().dims

        assert lat_dims == time_dims


def test_get_indexers_nd(data_dir, subset_output_dir):
    """test that the time coordinate is not included in the indexers. Also test that the dimensions are the same for
       a global box subset"""
    tropomi_file_name = 'S5P_OFFL_L2__SO2____20200713T002730_20200713T020900_14239_01_020103_20200721T191355_subset.nc4'
    shutil.copyfile(os.path.join(data_dir, 'tropomi', tropomi_file_name),
                    os.path.join(subset_output_dir, tropomi_file_name))

    file = os.path.join(subset_output_dir, tropomi_file_name)
    args = {
        'decode_coords': False,
        'mask_and_scale': False,
        'decode_times': False
    }

    with xr.open_datatree(
            file,
            **args
    ) as datatree:
        lat_var_name = datatree_subset.compute_coordinate_variable_names_from_tree(datatree)[1][0]
        lon_var_name = datatree_subset.compute_coordinate_variable_names_from_tree(datatree)[0][0]
        time_var_name = datatree_subset.compute_time_variable_name_tree(datatree, datatree[lat_var_name], [])
        oper = operator.and_

        dataset = datatree['/PRODUCT'].ds
        cond = oper(
            (datatree[lon_var_name] >= -180),
            (datatree[lon_var_name] <= 180)
        ) & (datatree[lat_var_name] >= -90) & (datatree[lat_var_name] <= 90) & True

        indexers = get_indexers_from_nd(cond, True)
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
    output_file = f"{request.node.name}_{file}"
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

    start_dt = time_utils._translate_timestamp(min_time)
    end_dt = time_utils._translate_timestamp(max_time)

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

    file = os.path.join(subset_output_dir, tropomi_file)

    args = {
        'decode_coords': False,
        'mask_and_scale': False,
        'decode_times': False
    }

    with xr.open_datatree(file, **args) as dataset:

        lat_var_names, lon_var_names, time_var_names = coordinate_utils.get_coordinate_variable_names(
            dataset=dataset
        )
        epoch_time_var = time_utils._get_time_epoch_var(dataset, time_var_names[0])
        assert epoch_time_var.split('/')[-1] == 'time'


def test_temporal_variable_subset(data_dir, subset_output_dir, request):
    """
    Test that both a temporal and variable subset can be executed
    on a granule, and that all of the data within that granule is
    subsetted as expected.
    """
    bbox = np.array(((-180, 180), (-90, 90)))
    file = 'ascat_20150702_084200_metopa_45145_eps_o_250_2300_ovw.l2.nc'
    output_file = f"{request.node.name}_{file}"
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

    start_dt = time_utils._translate_timestamp(min_time)
    end_dt = time_utils._translate_timestamp(max_time)

    # All dates should be within the given temporal bounds.
    assert (out_ds.time >= pd.to_datetime(start_dt)).all()
    assert (out_ds.time <= pd.to_datetime(end_dt)).all()

    # Only coordinate variables and variables requested in variable
    # subset should be present.
    assert set(np.append(['lat', 'lon', 'time'], variables)) == set(out_ds.data_vars.keys())

    assert set(np.append(["lat", "lon", "time"], variables)) == set(
        out_ds.data_vars.keys()
    )


@pytest.mark.parametrize("fixture_name, time_var_path", [
    (
        "fake_omi_pixcor_file",
        "/HDFEOS/SWATHS/OMI Ground Pixel Corners UV-1/Geolocation Fields/TimeUTC",
    ),
    (
        "fake_omi_bro_file",
        "/HDFEOS/SWATHS/OMI Total Column Amount BrO/Geolocation Fields/TimeUTC",
    ),
    (
        "fake_mls_aura_l2gp_oh_file",
        "/HDFEOS/SWATHS/OH/Geolocation Fields/Time",
    ),
])
def test_temporal_he5file_subset(
    fixture_name: str,
    time_var_path: str,
    fake_omi_pixcor_file: Path,
    fake_omi_bro_file: Path,
    fake_mls_aura_l2gp_oh_file: Path,
    tmp_path: Path,
):
    """
    Test that temporal subsetting correctly filters scan lines to only those
    that fall within the requested time range.
    """
    min_time = "2020-01-16T12:31:00Z"
    max_time = "2020-01-16T12:39:00Z"

    time_min = np.datetime64("2020-01-16T12:31:00", "ns")
    time_max = np.datetime64("2020-01-16T12:39:00", "ns")

    fixture_map: dict[str, Path] = {
        "fake_omi_pixcor_file": fake_omi_pixcor_file,
        "fake_omi_bro_file": fake_omi_bro_file,
        "fake_mls_aura_l2gp_oh_file": fake_mls_aura_l2gp_oh_file,
    }
    fixture_path = fixture_map[fixture_name]
    output_path = tmp_path / f"{fixture_path.stem}_temporal_subset.he5"

    subset.subset(
        file_to_subset=str(fixture_path),
        bbox=np.array(((-180, 180), (-90, 90))),
        output_file=str(output_path),
        min_time=min_time,
        max_time=max_time,
    )

    assert output_path.exists(), f"subset output file was not created at {output_path}"
    assert output_path.stat().st_size > 0, (
        f"subset output file is empty at {output_path}"
    )

    group_path, var_name = time_var_path.rsplit("/", 1)

    with xr.open_datatree(str(output_path), engine="netcdf4") as dtree:
        assert_phony_dims(dtree, present=False)

        assert group_path in dtree.groups, (
            f"expected group {group_path!r} not found in subsetted output"
        )

        ds = dtree[group_path].ds
        assert var_name in ds, (
            f"expected variable {var_name!r} not found in {group_path} after subset"
        )

        values = ds[var_name].values.ravel().astype(np.datetime64)
        assert (values >= time_min).all() and (values <= time_max).all(), (
            f"time values outside [{time_max}, {time_min}] found in "
            f"{time_var_path} after temporal subset. "
            f"min={values.min()}, max={values.max()}"
        )

def test_MLS_levels(fake_mls_aura_l2gp_oh_file: Path, tmp_path: Path):
    """
    Test that the unique groups are determined before bounding box
    subsetting
    """
    bbox = np.array(((-180, 180), (-90, 90)))

    subset_output_file = tmp_path / "mls_test_subset.nc"

    subset.subset(
        file_to_subset=fake_mls_aura_l2gp_oh_file,
        bbox=bbox,
        output_file=subset_output_file,
    )

    assert os.path.exists(subset_output_file), (
        f"subset output file was not created at {subset_output_file}"
    )
    assert os.path.getsize(subset_output_file) > 0, (
        f"subset output file is empty at {subset_output_file}"
    )

    with xr.open_datatree(fake_mls_aura_l2gp_oh_file, engine="netcdf4") as in_dt, \
         xr.open_datatree(str(subset_output_file), engine="netcdf4") as out_dt:

        in_geo  = in_dt["/HDFEOS/SWATHS/OH/Geolocation Fields"].ds
        out_geo = out_dt["/HDFEOS/SWATHS/OH/Geolocation Fields"].ds

        for var_name in in_geo.data_vars:
            assert var_name in out_geo.data_vars, (
                f"geolocation variable {var_name!r} missing from subsetted output"
            )
            assert in_geo[var_name].shape == out_geo[var_name].shape, (
                f"shape mismatch for geolocation variable {var_name!r}: "
                f"{in_geo[var_name].shape} != {out_geo[var_name].shape}"
            )

        in_data  = in_dt["/HDFEOS/SWATHS/OH/Data Fields"].ds
        out_data = out_dt["/HDFEOS/SWATHS/OH/Data Fields"].ds

        for var_name in in_data.data_vars:
            assert var_name in out_data.data_vars, (
                f"data variable {var_name!r} missing from subsetted output"
            )
            assert in_data[var_name].shape == out_data[var_name].shape, (
                f"shape mismatch for data variable {var_name!r}: "
                f"{in_data[var_name].shape} != {out_data[var_name].shape}"
            )


def test_he5_timeattrs_output(fake_omi_bro_file: Path, tmp_path: Path):
    """Test that the time attributes in the output match the attributes of the input for OMI test files"""

    output_file = tmp_path / "omi_bro_timeattrs.he5"

    min_time = "2020-01-16T12:30:00Z"
    max_time = "2020-01-16T12:40:00Z"
    bbox = np.array(((-180, 180), (-90, 90)))
    nc_dataset_input = nc.Dataset(fake_omi_bro_file)
    incut_set = (
        nc_dataset_input.groups["HDFEOS"]
        .groups["SWATHS"]
        .groups["OMI Total Column Amount BrO"]
        .groups["Geolocation Fields"]
    )
    xr_dataset_input = xr.open_dataset(xr.backends.NetCDF4DataStore(incut_set))
    inattrs = xr_dataset_input["Time"].attrs

    subset.subset(
        file_to_subset=fake_omi_bro_file,
        bbox=bbox,
        output_file=output_file,
        min_time=min_time,
        max_time=max_time,
    )

    output_ncdataset = nc.Dataset(output_file)
    outcut_set = (
        output_ncdataset.groups["HDFEOS"]
        .groups["SWATHS"]
        .groups["OMI Total Column Amount BrO"]
        .groups["Geolocation Fields"]
    )
    xrout_dataset = xr.open_dataset(xr.backends.NetCDF4DataStore(outcut_set))
    outattrs = xrout_dataset["Time"].attrs

    for key in inattrs.keys():
        if isinstance(inattrs[key], np.ndarray):
            if np.array_equal(inattrs[key], outattrs[key]):
                pass
            else:
                raise AssertionError(
                    "Attributes for {} do not equal each other".format(key)
                )
        else:
            assert inattrs[key] == outattrs[key]


def test_temporal_subset_lines(data_dir, subset_output_dir, request):
    """TODO: Give description to this test function."""
    bbox = np.array(((-180, 180), (-90, 90)))
    file = 'SWOT_L2_LR_SSH_Expert_368_012_20121111T235910_20121112T005015_DG10_01.nc'
    output_file = f"{request.node.name}_{file}"
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
    output_file = f"{request.node.name}_{file}"

    shutil.copyfile(os.path.join(data_dir, 'sentinel_6', file),
                    os.path.join(subset_output_dir, file))

    with pytest.raises(NoDataException, match="No data in subsetted granule."):
        spatial_bounds = subset.subset(
            file_to_subset=join(subset_output_dir, file),
            bbox=bbox,
            output_file=join(subset_output_dir, output_file)
        )

       #assert spatial_bounds is None


def test_get_time_OMI(fake_omi_bro_file, data_dir, subset_output_dir):
    """
    Test that the code can get time variables for OMI .he5 files"
    """

    args = {"decode_coords": False, "mask_and_scale": False, "decode_times": False}

    with xr.open_datatree(fake_omi_bro_file, **args) as dataset:
        lon_var_names, lat_var_names = (
            datatree_subset.compute_coordinate_variable_names_from_tree(dataset)
        )
        time_var_names = []
        for lat_var_name in lat_var_names:
            time_var_names.append(
                datatree_subset.compute_time_variable_name_tree(
                    dataset, dataset[lat_var_name], time_var_names
                )
            )
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
    output_file = f"{request.node.name}_{file}"
    min_time = '2019-09-01'
    max_time = '2019-09-30'

    with pytest.raises(NoDataException, match="No data in subsetted granule."):

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
        #ds = xr.open_dataset(
        #    join(subset_output_dir, output_file),
        #    decode_coords=False
        #)

        #assert all(dim_size == 1 for dim_size in ds.dims.values())


def test_passed_coords(data_dir):
    """
    Ensure the coordinates passed in to the subsetter are
    utilized and not manually calculated.
    """
    file = 'ascat_20150702_084200_metopa_45145_eps_o_250_2300_ovw.l2.nc'

    dataset = xr.open_dataset(join(data_dir, file),
                              decode_times=False,
                              decode_coords=False)

    dummy_lats = ['/dummy_lat']
    dummy_lons = ['/dummy_lon']
    dummy_times = ['/dummy_time']

    actual_lats = ['/lat']
    actual_lons = ['/lon']
    actual_times = ['/time']

    # coordinates now come with a leading / for groups
    # When none are passed in, variables are computed manually
    lats, lons, times = coordinate_utils.get_coordinate_variable_names(
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

    lats, lons, times = coordinate_utils.get_coordinate_variable_names(
        dataset,
        lat_var_names=actual_lats,
        lon_var_names=dummy_lons,
        time_var_names=None,
    )

    assert lats == actual_lats
    assert lons == dummy_lons
    assert times == actual_times
    # When only time is passed in, lats and lons are computed manually
    lats, lons, times = coordinate_utils.get_coordinate_variable_names(
        dataset,
        lat_var_names=None,
        lon_var_names=None,
        time_var_names=dummy_times
    )
    assert lats == actual_lats
    assert lons == actual_lons
    assert times == dummy_times

    # When time, lats, and lons are passed in, nothing is computed manually
    lats, lons, times = coordinate_utils.get_coordinate_variable_names(
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
    output_file_slash = f"{request.node.name}_{trop_file}"
    output_file_noslash = f"{request.node.name}_noslash_{trop_file}"
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
    output_file = f"{request.node.name}_{trop_file}"
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

    file_utils.override_decode_cf_datetime()

    ds_test = xr.open_dataset(nc_out_location)
    ds_test.close()



def test_subset_gpm_compute_new_var_data(fake_gpm_2adprenv_07_file: Path, tmp_path: Path):

    with xr.open_datatree(fake_gpm_2adprenv_07_file, engine="netcdf4") as dtree:
        assert_phony_dims(dtree, present = True)

    # Correct bbox with (min_lon, min_lat) and (max_lon, max_lat)
    bbox = np.array(((-180, 90), (-90, 90)))

    subset_output_file = tmp_path / "gpm_test_subset.HDF5"
    subset.subset(
        file_to_subset=fake_gpm_2adprenv_07_file,
        bbox=bbox,
        output_file=subset_output_file
    )

    # Open subsetted file using xarray-datatree
    dtree = xr.open_datatree(subset_output_file)

    # Check the dimensions
    assert_phony_dims(dtree, present=False)

    for group in dtree.groups:
        if "ScanTime" in group:
            assert int(dtree[group].ds.variables["timeMidScan"][:][0]) == 1306403820


def test_subset_gpm_mhs_compute_new_var_data(
    fake_gpm_2agprofmetopbmhs_08_file: Path, tmp_path: Path
):
    """
    Tests that a GPM v08 file with NetCDF4 format can run through subset and produce correct variables
    """

    subset_output_file = tmp_path / "mhs_test_subset.nc"

    # perform the subsetting operation on the fake data, make sure to
    # keep created timeMidScan var created by l2ss to compare
    subset.subset(
        file_to_subset=fake_gpm_2agprofmetopbmhs_08_file,
        bbox=np.array(((-90.1, 90.1), (-45.1, 45.1))),
        variables=[
            "S1/surfacePrecipitation",
            "S1/ScanTime/timeMidScan",
        ],
        output_file=subset_output_file,
        cut=True,
    )
    dtree = xr.open_datatree(subset_output_file)

    # are /only/ the requested variables + neccisary location data present?
    assert set(dtree["S1"].variables) == {
        "surfacePrecipitation",
        "Latitude",
        "Longitude",
    }
    assert "timeMidScan" in dtree["S1/ScanTime"].variables

    # are the requested variables dims the correct size?
    assert dtree["S1/Latitude"].shape == (6, 6)
    assert dtree["S1/Longitude"].shape == (6, 6)
    assert dtree["S1/surfacePrecipitation"].shape == (6, 6)

    # timeMidScan created, and correct?
    assert int(dtree["S1/ScanTime"].variables["timeMidScan"][0]) == 1456356716


def test_omi_novars_subset(fake_omi_bro_file: Path, tmp_path: Path):

    expected_data_vars = frozenset(
        {
            "AMFCloudFraction",
            "AMFCloudPressure",
            "AdjustedSceneAlbedo",
            "AirMassFactor",
            "AirMassFactorDiagnosticFlag",
            "AirMassFactorGeometric",
            "AverageColumnAmount",
            "AverageColumnUncertainty",
            "AverageFittingRMS",
            "ColumnAmount",
            "ColumnAmountDestriped",
            "ColumnUncertainty",
            "FitConvergenceFlag",
            "FittingRMS",
            "MainDataQualityFlag",
            "MaximumColumnAmount",
            "PixelArea",
            "PixelCornerLatitudes",
            "PixelCornerLongitudes",
            "RadianceReferenceColumnAmount",
            "RadianceReferenceColumnUncertainty",
            "RadianceReferenceColumnXTRFit",
            "RadianceReferenceConvergenceFlag",
            "RadianceReferenceFittingRMS",
            "RadianceReferenceLatitudeRange",
            "RadianceWavCalConvergenceFlag",
            "RadianceWavCalLatitudeRange",
            "SlantColumnAmount",
            "SlantColumnAmountDestriped",
            "SlantColumnUncertainty",
            "SlantFitConvergenceFlag",
            "SlantFittingRMS",
            "SolarWavCalConvergenceFlag",
        }
    )

    expected_geo_vars = frozenset(
        {
            "Latitude",
            "Longitude",
            "SolarAzimuthAngle",
            "SolarZenithAngle",
            "SpacecraftAltitude",
            "TerrainHeight",
            "Time",
            "TimeUTC",
            "ViewingAzimuthAngle",
            "ViewingZenithAngle",
            "XTrackQualityFlags",
            "XTrackQualityFlagsExpanded",
        }
    )

    with xr.open_datatree(fake_omi_bro_file, engine="netcdf4") as dtree:
        assert_phony_dims(dtree, present=True)

    bbox = np.array(((-90.1, 90.1), (-45.1, 45.1)))

    subset_output_file = tmp_path / "omibro_test_subset.hdf5"
    subset.subset(
        file_to_subset=fake_omi_bro_file,
        bbox=bbox,
        output_file=subset_output_file,
    )

    dtree = xr.open_datatree(subset_output_file)

    assert_phony_dims(dtree, present=False)

    data_ds = dtree["HDFEOS/SWATHS/OMI Total Column Amount BrO/Data Fields"].ds
    geo_ds = dtree["HDFEOS/SWATHS/OMI Total Column Amount BrO/Geolocation Fields"].ds

    missing_data_vars = expected_data_vars - set(data_ds.data_vars)
    missing_geo_vars = expected_geo_vars - set(geo_ds.data_vars)

    assert not missing_data_vars, (
        f"missing data field variables after subset: {sorted(missing_data_vars)}"
    )
    assert not missing_geo_vars, (
        f"missing geolocation field variables after subset: {sorted(missing_geo_vars)}"
    )

    assert data_ds["AMFCloudFraction"].dims == ("nTimes", "nXtrack")
    assert geo_ds["Latitude"].dims == ("nTimes", "nXtrack")
    assert geo_ds["Longitude"].dims == ("nTimes", "nXtrack")
    assert geo_ds["Time"].dims == ("nTimes",)



def test_subset_omipixcor_multi_swath(fake_omi_pixcor_file: Path, tmp_path: Path):

    with xr.open_datatree(fake_omi_pixcor_file, engine="netcdf4") as dtree:
        assert_phony_dims(dtree, present=True)

    bbox = np.array(((-3, 5), (-45.1, 45.1)))

    subset_output_file = tmp_path / "omipixcor_test_subset.he5"
    subset.subset(
        file_to_subset=fake_omi_pixcor_file,
        bbox=bbox,
        output_file=subset_output_file,
    )

    dtree = xr.open_datatree(subset_output_file)

    assert_phony_dims(dtree, present=False)

    # UV-1 and UV-2 share dim names but must retain their distinct sizes
    uv1_var = dtree["/HDFEOS/SWATHS/OMI Ground Pixel Corners UV-1/Data Fields"].ds[
        "FoV75CornerLatitude"
    ]
    uv2_var = dtree["/HDFEOS/SWATHS/OMI Ground Pixel Corners UV-2/Data Fields"].ds[
        "FoV75CornerLatitude"
    ]
    uv1_shape = uv1_var.shape
    uv2_shape = uv2_var.shape

    assert uv1_shape[-1] != uv2_shape[-1], (
        "UV-1 and UV-2 nXtrack sizes should differ but both resolved to the same size"
    )

    # they retained differences, but are the requested variables dims the correct size?
    assert uv1_shape == (2, 2, 3)
    assert uv2_shape == (2, 2, 5)


def test_temporal_subset_tempo(data_dir, subset_output_dir, request):

    tempo_file = "TEMPO_HCHO_L2_V01_20240110T170237Z_S005G08.nc"
    output_file = f"tempo_test_{tempo_file}"
    subset_output_file = join(subset_output_dir, output_file)
    bbox = np.array(((-180, 180), (-90, 90)))

    min_time = "2024-01-10T17:02:55.500000Z"
    max_time = "2024-01-10T17:03:31.900000Z"

    subset.subset(
        file_to_subset=join(data_dir, tempo_file),
        bbox=bbox,
        min_time=min_time,
        max_time=max_time,
        output_file=subset_output_file,
    )

    dtree = xr.open_datatree(subset_output_file, decode_times=False)

    assert dtree["/geolocation/time"].attrs["calendar"] == "gregorian"
    assert (
        dtree["/geolocation/time"].attrs["units"]
        == "seconds since 1980-01-06T00:00:00Z"
    )
    assert dtree["/geolocation/time"].dtype == np.float64

    assert dtree["/geolocation/time"].values[0] == 1388941375.536457
    assert dtree["/geolocation/time"].values[1] == 1388941378.569526
    assert dtree["/geolocation/time"].values[2] == 1388941381.602584
    assert dtree["/geolocation/time"].values[3] == 1388941384.635648
    assert dtree["/geolocation/time"].values[4] == 1388941387.6687133
    assert dtree["/geolocation/time"].values[5] == 1388941390.701774
    assert dtree["/geolocation/time"].values[6] == 1388941393.734837
    assert dtree["/geolocation/time"].values[7] == 1388941396.7679
    assert dtree["/geolocation/time"].values[8] == 1388941399.800967
    assert dtree["/geolocation/time"].values[9] == 1388941402.834026
    assert dtree["/geolocation/time"].values[10] == 1388941405.867095
    assert dtree["/geolocation/time"].values[11] == 1388941408.900158


# --- Constants for Expected Values ---
EXPECTED_1 = "bbox=[[-180.0, 180.0], [-90.0, 90.0]] cut=True"
EXPECTED_2 = "{'bbox': [[-180.0, 180.0], [-90.0, 90.0]], 'cut': True, 'pixel_subset': False, 'variables': []}"
EXPECTED_PARAMETERS = {
    "bbox": [[-180.0, 180.0], [-90.0, 90.0]],
    "cut": True,
    "pixel_subset": False,
    "variables": [],
}
JSON_HISTORY_TARGET_GRANULE = '20180101005944-REMSS-L2P_GHRSST-SSTsubskin-AMSR2-L2B_rt_r29918-v02.0-fv01.0.nc'

# --- Helper Functions ---
def get_json_history_test_file():
    """Helper to fetch the specific test file from the file list."""
    return next(filter(lambda f: JSON_HISTORY_TARGET_GRANULE in f, TEST_DATA_FILES))

def assert_history_metadata(history_json, schema, expected_len, expected_derived_from):
    """Encapsulates the repetitive validation and assertion logic for history payloads."""
    validate(instance=history_json, schema=schema)
    assert len(history_json) == expected_len

    for history in history_json:
        assert "date_time" in history
        assert history.get('program') == SERVICE_NAME
        assert history.get('derived_from') == expected_derived_from
        assert history.get('version') == importlib_metadata.distribution(SERVICE_NAME).version
        assert history.get('program_ref') == "https://cmr.earthdata.nasa.gov:443/search/concepts/S1962070864-POCLOUD"
        assert history.get('$schema') == "https://harmony.earthdata.nasa.gov/schemas/history/0.1.0/history-v0.1.0.json"

        params = history.get('parameters')
        if isinstance(params, str):
            assert params in (EXPECTED_1, EXPECTED_2)
        else:
            assert params == EXPECTED_PARAMETERS

def test_json_history_metadata_append(history_json_schema, data_dir, subset_output_dir, request):
    """
    Tests that the json history metadata header is appended to when it
    already exists. First we create a fake json_history header for input file.
    """
    test_file = get_json_history_test_file()
    output_file = f"{request.node.name}_{test_file}"
    input_file_subset = join(subset_output_dir, f"int_{output_file}")

    fake_history = [{
        "date_time": "2021-05-10T14:30:24.553263",
        "derived_from": basename(input_file_subset),
        "program": SERVICE_NAME,
        "version": importlib_metadata.distribution(SERVICE_NAME).version,
        "parameters": "bbox=[[-180.0, 180.0], [-90.0, 90.0]] cut=True",
        "program_ref": "https://cmr.earthdata.nasa.gov:443/search/concepts/S1962070864-POCLOUD",
        "$schema": "https://harmony.earthdata.nasa.gov/schemas/history/0.1.0/history-v0.1.0.json"
    }]

    # Use context managers for xarray to prevent unclosed file leaks
    with xr.open_dataset(join(data_dir, test_file)) as in_nc:
        in_nc.attrs['history_json'] = json.dumps(fake_history)
        in_nc.to_netcdf(input_file_subset, 'w')

    subset.subset(
        file_to_subset=input_file_subset,
        bbox=np.array(((-180, 180), (-90.0, 90))),
        output_file=join(subset_output_dir, output_file)
    )

    with xr.open_dataset(join(subset_output_dir, output_file)) as out_nc:
        history_json = json.loads(out_nc.attrs['history_json'])

    assert_history_metadata(
        history_json, 
        history_json_schema, 
        expected_len=2, 
        expected_derived_from=basename(input_file_subset)
    )


def test_json_history_metadata_create(history_json_schema, data_dir, subset_output_dir, request):
    """
    Tests that the json history metadata header is created when it does
    not exist. All test granules do not contain this header.
    """
    test_file = get_json_history_test_file()
    output_file = f"{request.node.name}_{test_file}"
    input_file_subset = join(subset_output_dir, f"int_{output_file}")

    # Open, save without history, and close safely
    with xr.open_dataset(join(data_dir, test_file)) as in_nc:
        in_nc.to_netcdf(input_file_subset, 'w')

    subset.subset(
        file_to_subset=input_file_subset,
        bbox=np.array(((-180, 180), (-90.0, 90))),
        output_file=join(subset_output_dir, output_file)
    )

    with xr.open_dataset(join(subset_output_dir, output_file)) as out_nc:
        history_json = json.loads(out_nc.attrs['history_json'])

    assert_history_metadata(
        history_json, 
        history_json_schema, 
        expected_len=1, 
        expected_derived_from=basename(input_file_subset)
    )


def test_json_history_metadata_create_origin_source(history_json_schema, data_dir, subset_output_dir, request):
    """
    Tests that the json history metadata header is created when it does
    not exist, specifically testing the origin_source injection.
    """
    test_file = get_json_history_test_file()
    output_file = f"{request.node.name}_{test_file}"
    input_file_subset = join(subset_output_dir, f"int_{output_file}")

    with xr.open_dataset(join(data_dir, test_file)) as in_nc:
        in_nc.to_netcdf(input_file_subset, 'w')

    subset.subset(
        file_to_subset=input_file_subset,
        bbox=np.array(((-180, 180), (-90.0, 90))),
        output_file=join(subset_output_dir, output_file),
        origin_source="fake_original_file.nc"
    )

    with xr.open_dataset(join(subset_output_dir, output_file)) as out_nc:
        history_json = json.loads(out_nc.attrs['history_json'])

    assert_history_metadata(
        history_json, 
        history_json_schema, 
        expected_len=1, 
        expected_derived_from="fake_original_file.nc"
    )
