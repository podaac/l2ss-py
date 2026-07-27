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
test_subset_harmony.py
==============

Test the harmony service
"""
# Standard library imports
import json
import os
import shutil
import sys
import tempfile
from datetime import datetime
from typing import Any
from unittest.mock import patch

# Third-party imports
import numpy as np
import pystac
import pytest
from harmony_service_lib.exceptions import NoDataException

# Local/Project imports
from harmony_service_lib.message import Message

from podaac.subsetter.subset_harmony import L2SubsetterService


@pytest.fixture
def temp_dir(request):
    """Create and clean up a temporary directory for tests."""
    test_data_dir = os.path.dirname(os.path.realpath(__file__))
    temp_dir = tempfile.mkdtemp(dir=test_data_dir)
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def test_env_vars(temp_dir: str) -> dict[str, str]:
    """Set up test environment variables."""
    return {
        'DATA_DIRECTORY': temp_dir,
        'STAGING_PATH': temp_dir,
        'WORKER_DIR': temp_dir,
        'STAGING_BUCKET': 'test-staging-bucket',
        'AWS_ACCESS_KEY_ID': 'test-key',
        'AWS_SECRET_ACCESS_KEY': 'test-secret',
        'AWS_REGION': 'us-west-2',
        'ENV': 'test'
    }


@pytest.fixture
def harmony_message_base() -> dict[str, Any]:
    """Create base Harmony message for tests."""
    return {
        "accessToken": "fake-token",
        "user": "test_user",
        "callback": "http://example.com/callback",
        "subset": {
            "bbox": [-91.1, -43.8, -73.0, -8.8]
        },
        "stagingLocation": "s3://test-staging-bucket/test-location",
        "format": {
            "mime": "application/x-netcdf4",
            "scaleOffsets": True,
            "srsId": 4326
        }
    }


def create_test_stac_item(test_dir: str, bbox: list[float]) -> pystac.Item:
    """Create a STAC item for testing."""
    test_granule = os.path.join(
        test_dir,
        'data',
        'JA1_GPN_2PeP001_002_20020115_060706_20020115_070316.nc'
    )

    # Ensure test data directory exists
    os.makedirs(os.path.dirname(test_granule), exist_ok=True)

    item = pystac.Item(
        id="test-granule",
        geometry={
            "type": "Polygon",
            "coordinates": [[
                [bbox[0], bbox[1]],
                [bbox[0], bbox[3]],
                [bbox[2], bbox[3]],
                [bbox[2], bbox[1]],
                [bbox[0], bbox[1]]
            ]]
        },
        bbox=bbox,
        datetime=datetime.strptime("2025-03-06T00:48:19Z", "%Y-%m-%dT%H:%M:%SZ"),
        properties={}
    )

    item.add_asset(
        "data",
        pystac.Asset(
            href=f"file://{test_granule}",
            media_type="application/x-netcdf4",
            roles=["data"]
        )
    )

    return item


def create_test_catalog(item: pystac.Item) -> pystac.Catalog:
    """Create a test STAC catalog with the given item."""
    catalog = pystac.Catalog(
        id="test-catalog",
        description="Test catalog for L2SubsetterService"
    )
    catalog.add_item(item)
    return catalog


@pytest.mark.parametrize("with_coord_vars", [True, False])
def test_service_invoke(
        mock_environ,
        temp_dir: str,
        test_env_vars: dict[str, str],
        harmony_message_base: dict[str, Any],
        with_coord_vars: bool
):
    """Test service invoke with and without coordinate variables."""
    test_dir = os.path.dirname(os.path.realpath(__file__))
    harmony_bbox = harmony_message_base["subset"]["bbox"]

    # Prepare source variables
    base_variable = {
        "id": "V0001-EXAMPLE",
        "name": "bathymetry",
        "type": "SCIENCE"
    }

    coord_variables = [] if not with_coord_vars else [
        {
            "id": "V0001-EXAMPLE",
            "name": name,
            "fullPath": f"example/group/path/ExampleVar{i + 2}",
            "type": "COORDINATE",
            "subtype": subtype
        }
        for i, (name, subtype) in enumerate([
            ("lat", "LATITUDE"),
            ("lon", "LONGITUDE"),
            ("time", "TIME")
        ])
    ]

    # Create input message
    input_json = harmony_message_base.copy()
    input_json["sources"] = [{
        "collection": "test-collection",
        "variables": [base_variable],
        "coordinateVariables": coord_variables
    }]

    # Create test setup
    item = create_test_stac_item(test_dir, harmony_bbox)
    catalog = create_test_catalog(item)
    message = Message(input_json)
    service = L2SubsetterService(message, catalog=catalog)

    # Create test arguments
    test_args = [
        "podaac.subsetter.subset_harmony",
        "--harmony-action", "invoke",
        "--harmony-input", json.dumps(input_json),
        "--harmony-metadata-dir", temp_dir,
        "--harmony-service-id", "l2ss-py"
    ]

    with patch.dict(os.environ, test_env_vars), \
            patch.object(sys, 'argv', test_args):
        try:
            # Process the item
            source = message.sources[0]
            result_item = service.process_item(item, source)

            # Verify results
            assert isinstance(result_item, pystac.Item)
            assert 'data' in result_item.assets

            # Verify bbox
            if result_item.bbox:
                np.testing.assert_almost_equal(harmony_bbox, result_item.bbox, decimal=1)

            # Verify output file
            output_asset = result_item.assets['data']
            output_path = output_asset.href.replace('file://', '')
            assert output_path.endswith('.nc')

        except Exception as e:
            pytest.fail(f"Test failed: {str(e)}")


@pytest.mark.parametrize("with_coord_vars", [True])
def test_service_invoke_pixel_subset(
        mock_environ,
        temp_dir: str,
        test_env_vars: dict[str, str],
        harmony_message_base: dict[str, Any],
        with_coord_vars: bool
):
    """Test service invoke with and without coordinate variables."""
    test_dir = os.path.dirname(os.path.realpath(__file__))
    harmony_bbox = harmony_message_base["subset"]["bbox"]
    harmony_message_base['pixelSubset'] = True

    # Prepare source variables
    base_variable = {
        "id": "V0001-EXAMPLE",
        "name": "bathymetry",
        "type": "SCIENCE"
    }

    coord_variables = [] if not with_coord_vars else [
        {
            "id": "V0001-EXAMPLE",
            "name": name,
            "fullPath": f"example/group/path/ExampleVar{i + 2}",
            "type": "COORDINATE",
            "subtype": subtype
        }
        for i, (name, subtype) in enumerate([
            ("lat", "LATITUDE"),
            ("lon", "LONGITUDE"),
            ("time", "TIME")
        ])
    ]

    # Create input message
    input_json = harmony_message_base.copy()
    input_json["sources"] = [{
        "collection": "test-collection",
        "variables": [base_variable],
        "coordinateVariables": coord_variables
    }]

    # Create test setup
    item = create_test_stac_item(test_dir, harmony_bbox)
    catalog = create_test_catalog(item)
    message = Message(input_json)
    service = L2SubsetterService(message, catalog=catalog)

    # Create test arguments
    test_args = [
        "podaac.subsetter.subset_harmony",
        "--harmony-action", "invoke",
        "--harmony-input", json.dumps(input_json),
        "--harmony-metadata-dir", temp_dir,
        "--harmony-service-id", "l2ss-py"
    ]

    with patch.dict(os.environ, test_env_vars), \
            patch.object(sys, 'argv', test_args):
        try:
            # Process the item
            source = message.sources[0]
            result_item = service.process_item(item, source)

            # Verify results
            assert isinstance(result_item, pystac.Item)
            assert 'data' in result_item.assets

            # Verify bbox
            if result_item.bbox:
                np.testing.assert_almost_equal(harmony_bbox, result_item.bbox, decimal=1)

            # Verify output file
            output_asset = result_item.assets['data']
            output_path = output_asset.href.replace('file://', '')
            assert output_path.endswith('.nc')

        except Exception as e:
            pytest.fail(f"Test failed: {str(e)}")


def test_harmony_exception_raised(mock_environ,
                                  temp_dir: str,
                                  test_env_vars: dict[str, str],
                                  harmony_message_base: dict[str, Any]):
    """Test that a HarmonyException is raised by the subset_harmony module."""

    input_json = harmony_message_base.copy()
    input_json["sources"] = [{
        "collection": "test-collection",
        "variables": [],
        "coordinateVariables": []
    }]

    # Create a minimal STAC catalog
    test_dir = os.path.dirname(os.path.realpath(__file__))
    item = create_test_stac_item(test_dir, harmony_message_base["subset"]["bbox"])
    catalog = create_test_catalog(item)
    message = Message(input_json)
    service = L2SubsetterService(message, catalog=catalog)

    # Attempt to invoke the service and verify the exception
    test_args = [
        "podaac.subsetter.subset_harmony",
        "--harmony-action", "invoke",
        "--harmony-input", json.dumps(input_json),
        "--harmony-metadata-dir", temp_dir,
        "--harmony-service-id", "l2ss-py"
    ]
    with patch.dict(os.environ, test_env_vars), patch.object(sys, 'argv', test_args), patch('podaac.subsetter.subset.subset', return_value=None):
        with pytest.raises(NoDataException):
            service.invoke()


def test_service_invoke_vertical_dimension(mock_environ, temp_dir, test_env_vars, harmony_message_base):
    """Test Harmony service vertical dimension subsetting using a STAC item."""
    import os
    from datetime import datetime

    import numpy as np
    import pystac
    import xarray as xr
    from harmony_service_lib.message import Message

    from podaac.subsetter.subset_harmony import L2SubsetterService

    # Create dummy dataset with vertical dimension
    lat = np.linspace(-90, 90, 4)
    lon = np.linspace(-180, 180, 6)
    depth = np.array([0, 10, 20, 30, 40])
    data = np.random.rand(4, 6, 5)
    data = np.broadcast_to(depth, (len(lat), len(lon), len(depth)))  # shape: (lat, lon, depth)
    lat_data = np.tile(lat[:, None], (1, 6))  # shape (4, 6)
    lon_data = np.tile(lon[None, :], (4, 1))   # shape (4, 6)
    
    ds = xr.Dataset(
        {
            "temperature": (("lat", "lon", "depth"), data),
        },
        coords={
            "lat": lat,
            "lon": lon,
            "depth": depth,
        }
    )

    nc_path = os.path.join(temp_dir, "vertical_test.nc")
    ds.to_netcdf(nc_path)

    # Create STAC item for the test file
    bbox = [-180, -90, 180, 90]
    item = pystac.Item(
        id="vertical-test",
        geometry={
            "type": "Polygon",
            "coordinates": [[
                [bbox[0], bbox[1]],
                [bbox[0], bbox[3]],
                [bbox[2], bbox[3]],
                [bbox[2], bbox[1]],
                [bbox[0], bbox[1]]
            ]]
        },
        bbox=bbox,
        datetime=datetime.strptime("2025-03-06T00:48:19Z", "%Y-%m-%dT%H:%M:%SZ"),
        properties={}
    )
    item.add_asset(
        "data",
        pystac.Asset(
            href=nc_path,  # Use plain file path, not file:// URI
            media_type="application/x-netcdf4",
            roles=["data"]
        )
    )

    # Add vertical dimension to Harmony message
    harmony_message = harmony_message_base.copy()
    harmony_message["subset"]["dimensions"] = [
        {"name": "temperature", "min": 10, "max": 30}
    ]
    # Ensure 'depth' is not included as a variable or coordinate variable
    harmony_message["sources"] = [{
        "collection": "test-collection",
        "variables": [
            {"id": "V0001-EXAMPLE", "name": "temperature", "type": "SCIENCE"},
            {"id": "V0001-EXAMPLE", "name": "lat", "type": "COORDINATE", "subtype": "LATITUDE"},
            {"id": "V0001-EXAMPLE", "name": "lon", "type": "COORDINATE", "subtype": "LONGITUDE"}
        ],
        "coordinateVariables": [
            {"id": "V0001-EXAMPLE", "name": "lat", "type": "COORDINATE", "subtype": "LATITUDE"},
            {"id": "V0001-EXAMPLE", "name": "lon", "type": "COORDINATE", "subtype": "LONGITUDE"}
        ]
    }]
    # Sanity check: depth should not be in variables or coordinateVariables
    for var in harmony_message["sources"][0]["variables"]:
        assert var["name"] != "depth", "'depth' should not be in variables list"
    for var in harmony_message["sources"][0]["coordinateVariables"]:
        assert var["name"] != "depth", "'depth' should not be in coordinateVariables list"

    harmony_message['subset']['bbox'] = [-180, -90, 180, 90]
    # Create catalog and wrap message
    catalog = pystac.Catalog(id="test-catalog", description="Test catalog for vertical dimension")
    catalog.add_item(item)
    message = Message(harmony_message)
    service = L2SubsetterService(message, catalog=catalog)

    # Patch download/stage/output filename
    import podaac.subsetter.subset_harmony as sh
    sh.download = lambda href, *args, **kwargs: href
    sh.stage = lambda output_file, staged_filename, mime, **kwargs: output_file
    sh.generate_output_filename = lambda href, ext, **ops: 'output.nc'

    # Run process_item
    source = message.sources[0]
    result_item = service.process_item(item, source)
    # Check that the bbox and geometry are set
    assert result_item.bbox is not None
    assert result_item.geometry is not None
    # Check output file exists and has correct depth subset
    output_asset = result_item.assets['data']
    output_path = output_asset.href.replace('file://', '')
    ds_out = xr.open_dataset(output_path)
    arr = ds_out['temperature'].values
    assert np.any((arr >= 10) & (arr <= 30)), "There should be values between 10 and 30"
    ds_out.close()
