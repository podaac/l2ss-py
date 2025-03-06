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
from typing import Dict, Any, List

# Third-party imports
import numpy as np
import pytest
import pystac
from unittest.mock import patch

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
def test_env_vars(temp_dir: str) -> Dict[str, str]:
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
def harmony_message_base() -> Dict[str, Any]:
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

def create_test_stac_item(test_dir: str, bbox: List[float]) -> pystac.Item:
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
    test_env_vars: Dict[str, str],
    harmony_message_base: Dict[str, Any],
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
            "fullPath": f"example/group/path/ExampleVar{i+2}",
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
            assert output_path.endswith('.nc4')
            
        except Exception as e:
            pytest.fail(f"Test failed: {str(e)}")

@pytest.mark.parametrize("with_coord_vars", [True])
def test_service_invoke_pixel_subset(
    mock_environ,
    temp_dir: str,
    test_env_vars: Dict[str, str],
    harmony_message_base: Dict[str, Any],
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
            "fullPath": f"example/group/path/ExampleVar{i+2}",
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
            assert output_path.endswith('.nc4')
            
        except Exception as e:
            pytest.fail(f"Test failed: {str(e)}")
