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
import numpy as np
import json
import os.path
import sys
from unittest.mock import patch, MagicMock

import podaac.subsetter
from podaac.subsetter.subset_harmony import L2SubsetterService
from harmony.util import config


def spy_on(method):
    """
    Creates a spy for the given object instance method which records results
    and return values while letting the call run as normal.  Calls are recorded
    on `spy_on(A.b).mock` (MagicMock) and return values are appended to the
    array `spy_on(A.b).return_values`.

    The return value should be passed as the third argument to patch.object in
    order to begin recording calls

    Parameters
    ----------
    method : function
        The method to spy on

    Returns
    -------
    function
        A wrapper function that can be passed to patch.object to record calls
    """
    mock = MagicMock()
    return_values = []

    def wrapper(self, *args, **kwargs):
        mock(*args, **kwargs)
        result = method(self, *args, **kwargs)
        return_values.append(result)
        return result
    wrapper.mock = mock
    wrapper.return_values = return_values
    return wrapper


def test_service_invoke(mock_environ):
    test_dir = os.path.dirname(os.path.realpath(__file__))
    input_json = json.load(
        open(os.path.join(test_dir, 'data', 'test_subset_harmony', 'test_service_invoke.input.json')))

    test_granule = os.path.join(test_dir, 'data', 'JA1_GPN_2PeP001_002_20020115_060706_20020115_070316.nc')
    input_json['sources'][0]['granules'][0]['url'] = f'file://{test_granule}'
    input_json['sources'][0]['variables'][0]['name'] = 'bathymetry'

    test_args = [
        podaac.subsetter.subset_harmony.__file__,
        "--harmony-action", "invoke",
        "--harmony-input", json.dumps(input_json)
    ]

    process_item_spy = spy_on(L2SubsetterService.process_item)
    with patch.object(sys, 'argv', test_args), \
         patch.object(L2SubsetterService, 'process_item', process_item_spy):
        # Mocks / spies
        podaac.subsetter.subset_harmony.main(config(False))
        expected_bbox = [-91.1, -43.8, -73.0, -8.8]

        process_item_spy.mock.assert_called_once()

        result = process_item_spy.return_values[0]

        filename = 'JA1_GPN_2PeP001_002_20020115_060706_20020115_070316_bathymetry_subsetted.nc4'
        assert result.assets['data'].to_dict() == {
            'href': 'http://example.com/public/some-org/some-service/some-uuid/' + filename,
            'type': 'application/x-netcdf4',
            'title': filename,
            'roles': ['data'],
        }
        assert result.properties['start_datetime'] == '2001-01-01T01:01:01Z'
        assert result.properties['end_datetime'] == '2002-02-02T02:02:02Z'
        np.testing.assert_almost_equal(expected_bbox, result.bbox, decimal=1)

    # When subset function returns 'None', bbox should not be passed to
    # the Harmony service lib 'async_add_local_file_partial_result'
    # function
    process_item_spy = spy_on(L2SubsetterService.process_item)
    with patch.object(sys, 'argv', test_args), \
         patch.object(L2SubsetterService, 'process_item', process_item_spy), \
         patch('podaac.subsetter.subset.subset') as mock_subsetter:

        mock_subsetter.return_value = None

        podaac.subsetter.subset_harmony.main(config(False))

        process_item_spy.mock.assert_called_once()

        result = process_item_spy.return_values[0]

        # Outputs the original input granule bounding box instead of the subset one
        assert result.bbox == [-1, -2, 3, 4]

        # Uses a filename that indicates no spatial subsetting
        filename = 'JA1_GPN_2PeP001_002_20020115_060706_20020115_070316_bathymetry.nc4'
        assert result.assets['data'].title == filename
