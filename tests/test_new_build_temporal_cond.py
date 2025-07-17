import pytest
import numpy as np
import xarray as xr
import pandas as pd
from unittest.mock import patch, MagicMock
from podaac.subsetter.subset import build_temporal_cond


class TestNewBuildTemporalCond:
    """Test cases for build_temporal_cond function."""

    def test_float64_with_description_attribute(self):
        """
        Test the case where time_data is float64/float32 and has a description attribute.
        This should use extract_epoch and convert_time_from_description.
        """
        # Create a mock dataset with time variable that has description attribute
        time_data = xr.DataArray(
            data=np.array([0.0, 3600.0, 7200.0, 10800.0], dtype=np.float64),
            dims=['time'],
            attrs={
                'description': 'seconds since 1990-01-01T00:00:00',
                'units': 'seconds'
            }
        )
        
        dataset = xr.Dataset({
            'time_var': time_data
        })
        
        min_time = '1990-01-01T01:00:00Z'
        max_time = '1990-01-01T02:00:00Z'
        
        # Test with both min and max time
        result = build_temporal_cond(min_time, max_time, dataset, 'time_var')
        
        # The result should be a DataArray with boolean values
        assert isinstance(result, xr.DataArray)
        assert result.dtype == bool
        assert result.shape == (4,)
        
        # The first time point (0 seconds) should be False (before min_time)
        # The second time point (3600 seconds = 1 hour) should be True
        # The third time point (7200 seconds = 2 hours) should be True  
        # The fourth time point (10800 seconds = 3 hours) should be False (after max_time)
        expected = np.array([False, True, True, False])
        np.testing.assert_array_equal(result.values, expected)

    def test_float64_without_description_but_with_long_name(self):
        """
        Test the case where time_data is float64/float32, has no description,
        but has long_name "Approximate observation time for each row".
        This should use REV_START_TIME from dataset attributes.
        """
        # Create time data with seconds since start
        time_data = xr.DataArray(
            data=np.array([0.0, 60.0, 120.0, 180.0], dtype=np.float64),
            dims=['time'],
            attrs={
                'long_name': 'Approximate observation time for each row',
                'units': 'seconds'
            }
        )
        
        # Create dataset with REV_START_TIME attribute (midnight)
        dataset = xr.Dataset({
            'time_var': time_data
        })
        dataset.attrs['REV_START_TIME'] = '2020-001T00:00:00'  # Day 1 of 2020, midnight
        
        min_time = '2020-01-01T00:01:00Z'  # 1 minute after start
        max_time = '2020-01-01T00:02:00Z'  # 2 minutes after start
        
        result = build_temporal_cond(min_time, max_time, dataset, 'time_var')
        
        # The result should be a numpy array with boolean values
        assert isinstance(result, np.ndarray)
        assert result.dtype == bool
        assert result.shape == (4,)
        
        # The first time point (0 seconds) should be False (before min_time)
        # The second time point (60 seconds = 1 minute) should be True
        # The third time point (120 seconds = 2 minutes) should be True
        # The fourth time point (180 seconds = 3 minutes) should be False (after max_time)
        expected = np.array([False, True, True, False])
        np.testing.assert_array_equal(result, expected)

    def test_float32_with_description_attribute(self):
        """
        Test the case where time_data is float32 and has a description attribute.
        """
        time_data = xr.DataArray(
            data=np.array([0.0, 3600.0, 7200.0], dtype=np.float32),
            dims=['time'],
            attrs={
                'description': 'seconds since 2000-01-01T00:00:00',
                'units': 'seconds'
            }
        )
        
        dataset = xr.Dataset({
            'time_var': time_data
        })
        
        min_time = '2000-01-01T01:00:00Z'
        max_time = '2000-01-01T02:00:00Z'
        
        result = build_temporal_cond(min_time, max_time, dataset, 'time_var')
        
        assert isinstance(result, xr.DataArray)
        assert result.dtype == bool
        assert result.shape == (3,)
        
        expected = np.array([False, True, True])
        np.testing.assert_array_equal(result.values, expected)

    def test_float64_without_description_and_wrong_long_name(self):
        """
        Test the case where time_data is float64/float32, has no description,
        and has a different long_name (not "Approximate observation time for each row").
        This should not trigger the special case and should raise an error.
        """
        time_data = xr.DataArray(
            data=np.array([0.0, 3600.0, 7200.0], dtype=np.float64),
            dims=['time'],
            attrs={
                'long_name': 'Some other time variable',
                'units': 'seconds'
            }
        )
        
        dataset = xr.Dataset({
            'time_var': time_data
        })
        
        min_time = '2000-01-01T01:00:00Z'
        max_time = '2000-01-01T02:00:00Z'
        
        # This should raise a TypeError when trying to compare float with datetime
        with pytest.raises(TypeError):
            build_temporal_cond(min_time, max_time, dataset, 'time_var')

    def test_float64_with_description_and_long_name(self):
        """
        Test that description takes precedence over long_name when both are present.
        """
        time_data = xr.DataArray(
            data=np.array([0.0, 3600.0, 7200.0], dtype=np.float64),
            dims=['time'],
            attrs={
                'description': 'seconds since 1990-01-01T00:00:00',
                'long_name': 'Approximate observation time for each row',
                'units': 'seconds'
            }
        )
        
        dataset = xr.Dataset({
            'time_var': time_data
        })
        dataset.attrs['REV_START_TIME'] = '2020-001T12:00:00'
        
        min_time = '1990-01-01T01:00:00Z'
        max_time = '1990-01-01T02:00:00Z'
        
        result = build_temporal_cond(min_time, max_time, dataset, 'time_var')
        
        # Should use description (1990 epoch), not REV_START_TIME (2020 epoch)
        assert isinstance(result, xr.DataArray)
        assert result.dtype == bool
        assert result.shape == (3,)
        
        expected = np.array([False, True, True])
        np.testing.assert_array_equal(result.values, expected)

    def test_no_temporal_bounds(self):
        """
        Test that when no temporal bounds are provided, the function returns True.
        """
        time_data = xr.DataArray(
            data=np.array([0.0, 3600.0, 7200.0], dtype=np.float64),
            dims=['time'],
            attrs={
                'description': 'seconds since 1990-01-01T00:00:00',
                'units': 'seconds'
            }
        )
        
        dataset = xr.Dataset({
            'time_var': time_data
        })
        
        # No temporal bounds
        result = build_temporal_cond(None, None, dataset, 'time_var')
        
        assert result is True

    def test_only_min_time(self):
        """
        Test with only min_time provided.
        """
        time_data = xr.DataArray(
            data=np.array([0.0, 3600.0, 7200.0], dtype=np.float64),
            dims=['time'],
            attrs={
                'description': 'seconds since 1990-01-01T00:00:00',
                'units': 'seconds'
            }
        )
        
        dataset = xr.Dataset({
            'time_var': time_data
        })
        
        min_time = '1990-01-01T01:00:00Z'
        
        result = build_temporal_cond(min_time, None, dataset, 'time_var')
        
        assert isinstance(result, xr.DataArray)
        assert result.dtype == bool
        assert result.shape == (3,)
        
        # All times >= min_time should be True
        expected = np.array([False, True, True])
        np.testing.assert_array_equal(result.values, expected)

    def test_only_max_time(self):
        """
        Test with only max_time provided.
        """
        time_data = xr.DataArray(
            data=np.array([0.0, 3600.0, 7200.0], dtype=np.float64),
            dims=['time'],
            attrs={
                'description': 'seconds since 1990-01-01T00:00:00',
                'units': 'seconds'
            }
        )
        
        dataset = xr.Dataset({
            'time_var': time_data
        })
        
        max_time = '1990-01-01T01:00:00Z'
        
        result = build_temporal_cond(None, max_time, dataset, 'time_var')
        
        assert isinstance(result, xr.DataArray)
        assert result.dtype == bool
        assert result.shape == (3,)
        
        # All times <= max_time should be True
        expected = np.array([True, True, False])
        np.testing.assert_array_equal(result.values, expected)

    def test_malformed_rev_start_time(self):
        """
        Test that malformed REV_START_TIME raises appropriate error.
        """
        time_data = xr.DataArray(
            data=np.array([0.0, 60.0, 120.0], dtype=np.float64),
            dims=['time'],
            attrs={
                'long_name': 'Approximate observation time for each row',
                'units': 'seconds'
            }
        )
        
        dataset = xr.Dataset({
            'time_var': time_data
        })
        dataset.attrs['REV_START_TIME'] = 'invalid-date-format'
        
        min_time = '2020-01-01T12:01:00Z'
        max_time = '2020-01-01T12:02:00Z'
        
        # This should raise a ValueError when trying to parse the date
        with pytest.raises(ValueError):
            build_temporal_cond(min_time, max_time, dataset, 'time_var')

    def test_missing_rev_start_time(self):
        """
        Test that missing REV_START_TIME raises appropriate error.
        """
        time_data = xr.DataArray(
            data=np.array([0.0, 60.0, 120.0], dtype=np.float64),
            dims=['time'],
            attrs={
                'long_name': 'Approximate observation time for each row',
                'units': 'seconds'
            }
        )
        
        dataset = xr.Dataset({
            'time_var': time_data
        })
        # No REV_START_TIME attribute
        
        min_time = '2020-01-01T12:01:00Z'
        max_time = '2020-01-01T12:02:00Z'
        
        # This should raise an AttributeError when trying to access REV_START_TIME
        with pytest.raises(AttributeError):
            build_temporal_cond(min_time, max_time, dataset, 'time_var')

    def test_malformed_description(self):
        """
        Test that malformed description raises appropriate error.
        """
        time_data = xr.DataArray(
            data=np.array([0.0, 3600.0, 7200.0], dtype=np.float64),
            dims=['time'],
            attrs={
                'description': 'invalid description format',
                'units': 'seconds'
            }
        )
        
        dataset = xr.Dataset({
            'time_var': time_data
        })
        
        min_time = '1990-01-01T01:00:00Z'
        max_time = '1990-01-01T02:00:00Z'
        
        # This should raise a ValueError when trying to extract epoch
        with pytest.raises(ValueError):
            build_temporal_cond(min_time, max_time, dataset, 'time_var') 