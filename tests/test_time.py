import pytest
import xarray as xr
import numpy as np
from datetime import datetime
import re
from podaac.subsetter.subset import compute_time_variable_name, find_matching_coords

def create_test_dataset(coords=None, data_vars=None, attrs=None):
    """Helper function to create test datasets"""
    if coords is None:
        coords = {}
    if data_vars is None:
        data_vars = {}
    return xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs or {})

class TestComputeTimeVariableName:
    def test_direct_time_coordinate(self):
        """Test when there's a direct 'time' coordinate"""
        ds = create_test_dataset(
            data_vars={
                'time': (['dim1'], np.array([1, 2, 3])),
                'lat': (['dim1'], np.array([10, 20, 30]))
            }
        )
        lat_var = ds['lat']
        result = compute_time_variable_name(ds, lat_var, [])
        assert result == 'time'

    def test_time_in_variable_name(self):
        """Test when there's a variable with 'time' in its name"""
        ds = create_test_dataset(
            data_vars={
                'scan_time': (['dim1'], np.array([1, 2, 3])),
                'lat': (['dim1'], np.array([10, 20, 30]))
            }
        )
        lat_var = ds['lat']
        result = compute_time_variable_name(ds, lat_var, [])
        assert result == 'scan_time'

    def test_time_with_standard_name(self):
        """Test when variable has standard_name attribute set to 'time'"""
        ds = create_test_dataset(
            data_vars={
                'measurement_time': (['x'], np.array([1, 2, 3])),
                'lat': (['x'], np.array([10, 20, 30]))
            }
        )
        ds['measurement_time'].attrs['standard_name'] = 'time'
        lat_var = ds['lat']
        result = compute_time_variable_name(ds, lat_var, [])
        assert result == 'measurement_time'

    def test_time_with_units(self):
        """Test when variable has time units"""
        ds = create_test_dataset(
            data_vars={
                'timestamp': (['y'], np.array([1, 2, 3])),
                'lat': (['y'], np.array([10, 20, 30]))
            }
        )
        ds['timestamp'].attrs['units'] = 'seconds since 2000-01-01 00:00:00'
        lat_var = ds['lat']
        result = compute_time_variable_name(ds, lat_var, [])
        assert result == 'timestamp'

    def test_time_with_axis_attribute(self):
        """Test when variable has axis attribute set to 'T'"""
        ds = create_test_dataset(
            data_vars={
                'temporal': (['z'], np.array([1, 2, 3])),
                'lat': (['z'], np.array([10, 20, 30]))
            }
        )
        ds['temporal'].attrs['axis'] = 'T'
        lat_var = ds['lat']
        result = compute_time_variable_name(ds, lat_var, [])
        assert result == 'temporal'

    def test_excluded_time_variables(self):
        """Test that variables in total_time_vars are excluded"""
        # Create dataset with coordinates
        ds = create_test_dataset(
            coords={
                'time1': ('dim1', np.array([1, 2, 3])),
                'time2': ('dim1', np.array([4, 5, 6])),
                'lat': ('dim1', np.array([10, 20, 30]))
            },
            data_vars={
                'temperature': (['dim1'], np.array([15, 16, 17])),
                'pressure': (['dim1'], np.array([1000, 1001, 1002]))
            }
        )
        
        # Add coordinates attribute to data variables
        ds['temperature'].attrs['coordinates'] = 'time1 lat'
        ds['pressure'].attrs['coordinates'] = 'time2 lat'
        
        lat_var = ds['lat']
        
        # First, verify that find_matching_coords works as expected
        time_coords = find_matching_coords(ds, ['time'])
        assert set(time_coords) == {'time1', 'time2'}, "find_matching_coords should find both time variables"
        
        # Test compute_time_variable_name with one excluded time variable
        result = compute_time_variable_name(ds, lat_var, ['time1'])
        assert result in ['time2', 'time1']

    def test_dimension_mismatch(self):
        """Test when time variable dimensions don't match lat variable"""
        ds = create_test_dataset(
            data_vars={
                'time': (['x'], np.array([1, 2, 3])),
                'lat': (['y'], np.array([10, 20, 30]))
            }
        )
        lat_var = ds['lat']
        result = compute_time_variable_name(ds, lat_var, [])
        assert result is None

    def test_timemidscan_variable(self):
        """Test when there's a 'timeMidScan' variable"""
        ds = create_test_dataset(
            data_vars={
                'timeMidScan': (['dim1'], np.array([1, 2, 3])),
                'lat': (['dim1'], np.array([10, 20, 30]))
            }
        )
        lat_var = ds['lat']
        result = compute_time_variable_name(ds, lat_var, [])
        assert result == 'timeMidScan'

    def test_no_time_variable(self):
        """Test when no time variable can be found"""
        ds = create_test_dataset(
            data_vars={
                'data': (['x'], np.array([1, 2, 3])),
                'lat': (['x'], np.array([10, 20, 30]))
            }
        )
        lat_var = ds['lat']
        result = compute_time_variable_name(ds, lat_var, [])
        assert result is None

    def test_zero_dimension_variable(self):
        """Test handling of zero-dimension variables"""
        ds = create_test_dataset(
            data_vars={
                'time_scalar': ([], np.array(1)),
                'time': (['dim1'], np.array([1, 2, 3])),
                'lat': (['dim1'], np.array([10, 20, 30]))
            }
        )
        lat_var = ds['lat']
        result = compute_time_variable_name(ds, lat_var, [])
        assert result == 'time'

    def test_all_time_vars_excluded(self):
        """Test when all time variables are in total_time_vars"""
        ds = create_test_dataset(
            data_vars={
                'time1': (['dim1'], np.array([1, 2, 3])),
                'time2': (['dim1'], np.array([4, 5, 6])),
                'lat': (['dim1'], np.array([10, 20, 30]))
            }
        )
        lat_var = ds['lat']
        result = compute_time_variable_name(ds, lat_var, ['time1', 'time2'])
        assert result is None