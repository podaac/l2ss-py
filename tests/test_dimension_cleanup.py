import pytest
import numpy as np
import netCDF4 as nc
import xarray as xr
import uuid
from podaac.subsetter import dimension_cleanup


def test_sync_dims_inplace():
    # original: variable 'a' with dims ('x',)
    orig = xr.Dataset({'a': (('x',), [1, 2, 3])})
    # new: variable 'a' with dims ('x', 'y')
    new = xr.Dataset({'a': (('x', 'y'), np.ones((3, 2)))})
    dimension_cleanup.sync_dims_inplace(orig, new)
    # After sync, 'a' in new should have shape (3,)
    assert new['a'].shape == (3,)

def test_remove_duplicate_dims_xarray():
    # Create xarray Dataset with a variable with duplicate dims
    arr = np.arange(9).reshape(3, 3)
    ds = xr.Dataset({'foo': (('x', 'x'), arr)})
    ds['x'] = ('x', [0, 1, 2])
    ds['foo'].attrs['units'] = 'test_units'
    ds['foo'].encoding['zlib'] = True
    ds2 = dimension_cleanup.remove_duplicate_dims_xarray(ds)
    # Should have renamed one of the duplicate dims
    assert ds2['foo'].dims == ('x', 'x_1') or ds2['foo'].dims == ('x', 'x_1')
    # Data and attrs preserved
    np.testing.assert_array_equal(ds2['foo'].values, arr)
    assert ds2['foo'].attrs['units'] == 'test_units'
    assert ds2['foo'].encoding.get('zlib', False) is True 

def test_remove_duplicate_dims_xarray_no_duplicates():
    ds = xr.Dataset({'foo': (('x',), [1, 2, 3])})
    ds2 = dimension_cleanup.remove_duplicate_dims_xarray(ds)
    assert ds2['foo'].dims == ('x',)
    np.testing.assert_array_equal(ds2['foo'].values, [1, 2, 3])

def test_remove_duplicate_dims_xarray_multiple_duplicates():
    arr = np.zeros((2, 2, 2))
    ds = xr.Dataset({'foo': (('x', 'x', 'x'), arr)})
    ds2 = dimension_cleanup.remove_duplicate_dims_xarray(ds)
    # Should have renamed two of the dims
    assert len(set(ds2['foo'].dims)) == 3
    np.testing.assert_array_equal(ds2['foo'].values, arr)

def test_sync_dims_inplace_no_extra_dims():
    orig = xr.Dataset({'a': (('x',), [1, 2, 3])})
    new = xr.Dataset({'a': (('x',), [1, 2, 3])})
    dimension_cleanup.sync_dims_inplace(orig, new)
    assert new['a'].shape == (3,)

def test_sync_dims_inplace_var_not_in_original():
    orig = xr.Dataset({'a': (('x',), [1, 2, 3])})
    new = xr.Dataset({'b': (('x', 'y'), np.ones((3, 2)))})
    dimension_cleanup.sync_dims_inplace(orig, new)
    assert new['b'].shape == (3, 2)
