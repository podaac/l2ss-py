import pytest
import numpy as np
import xarray as xr
import operator
from podaac.subsetter.xarray_enhancements import get_indexers_from_nd

def create_lon_lat_condition_dataset(lon_bounds, lat_bounds, lon_var_name='longitude', lat_var_name='latitude'):
    """
    Create a test dataset using xarray with longitude and latitude coordinates
    """
    # Create coordinates
    lats = np.linspace(-90, 90, 181)
    lons = np.linspace(-180, 180, 361)
    
    # Create the dataset with coordinates
    ds = xr.Dataset(
        coords={
            'lat': lats,
            'lon': lons
        }
    )
    
    # Add longitude and latitude as variables
    ds[lon_var_name] = xr.DataArray(
        data=np.tile(lons[None, :], (len(lats), 1)),
        dims=['lat', 'lon']
    )
    
    ds[lat_var_name] = xr.DataArray(
        data=np.tile(lats[:, None], (1, len(lons))),
        dims=['lat', 'lon']
    )
    
    # Determine operator based on longitude bounds
    oper = operator.and_
    if lon_bounds[0] > lon_bounds[1]:
        oper = operator.or_
    
    # Create condition based on bounds
    if oper == operator.and_:
        condition = (
            (ds[lon_var_name] >= lon_bounds[0]) & 
            (ds[lon_var_name] <= lon_bounds[1]) &
            (ds[lat_var_name] >= lat_bounds[0]) & 
            (ds[lat_var_name] <= lat_bounds[1])
        )
    else:
        condition = (
            ((ds[lon_var_name] >= lon_bounds[0]) | 
             (ds[lon_var_name] <= lon_bounds[1])) &
            (ds[lat_var_name] >= lat_bounds[0]) & 
            (ds[lat_var_name] <= lat_bounds[1])
        )
    
    return condition

def test_standard_longitude_bounds():
    """
    Test standard longitude bounds (no antimeridian crossing)
    """
    lon_bounds = (30, 60)
    lat_bounds = (-30, 30)
    
    cond = create_lon_lat_condition_dataset(lon_bounds, lat_bounds)
    indexers = get_indexers_from_nd(cond, cut=True)
    
    assert 'lat' in indexers
    assert 'lon' in indexers
    assert len(indexers['lat']) > 0
    assert len(indexers['lon']) > 0

def test_antimeridian_crossing():
    """
    Test longitude bounds crossing the antimeridian
    """
    lon_bounds = (170, -170)
    lat_bounds = (-30, 30)
    
    cond = create_lon_lat_condition_dataset(lon_bounds, lat_bounds)
    indexers = get_indexers_from_nd(cond, cut=True)
    
    assert 'lat' in indexers
    assert 'lon' in indexers
    assert len(indexers['lat']) > 0
    assert len(indexers['lon']) > 0

def test_full_latitude_range():
    """
    Test full latitude range
    """
    lon_bounds = (0, 60)
    lat_bounds = (-90, 90)
    
    cond = create_lon_lat_condition_dataset(lon_bounds, lat_bounds)
    indexers = get_indexers_from_nd(cond, cut=True)
    
    assert 'lat' in indexers
    assert len(indexers['lat']) == 181  # Full latitude range

def test_no_data_region():
    """
    Test region with no data
    """
    lon_bounds = (190, 200)  # Outside our longitude range
    lat_bounds = (91, 92)    # Outside our latitude range
    
    cond = create_lon_lat_condition_dataset(lon_bounds, lat_bounds)
    indexers = get_indexers_from_nd(cond, cut=True)
    
    assert 'lat' in indexers
    assert 'lon' in indexers
    assert len(indexers['lat']) == 0
    assert len(indexers['lon']) == 0

def test_custom_variable_names():
    """
    Test with custom longitude and latitude variable names
    """
    lon_bounds = (0, 60)
    lat_bounds = (-30, 30)
    
    cond = create_lon_lat_condition_dataset(
        lon_bounds, lat_bounds,
        lon_var_name='lon_custom',
        lat_var_name='lat_custom'
    )
    indexers = get_indexers_from_nd(cond, cut=True)
    
    assert 'lat' in indexers
    assert 'lon' in indexers
    assert len(indexers['lat']) > 0
    assert len(indexers['lon']) > 0