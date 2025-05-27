import pytest
import numpy as np
import xarray as xr
import operator
from podaac.subsetter.datatree_subset import get_indexers_from_nd

def create_lon_lat_condition_dataset(
    lon_bounds, lat_bounds, 
    lon_var_name='longitude', lat_var_name='latitude', 
    lon_dim_name='lon', lat_dim_name='lat',
    third_dim_name='level', third_dim_values=None, include_third_dim=False
):
    """
    Create a test dataset using xarray with customizable longitude, latitude, and optional third dimension.
    
    Parameters:
    - lon_bounds: Tuple of longitude bounds (start, end)
    - lat_bounds: Tuple of latitude bounds (start, end)
    - lon_var_name: Name of the longitude variable
    - lat_var_name: Name of the latitude variable
    - lon_dim_name: Name of the longitude dimension
    - lat_dim_name: Name of the latitude dimension
    - third_dim_name: Name of the third dimension
    - third_dim_values: Values for the third dimension (e.g., levels, time). Defaults to range(0, 10).
    - include_third_dim: Whether to include the third dimension (True for 3D, False for 2D)
    """
    # Default values for the third dimension if not provided
    if third_dim_values is None:
        third_dim_values = np.linspace(0, 1000, 11)  # Example: 11 levels from 0 to 1000
    
    # Create coordinates
    lats = np.linspace(-90, 90, 181)
    lons = np.linspace(-180, 180, 361)
    
    if include_third_dim:
        coords = {lat_dim_name: lats, lon_dim_name: lons, third_dim_name: third_dim_values}
    else:
        coords = {lat_dim_name: lats, lon_dim_name: lons}
    
    # Create the dataset with coordinates
    ds = xr.Dataset(coords=coords)
    
    # Add longitude and latitude as variables
    if include_third_dim:
        ds[lon_var_name] = xr.DataArray(
            data=np.tile(lons[None, None, :], (len(third_dim_values), len(lats), 1)),
            dims=[third_dim_name, lat_dim_name, lon_dim_name]
        )
        ds[lat_var_name] = xr.DataArray(
            data=np.tile(lats[None, :, None], (len(third_dim_values), 1, len(lons))),
            dims=[third_dim_name, lat_dim_name, lon_dim_name]
        )
        ds[third_dim_name] = xr.DataArray(
            data=np.tile(third_dim_values[:, None, None], (1, len(lats), len(lons))),
            dims=[third_dim_name, lat_dim_name, lon_dim_name]
        )
    else:
        ds[lon_var_name] = xr.DataArray(
            data=np.tile(lons[None, :], (len(lats), 1)),
            dims=[lat_dim_name, lon_dim_name]
        )
        ds[lat_var_name] = xr.DataArray(
            data=np.tile(lats[:, None], (1, len(lons))),
            dims=[lat_dim_name, lon_dim_name]
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

def test_random_dim():
    """
    Test with custom longitude and latitude variable names
    """
    lon_bounds = (0, 60)
    lat_bounds = (-30, 30)
    
    cond = create_lon_lat_condition_dataset(
        lon_bounds, lat_bounds,
        lon_var_name='lon_custom',
        lat_var_name='lat_custom',
        lon_dim_name='x',
        lat_dim_name='y',
        third_dim_name='ztrack',
        include_third_dim=True
    )
    indexers = get_indexers_from_nd(cond, cut=True)
    
    print(indexers)
    assert 'x' in indexers
    assert 'y' in indexers
    assert len(indexers['x']) > 0
    assert len(indexers['y']) > 0

def test_xdim_grid():
    """
    Test with custom longitude and latitude variable names
    """
    lon_bounds = (0, 60)
    lat_bounds = (-30, 30)
    
    cond = create_lon_lat_condition_dataset(
        lon_bounds, lat_bounds,
        lon_var_name='lon_custom',
        lat_var_name='lat_custom',
        lon_dim_name='xdim_grid',
        lat_dim_name='ydim_grid',
        third_dim_name='ztrack',
        include_third_dim=True
    )
    indexers = get_indexers_from_nd(cond, cut=True)
    
    assert 'xdim_grid' in indexers
    assert 'ydim_grid' in indexers
    assert len(indexers['xdim_grid']) > 0
    assert len(indexers['ydim_grid']) > 0