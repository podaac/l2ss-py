# In tests/conftest.py
import pytest
import tempfile
import shutil
from os import listdir
from os.path import dirname, isfile, join, realpath
import gc as garbage_collection

import netCDF4 as nc
import xarray as xr

@pytest.fixture(scope='class')
def data_dir():
    """Gets the directory containing data files used for tests."""
    test_dir = dirname(realpath(__file__))
    return join(test_dir, '..', 'data')

@pytest.fixture
def subset_output_dir(data_dir):
    subset_output_dir = tempfile.mkdtemp(dir=data_dir)
    yield subset_output_dir
    shutil.rmtree(subset_output_dir)

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

def data_files():
    """Get all the netCDF files from the test data directory."""
    test_dir = dirname(realpath(__file__))
    test_data_dir = join(test_dir, '..' , 'data')
    return [f for f in listdir(test_data_dir) if isfile(join(test_data_dir, f)) and f.endswith(".nc")]