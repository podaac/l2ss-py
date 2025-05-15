import shutil
import tempfile
import os
from os import listdir
from os.path import dirname, isfile, join, realpath
from pathlib import Path
from unittest import TestCase

import pytest
import xarray as xr

from podaac.subsetter import subset
from podaac.subsetter import datatree_subset

@pytest.fixture(scope='class')
def data_dir():
    """Gets the directory containing data files used for tests."""
    test_dir = dirname(realpath(__file__))
    return join(test_dir, '..', 'data')


@pytest.fixture(scope='class')
def subset_output_dir(data_dir):
    """Makes a new temporary directory to hold the subset results while tests are running."""
    subset_output_dir = tempfile.mkdtemp(dir=data_dir)
    yield subset_output_dir
    shutil.rmtree(subset_output_dir)


def data_files():
    """Get all the netCDF files from the test data directory."""
    test_dir = dirname(realpath(__file__))
    test_data_dir = join(test_dir, '..' , 'data')
    return [f for f in listdir(test_data_dir) if isfile(join(test_data_dir, f)) and f.endswith(".nc")]


TEST_DATA_FILES = data_files()


@pytest.mark.parametrize('test_file', TEST_DATA_FILES)
def test_get_time_variable_name(test_file, data_dir):
    """Ensures that the name of the time variable can be retrieved."""
    args = {
        'decode_coords': False,
        'mask_and_scale': False,
        'decode_times': True
    }
    ds, _, file_ext = subset.open_as_nc_dataset(os.path.join(data_dir, test_file))
    ds = xr.open_dataset(xr.backends.NetCDF4DataStore(ds), **args)

    lat_var_name = subset.compute_coordinate_variable_names(ds)[0][0]
    time_var_name = datatree_subset.compute_time_variable_name_tree(ds, ds[lat_var_name], [])

    assert time_var_name is not None
    assert 'time' in time_var_name

    ds.close()