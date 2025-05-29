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
from conftest import data_files 

@pytest.mark.parametrize("test_file", data_files())
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