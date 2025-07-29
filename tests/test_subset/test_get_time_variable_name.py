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

    in_ds_tree = xr.open_datatree(join(data_dir, test_file),
                            decode_times=False,
                            decode_coords=False)

    lat_var_name = datatree_subset.compute_coordinate_variable_names_from_tree(in_ds_tree)[1][0].strip('/')
    time_var_name = datatree_subset.compute_time_variable_name_tree(in_ds_tree, in_ds_tree[lat_var_name], [])

    assert time_var_name is not None
    assert 'time' in time_var_name

    in_ds_tree.close()