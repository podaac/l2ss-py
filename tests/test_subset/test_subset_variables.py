import shutil
import tempfile
from os import listdir
from os.path import dirname, isfile, join, realpath
from unittest import TestCase

import netCDF4 as nc
import numpy as np
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


@pytest.mark.parametrize("test_file", TEST_DATA_FILES)
def test_subset_variables(test_file, data_dir, subset_output_dir, request):
    """
    Test that all variables present in the original NetCDF file
    are present after the subset takes place, and with the same
    attributes.
    """

    bbox = np.array(((-180, 90), (-90, 90)))
    output_file = "{}_{}".format(request.node.name, test_file)
    subset.subset(
        file_to_subset=join(data_dir, test_file),
        bbox=bbox,
        output_file=join(subset_output_dir, output_file)
    )

    in_ds = xr.open_dataset(join(data_dir, test_file),
                            decode_times=False,
                            decode_coords=False)
    out_ds = xr.open_dataset(join(subset_output_dir, output_file),
                             decode_times=False,
                             decode_coords=False)


    nc_in_ds = nc.Dataset(join(data_dir, test_file))
    nc_out_ds = nc.Dataset(join(subset_output_dir, output_file))

    time_var_name = None
    try:
        lat_var_name = subset.compute_coordinate_variable_names(in_ds)[0][0]
        time_var_name = datatree_subset.compute_time_variable_name_tree(in_ds, in_ds[lat_var_name], [])
    except ValueError:
        # unable to determine lon lat vars
        pass

    if time_var_name:
        assert nc_in_ds[time_var_name].units == nc_out_ds[time_var_name].units

    nc_in_ds.close()
    nc_out_ds.close()

    for in_var, out_var in zip(in_ds.data_vars.items(), out_ds.data_vars.items()):
        # compare names
        assert in_var[0] == out_var[0]

        # compare attributes    
        np.testing.assert_equal(in_var[1].attrs, out_var[1].attrs)

        # compare type and dimension names
        assert in_var[1].dtype == out_var[1].dtype
        assert in_var[1].dims == out_var[1].dims

    in_ds.close()
    out_ds.close()