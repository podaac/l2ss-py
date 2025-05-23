import operator
import shutil
import tempfile
import os
from os import listdir
from os.path import dirname, isfile, join, realpath
from pathlib import Path
from unittest import TestCase

import netCDF4 as nc
import numpy as np
import pytest
import xarray as xr

from podaac.subsetter import subset
from podaac.subsetter.group_handling import GROUP_DELIM


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
def test_specified_variables(test_file, data_dir, subset_output_dir, request):
    """
    Test that the variables which are specified when calling the subset
    operation are present in the resulting subsetted data file,
    and that the variables which are specified are not present.
    """
    nc_copy_for_expected_results = os.path.join(subset_output_dir, Path(test_file).stem + "_dup.nc")
    shutil.copyfile(os.path.join(data_dir, test_file),
                    nc_copy_for_expected_results)

    bbox = np.array(((-180, 180), (-90, 90)))
    output_file = "{}_{}".format(request.node.name, test_file)

    in_ds, _, file_ext = subset.open_as_nc_dataset(nc_copy_for_expected_results)
    in_ds = xr.open_dataset(xr.backends.NetCDF4DataStore(in_ds),
                            decode_times=False,
                            decode_coords=False)
    # Non-data vars are by default included in the result
    non_data_vars = set(in_ds.variables.keys()) - set(in_ds.data_vars.keys())

    # Coordinate variables are always included in the result
    lat_var_names, lon_var_names, time_var_names = subset.get_coordinate_variable_names(in_ds)
    coordinate_variables = lat_var_names + lon_var_names + time_var_names

    # Pick some variables to include in the result (every other variable: first, third, fifth, etc.)
    included_variables = set([variable[0] for variable in in_ds.data_vars.items()][::2])
    included_variables = list(included_variables)

    # All other data variables should be dropped
    expected_excluded_variables = list(set(variable[0] for variable in in_ds.data_vars.items())
                                       - set(included_variables) - set(coordinate_variables))

    subset.subset(
        file_to_subset=join(data_dir, test_file),
        bbox=bbox,
        output_file=join(subset_output_dir, output_file),
        variables=[var.replace(GROUP_DELIM, '/') for var in included_variables]
    )

    out_ds, _, file_ext = subset.open_as_nc_dataset(join(subset_output_dir, output_file))
    out_ds = xr.open_dataset(xr.backends.NetCDF4DataStore(out_ds),
                             decode_times=False,
                             decode_coords=False)

    out_vars = list(out_ds.variables.keys())

    assert set(out_vars) == set(included_variables + coordinate_variables).union(non_data_vars)
    assert set(out_vars).isdisjoint(expected_excluded_variables)

    in_ds.close()
    out_ds.close()