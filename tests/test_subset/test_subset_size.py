import shutil
import tempfile
import os
from os import listdir
from os.path import dirname, isfile, join, realpath
from pathlib import Path
from unittest import TestCase

import pytest
import numpy as np
import xarray as xr

from podaac.subsetter import subset
from harmony_service_lib.exceptions import NoDataException

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
def test_subset_size(test_file, data_dir, subset_output_dir, request):
    """Verifies that the subsetted file is smaller in size than the original file."""
    bbox = np.array(((-180, 0), (-30, 90)))
    output_file = "{}_{}".format(request.node.name, test_file)
    input_file_path = os.path.join(data_dir, test_file)
    output_file_path = os.path.join(subset_output_dir, output_file)

    try:
        subset.subset(
            file_to_subset=input_file_path,
            bbox=bbox,
            output_file=output_file_path
        )

        original_file_size = os.path.getsize(input_file_path)
        subset_file_size = os.path.getsize(output_file_path)

        assert subset_file_size < original_file_size
    except NoDataException as e:
        assert True