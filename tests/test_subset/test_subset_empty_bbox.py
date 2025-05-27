import operator
import shutil
import tempfile
import warnings
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

 
@pytest.mark.parametrize("test_file", TEST_DATA_FILES)
def test_subset_empty_bbox(test_file, data_dir, subset_output_dir, request):
    """Test that an empty file is returned when the bounding box
    contains no data."""
    nc_copy_for_expected_results = os.path.join(subset_output_dir, Path(test_file).stem + "_dup.nc")
    shutil.copyfile(os.path.join(data_dir, test_file),
                    nc_copy_for_expected_results)

    bbox = np.array(((120, 125), (-90, -85)))
    output_file = "{}_{}".format(request.node.name, test_file)

    with pytest.raises(NoDataException, match="No data in subsetted granule."):

        subset.subset(
            file_to_subset=join(data_dir, test_file),
            bbox=bbox,
            output_file=join(subset_output_dir, output_file)
        )

    """
    test_input_dataset = xr.open_dataset(
        nc_copy_for_expected_results,
        decode_times=False,
        decode_coords=False,
        mask_and_scale=False
    )
    empty_dataset = xr.open_dataset(
        join(subset_output_dir, output_file),
        decode_times=False,
        decode_coords=False,
        mask_and_scale=False
    )

    # Ensure all variables are present but empty.
    for _, variable in empty_dataset.data_vars.items():
        fill_value = variable.attrs.get('_FillValue', np.nan)
        data = variable.data
        
        # Perform the main check
        condition = np.all(data == fill_value) or np.all(np.isnan(data))
        
        # Handle the specific integer dtype case
        if not condition and not (np.isnan(fill_value) and np.issubdtype(variable.dtype, np.integer)):
            assert condition, f"Data does not match fill value for variable: {variable}"

    assert test_input_dataset.dims.keys() == empty_dataset.dims.keys()
    
    test_input_dataset.close()
    empty_dataset.close()

    """
