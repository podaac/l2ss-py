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
from conftest import data_files 


@pytest.mark.parametrize("test_file", data_files())
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
