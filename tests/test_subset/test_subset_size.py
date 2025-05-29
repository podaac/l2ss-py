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
import gc as garbage_collection

from podaac.subsetter import subset
from harmony_service_lib.exceptions import NoDataException
from conftest import data_files 


@pytest.mark.parametrize("test_file", data_files())
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