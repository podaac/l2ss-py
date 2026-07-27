import os

import numpy as np
import pytest
from conftest import data_files
from harmony_service_lib.exceptions import NoDataException

from podaac.subsetter import subset


@pytest.mark.parametrize("test_file", data_files())
def test_subset_size(test_file, data_dir, subset_output_dir, request):
    """Verifies that the subsetted file is smaller in size than the original file."""
    bbox = np.array(((-180, 0), (-30, 90)))
    output_file = f"{request.node.name}_{test_file}"
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
    except NoDataException:
        assert True