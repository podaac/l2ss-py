import operator
import shutil
import tempfile
import os
from os import listdir
from os.path import dirname, isfile, join, realpath
from pathlib import Path
from unittest import TestCase
import h5py

import netCDF4 as nc
import numpy as np
import pytest
import xarray as xr

from podaac.subsetter import subset
from podaac.subsetter.utils.coordinate_utils import get_coordinate_variable_names
from podaac.subsetter.utils.variables_utils import get_all_variable_names_from_dtree

from conftest import data_files 

import xarray as xr




def get_non_variable_names_from_dtree(dtree: xr.DataTree):
    """
    Recursively extract all non-variable names (with full paths) from an xarray DataTree.
    This includes coordinates, dimensions, and other variables that are not data_vars.
    
    Parameters
    ----------
    dtree : xr.DataTree
        The root of the DataTree.
    Returns
    -------
    List[str]
        A list of non-variable full paths (e.g. '/group1/coord').
    """
    non_var_names = []
    
    def recurse(node: xr.DataTree):
        group_path = node.path
        
        # Get all variables that are NOT data_vars
        if node.ds is not None:  # Check if node has a dataset
            for var_name in node.ds.variables:
                if var_name not in node.ds.data_vars:
                    if group_path in ("", "/"):
                        full_path = f"/{var_name}"
                    else:
                        full_path = f"{group_path}/{var_name}"
                    non_var_names.append(full_path)
        
        for child in node.children.values():
            recurse(child)
    
    recurse(dtree)
    return non_var_names



@pytest.mark.parametrize("test_file", data_files())
def test_specified_variables(test_file, data_dir, subset_output_dir, request):
    """
    Test that the variables which are specified when calling the subset
    operation are present in the resulting subsetted data file,
    and that the variables which are specified are not present.
    """
    nc_copy_for_expected_results = os.path.join(subset_output_dir, Path(test_file).stem + "_dup.nc")
    shutil.copyfile(os.path.join(data_dir, test_file), nc_copy_for_expected_results)

    bbox = np.array(((-180, 180), (-90, 90)))
    output_file = "{}_{}".format(request.node.name, test_file)
    
    in_ds_tree = xr.open_datatree(nc_copy_for_expected_results, decode_times=False, decode_coords=False)

    # Coordinate variables are always included in the result
    lat_var_names, lon_var_names, time_var_names = get_coordinate_variable_names(in_ds_tree)

    coordinate_variables = lat_var_names + lon_var_names + time_var_names
    all_variables = get_all_variable_names_from_dtree(in_ds_tree)
    non_coordinate_vars = [
        var for var in all_variables if var not in coordinate_variables
    ]

    included_variables = non_coordinate_vars[::2] + coordinate_variables

    non_vars = get_non_variable_names_from_dtree(in_ds_tree)

    subset.subset(
        file_to_subset=join(data_dir, test_file),
        bbox=bbox,
        output_file=join(subset_output_dir, output_file),
        variables=included_variables
    )

    out_ds_tree = xr.open_datatree(join(subset_output_dir, output_file), decode_times=False, decode_coords=False)
    out_lat_var_names, out_lon_var_names, out_time_var_names = get_coordinate_variable_names(out_ds_tree)
    out_coordinate_variables = out_lat_var_names + out_lon_var_names + out_time_var_names

    subsetted_vars = get_all_variable_names_from_dtree(out_ds_tree)
    subsetted_non_vars = get_non_variable_names_from_dtree(out_ds_tree) 

    assert set(subsetted_vars + subsetted_non_vars) == set(included_variables + non_vars)

    in_ds_tree.close()
    out_ds_tree.close()
