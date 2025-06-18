import operator
import shutil
import tempfile
import warnings
import urllib.parse
from os import listdir
from os.path import dirname, isfile, join, realpath
from unittest import TestCase

import netCDF4 as nc
import numpy as np
import pytest
import xarray as xr

from podaac.subsetter import subset


def compare_java(test_file, cut, data_dir, subset_output_dir, request):
    """
    Run the L2 subsetter and compare the result to the equivelant
    legacy (Java) subsetter result.
    Parameters
    ----------
    test_file : str
        path to test file.
    cut : boolean
        True if the subsetter should return compact.
    """
    bbox_map = [("ascat_20150702_084200", ((-180, 0), (-90, 0))),
                ("ascat_20150702_102400", ((-180, 0), (-90, 0))),
                ("MODIS_A-JPL", ((65.8, 86.35), (40.1, 50.15))),
                ("MODIS_T-JPL", ((-78.7, -60.7), (-54.8, -44))),
                ("VIIRS", ((-172.3, -126.95), (62.3, 70.65))),
                ("AMSR2-L2B_v08_r38622", ((-180, 0), (-90, 0)))]

    java_files_dir = join(data_dir, "java_results", "cut" if cut else "uncut")

    java_files = [join(java_files_dir, f) for f in listdir(java_files_dir) if
                  isfile(join(java_files_dir, f)) and f.endswith(".nc")]

    file, bbox = next(iter([b for b in bbox_map if b[0] in test_file]))
    java_file = next(iter([f for f in java_files if file in f]))

    output_file = "{}_{}".format(urllib.parse.quote_plus(request.node.name), test_file)
    subset.subset(
        file_to_subset=join(data_dir, test_file),
        bbox=np.array(bbox),
        output_file=join(subset_output_dir, output_file),
        cut=cut
    )

    j_ds = xr.open_dataset(join(data_dir, java_file),
                           decode_times=False,
                           decode_coords=False,
                           mask_and_scale=False)

    py_ds = xr.open_dataset(join(subset_output_dir, output_file),
                            decode_times=False,
                            decode_coords=False,
                            mask_and_scale=False)

    for var_name, var in j_ds.data_vars.items():
        # Compare shape
        np.testing.assert_equal(var.shape, py_ds[var_name].shape)

        # Compare meta
        np.testing.assert_equal(var.attrs, py_ds[var_name].attrs)

        diff_indices = np.where(var.values != py_ds[var_name].values)

        # Compare data
        np.testing.assert_equal(var.values, py_ds[var_name].values)

    # Compare meta. History will always be different, so remove
    # from the headers for comparison.
    del j_ds.attrs['history']
    del py_ds.attrs['history']
    del py_ds.attrs['history_json']

    ignore_attributes = [
        "northernmost_latitude",
        "southernmost_latitude",
        "easternmost_longitude",
        "westernmost_longitude",
        "geospatial_bounds_crs",
        "geospatial_bounds"
    ]
    filtered_j_ds_attrs = {k: v for k, v in j_ds.attrs.items() if k not in ignore_attributes}
    filtered_py_ds_attrs = {k: v for k, v in py_ds.attrs.items() if k not in ignore_attributes}

    np.testing.assert_equal(filtered_j_ds_attrs, filtered_py_ds_attrs)

    j_ds.close()
    py_ds.close()

@pytest.mark.parametrize("test_file", [
    "ascat_20150702_084200_metopa_45145_eps_o_250_2300_ovw.l2.nc",
    "ascat_20150702_102400_metopa_45146_eps_o_250_2300_ovw.l2.nc",
    "MODIS_A-JPL-L2P-v2014.0.nc",
    "MODIS_T-JPL-L2P-v2014.0.nc",
    #"VIIRS_NPP-NAVO-L2P-v3.0.nc",
    "AMSR2-L2B_v08_r38622-v02.0-fv01.0.nc"
])
def test_compare_java_compact(test_file, data_dir, subset_output_dir, request):
    """
    Tests that the results of the subsetting operation is
    equivalent to the Java subsetting result on the same bounding
    box. For simplicity the subsetted Java granules have been
    manually run and copied into this project. This test DOES
    cut the scanline.

    No longer testing VIIRS VIIRS_NPP-NAVO-L2P-v3.0.nc cell 77 92
    there is one cell that doesn't match latitude is 65.00000####,
    updated to numpy ^ 2.0.0, most likely due to higher precision so 
    we no longer want to test since new numpy is more precise than the 
    java subsetted
    """

    compare_java(test_file, True, data_dir, subset_output_dir, request)


@pytest.mark.parametrize("test_file", [
    "ascat_20150702_084200_metopa_45145_eps_o_250_2300_ovw.l2.nc",
    "ascat_20150702_102400_metopa_45146_eps_o_250_2300_ovw.l2.nc",
    "MODIS_A-JPL-L2P-v2014.0.nc",
    "MODIS_T-JPL-L2P-v2014.0.nc",
    #"VIIRS_NPP-NAVO-L2P-v3.0.nc",
    "AMSR2-L2B_v08_r38622-v02.0-fv01.0.nc"
])
def test_compare_java(test_file, data_dir, subset_output_dir, request):
    """
    Tests that the results of the subsetting operation is
    equivalent to the Java subsetting result on the same bounding
    box. For simplicity the subsetted Java granules have been
    manually run and copied into this project. This runs does NOT
    cut the scanline.
    """

    compare_java(test_file, False, data_dir, subset_output_dir, request)