import operator
import shutil
import tempfile
import warnings
from os import listdir
from os.path import dirname, isfile, join, realpath
from unittest import TestCase

import netCDF4 as nc
import numpy as np
import pytest
import xarray as xr

from podaac.subsetter import subset
from conftest import data_files 
from podaac.subsetter.utils.coordinate_utils import compute_coordinate_variable_names, convert_bbox

@pytest.mark.parametrize("test_file", data_files())
def test_subset_bbox(test_file, data_dir, subset_output_dir, request):
    """
    Test that all data present is within the bounding box given,
    and that the correct bounding box is used. This test assumed
    that the scanline *is* being cut.
    """

    # pylint: disable=too-many-locals
    bbox = np.array(((-180, 90), (-90, 90)))
    output_file = "{}_{}".format(request.node.name, test_file)
    subset_output_file = join(subset_output_dir, output_file)
    subset.subset(
        file_to_subset=join(data_dir, test_file),
        bbox=bbox,
        output_file=subset_output_file
    )

    out_ds, _, file_ext = subset.open_as_nc_dataset(subset_output_file)
    out_ds = xr.open_dataset(xr.backends.NetCDF4DataStore(out_ds),
                             decode_times=False,
                             decode_coords=False,
                             mask_and_scale=False)

    lat_var_name, lon_var_name = compute_coordinate_variable_names(out_ds)

    lat_var_name = lat_var_name[0]
    lon_var_name = lon_var_name[0]

    lon_bounds, lat_bounds = convert_bbox(bbox, out_ds, lat_var_name, lon_var_name)

    lats = out_ds[lat_var_name].values
    lons = out_ds[lon_var_name].values

    warnings.filterwarnings('ignore')

    # Step 1: Get mask of values which aren't in the bounds.

    # For lon spatial condition, need to consider the
    # lon_min > lon_max case. If that's the case, should do
    # an 'or' instead.
    oper = operator.and_ if lon_bounds[0] < lon_bounds[1] else operator.or_

    # In these two masks, True == valid and False == invalid
    lat_truth = np.ma.masked_where((lats >= lat_bounds[0])
                                   & (lats <= lat_bounds[1]), lats).mask
    lon_truth = np.ma.masked_where(oper((lons >= lon_bounds[0]),
                                        (lons <= lon_bounds[1])), lons).mask

    # combine masks
    spatial_mask = np.bitwise_and(lat_truth, lon_truth)

    # Create a mask which represents the valid matrix bounds of
    # the spatial mask. This is used in the case where a var
    # has no _FillValue.
    if lon_truth.ndim == 1:
        bound_mask = spatial_mask
    else:
        rows = np.any(spatial_mask, axis=1)
        cols = np.any(spatial_mask, axis=0)
        bound_mask = np.array([[r & c for c in cols] for r in rows])

    # If all the lat/lon values are valid, the file is valid and
    # there is no need to check individual variables.
    if np.all(spatial_mask):
        return

    # Step 2: Get mask of values which are NaN or "_FillValue in
    # each variable.
    for _, var in out_ds.data_vars.items():
        # remove dimension of '1' if necessary
        vals = np.squeeze(var.values)

        # Get the Fill Value
        fill_value = var.attrs.get('_FillValue')

        # If _FillValue isn't provided, check that all values
        # are in the valid matrix bounds go to the next variable
        if fill_value is None:
            combined_mask = np.ma.mask_or(spatial_mask, bound_mask)
            np.testing.assert_equal(bound_mask, combined_mask)
            continue

        # If the shapes of this var doesn't match the mask,
        # reshape the var so the comparison can be made. Take
        # the first index of the unknown dims. This makes
        # assumptions about the ordering of the dimensions.
        if vals.shape != out_ds[lat_var_name].shape and vals.shape:
            slice_list = []
            for dim in var.dims:
                if dim in out_ds[lat_var_name].dims:
                    slice_list.append(slice(None))
                else:
                    slice_list.append(slice(0, 1))
            vals = np.squeeze(vals[tuple(slice_list)])

        # Skip for byte type.
        if vals.dtype == 'S1':
            continue

        # In this mask, False == NaN and True = valid
        var_mask = np.invert(np.ma.masked_invalid(vals).mask)
        fill_mask = np.invert(np.ma.masked_values(vals, fill_value).mask)

        var_mask = np.bitwise_and(var_mask, fill_mask)

        if var_mask.shape != spatial_mask.shape:
            # This may be a case where the time represents lines,
            # or some other case where the variable doesn't share
            # a shape with the coordinate variables.
            continue

        # Step 3: Combine the spatial and var mask with 'or'
        combined_mask = np.ma.mask_or(var_mask, spatial_mask)

        # Step 4: compare the newly combined mask and the
        # spatial mask created from the lat/lon masks. They
        # should be equal, because the 'or' of the two masks
        # where out-of-bounds values are 'False' will leave
        # those values assuming there are only NaN values
        # in the data at those locations.
        np.testing.assert_equal(spatial_mask, combined_mask)

    out_ds.close()