# Copyright 2019, by the California Institute of Technology.
# ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.
# Any commercial use must be negotiated with the Office of Technology
# Transfer at the California Institute of Technology.
#
# This software may be subject to U.S. export control laws. By accepting
# this software, the user agrees to comply with all applicable U.S. export
# laws and regulations. User has the responsibility to obtain export
# licenses, or other export authority as may be required before exporting
# such information to foreign countries or providing access to foreign
# persons.

"""
==============
test_subset.py
==============
Test the subsetter functionality.
"""
import json
import operator
import os
import shutil
import tempfile
import unittest
from os import listdir
from os.path import dirname, join, realpath, isfile, basename

import geopandas as gpd
import importlib_metadata
import netCDF4 as nc
import h5py
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from jsonschema import validate
from shapely.geometry import Point

from podaac.subsetter import subset
from podaac.subsetter.subset import SERVICE_NAME
from podaac.subsetter import xarray_enhancements as xre
from podaac.subsetter import dimension_cleanup as dc


class TestSubsetter(unittest.TestCase):
    """
    Unit tests for the L2 subsetter. These tests are all related to the
    subsetting functionality itself, and should provide coverage on the
    following files:
    - podaac.subsetter.subset.py
    - podaac.subsetter.xarray_enhancements.py
    """

    @classmethod
    def setUpClass(cls):
        cls.test_dir = dirname(realpath(__file__))
        cls.test_data_dir = join(cls.test_dir, 'data')
        cls.subset_output_dir = tempfile.mkdtemp(dir=cls.test_data_dir)
        cls.test_files = [f for f in listdir(cls.test_data_dir)
                          if isfile(join(cls.test_data_dir, f)) and f.endswith(".nc")]

        cls.history_json_schema = {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "$id": "https://harmony.earthdata.nasa.gov/history.schema.json",
            "title": "Data Processing History",
            "description": "A history record of processing that produced a given data file. For more information, see: https://wiki.earthdata.nasa.gov/display/TRT/In-File+Provenance+Metadata+-+TRT-42",
            "type": ["array", "object"],
            "items": {"$ref": "#/definitions/history_record"},

            "definitions": {
                "history_record": {
                    "type": "object",
                    "properties": {
                        "date_time": {
                            "description": "A Date/Time stamp in ISO-8601 format, including time-zone, GMT (or Z) preferred",
                            "type": "string",
                            "format": "date-time"
                        },
                        "derived_from": {
                            "description": "List of source data files used in the creation of this data file",
                            "type": ["array", "string"],
                            "items": {"type": "string"}
                        },
                        "program": {
                            "description": "The name of the program which generated this data file",
                            "type": "string"
                        },
                        "version": {
                            "description": "The version identification of the program which generated this data file",
                            "type": "string"
                        },
                        "parameters": {
                            "description": "The list of parameters to the program when generating this data file",
                            "type": ["array", "string"],
                            "items": {"type": "string"}
                        },
                        "program_ref": {
                            "description": "A URL reference that defines the program, e.g., a UMM-S reference URL",
                            "type": "string"
                        },
                        "$schema": {
                            "description": "The URL to this schema",
                            "type": "string"
                        }
                    },
                    "required": ["date_time", "program"],
                    "additionalProperties": False
                }
            }
        }

    @classmethod
    def tearDownClass(cls):
        # Remove the temporary directories used to house subset data
        shutil.rmtree(cls.subset_output_dir)

    def test_subset_variables(self):
        """
        Test that all variables present in the original NetCDF file
        are present after the subset takes place, and with the same
        attributes.
        """

        bbox = np.array(((-180, 90), (-90, 90)))
        for file in self.test_files:
            output_file = "{}_{}".format(self._testMethodName, file)
            subset.subset(
                file_to_subset=join(self.test_data_dir, file),
                bbox=bbox,
                output_file=join(self.subset_output_dir, output_file)
            )

            in_ds = xr.open_dataset(join(self.test_data_dir, file),
                                    decode_times=False,
                                    decode_coords=False)
            out_ds = xr.open_dataset(join(self.subset_output_dir, output_file),
                                     decode_times=False,
                                     decode_coords=False)

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
            

    def test_subset_bbox(self):
        """
        Test that all data present is within the bounding box given,
        and that the correct bounding box is used. This test assumed
        that the scanline *is* being cut.
        """

        # pylint: disable=too-many-locals
        bbox = np.array(((-180, 90), (-90, 90)))
        for file in self.test_files:
            output_file = "{}_{}".format(self._testMethodName, file)
            subset.subset(
                file_to_subset=join(self.test_data_dir, file),
                bbox=bbox,
                output_file=join(self.subset_output_dir, output_file)
            )

            out_ds = xr.open_dataset(join(self.subset_output_dir, output_file),
                                     decode_times=False,
                                     decode_coords=False,
                                     mask_and_scale=False)

            lat_var_name, lon_var_name = subset.compute_coordinate_variable_names(out_ds)

            lat_var_name = lat_var_name[0]
            lon_var_name = lon_var_name[0]

            lon_bounds, lat_bounds = subset.convert_bbox(bbox, out_ds, lat_var_name, lon_var_name)

            lats = out_ds[lat_var_name].values
            lons = out_ds[lon_var_name].values

            np.warnings.filterwarnings('ignore')

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
                continue

            # Step 2: Get mask of values which are NaN or "_FillValue in
            # each variable.
            for var_name, var in out_ds.data_vars.items():
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

    @pytest.mark.skip(reason="This is being tested currently.  Temporarily skipped.")
    def test_subset_no_bbox(self):
        """
        Test that the subsetted file is identical to the given file
        when a 'full' bounding box is given.
        """

        bbox = np.array(((-180, 180), (-90, 90)))
        for file in self.test_files:
            output_file = "{}_{}".format(self._testMethodName, file)
            subset.subset(
                file_to_subset=join(self.test_data_dir, file),
                bbox=bbox,
                output_file=join(self.subset_output_dir, output_file)
            )

            # pylint: disable=no-member
            in_nc = nc.Dataset(join(self.test_data_dir, file), 'r')
            out_nc = nc.Dataset(join(self.subset_output_dir, output_file), 'r')

            # Make sure the output dimensions match the input
            # dimensions, which means the full file was returned.
            for name, dimension in in_nc.dimensions.items():
                assert dimension.size == out_nc.dimensions[name].size

            in_nc.close()
            out_nc.close()

    def test_subset_empty_bbox(self):
        """
        Test that an empty file is returned when the bounding box
        contains no data.
        """

        bbox = np.array(((120, 125), (-90, -85)))
        for file in self.test_files:
            output_file = "{}_{}".format(self._testMethodName, file)
            subset.subset(
                file_to_subset=join(self.test_data_dir, file),
                bbox=bbox,
                output_file=join(self.subset_output_dir, output_file)
            )
            test_input_dataset = xr.open_dataset(
                join(self.test_data_dir, file),
                decode_times=False,
                decode_coords=False,
                mask_and_scale=False
            )
            empty_dataset = xr.open_dataset(
                join(self.subset_output_dir, output_file),
                decode_times=False,
                decode_coords=False,
                mask_and_scale=False
            )

            # Ensure all variables are present but empty.
            for variable_name, variable in empty_dataset.data_vars.items():
                assert np.all(variable.data == variable.attrs.get('_FillValue', np.nan) or np.isnan(variable.data))

            assert test_input_dataset.dims.keys() == empty_dataset.dims.keys()


    def test_bbox_conversion(self):
        """
        Test that the bounding box conversion returns expected
        results. Expected results are hand-calculated.
        """

        ds_180 = xr.open_dataset(join(self.test_data_dir,
                                      "MODIS_A-JPL-L2P-v2014.0.nc"),
                                 decode_times=False,
                                 decode_coords=False)

        ds_360 = xr.open_dataset(join(
            self.test_data_dir,
            "ascat_20150702_084200_metopa_45145_eps_o_250_2300_ovw.l2.nc"),
            decode_times=False,
            decode_coords=False)

        # Elements in each tuple are:
        # ds type, lon_range, expected_result
        test_bboxes = [
            (ds_180, (-180, 180), (-180, 180)),
            (ds_360, (-180, 180), (0, 360)),
            (ds_180, (-180, 0), (-180, 0)),
            (ds_360, (-180, 0), (180, 360)),
            (ds_180, (-80, 80), (-80, 80)),
            (ds_360, (-80, 80), (280, 80)),
            (ds_180, (0, 180), (0, 180)),
            (ds_360, (0, 180), (0, 180)),
            (ds_180, (80, -80), (80, -80)),
            (ds_360, (80, -80), (80, 280)),
            (ds_180, (-80, -80), (-180, 180)),
            (ds_360, (-80, -80), (0, 360))
        ]

        lat_var = 'lat'
        lon_var = 'lon'

        for test_bbox in test_bboxes:
            dataset = test_bbox[0]
            lon_range = test_bbox[1]
            expected_result = test_bbox[2]
            actual_result, _ = subset.convert_bbox(np.array([lon_range, [0, 0]]),
                                                   dataset, lat_var, lon_var)

            np.testing.assert_equal(actual_result, expected_result)

    def compare_java(self, java_files, cut):
        """
        Run the L2 subsetter and compare the result to the equivelant
        legacy (Java) subsetter result.
        Parameters
        ----------
        java_files : list of strings
            List of paths to each subsetted Java file.
        cut : boolean
            True if the subsetter should return compact.
        """
        bbox_map = [("ascat_20150702_084200", ((-180, 0), (-90, 0))),
                    ("ascat_20150702_102400", ((-180, 0), (-90, 0))),
                    ("MODIS_A-JPL", ((65.8, 86.35), (40.1, 50.15))),
                    ("MODIS_T-JPL", ((-78.7, -60.7), (-54.8, -44))),
                    ("VIIRS", ((-172.3, -126.95), (62.3, 70.65))),
                    ("AMSR2-L2B_v08_r38622", ((-180, 0), (-90, 0)))]

        for file_str, bbox in bbox_map:
            java_file = [file for file in java_files if file_str in file][0]
            test_file = [file for file in self.test_files if file_str in file][0]
            output_file = "{}_{}".format(self._testMethodName, test_file)
            subset.subset(
                file_to_subset=join(self.test_data_dir, test_file),
                bbox=np.array(bbox),
                output_file=join(self.subset_output_dir, output_file),
                cut=cut
            )

            j_ds = xr.open_dataset(join(self.test_data_dir, java_file),
                                   decode_times=False,
                                   decode_coords=False,
                                   mask_and_scale=False)

            py_ds = xr.open_dataset(join(self.subset_output_dir, output_file),
                                    decode_times=False,
                                    decode_coords=False,
                                    mask_and_scale=False)

            for var_name, var in j_ds.data_vars.items():
                # Compare shape
                np.testing.assert_equal(var.shape, py_ds[var_name].shape)

                # Compare meta
                np.testing.assert_equal(var.attrs, py_ds[var_name].attrs)

                # Compare data
                np.testing.assert_equal(var.values, py_ds[var_name].values)

            # Compare meta. History will always be different, so remove
            # from the headers for comparison.
            del j_ds.attrs['history']
            del py_ds.attrs['history']
            del py_ds.attrs['history_json']
            np.testing.assert_equal(j_ds.attrs, py_ds.attrs)

    def test_compare_java_compact(self):
        """
        Tests that the results of the subsetting operation is
        equivalent to the Java subsetting result on the same bounding
        box. For simplicity the subsetted Java granules have been
        manually run and copied into this project. This test DOES
        cut the scanline.
        """

        java_result_files = [join("java_results", "cut", f) for f in
                             listdir(join(self.test_data_dir, "java_results", "cut")) if
                             isfile(join(self.test_data_dir, "java_results", "cut", f))
                             and f.endswith(".nc")]

        self.compare_java(java_result_files, cut=True)

    def test_compare_java(self):
        """
        Tests that the results of the subsetting operation is
        equivalent to the Java subsetting result on the same bounding
        box. For simplicity the subsetted Java granules have been
        manually run and copied into this project. This runs does NOT
        cut the scanline.
        """

        java_result_files = [join("java_results", "uncut", f) for f in
                             listdir(join(self.test_data_dir, "java_results", "uncut")) if
                             isfile(join(self.test_data_dir, "java_results", "uncut", f))
                             and f.endswith(".nc")]

        self.compare_java(java_result_files, cut=False)

    def test_history_metadata_append(self):
        """
        Tests that the history metadata header is appended to when it
        already exists.
        """
        test_file = next(filter(
            lambda f: '20180101005944-REMSS-L2P_GHRSST-SSTsubskin-AMSR2-L2B_rt_r29918-v02.0-fv01.0.nc' in f
            , self.test_files))
        output_file = "{}_{}".format(self._testMethodName, test_file)
        subset.subset(
            file_to_subset=join(self.test_data_dir, test_file),
            bbox=np.array(((-180, 180), (-90.0, 90))),
            output_file=join(self.subset_output_dir, output_file)
        )

        in_nc = xr.open_dataset(join(self.test_data_dir, test_file))
        out_nc = xr.open_dataset(join(self.subset_output_dir, output_file))

        # Assert that the original granule contains history
        assert in_nc.attrs.get('history') is not None

        # Assert that input and output files have different history
        self.assertNotEqual(in_nc.attrs['history'], out_nc.attrs['history'])

        # Assert that last line of history was created by this service
        assert SERVICE_NAME in out_nc.attrs['history'].split('\n')[-1]

        # Assert that the old history is still in the subsetted granule
        assert in_nc.attrs['history'] in out_nc.attrs['history']

    def test_history_metadata_create(self):
        """
        Tests that the history metadata header is created when it does
        not exist. All test granules contain this header already, so
        for this test the header will be removed manually from a granule.
        """
        test_file = next(filter(
            lambda f: '20180101005944-REMSS-L2P_GHRSST-SSTsubskin-AMSR2-L2B_rt_r29918-v02.0-fv01.0.nc' in f
            , self.test_files))
        output_file = "{}_{}".format(self._testMethodName, test_file)

        # Remove the 'history' metadata from the granule
        in_nc = xr.open_dataset(join(self.test_data_dir, test_file))
        del in_nc.attrs['history']
        in_nc.to_netcdf(join(self.subset_output_dir, 'int_{}'.format(output_file)), 'w')

        subset.subset(
            file_to_subset=join(self.subset_output_dir, "int_{}".format(output_file)),
            bbox=np.array(((-180, 180), (-90.0, 90))),
            output_file=join(self.subset_output_dir, output_file)
        )

        out_nc = xr.open_dataset(join(self.subset_output_dir, output_file))

        # Assert that the input granule contains no history
        assert in_nc.attrs.get('history') is None

        # Assert that the history was created by this service
        assert SERVICE_NAME in out_nc.attrs['history']

        # Assert that the history created by this service is the only
        # line present in the history.
        assert '\n' not in out_nc.attrs['history']

    def test_specified_variables(self):
        """
        Test that the variables which are specified when calling the subset
        operation are present in the resulting subsetted data file,
        and that the variables which are specified are not present.
        """
        bbox = np.array(((-180, 180), (-90, 90)))
        for file in self.test_files:
            output_file = "{}_{}".format(self._testMethodName, file)

            in_ds = xr.open_dataset(join(self.test_data_dir, file),
                                    decode_times=False,
                                    decode_coords=False)

            included_variables = set([variable[0] for variable in in_ds.data_vars.items()][::2])
            included_variables = list(included_variables)

            excluded_variables = list(set(variable[0] for variable in in_ds.data_vars.items())
                                      - set(included_variables))

            subset.subset(
                file_to_subset=join(self.test_data_dir, file),
                bbox=bbox,
                output_file=join(self.subset_output_dir, output_file),
                variables=included_variables
            )

            # Get coord variables
            time_var_name = []
            lat_var_names, lon_var_names = subset.compute_coordinate_variable_names(in_ds)
            lat_var_name = lat_var_names[0]
            lon_var_name = lon_var_names[0]
            time_var_name = subset.compute_time_variable_name(in_ds, in_ds[lat_var_name])

            included_variables.append(lat_var_name)
            included_variables.append(lon_var_name)
            included_variables.append(time_var_name)
            included_variables.extend(in_ds.coords.keys())

            if lat_var_name in excluded_variables:
                excluded_variables.remove(lat_var_name)
            if lon_var_name in excluded_variables:
                excluded_variables.remove(lon_var_name)
            if time_var_name in excluded_variables:
                excluded_variables.remove(time_var_name)

            out_ds = xr.open_dataset(join(self.subset_output_dir, output_file),
                                     decode_times=False,
                                     decode_coords=False)

            out_vars = [out_var for out_var in out_ds.data_vars.keys()]
            out_vars.extend(out_ds.coords.keys())

            assert set(out_vars) == set(included_variables)
            assert set(out_vars).isdisjoint(excluded_variables)

            in_ds.close()
            out_ds.close()

    def test_calculate_chunks(self):
        """
        Test that the calculate chunks function in the subset module
        correctly calculates and returns the chunks dims dictionary.
        """
        rs = np.random.RandomState(0)
        dataset = xr.DataArray(
            rs.randn(2, 4000, 4001),
            dims=['x', 'y', 'z']
        ).to_dataset(name='foo')

        chunk_dict = subset.calculate_chunks(dataset)

        assert chunk_dict.get('x') is None
        assert chunk_dict.get('y') is None
        assert chunk_dict.get('z') == 4000

    def test_missing_coord_vars(self):
        """
        As of right now, the subsetter expects the data to contain lat
        and lon variables. If not present, an error is thrown.
        """
        file = 'MODIS_T-JPL-L2P-v2014.0.nc'
        ds = xr.open_dataset(join(self.test_data_dir, file),
                             decode_times=False,
                             decode_coords=False,
                             mask_and_scale=False)

        # Manually remove var which will cause error when attempting
        # to subset.
        ds = ds.drop_vars(['lat'])

        output_file = '{}_{}'.format('missing_coords', file)
        ds.to_netcdf(join(self.subset_output_dir, output_file))

        bbox = np.array(((-180, 180), (-90, 90)))

        with pytest.raises(ValueError):
            subset.subset(
                file_to_subset=join(self.subset_output_dir, output_file),
                bbox=bbox,
                output_file=''
            )

    def test_data_1D(self):
        """
        Test that subsetting a 1-D granule does not result in failure.
        """
        merged_jason_filename = 'JA1_GPN_2PeP001_002_20020115_060706_20020115_070316.nc'
        output_file = "{}_{}".format(self._testMethodName, merged_jason_filename)

        subset.subset(
            file_to_subset=join(self.test_data_dir, merged_jason_filename),
            bbox=np.array(((-180, 0), (-90, 0))),
            output_file=join(self.subset_output_dir, output_file)
        )

        xr.open_dataset(join(self.subset_output_dir, output_file))

    def test_get_coord_variable_names(self):
        """
        Test that the expected coord variable names are returned
        """
        file = 'MODIS_T-JPL-L2P-v2014.0.nc'
        ds = xr.open_dataset(join(self.test_data_dir, file),
                             decode_times=False,
                             decode_coords=False,
                             mask_and_scale=False)

        old_lat_var_name = 'lat'
        old_lon_var_name = 'lon'

        lat_var_name, lon_var_name = subset.compute_coordinate_variable_names(ds)

        assert lat_var_name[0] == old_lat_var_name
        assert lon_var_name[0] == old_lon_var_name

        new_lat_var_name = 'latitude'
        new_lon_var_name = 'x'
        ds = ds.rename({old_lat_var_name: new_lat_var_name,
                        old_lon_var_name: new_lon_var_name})

        lat_var_name, lon_var_name = subset.compute_coordinate_variable_names(ds)

        assert lat_var_name[0] == new_lat_var_name
        assert lon_var_name[0] == new_lon_var_name

    def test_cannot_get_coord_variable_names(self):
        """
        Test that, when given a dataset with coord vars which are not
        expected, a ValueError is raised.
        """
        file = 'MODIS_T-JPL-L2P-v2014.0.nc'
        ds = xr.open_dataset(join(self.test_data_dir, file),
                             decode_times=False,
                             decode_coords=False,
                             mask_and_scale=False)

        old_lat_var_name = 'lat'
        new_lat_var_name = 'foo'

        ds = ds.rename({old_lat_var_name: new_lat_var_name})
        # Remove 'coordinates' attribute
        for var_name, var in ds.items():
            if 'coordinates' in var.attrs:
                del var.attrs['coordinates']

        self.assertRaises(ValueError, subset.compute_coordinate_variable_names, ds)

    def test_get_spatial_bounds(self):
        """
        Test that the get_spatial_bounds function works as expected.
        The get_spatial_bounds function should return lat/lon min/max
        which is masked and scaled for both variables. The values
        should also be adjusted for -180,180/-90,90 coordinate types
        """
        ascat_filename = 'ascat_20150702_084200_metopa_45145_eps_o_250_2300_ovw.l2.nc'
        ghrsst_filename = '20190927000500-JPL-L2P_GHRSST-SSTskin-MODIS_A-D-v02.0-fv01.0.nc'

        ascat_dataset = xr.open_dataset(
            join(self.test_data_dir, ascat_filename),
            decode_times=False,
            decode_coords=False,
            mask_and_scale=False
        )
        ghrsst_dataset = xr.open_dataset(
            join(self.test_data_dir, ghrsst_filename),
            decode_times=False,
            decode_coords=False,
            mask_and_scale=False
        )

        # ascat1 longitude is -0 360, ghrsst modis A is -180 180
        # Both have metadata for valid_min

        # Manually calculated spatial bounds
        ascat_expected_lat_min = -89.4
        ascat_expected_lat_max = 89.2
        ascat_expected_lon_min = -180.0
        ascat_expected_lon_max = 180.0

        ghrsst_expected_lat_min = -77.2
        ghrsst_expected_lat_max = -53.6
        ghrsst_expected_lon_min = -170.5
        ghrsst_expected_lon_max = -101.7

        min_lon, max_lon, min_lat, max_lat = subset.get_spatial_bounds(
            dataset=ascat_dataset,
            lat_var_names=['lat'],
            lon_var_names=['lon']
        ).flatten()

        assert np.isclose(min_lat, ascat_expected_lat_min)
        assert np.isclose(max_lat, ascat_expected_lat_max)
        assert np.isclose(min_lon, ascat_expected_lon_min)
        assert np.isclose(max_lon, ascat_expected_lon_max)

        # Remove the label from the dataset coordinate variables indicating the valid_min.
        del ascat_dataset['lat'].attrs['valid_min']
        del ascat_dataset['lon'].attrs['valid_min']

        min_lon, max_lon, min_lat, max_lat = subset.get_spatial_bounds(
            dataset=ascat_dataset,
            lat_var_names=['lat'],
            lon_var_names=['lon']
        ).flatten()

        assert np.isclose(min_lat, ascat_expected_lat_min)
        assert np.isclose(max_lat, ascat_expected_lat_max)
        assert np.isclose(min_lon, ascat_expected_lon_min)
        assert np.isclose(max_lon, ascat_expected_lon_max)

        # Repeat test, but with GHRSST granule

        min_lon, max_lon, min_lat, max_lat = subset.get_spatial_bounds(
            dataset=ghrsst_dataset,
            lat_var_names=['lat'],
            lon_var_names=['lon']
        ).flatten()

        assert np.isclose(min_lat, ghrsst_expected_lat_min)
        assert np.isclose(max_lat, ghrsst_expected_lat_max)
        assert np.isclose(min_lon, ghrsst_expected_lon_min)
        assert np.isclose(max_lon, ghrsst_expected_lon_max)

        # Remove the label from the dataset coordinate variables indicating the valid_min.

        del ghrsst_dataset['lat'].attrs['valid_min']
        del ghrsst_dataset['lon'].attrs['valid_min']

        min_lon, max_lon, min_lat, max_lat = subset.get_spatial_bounds(
            dataset=ghrsst_dataset,
            lat_var_names=['lat'],
            lon_var_names=['lon']
        ).flatten()

        assert np.isclose(min_lat, ghrsst_expected_lat_min)
        assert np.isclose(max_lat, ghrsst_expected_lat_max)
        assert np.isclose(min_lon, ghrsst_expected_lon_min)
        assert np.isclose(max_lon, ghrsst_expected_lon_max)

    def test_shapefile_subset(self):
        """
        Test that using a shapefile to subset data instead of a bbox
        works as expected
        """
        shapefile = 'test.shp'
        ascat_filename = 'ascat_20150702_084200_metopa_45145_eps_o_250_2300_ovw.l2.nc'
        output_filename = f'{self._testMethodName}_{ascat_filename}'

        shapefile_file_path = join(self.test_data_dir, 'test_shapefile_subset', shapefile)
        ascat_file_path = join(self.test_data_dir, ascat_filename)
        output_file_path = join(self.subset_output_dir, output_filename)

        subset.subset(
            file_to_subset=ascat_file_path,
            bbox=None,
            output_file=output_file_path,
            shapefile=shapefile_file_path
        )

        # Check that each point of data is within the shapefile
        shapefile_df = gpd.read_file(shapefile_file_path)
        with xr.open_dataset(output_file_path) as result_dataset:
            def in_shape(lon, lat):
                if np.isnan(lon) or np.isnan(lat):
                    return
                point = Point(lon, lat)
                point_in_shapefile = shapefile_df.contains(point)
                assert point_in_shapefile[0]

            in_shape_vec = np.vectorize(in_shape)
            in_shape_vec(result_dataset.lon, result_dataset.lat)

    def test_variable_subset_oco2(self):
        """
        variable subsets for groups and root group using a '/'
        """

        oco2_file_name = 'oco2_LtCO2_190201_B10206Ar_200729175909s.nc4'
        output_file_name = 'oco2_test_out.nc'
        shutil.copyfile(os.path.join(self.test_data_dir, 'OCO2', oco2_file_name),
                        os.path.join(self.subset_output_dir, oco2_file_name))
        bbox = np.array(((-180,180),(-90.0,90)))
        variables = ['/xco2','/xco2_quality_flag','/Retrieval/water_height','/sounding_id']
        subset.subset(
            file_to_subset=join(self.test_data_dir, 'OCO2',oco2_file_name),
            bbox=bbox,
            variables=variables,
            output_file=join(self.subset_output_dir, output_file_name),
        )
        
        out_nc = nc.Dataset(join(self.subset_output_dir, output_file_name))
        var_listout = list(out_nc.groups['Retrieval'].variables.keys())
        assert ('water_height' in var_listout)

    def test_var_subsetting_tropomi(self):
        """
        Check that variable subsetting is the same if a leading slash is included
        """
        TROP_dir = join(self.test_data_dir, 'tropomi')
        trop_file = 'S5P_OFFL_L2__CH4____20190319T110835_20190319T125006_07407_01_010202_20190325T125810_subset.nc4'
        variable_slash = ['/PRODUCT/methane_mixing_ratio']
        variable_noslash = ['PRODUCT/methane_mixing_ratio']
        bbox = np.array(((-180, 180), (-90, 90)))
        output_file_slash = "{}_{}".format(self._testMethodName, trop_file)
        output_file_noslash = "{}_noslash_{}".format(self._testMethodName, trop_file)
        shutil.copyfile(
            os.path.join(TROP_dir, trop_file),
            os.path.join(self.subset_output_dir, trop_file)
        )
        shutil.copyfile(
            os.path.join(TROP_dir, trop_file),
            os.path.join(self.subset_output_dir,'slashtest'+trop_file)
        )
        slash_test = subset.subset(
            file_to_subset=join(self.subset_output_dir, trop_file),
            bbox=bbox,
            output_file=join(self.subset_output_dir, output_file_slash),
            variables = variable_slash
        )
        noslash_test = subset.subset(
            file_to_subset=join(self.subset_output_dir, 'slashtest'+trop_file),
            bbox=bbox,
            output_file=join(self.subset_output_dir, output_file_noslash),
            variables = variable_noslash
        )

        slash_dataset = nc.Dataset(join(self.subset_output_dir, output_file_slash))
        noslash_dataset = nc.Dataset(join(self.subset_output_dir, output_file_noslash))

        assert list(slash_dataset.groups['PRODUCT'].variables) == list(noslash_dataset.groups['PRODUCT'].variables)

