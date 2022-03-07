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

            lat_var_name, lon_var_name = subset.get_coord_variable_names(out_ds)

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
            lat_var_names, lon_var_names = subset.get_coord_variable_names(in_ds)
            lat_var_name = lat_var_names[0]
            lon_var_name = lon_var_names[0]
            time_var_name = subset.get_time_variable_name(in_ds, in_ds[lat_var_name])

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

        lat_var_name, lon_var_name = subset.get_coord_variable_names(ds)

        assert lat_var_name[0] == old_lat_var_name
        assert lon_var_name[0] == old_lon_var_name

        new_lat_var_name = 'latitude'
        new_lon_var_name = 'x'
        ds = ds.rename({old_lat_var_name: new_lat_var_name,
                        old_lon_var_name: new_lon_var_name})

        lat_var_name, lon_var_name = subset.get_coord_variable_names(ds)

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

        self.assertRaises(ValueError, subset.get_coord_variable_names, ds)

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

    def test_variable_subset_oco3(self):
        """
        multiple variable subset of variables in different groups in oco3
        """

        oco3_file_name = 'oco3_LtSIF_200226_B10206r_200709053505s.nc4'
        output_file_name = 'oco3_test_out.nc'
        shutil.copyfile(os.path.join(self.test_data_dir, 'OCO3/OCO3_L2_LITE_SIF.EarlyR', oco3_file_name),
                        os.path.join(self.subset_output_dir, oco3_file_name))
        bbox = np.array(((-180,180),(-90.0,90)))
        variables = ['/Science/IGBP_index', '/Offset/SIF_Relative_SDev_757nm','/Meteo/temperature_skin']
        subset.subset(
            file_to_subset=join(self.subset_output_dir, oco3_file_name),
            bbox=bbox,
            variables=variables,
            output_file=join(self.subset_output_dir, output_file_name),
        )
        
        out_nc = nc.Dataset(join(self.subset_output_dir, output_file_name))
        var_listout =list(out_nc.groups['Science'].variables.keys())
        var_listout.extend(list(out_nc.groups['Offset'].variables.keys()))
        var_listout.extend(list(out_nc.groups['Meteo'].variables.keys()))
        assert ('IGBP_index' in var_listout)
        assert ('SIF_Relative_SDev_757nm' in var_listout)
        assert ('temperature_skin' in var_listout)

    def test_variable_subset_s6(self):
        """
        multiple variable subset of variables in different groups in oco3
        """

        s6_file_name = 'S6A_P4_2__LR_STD__ST_002_140_20201207T011501_20201207T013023_F00.nc'
        output_file_name = 's6_test_out.nc'
        shutil.copyfile(os.path.join(self.test_data_dir, 'sentinel_6', s6_file_name),
                        os.path.join(self.subset_output_dir, s6_file_name))
        bbox = np.array(((-180,180),(-90.0,90)))
        variables = ['/data_01/ku/range_ocean_mle3_rms', '/data_20/ku/range_ocean']
        subset.subset(
            file_to_subset=join(self.subset_output_dir, s6_file_name),
            bbox=bbox,
            variables=variables,
            output_file=join(self.subset_output_dir, output_file_name),
        )
        
        out_nc = nc.Dataset(join(self.subset_output_dir, output_file_name))
        var_listout =list(out_nc.groups['data_01'].groups['ku'].variables.keys())
        var_listout.extend(list(out_nc.groups['data_20'].groups['ku'].variables.keys()))
        assert ('range_ocean_mle3_rms' in var_listout)
        assert ('range_ocean' in var_listout)


    def test_transform_grouped_dataset(self):
        """
        Test that the transformation function results in a correctly
        formatted dataset.
        """
        s6_file_name = 'S6A_P4_2__LR_STD__ST_002_140_20201207T011501_20201207T013023_F00.nc'
        shutil.copyfile(os.path.join(self.test_data_dir, 'sentinel_6', s6_file_name),
                        os.path.join(self.subset_output_dir, s6_file_name))

        nc_ds = nc.Dataset(os.path.join(self.test_data_dir, 'sentinel_6', s6_file_name))
        nc_ds_transformed = subset.transform_grouped_dataset(
            nc.Dataset(os.path.join(self.subset_output_dir, s6_file_name), 'r'),
            os.path.join(self.subset_output_dir, s6_file_name)
        )

        # The original ds has groups
        assert nc_ds.groups

        # There should be no groups in the new ds
        assert not nc_ds_transformed.groups

        # The original ds has no variables in the root group
        assert not nc_ds.variables

        # The new ds has variables in the root group
        assert nc_ds_transformed.variables

        # Each var in the new ds should map to a variable in the old ds
        for var_name, var in nc_ds_transformed.variables.items():
            path = var_name.strip('__').split('__')

            group = nc_ds[path[0]]
            for g in path[1:-1]:
                group = group[g]
            assert var_name.strip('__').split('__')[-1] in group.variables.keys()


    def test_group_subset(self):
        """
        Ensure a subset function can be run on a granule that contains
        groups without errors, and that the subsetted data is within
        the given spatial bounds.
        """
        s6_file_name = 'S6A_P4_2__LR_STD__ST_002_140_20201207T011501_20201207T013023_F00.nc'
        s6_output_file_name = 'SS_S6A_P4_2__LR_STD__ST_002_140_20201207T011501_20201207T013023_F00.nc'
        # Copy S6 file to temp dir
        shutil.copyfile(
            os.path.join(self.test_data_dir, 'sentinel_6', s6_file_name),
            os.path.join(self.subset_output_dir, s6_file_name)
        )

        # Make sure it runs without errors
        bbox = np.array(((150, 180), (-90, -50)))
        bounds = subset.subset(
            file_to_subset=os.path.join(self.subset_output_dir, s6_file_name),
            bbox=bbox,
            output_file=os.path.join(self.subset_output_dir, s6_output_file_name)
        )

        # Check that bounds are within requested bbox
        assert bounds[0][0] >= bbox[0][0]
        assert bounds[0][1] <= bbox[0][1]
        assert bounds[1][0] >= bbox[1][0]
        assert bounds[1][1] <= bbox[1][1]

    def test_json_history_metadata_append(self):
        """
        Tests that the json history metadata header is appended to when it
        already exists. First we create a fake json_history header for input file.
        """
        test_file = next(filter(
            lambda f: '20180101005944-REMSS-L2P_GHRSST-SSTsubskin-AMSR2-L2B_rt_r29918-v02.0-fv01.0.nc' in f
            , self.test_files))
        output_file = "{}_{}".format(self._testMethodName, test_file)
        input_file_subset = join(self.subset_output_dir, "int_{}".format(output_file))

        fake_history = [
            {
                "date_time": "2021-05-10T14:30:24.553263",
                "derived_from": basename(input_file_subset),
                "program": SERVICE_NAME,
                "version": importlib_metadata.distribution(SERVICE_NAME).version,
                "parameters": "bbox=[[-180.0, 180.0], [-90.0, 90.0]] cut=True",
                "program_ref": "https://cmr.earthdata.nasa.gov:443/search/concepts/S1962070864-POCLOUD",
                "$schema": "https://harmony.earthdata.nasa.gov/schemas/history/0.1.0/history-v0.1.0.json"
            }
        ]

        in_nc = xr.open_dataset(join(self.test_data_dir, test_file))
        in_nc.attrs['history_json'] = json.dumps(fake_history)
        in_nc.to_netcdf(join(self.subset_output_dir, 'int_{}'.format(output_file)), 'w')

        subset.subset(
            file_to_subset=input_file_subset,
            bbox=np.array(((-180, 180), (-90.0, 90))),
            output_file=join(self.subset_output_dir, output_file)
        )

        out_nc = xr.open_dataset(join(self.subset_output_dir, output_file))

        history_json = json.loads(out_nc.attrs['history_json'])
        assert len(history_json) == 2

        is_valid_shema = validate(instance=history_json, schema=self.history_json_schema)
        assert is_valid_shema is None

        for history in history_json:
            assert "date_time" in history
            assert history.get('program') == SERVICE_NAME
            assert history.get('derived_from') == basename(input_file_subset)
            assert history.get('version') == importlib_metadata.distribution(SERVICE_NAME).version
            assert history.get('parameters') == 'bbox=[[-180.0, 180.0], [-90.0, 90.0]] cut=True'
            assert history.get(
                'program_ref') == "https://cmr.earthdata.nasa.gov:443/search/concepts/S1962070864-POCLOUD"
            assert history.get(
                '$schema') == "https://harmony.earthdata.nasa.gov/schemas/history/0.1.0/history-v0.1.0.json"

    def test_json_history_metadata_create(self):
        """
        Tests that the json history metadata header is created when it does
        not exist. All test granules does not contain this header.
        """
        test_file = next(filter(
            lambda f: '20180101005944-REMSS-L2P_GHRSST-SSTsubskin-AMSR2-L2B_rt_r29918-v02.0-fv01.0.nc' in f
            , self.test_files))
        output_file = "{}_{}".format(self._testMethodName, test_file)

        # Remove the 'history' metadata from the granule
        in_nc = xr.open_dataset(join(self.test_data_dir, test_file))
        in_nc.to_netcdf(join(self.subset_output_dir, 'int_{}'.format(output_file)), 'w')

        input_file_subset = join(self.subset_output_dir, "int_{}".format(output_file))
        subset.subset(
            file_to_subset=input_file_subset,
            bbox=np.array(((-180, 180), (-90.0, 90))),
            output_file=join(self.subset_output_dir, output_file)
        )

        out_nc = xr.open_dataset(join(self.subset_output_dir, output_file))

        history_json = json.loads(out_nc.attrs['history_json'])
        assert len(history_json) == 1

        is_valid_shema = validate(instance=history_json, schema=self.history_json_schema)
        assert is_valid_shema is None

        for history in history_json:
            assert "date_time" in history
            assert history.get('program') == SERVICE_NAME
            assert history.get('derived_from') == basename(input_file_subset)
            assert history.get('version') == importlib_metadata.distribution(SERVICE_NAME).version
            assert history.get('parameters') == 'bbox=[[-180.0, 180.0], [-90.0, 90.0]] cut=True'
            assert history.get(
                'program_ref') == "https://cmr.earthdata.nasa.gov:443/search/concepts/S1962070864-POCLOUD"
            assert history.get(
                '$schema') == "https://harmony.earthdata.nasa.gov/schemas/history/0.1.0/history-v0.1.0.json"

    def test_json_history_metadata_create_origin_source(self):
        """
        Tests that the json history metadata header is created when it does
        not exist. All test granules does not contain this header.
        """
        test_file = next(filter(
            lambda f: '20180101005944-REMSS-L2P_GHRSST-SSTsubskin-AMSR2-L2B_rt_r29918-v02.0-fv01.0.nc' in f
            , self.test_files))
        output_file = "{}_{}".format(self._testMethodName, test_file)

        # Remove the 'history' metadata from the granule
        in_nc = xr.open_dataset(join(self.test_data_dir, test_file))
        in_nc.to_netcdf(join(self.subset_output_dir, 'int_{}'.format(output_file)), 'w')

        input_file_subset = join(self.subset_output_dir, "int_{}".format(output_file))
        subset.subset(
            file_to_subset=input_file_subset,
            bbox=np.array(((-180, 180), (-90.0, 90))),
            output_file=join(self.subset_output_dir, output_file),
            origin_source="fake_original_file.nc"
        )

        out_nc = xr.open_dataset(join(self.subset_output_dir, output_file))

        history_json = json.loads(out_nc.attrs['history_json'])
        assert len(history_json) == 1

        is_valid_shema = validate(instance=history_json, schema=self.history_json_schema)
        assert is_valid_shema is None

        for history in history_json:
            assert "date_time" in history
            assert history.get('program') == SERVICE_NAME
            assert history.get('derived_from') == "fake_original_file.nc"
            assert history.get('version') == importlib_metadata.distribution(SERVICE_NAME).version
            assert history.get('parameters') == 'bbox=[[-180.0, 180.0], [-90.0, 90.0]] cut=True'
            assert history.get(
                'program_ref') == "https://cmr.earthdata.nasa.gov:443/search/concepts/S1962070864-POCLOUD"
            assert history.get(
                '$schema') == "https://harmony.earthdata.nasa.gov/schemas/history/0.1.0/history-v0.1.0.json"

    def test_temporal_subset_ascat(self):
        """
        Test that a temporal subset results in a granule that only
        contains times within the given bounds.
        """
        bbox = np.array(((-180, 180), (-90, 90)))
        file = 'ascat_20150702_084200_metopa_45145_eps_o_250_2300_ovw.l2.nc'
        output_file = "{}_{}".format(self._testMethodName, file)
        min_time = '2015-07-02T09:00:00'
        max_time = '2015-07-02T10:00:00'

        subset.subset(
            file_to_subset=join(self.test_data_dir, file),
            bbox=bbox,
            output_file=join(self.subset_output_dir, output_file),
            min_time=min_time,
            max_time=max_time
        )

        in_ds = xr.open_dataset(join(self.test_data_dir, file),
                                decode_times=False,
                                decode_coords=False)

        out_ds = xr.open_dataset(join(self.subset_output_dir, output_file),
                                 decode_times=False,
                                 decode_coords=False)

        # Check that 'time' types match
        assert in_ds.time.dtype == out_ds.time.dtype

        in_ds.close()
        out_ds.close()

        # Check that all times are within the given bounds. Open
        # dataset using 'decode_times=True' for auto-conversions to
        # datetime
        out_ds = xr.open_dataset(join(self.subset_output_dir, output_file),
                                 decode_coords=False)

        start_dt = subset.translate_timestamp(min_time)
        end_dt = subset.translate_timestamp(max_time)

        # All dates should be within the given temporal bounds.
        assert (out_ds.time >= pd.to_datetime(start_dt)).all()
        assert (out_ds.time <= pd.to_datetime(end_dt)).all()

    def test_temporal_subset_modis_a(self):
        """
        Test that a temporal subset results in a granule that only
        contains times within the given bounds.
        """
        bbox = np.array(((-180, 180), (-90, 90)))
        file = 'MODIS_A-JPL-L2P-v2014.0.nc'
        output_file = "{}_{}".format(self._testMethodName, file)
        min_time = '2019-08-05T06:57:00'
        max_time = '2019-08-05T06:58:00'
        # Actual min is 2019-08-05T06:55:01.000000000
        # Actual max is 2019-08-05T06:59:57.000000000

        subset.subset(
            file_to_subset=join(self.test_data_dir, file),
            bbox=bbox,
            output_file=join(self.subset_output_dir, output_file),
            min_time=min_time,
            max_time=max_time
        )

        in_ds = xr.open_dataset(join(self.test_data_dir, file),
                                decode_times=False,
                                decode_coords=False)

        out_ds = xr.open_dataset(join(self.subset_output_dir, output_file),
                                 decode_times=False,
                                 decode_coords=False)

        # Check that 'time' types match
        assert in_ds.time.dtype == out_ds.time.dtype

        in_ds.close()
        out_ds.close()

        # Check that all times are within the given bounds. Open
        # dataset using 'decode_times=True' for auto-conversions to
        # datetime
        out_ds = xr.open_dataset(join(self.subset_output_dir, output_file),
                                 decode_coords=False)

        start_dt = subset.translate_timestamp(min_time)
        end_dt = subset.translate_timestamp(max_time)

        epoch_dt = out_ds['time'].values[0]

        # All timedelta + epoch should be within the given temporal bounds.
        assert out_ds.sst_dtime.min() + epoch_dt >= np.datetime64(start_dt)
        assert out_ds.sst_dtime.min() + epoch_dt <= np.datetime64(end_dt)

    def test_temporal_subset_s6(self):
        """
        Test that a temporal subset results in a granule that only
        contains times within the given bounds.
        """
        bbox = np.array(((-180, 180), (-90, 90)))
        file = 'S6A_P4_2__LR_STD__ST_002_140_20201207T011501_20201207T013023_F00.nc'
        # Copy S6 file to temp dir
        shutil.copyfile(
            os.path.join(self.test_data_dir, 'sentinel_6', file),
            os.path.join(self.subset_output_dir, file)
        )
        output_file = "{}_{}".format(self._testMethodName, file)
        min_time = '2020-12-07T01:20:00'
        max_time = '2020-12-07T01:25:00'
        # Actual min is 2020-12-07T01:15:01.000000000
        # Actual max is 2020-12-07T01:30:23.000000000

        subset.subset(
            file_to_subset=join(self.subset_output_dir, file),
            bbox=bbox,
            output_file=join(self.subset_output_dir, output_file),
            min_time=min_time,
            max_time=max_time
        )

        # Check that all times are within the given bounds. Open
        # dataset using 'decode_times=True' for auto-conversions to
        # datetime
        out_ds = xr.open_dataset(
            join(self.subset_output_dir, output_file),
            decode_coords=False,
            group='data_01'
        )

        start_dt = subset.translate_timestamp(min_time)
        end_dt = subset.translate_timestamp(max_time)

        # All dates should be within the given temporal bounds.
        assert (out_ds.time >= pd.to_datetime(start_dt)).all()
        assert (out_ds.time <= pd.to_datetime(end_dt)).all()

    def test_get_time_variable_name(self):
        for test_file in self.test_files:
            args = {
                'decode_coords': False,
                'mask_and_scale': False,
                'decode_times': True
            }

            ds = xr.open_dataset(os.path.join(self.test_data_dir, test_file), **args)
            lat_var_name = subset.get_coord_variable_names(ds)[0][0]
            time_var_name = subset.get_time_variable_name(ds, ds[lat_var_name])
            assert time_var_name is not None
            assert 'time' in time_var_name

    def test_subset_jason(self):
        bbox = np.array(((-180, 0), (-90, 90)))
        file = 'JA1_GPN_2PeP001_002_20020115_060706_20020115_070316.nc'
        output_file = "{}_{}".format(self._testMethodName, file)
        min_time = "2002-01-15T06:07:06Z"
        max_time = "2002-01-15T06:30:16Z"

        subset.subset(
            file_to_subset=os.path.join(self.test_data_dir, file),
            bbox=bbox,
            min_time=min_time,
            max_time=max_time,
            output_file=os.path.join(self.subset_output_dir, output_file)
        )

    def test_subset_size(self):

        for file in self.test_files:
            bbox = np.array(((-180, 0), (-30, 90)))
            output_file = "{}_{}".format(self._testMethodName, file)
            input_file_path = os.path.join(self.test_data_dir, file)
            output_file_path = os.path.join(self.subset_output_dir, output_file)

            subset.subset(
                file_to_subset=input_file_path,
                bbox=bbox,
                output_file=output_file_path
            )

            original_file_size = os.path.getsize(input_file_path)
            subset_file_size = os.path.getsize(output_file_path)

            assert subset_file_size < original_file_size

    def test_duplicate_dims_sndr(self):
        """
        Check if SNDR Climcaps files run successfully even though
        these files have variables with duplicate dimensions
        """
        SNDR_dir = join(self.test_data_dir, 'SNDR')
        sndr_files = [f for f in listdir(SNDR_dir)
                          if isfile(join(SNDR_dir, f)) and f.endswith(".nc")]

        bbox = np.array(((-180, 90), (-90, 90)))
        for file in sndr_files:
            output_file = "{}_{}".format(self._testMethodName, file)
            shutil.copyfile(
                os.path.join(SNDR_dir, file),
                os.path.join(self.subset_output_dir, file)
            )
            box_test = subset.subset(
                file_to_subset=join(self.subset_output_dir, file),
                bbox=bbox,
                output_file=join(self.subset_output_dir, output_file),
            )
            # check if the box_test is
            assert len(box_test)==2

    def test_root_group(self):
        """test that the GROUP_DELIM string, '__', is added to variables in the root group"""

        sndr_file_name = 'SNDR.SNPP.CRIMSS.20200118T0024.m06.g005.L2_CLIMCAPS_RET.std.v02_28.G.200314032326_subset.nc'
        shutil.copyfile(os.path.join(self.test_data_dir, 'SNDR', sndr_file_name),
                        os.path.join(self.subset_output_dir, sndr_file_name))

        nc_dataset = nc.Dataset(os.path.join(self.subset_output_dir, sndr_file_name))

        args = {
                'decode_coords': False,
                'mask_and_scale': False,
                'decode_times': False
            }
        nc_dataset = subset.transform_grouped_dataset(nc_dataset, os.path.join(self.subset_output_dir, sndr_file_name))
        with xr.open_dataset(
            xr.backends.NetCDF4DataStore(nc_dataset),
            **args
        ) as dataset:
            var_list = list(dataset.variables)
            assert (var_list[0][0:2] == subset.GROUP_DELIM)
            group_lst = []
            for var_name in dataset.variables.keys(): #need logic if there is data in the top level not in a group
                group_lst.append('/'.join(var_name.split(subset.GROUP_DELIM)[:-1]))
            group_lst = ['/' if group=='' else group for group in group_lst]
            groups = set(group_lst)
            expected_group = {'/mw', '/ave_kern', '/', '/mol_lay', '/aux'}
            assert (groups == expected_group)

    def test_get_time_squeeze(self):
        """test builtin squeeze method on the lat and time variables so 
        when the two have the same shape with a time and delta time in
        the tropomi product granuales the get_time_variable_name returns delta time as well"""

        tropomi_file_name = 'S5P_OFFL_L2__SO2____20200713T002730_20200713T020900_14239_01_020103_20200721T191355_subset.nc4'
        shutil.copyfile(os.path.join(self.test_data_dir, 'tropomi', tropomi_file_name),
                        os.path.join(self.subset_output_dir, tropomi_file_name))

        nc_dataset = nc.Dataset(os.path.join(self.subset_output_dir, tropomi_file_name))

        args = {
                'decode_coords': False,
                'mask_and_scale': False,
                'decode_times': False
            }
        nc_dataset = subset.transform_grouped_dataset(nc_dataset, os.path.join(self.subset_output_dir, tropomi_file_name))
        with xr.open_dataset(
            xr.backends.NetCDF4DataStore(nc_dataset),
            **args
        ) as dataset:
            lat_var_name = subset.get_coord_variable_names(dataset)[0][0]
            time_var_name = subset.get_time_variable_name(dataset, dataset[lat_var_name])
            lat_dims = dataset[lat_var_name].squeeze().dims
            time_dims = dataset[time_var_name].squeeze().dims
            assert (lat_dims == time_dims)

    def test_get_indexers_nd(self):
        """test that the time coordinate is not included in the indexers. Also test that the dimensions are the same for
           a global box subset"""
        tropomi_file_name = 'S5P_OFFL_L2__SO2____20200713T002730_20200713T020900_14239_01_020103_20200721T191355_subset.nc4'
        shutil.copyfile(os.path.join(self.test_data_dir, 'tropomi', tropomi_file_name),
                        os.path.join(self.subset_output_dir, tropomi_file_name))

        nc_dataset = nc.Dataset(os.path.join(self.subset_output_dir, tropomi_file_name))

        args = {
                'decode_coords': False,
                'mask_and_scale': False,
                'decode_times': False
            }
        nc_dataset = subset.transform_grouped_dataset(nc_dataset, os.path.join(self.subset_output_dir, tropomi_file_name))
        with xr.open_dataset(
            xr.backends.NetCDF4DataStore(nc_dataset),
            **args
        ) as dataset:
            lat_var_name = subset.get_coord_variable_names(dataset)[0][0]
            lon_var_name = subset.get_coord_variable_names(dataset)[1][0]
            time_var_name = subset.get_time_variable_name(dataset, dataset[lat_var_name])
            oper = operator.and_

            cond = oper(
                (dataset[lon_var_name] >= -180),
                (dataset[lon_var_name] <= 180)
                ) & (dataset[lat_var_name] >= -90) & (dataset[lat_var_name] <= 90) & True

            indexers = xre.get_indexers_from_nd(cond, True)
            indexed_cond = cond.isel(**indexers)
            indexed_ds = dataset.isel(**indexers)
            new_dataset = indexed_ds.where(indexed_cond)

            assert ((time_var_name not in indexers.keys()) == True) #time can't be in the index
            assert (new_dataset.dims == dataset.dims)

    def test_variable_type_string_oco2(self):
        """Code passes a ceating a variable that is type object in oco2 file"""

        oco2_file_name = 'oco2_LtCO2_190201_B10206Ar_200729175909s.nc4'
        output_file_name = 'oco2_test_out.nc'
        shutil.copyfile(os.path.join(self.test_data_dir, 'OCO2', oco2_file_name),
                        os.path.join(self.subset_output_dir, oco2_file_name))
        bbox = np.array(((-180,180),(-90.0,90)))

        subset.subset(
            file_to_subset=join(self.test_data_dir, 'OCO2',oco2_file_name),
            bbox=bbox,
            output_file=join(self.subset_output_dir, output_file_name),
        )

        in_nc = xr.open_dataset(join(self.test_data_dir, 'OCO2',oco2_file_name))
        out_nc = xr.open_dataset(join(self.subset_output_dir, output_file_name))
        assert (in_nc.variables['source_files'].dtype == out_nc.variables['source_files'].dtype)

    def test_transform_h5py_dataset(self):
        """
        Test that the transformation function results in a correctly
        formatted dataset for h5py files
        """
        OMI_file_name = 'OMI-Aura_L2-OMSO2_2020m0116t1207-o82471_v003-2020m0223t142939.he5'
        shutil.copyfile(os.path.join(self.test_data_dir, 'OMSO2', OMI_file_name),
                        os.path.join(self.subset_output_dir, OMI_file_name))

        h5_ds = h5py.File(os.path.join(self.test_data_dir, 'OMSO2', OMI_file_name), 'r')

        entry_lst = []
        # Get root level objects
        key_lst = list(h5_ds.keys())
        
        # Go through every level of the file to fill out the remaining objects
        for entry_str in key_lst:
            # If object is a group, add it to the loop list
            if (isinstance(h5_ds[entry_str],h5py.Group)):
                for group_keys in list(h5_ds[entry_str].keys()):
                    if (isinstance(h5_ds[entry_str + "/" + group_keys], h5py.Dataset)):
                        entry_lst.append(entry_str + "/" + group_keys)
                    key_lst.append(entry_str + "/" + group_keys)
        

        nc_dataset = subset.h5file_transform(os.path.join(self.subset_output_dir, OMI_file_name))

        nc_vars_flattened = list(nc_dataset.variables.keys())
        for i in range(len(entry_lst)): # go through all the datasets in h5py file
            input_variable = '__'+entry_lst[i].replace('/', '__')
            output_variable = nc_vars_flattened[i]
            assert (input_variable == output_variable)

        nc_dataset.close()
        h5_ds.close()


    def test_variable_dims_matched_tropomi(self):
        """
        Code must match the dimensions for each variable rather than
        assume all dimensions in a group are the same
        """

        tropomi_file_name = 'S5P_OFFL_L2__SO2____20200713T002730_20200713T020900_14239_01_020103_20200721T191355_subset.nc4'
        output_file_name = 'tropomi_test_out.nc'
        shutil.copyfile(os.path.join(self.test_data_dir, 'tropomi', tropomi_file_name),
                        os.path.join(self.subset_output_dir, tropomi_file_name))

        in_nc = nc.Dataset(os.path.join(self.subset_output_dir, tropomi_file_name))

        # Get variable dimensions from input dataset
        in_var_dims = {
            var_name: [dim.split(subset.GROUP_DELIM)[-1] for dim in var.dimensions]
            for var_name, var in in_nc.groups['PRODUCT'].variables.items()
        }
        
        # Get variables from METADATA group
        in_var_dims.update(
            {
                var_name: [dim.split(subset.GROUP_DELIM)[-1] for dim in var.dimensions]
                for var_name, var in in_nc.groups['METADATA'].groups['QA_STATISTICS'].variables.items()
            }
        )
        # Include PRODUCT>SUPPORT_DATA>GEOLOCATIONS location
        in_var_dims.update(
            {
                var_name: [dim.split(subset.GROUP_DELIM)[-1] for dim in var.dimensions]
                for var_name, var in in_nc.groups['PRODUCT'].groups['SUPPORT_DATA'].groups['GEOLOCATIONS'].variables.items()
            }
        )

        out_nc = subset.transform_grouped_dataset(
            in_nc, os.path.join(self.subset_output_dir, tropomi_file_name)
        )

        # Get variable dimensions from output dataset
        out_var_dims = {
            var_name.split(subset.GROUP_DELIM)[-1]: [dim.split(subset.GROUP_DELIM)[-1] for dim in var.dimensions]
            for var_name, var in out_nc.variables.items()
        }

        self.assertDictEqual(in_var_dims, out_var_dims)


    def test_temporal_merged_topex(self):
        """
        Test that a temporal subset results in a granule that only
        contains times within the given bounds.
        """
        bbox = np.array(((-180, 180), (-90, 90)))
        file = 'Merged_TOPEX_Jason_OSTM_Jason-3_Cycle_002.V4_2.nc'
        # Copy S6 file to temp dir
        shutil.copyfile(
            os.path.join(self.test_data_dir, file),
            os.path.join(self.subset_output_dir, file)
        )
        output_file = "{}_{}".format(self._testMethodName, file)
        min_time = '1992-01-01T00:00:00'
        max_time = '1992-11-01T00:00:00'
        # Actual min is 2020-12-07T01:15:01.000000000
        # Actual max is 2020-12-07T01:30:23.000000000

        subset.subset(
            file_to_subset=join(self.subset_output_dir, file),
            bbox=bbox,
            output_file=join(self.subset_output_dir, output_file),
            min_time=min_time,
            max_time=max_time
        )

        # Check that all times are within the given bounds. Open
        # dataset using 'decode_times=True' for auto-conversions to
        # datetime
        out_ds = xr.open_dataset(
            join(self.subset_output_dir, output_file),
            decode_coords=False
        )

        start_dt = subset.translate_timestamp(min_time)
        end_dt = subset.translate_timestamp(max_time)

        # delta time from the MJD of this data collection
        mjd_dt = np.datetime64("1992-01-01")
        start_delta_dt = np.datetime64(start_dt) - mjd_dt
        end_delta_dt = np.datetime64(end_dt) - mjd_dt

        # All dates should be within the given temporal bounds.
        assert (out_ds.time.values >= start_delta_dt).all()
        assert (out_ds.time.values <= end_delta_dt).all()

    def test_get_time_epoch_var(self):
        """
        Test that get_time_epoch_var method returns the 'time' variable for the tropomi CH4 granule"
        """
        bbox = np.array(((-180, 180), (-90, 90)))
        tropomi_file = 'S5P_OFFL_L2__CH4____20190319T110835_20190319T125006_07407_01_010202_20190325T125810_subset.nc4'

        shutil.copyfile(os.path.join(self.test_data_dir, 'tropomi', tropomi_file),
                        os.path.join(self.subset_output_dir, tropomi_file))


        nc_dataset = nc.Dataset(os.path.join(self.subset_output_dir, tropomi_file), mode='r')

        nc_dataset = subset.transform_grouped_dataset(nc_dataset, os.path.join(self.subset_output_dir, tropomi_file))

        args = {
            'decode_coords': False,
            'mask_and_scale': False,
            'decode_times': False
        }

        with xr.open_dataset(
                xr.backends.NetCDF4DataStore(nc_dataset),
                **args
        ) as dataset:

            lat_var_names, lon_var_names = subset.get_coord_variable_names(dataset)
            time_var_names = [
                subset.get_time_variable_name(
                    dataset, dataset[lat_var_name]
                ) for lat_var_name in lat_var_names
            ]
            epoch_time_var = subset.get_time_epoch_var(dataset, time_var_names[0])
            
            assert epoch_time_var.split('__')[-1] == 'time'

    def test_temporal_variable_subset(self):
        """
        Test that both a temporal and variable subset can be executed
        on a granule, and that all of the data within that granule is
        subsetted as expected.
        """
        bbox = np.array(((-180, 180), (-90, 90)))
        file = 'ascat_20150702_084200_metopa_45145_eps_o_250_2300_ovw.l2.nc'
        output_file = "{}_{}".format(self._testMethodName, file)
        min_time = '2015-07-02T09:00:00'
        max_time = '2015-07-02T10:00:00'
        variables = [
            'wind_speed',
            'wind_dir'
        ]

        subset.subset(
            file_to_subset=join(self.test_data_dir, file),
            bbox=bbox,
            output_file=join(self.subset_output_dir, output_file),
            min_time=min_time,
            max_time=max_time,
            variables=variables
        )

        in_ds = xr.open_dataset(join(self.test_data_dir, file),
                                decode_times=False,
                                decode_coords=False)

        out_ds = xr.open_dataset(join(self.subset_output_dir, output_file),
                                 decode_times=False,
                                 decode_coords=False)

        # Check that 'time' types match
        assert in_ds.time.dtype == out_ds.time.dtype

        in_ds.close()
        out_ds.close()

        # Check that all times are within the given bounds. Open
        # dataset using 'decode_times=True' for auto-conversions to
        # datetime
        out_ds = xr.open_dataset(join(self.subset_output_dir, output_file),
                                 decode_coords=False)

        start_dt = subset.translate_timestamp(min_time)
        end_dt = subset.translate_timestamp(max_time)

        # All dates should be within the given temporal bounds.
        assert (out_ds.time >= pd.to_datetime(start_dt)).all()
        assert (out_ds.time <= pd.to_datetime(end_dt)).all()

        # Only coordinate variables and variables requested in variable
        # subset should be present.
        assert set(np.append(['lat', 'lon', 'time'], variables)) == set(out_ds.data_vars.keys())

    def test_temporal_subset_lines(self):
        bbox = np.array(((-180, 180), (-90, 90)))
        file = 'SWOT_L2_LR_SSH_Expert_368_012_20121111T235910_20121112T005015_DG10_01.nc'
        output_file = "{}_{}".format(self._testMethodName, file)
        min_time = '2012-11-11T23:59:10'
        max_time = '2012-11-12T00:20:10'

        subset.subset(
            file_to_subset=join(self.test_data_dir, file),
            bbox=bbox,
            output_file=join(self.subset_output_dir, output_file),
            min_time=min_time,
            max_time=max_time
        )

        ds = xr.open_dataset(
            join(self.subset_output_dir, output_file),
            decode_times=False,
            decode_coords=False
        )

        assert ds.time.dims != ds.latitude.dims

    def test_grouped_empty_subset(self):
        """
        Test that an empty subset of a grouped dataset returns 'None'
        spatial bounds.
        """
        bbox = np.array(((-10, 10), (-10, 10)))
        file = 'S6A_P4_2__LR_STD__ST_002_140_20201207T011501_20201207T013023_F00.nc'
        output_file = "{}_{}".format(self._testMethodName, file)

        shutil.copyfile(os.path.join(self.test_data_dir, 'sentinel_6', file),
                        os.path.join(self.subset_output_dir, file))

        spatial_bounds = subset.subset(
            file_to_subset=join(self.subset_output_dir, file),
            bbox=bbox,
            output_file=join(self.subset_output_dir, output_file)
        )

        assert spatial_bounds is None

    def test_get_time_OMI(self):
        """
        Test that code get time variables for OMI .he5 files"
        """
        omi_file = 'OMI-Aura_L2-OMSO2_2020m0116t1207-o82471_v003-2020m0223t142939.he5'

        shutil.copyfile(os.path.join(self.test_data_dir, 'OMSO2', omi_file),
                        os.path.join(self.subset_output_dir, omi_file))

        nc_dataset = subset.h5file_transform(os.path.join(self.subset_output_dir, omi_file))

        args = {
            'decode_coords': False,
            'mask_and_scale': False,
            'decode_times': False
        }

        with xr.open_dataset(
                xr.backends.NetCDF4DataStore(nc_dataset),
                **args
        ) as dataset:

            lat_var_names, lon_var_names = subset.get_coord_variable_names(dataset)
            time_var_names = [
                subset.get_time_variable_name(
                    dataset, dataset[lat_var_name]
                ) for lat_var_name in lat_var_names
            ]
            assert "Time" in time_var_names[0]
            assert "Latitude" in lat_var_names[0]
