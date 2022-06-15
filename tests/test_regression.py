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
import sys
import urllib3
import certifi
import requests
from time import sleep
import argparse
import os
from os import listdir
from os.path import dirname, join, realpath, isfile, basename

#import geopandas as gpd
import importlib_metadata
import netCDF4 as nc
#import h5py
import numpy as np
import pandas as pd
import pytest
import datetime as dt



import xarray as xr
#from jsonschema import validate
#from shapely.geometry import Point

from podaac.subsetter import subset
#from podaac.subsetter.subset import SERVICE_NAME
#from podaac.subsetter import xarray_enhancements as xre
#from podaac.subsetter import dimension_cleanup as dc
from podaac.subsetter import ncls

from IPython.display import display, JSON
from harmony import BBox, Client, Collection, Request
from harmony.config import Environment



# This method POSTs formatted JSON WSP requests to the GES DISC endpoint URL and returns the response
def get_http_data(request, http, svcurl):
    hdrs = {'Content-Type': 'application/json',
            'Accept'      : 'application/json'}
    data = json.dumps(request)       
    r = http.request('POST', svcurl, body=data, headers=hdrs)
    response = json.loads(r.data)   
    # Check for errors
    if response['type'] == 'jsonwsp/fault' :
        print('API Error: faulty request')
        print('response was:')
        print(json.dumps(response,indent=2))
    return response

def get_product_name(satellite,level,product,version,filename):
    break_out_of_line = False

    with open(filename, 'r') as f:
        for line in f.readlines():

            for i in [satellite,'L'+str(level),product,str(version)]:
                if i in line:
                    if str(line[-2]) == i:
                        dir_name = product+'_'+str(version)
                        short_name = line[:-1]
                        break_out_of_line = True
                        break
                    continue
                else:
                    break
                    
            if break_out_of_line == True:
                break

            #try except block

    return short_name, dir_name

def run_api(outdir, satellite='S5P', product='CH4', level=2, version=1, begTime='2019-03-11T10:16:55.000Z',
            endTime='2019-03-11T11:58:25.999Z', bbox='[-10,0,20,30]',
            file_type='nc4', crop=True):

    short_name, dir_name = get_product_name(satellite,level,product,version,'product_table.txt')
    #vars = [i for i in args['variables'].split(',')]
    if type(bbox) == str:
        bbox = [float(i) for i in bbox.replace('[', '').replace(']', '').split(',')]
        print (bbox)
        
    API_dict = {
        'role'  : 'subset',
        'agent' : 'SUBSET_LEVEL'+str(level),
        'format': file_type,
        'start' : begTime,
        'end'   : endTime,
        'box'   : [bbox[0], bbox[1],
                   bbox[2], bbox[3]],  
        'crop'  : crop,
        'data'  : [{'datasetId': short_name}]        
    }

    # Create a urllib PoolManager instance to make requests.
    http = urllib3.PoolManager(cert_reqs='CERT_REQUIRED',ca_certs=certifi.where())

    # Set the URL for the GES DISC subset service endpoint
    svcurl = 'https://disc.gsfc.nasa.gov/service/subset/jsonwsp'
    #svcurl = 'https://uui-test.gesdisc.eosdis.nasa.gov/service/subset/jsonwsp'

    # Define the parameters for the data subset

    # Construct JSON WSP request for API method: subset
    print (API_dict)

    subset_request = {
        'methodname': 'subset',
        'type': 'jsonwsp/request',
        'version': '1.0',
        'args': API_dict
    }

    # Submit the subset request to the GES DISC Server
    response = get_http_data(subset_request, http, svcurl)
    print (response)
    # Report the JobID and initial status
    if response['result']: 
        myJobId = response['result']['jobId']
        print('Job ID: '+myJobId)
        print('Job status: '+response['result']['Status'])


    # Construct JSON WSP request for API method: GetStatus
    status_request = {
        'methodname': 'GetStatus',
        'version': '1.0',
        'type': 'jsonwsp/request',
        'args': {'jobId': myJobId}
    }

    # Check on the job status after a brief nap
    while response['result']['Status'] in ['Accepted', 'Running']:
        sleep(5)
        response = get_http_data(status_request,http,svcurl)
        status  = response['result']['Status']
        percent = response['result']['PercentCompleted']
        print ('Job status: %s (%d%c complete)' % (status,percent,'%'))

    if response['result'] and response['result']['Status'] == 'Succeeded' :
        print ('Job Finished:  %s' % response['result']['message'])
    else : 
        print('Something went wrong.')
        print('Here is the subset_request: \n%s' % json.dumps(subset_request,indent=2))
        print('Here is the response: \n%s' % json.dumps(response,indent=2))

    # Construct JSON WSP request for API method: GetResult
    batchsize = 20
    results_request = {
        'methodname': 'GetResult',
        'version': '1.0',
        'type': 'jsonwsp/request',
        'args': {
            'jobId': myJobId,
            'count': batchsize,
            'startIndex': 0
        }
    }

    # Retrieve the results in JSON in multiple batches 
    # Initialize variables, then submit the first GetResults request
    # Add the results from this batch to the list and increment the count
    results = []
    count = 0 
    response = get_http_data(results_request, http, svcurl) 
    count = count + response['result']['itemsPerPage']
    results.extend(response['result']['items']) 

    # Increment the startIndex and keep asking for more results until we have them all
    total = response['result']['totalResults']
    while count < total :
        results_request['args']['startIndex'] += batchsize 
        response = get_http_data(results_request, http, svcurl) 
        count = count + response['result']['itemsPerPage']
        results.extend(response['result']['items'])

    # Check on the bookkeeping
    print('Retrieved %d out of %d expected items' % (len(results), total))

    # Sort the results into documents and URLs
    docs = []
    urls = []
    for item in results :
        try:
            if item['start'] and item['end'] : urls.append(item) 
        except:
            docs.append(item)

    # Print out the documentation links, but do not download them
    print('\nDocumentation:')
    for item in docs : print(item['label']+': '+item['link'])
    print (urls)

    # Use the requests library to submit the HTTP_Services URLs and write out the results.
    print('\nHTTP_services output:')
    try:
        os.makedirs(outdir)
    except FileExistsError:
        pass
    for item in urls :
        URL = item['link'] 
        result = requests.get(URL)
        try:
            result.raise_for_status()
            outfn = item['label']
            f = open(outdir+'/'+outfn,'wb')
            f.write(result.content)
            f.close()
            print(outfn)
        except:
            print('Error! Status code is %d for this URL:\n%s' % (result.status.code,URL))
            print('Help for downloading data is at https://disc.gsfc.nasa.gov/data-access')

    return outfn


def get_reference_dir_name(satellite, level, product, version, filename):
    break_out_of_line = False

    with open(filename, 'r') as f:
        for line in f.readlines():

            for i in [satellite,'L'+str(level),product,str(version)]:
                if i in line:
                    if str(line[-2]) == i:
                        short_name = line.replace('\n', '')
                        dir_name = product+'_'+str(version)
                        break_out_of_line = True
                        break
                    continue
                else:
                    break
                    
            if break_out_of_line == True:
                break

    return dir_name

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
        
    @pytest.mark.skip(reason="This is being tested currently.  Temporarily skipped.")
    def test_temporal(self):
        
        pass
        
    #@pytest.mark.skip(reason="This is being tested currently.  Temporarily skipped.")
    def test_bounding_box(self,satellite='S5P',level=2, product='CH4',
                          version='1', filename='product_table.txt',
                          data_name='tropomi',bbox='[-10,0,20,30]', outputtype='nc4'):

        reference_file = run_api(outdir=self.test_data_dir)
        
        bbox = [float(i) for i in bbox.replace('[','').replace(']','').split(',')]
        box_array = np.asarray(((bbox[0],bbox[2]),(bbox[1],bbox[3])))

        #reference_name = get_reference_dir_name(satellite,level,product,version,filename)
        #reference_path = os.path.join(self.test_data_dir,data_name,reference_name, 'bbox')
        #reference_file = os.listdir(reference_path)[0]

        #input_file = [i for i in os.listdir(os.path.join(self.test_data_dir,data_name,reference_name)) if i.endswith('nc4')][0]


        """shutil.copyfile(
            os.path.join(self.test_data_dir, data_name, reference_name, input_file),
            os.path.join(self.subset_output_dir, input_file)
        )
        shutil.copyfile(
            os.path.join(reference_path,reference_file),
            os.path.join(self.subset_output_dir, reference_file)
        )"""
        
        #output_file = "{}_{}".format(self._testMethodName, input_file)
        #shapefile = None
        
        harmony_client = Client(env=Environment.UAT)

        # get subset from harmony uat
        
        collection = Collection(id='C1220280439-GES_DISC') # Tropomi_CH4_1
        request = Request(
            collection=collection,
            spatial=BBox(-10,0,20,30),
            temporal = {
                'start': dt.datetime(2019, 3, 11, 10, 16),
                'stop' : dt.datetime(2019, 3, 11, 11, 58)
            },
            max_results=1,
        )

        job_id = harmony_client.submit(request)

        myURL = 'https://harmony.uat.earthdata.nasa.gov/jobs/'+job_id

        results = harmony_client.download_all(job_id, directory=self.test_data_dir+'/tropomi', overwrite=True)
        output_file = [f.result() for f in results][0]

        spatial_bounds = subset.subset(
            file_to_subset=join(self.subset_output_dir, input_file),
            bbox=box_array,
            output_file=join(self.subset_output_dir, output_file)
        )



        
        if outputtype=='he5':

            hl2ss_nc = subset.h5file_transform(
                join(self.subset_output_dir, harmony_subsetter_file)    
                )[0]
            
            ref_nc = subset.h5file_transform(
                join(self.subset_output_dir, harmony_subsetter_file)    
                )[0]
            
            # open both files
            # utilize ncls.py
            
            pass
        
        elif outputtype=='nc4':

            hl2ss_nc = nc.Dataset(
                join(self.subset_output_dir, harmony_subsetter_file),
                decode_coords=False)
        
            # there should only be one reference file
            ref_nc = nc.Dataset(
                join(self.subset_output_dir, reference_api_file),
                decode_coords=False)

            nc_flattened_hl2ss = subset.transform_grouped_dataset(hl2ss_nc, os.path.join(self.subset_output_dir, harmony_subsetter_file))
            nc_flattened_ref = subset.transform_grouped_dataset(ref_nc, os.path.join(self.subset_output_dir, reference_api_file))
            
            # open both files
            #harmony_subsetter_file = 'S5P_OFFL_L2_CH4_20190311T101655_20190311T115825_07293_01_010202_20190317T121015_subsetted.nc4'
            #reference_api_file = 'S5P_OFFL_L2__CH4____20190311T101655_20190311T115825_07293_01_010202_20190317T121015.SUB.nc4'
            # utilize ncls.py
            #ncls_hl2ss = ncls.NCLS(os.path.join(self.test_data_dir,harmony_subsetter_file ))
            #ncls_ref = ncls.NCLS(os.path.join(self.test_data_dir, reference_api_file))

            
            """shutil.copyfile(
            os.path.join(self.test_data_dir, harmony_subsetter_file),
            os.path.join(self.subset_output_dir, harmony_subsetter_file)
            )
            shutil.copyfile(
            os.path.join(self.test_data_dir,reference_api_file),
            os.path.join(self.subset_output_dir, reference_api_file)
            )"""

        hl2ss_ds = xr.open_dataset(xr.backends.NetCDF4DataStore(nc_flattened_hl2ss),
                                   decode_times=False,
                                   decode_coords=False,
                                   mask_and_scale=False)
        ref_ds = xr.open_dataset(xr.backends.NetCDF4DataStore(nc_flattened_ref),
                                 decode_times=False,
                                 decode_coords=False,
                                 mask_and_scale=False)

        lat_var_name, lon_var_name = subset.compute_coordinate_variable_names(hl2ss_ds)

        lat_var_name = lat_var_name[0]
        lon_var_name = lon_var_name[0]

        lon_bounds, lat_bounds = subset.convert_bbox(box_array, hl2ss_ds, lat_var_name, lon_var_name)

        lats = hl2ss_ds[lat_var_name].values
        lons = hl2ss_ds[lon_var_name].values

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
            pass

        # Step 2: Get mask of values which are NaN or "_FillValue in
        # each variable.

        for var_name, var in hl2ss_ds.data_vars.items():

            assert(var.shape==ref_ds[var_name].shape)

            """if var_name == '__PRODUCT__delta_time' or var_name == '__PRODUCT__time_utc':
                print (var_name)
                print (np.array(var))
                print (np.array(ref_ds[var_name]))
                pass
            else:
                print (var_name)
                print (np.array(var))
                xr.testing.assert_equal(var,ref_ds[var_name])"""

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

            if vals.shape != hl2ss_ds[lat_var_name].shape and vals.shape:
                slice_list = []
                for dim in var.squeeze().dims:
                    if dim in hl2ss_ds[lat_var_name].squeeze().dims:
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

        hl2ss_ds.close()
