"""
Module designed for mapping the dimensions in GPM. Phony dimensions are changed
to nscan, nbin, nfreq by using the DimensionNames variable attribute
"""

import datetime
from netCDF4 import date2num  # pylint: disable=no-name-in-module
import numpy as np

dim_dict = {}


def compute_new_time_data(time_group, nc_dataset):
    """
    Create a time variable, timeMidScan, that is present in other
    GPM collections but not the ENV collections.
    """
    # Set the time unit for GPM
    time_unit_out = "seconds since 1980-01-06 00:00:00"

    new_time_list = []
    for i in range(len(nc_dataset[time_group+'__Year'][:])):
        try:
            # Safely convert milliseconds to microseconds
            millisecond = int(nc_dataset[time_group+'__MilliSecond'][:][i])  # Cast to int first
            microsecond = np.clip(millisecond * 1000, 0, 999999)  # Ensure within range

            dt = datetime.datetime(
                int(nc_dataset[time_group+'__Year'][:][i]),
                int(nc_dataset[time_group+'__Month'][:][i]),
                int(nc_dataset[time_group+'__DayOfMonth'][:][i]),
                hour=int(nc_dataset[time_group+'__Hour'][:][i]),
                minute=int(nc_dataset[time_group+'__Minute'][:][i]),
                second=int(nc_dataset[time_group+'__Second'][:][i]),
                microsecond=microsecond
            )

            new_time_list.append(date2num(dt, time_unit_out))

        except (ValueError, OverflowError) as e:
            print(f"Skipping invalid entry at index {i}: {e}")

    return new_time_list, time_unit_out


def change_var_dims(nc_dataset, variables=None, time_name="__timeMidScan"):
    """
    Go through each variable and get the dimension names from attribute "DimensionNames
    If the name is unique, add it as a dimension to the netCDF4 dataset. Then change the
    dimensions to have the name in the DimensionName attribute rather than phony_dim
    """
    var_list = list(nc_dataset.variables.keys())
    # loop through variable list to avoid netcdf4 runtime error
    for var_name in var_list:
        # GPM will always need to be cleaned up via netCDF
        # generalizing coordinate variables in netCDF file to speed variable subsetting up
        if variables:
            if var_name not in variables and 'lat' not in var_name.lower() and 'lon' not in var_name.lower() and 'time' not in var_name.lower():
                # delete the uneccesary variables
                del nc_dataset.variables[var_name]
                continue

        var = nc_dataset.variables[var_name]
        # get the DimensionName attribute from the variable
        if 'DimensionNames' in var.ncattrs():
            dim_list = var.getncattr('DimensionNames').split(',')
            # create dimension map
            for count, dim in enumerate(dim_list):
                # get unique group for new dimension name
                dim_prefix = var_name.split('__')[1]
                # new dimension name
                new_dim = '__'+dim_prefix+'__'+dim
                length = var.shape[count]
                # check if the dimension name created has already been created in the dataset
                if new_dim not in dim_dict:
                    # create the new dimension
                    nc_dataset.createDimension(new_dim, length)
                    dim_dict[new_dim] = length
            # utilized from Dimension Cleanup module
            attrs_contents = {}
            new_mapped_var = {}
            # if the variable has attributes, get the attributes to then be copied to the new variable
            if len(var.ncattrs()) > 0:
                for attrname in var.ncattrs():
                    if attrname != '_FillValue':
                        attrs_contents[attrname] = nc_dataset.variables[var_name].getncattr(attrname)

                fill_value = var._FillValue  # pylint: disable=W0212
                dim_tup = ('__'+dim_prefix+'__'+i for i in var.getncattr('DimensionNames').split(','))
                # delete the old variable with phony_dim names
                del nc_dataset.variables[var_name]

                # create the new variable with dimension names
                new_mapped_var[var_name] = nc_dataset.createVariable(var_name, str(var[:].dtype),
                                                                     dim_tup, fill_value=fill_value)
                for attr_name, contents in attrs_contents.items():
                    new_mapped_var[var_name].setncattr(attr_name, contents)

                # copy the data to the new variable with dimension names
                new_mapped_var[var_name][:] = var[:]

    if not any(time_name in var for var in var_list):
        # if there isn't any timeMidScan variables, create one
        scan_time_groups = ["__".join(i.split('__')[:-1]) for i in var_list if 'ScanTime' in i]
        for time_group in list(set(scan_time_groups)):
            # get the seconds since Jan 6, 1980
            time_data, time_unit = compute_new_time_data(time_group, nc_dataset)
            # make a new variable for each ScanTime group
            new_time_var_name = time_group+time_name
            # copy dimensions from the Year variable
            var_dims = nc_dataset.variables[time_group+'__Year'].dimensions
            comp_args = {"zlib": True, "complevel": 1}
            nc_dataset.createVariable(new_time_var_name, 'f8', var_dims, **comp_args)
            nc_dataset.variables[new_time_var_name].setncattr('unit', time_unit)
            # copy the data in
            nc_dataset.variables[new_time_var_name][:] = time_data

    return nc_dataset
