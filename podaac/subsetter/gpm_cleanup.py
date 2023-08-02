"""
Module designed for mapping the dimensions in GPM. Phony dimensions are changed
to nscan, nbin, nfreq by using the DimensionNames variable attribute
"""

dim_dict = {}


def change_var_dims(nc_dataset, variables=None):
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

    return nc_dataset
