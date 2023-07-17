import collections

import netCDF4 as nc

dim_dict = {}

def change_var_dims(nc_dataset):
    for var_name, var in nc_dataset.variables.items():
        #print (var.ncattrs())
        if 'DimensionNames' in var.ncattrs():
            dim_list = var.getncattr('DimensionNames').split(',')
            # create dimension map
            for i in range(len(dim_list)):
                dim_prefix = var_name.split('__')[1]
                dim_dict['__'+dim_prefix+'__'+dim_list[i]] = var.shape[i]

    for key, length in dim_dict.items():
        nc_dataset.createDimension(key, length)

    attrs_contents = {}
    new_mapped_var = {}
    var_list = list(nc_dataset.variables.keys())
    for var_name in var_list:
        var = nc_dataset.variables[var_name]
        # Attributes for the original variable are retrieved.
        if len(var.ncattrs()) > 0:
            for attrname in var.ncattrs():
                if attrname != '_FillValue':
                    attrs_contents[attrname] = nc_dataset.variables[var_name].getncattr(attrname)
            print (var.ncattrs())
            print (var_name)
            fill_value = var._FillValue  # pylint: disable=W0212
            if 'DimensionNames' in var.ncattrs():
                dim_prefix = var_name.split('__')[1]
                dim_tup = tuple(['__'+dim_prefix+'__'+i for i in var.getncattr('DimensionNames').split(',')])
            # Delete existing Variable
            del nc_dataset.variables[var_name]

            # Replace original *Variable* with new variable with no duplicated dimensions.
            new_mapped_var[var_name] = nc_dataset.createVariable(var_name, str(var[:].dtype),
                                                                dim_tup, fill_value=fill_value)
            for attr_name, contents in attrs_contents.items():
                new_mapped_var[var_name].setncattr(attr_name, contents)
            new_mapped_var[var_name][:] = var[:]
    print (nc_dataset.variables['__FS__SLV__precipRate'])

    return nc_dataset