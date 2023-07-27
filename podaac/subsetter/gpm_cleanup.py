"""
Module designed for mapping the dimensions in GPM. Phony dimensions are changed
to nscan, nbin, nfreq by using the DimensionNames variable attribute
"""

dim_dict = {}

def change_var_dims(nc_dataset, variables=None):
    var_list = list(nc_dataset.variables.keys())
    for var_name in var_list:
        # GPM will always need to be cleaned up via netCDF
        # generalizing coordinate variables in netCDF file to speed variable subsetting up
        if variables:
            if var_name not in variables and 'lat' not in var_name.lower() and \
                'lon' not in var_name.lower() and 'time' not in var_name.lower():
                del nc_dataset.variables[var_name]
                continue

        var = nc_dataset.variables[var_name]
        if 'DimensionNames' in var.ncattrs():
            dim_list = var.getncattr('DimensionNames').split(',')
            # create dimension map
            for i in range(len(dim_list)):
                dim_prefix = var_name.split('__')[1]
                key = '__'+dim_prefix+'__'+dim_list[i]
                length = var.shape[i]
                if key not in list(dim_dict.keys()):
                    nc_dataset.createDimension(key, length)
                    dim_dict[key] = length

            attrs_contents = {}
            new_mapped_var = {}

            if len(var.ncattrs()) > 0:
                for attrname in var.ncattrs():
                    if attrname != '_FillValue':
                        attrs_contents[attrname] = nc_dataset.variables[var_name].getncattr(attrname)

                fill_value = var._FillValue  # pylint: disable=W0212
                dim_tup = tuple(['__'+dim_prefix+'__'+i for i in var.getncattr('DimensionNames').split(',')])
                del nc_dataset.variables[var_name]

                new_mapped_var[var_name] = nc_dataset.createVariable(var_name, str(var[:].dtype),
                                                                    dim_tup, fill_value=fill_value)
                for attr_name, contents in attrs_contents.items():
                    new_mapped_var[var_name].setncattr(attr_name, contents)

                new_mapped_var[var_name][:] = var[:]

    return nc_dataset