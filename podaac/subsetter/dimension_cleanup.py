# This software may be subject to U.S. export control laws. By accepting
# this software, the user agrees to comply with all applicable U.S. export
# laws and regulations. User has the responsibility to obtain export
# licenses, or other export authority as may be required before exporting
# such information to foreign countries or providing access to foreign
# persons.

"""
======================
dimension_cleanup.py
======================
Functions which improve upon existing netCDF4 library existing functions
"""
import collections

import netCDF4 as nc
import xarray as xr


def remove_duplicate_dims(nc_dataset: nc.Dataset) -> nc.Dataset:
    """
    xarray cannot read netCDF4 datasets with duplicate dimensions.
    Function goes through a dataset to catch any variables with duplicate dimensions.
    creates an exact copy of the dimension duplicated with a new name. Variable
    is reset with new dimensions without duplicates. Old variable deleted, new variable's name
    is changed to the original name.
    """
    dup_vars = {}
    dup_new_varnames = []

    for var_name, var in nc_dataset.variables.items():
        dim_list = list(var.dimensions)
        if len(set(dim_list)) != len(dim_list):  # get true if var.dimensions has a duplicate
            dup_vars[var_name] = var  # populate dictionary with variables with vars with dup dims

    for dup_var_name, dup_var in dup_vars.items():
        dim_list = list(dup_var.dimensions)  # original dimensions of the variable with duplicated dims

        # Dimension(s) that are duplicated are retrieved.
        #   Note: this is not yet tested for more than one duplicated dimension.
        dim_dup = [item for item, count in collections.Counter(dim_list).items() if count > 1][0]
        dim_dup_length = dup_var.shape[dup_var.dimensions.index(dim_dup)]  # length of the duplicated dimension

        # New dimension and variable names are created.
        dim_dup_new = dim_dup+'_1'
        var_name_new = dup_var_name+'_1'
        dup_new_varnames.append(var_name_new)

        # The last dimension for the variable is replaced with the new name in a temporary list.
        new_dim_list = dim_list[:-1]
        new_dim_list.extend([dim_dup_new])

        new_dup_var = {}
        attrs_contents = {}

        # Attributes for the original variable are retrieved.
        for attrname in dup_var.ncattrs():
            if attrname != '_FillValue':
                attrs_contents[attrname] = nc_dataset.variables[dup_var_name].getncattr(attrname)

        fill_value = dup_var._FillValue  # pylint: disable=W0212

        # Only create a new *Dimension* if it doesn't already exist.
        if dim_dup_new not in nc_dataset.dimensions.keys():

            # New dimension is created by copying from the duplicated dimension.
            nc_dataset.createDimension(dim_dup_new, dim_dup_length)

            # Only create a new dimension *Variable* if it existed originally in the NetCDF structure.
            if dim_dup in nc_dataset.variables.keys():

                # New variable object is created for the renamed, previously duplicated dimension.
                new_dup_var[dim_dup_new] = nc_dataset.createVariable(dim_dup_new, nc_dataset.variables[dim_dup].dtype,
                                                                     (dim_dup_new,), fill_value=fill_value)
                # New variable's attributes are set to the original ones.
                for ncattr in nc_dataset.variables[dim_dup].ncattrs():
                    if ncattr != '_FillValue':
                        new_dup_var[dim_dup_new].setncattr(ncattr, nc_dataset.variables[dim_dup].getncattr(ncattr))
                new_dup_var[dim_dup_new][:] = nc_dataset.variables[dim_dup][:]

        # Delete existing Variable
        del nc_dataset.variables[dup_var_name]

        # Replace original *Variable* with new variable with no duplicated dimensions.
        new_dup_var[dup_var_name] = nc_dataset.createVariable(dup_var_name, str(dup_var[:].dtype),
                                                              tuple(new_dim_list), fill_value=fill_value)
        for attr_name, contents in attrs_contents.items():
            new_dup_var[dup_var_name].setncattr(attr_name, contents)
        new_dup_var[dup_var_name][:] = dup_var[:]

    return nc_dataset


def sync_dims_inplace(original_dataset: xr.Dataset, new_dataset: xr.Dataset) -> None:
    """
    Synchronize dimensions of variables in the new dataset with the original dataset.

    Parameters:
    original_dataset (xr.Dataset): The original dataset.
    new_dataset (xr.Dataset): The new dataset with possibly additional dimensions.
    """

    for variable_name in new_dataset.variables:
        original_variable_dims = original_dataset[variable_name].dims
        new_variable_dims = new_dataset[variable_name].dims

        for new_dim in new_variable_dims:
            if new_dim not in original_variable_dims:
                new_dataset[variable_name] = new_dataset[variable_name].isel({new_dim: 0})
