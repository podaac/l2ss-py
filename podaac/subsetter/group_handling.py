"""
group_handling.py

Functions for converting multidimensional data structures
 between a group hierarchy and a flat structure
"""
from shutil import copy
from typing import List, Tuple

import h5py
import netCDF4 as nc
import numpy as np
import xarray as xr

GROUP_DELIM = '__'


def transform_grouped_dataset(nc_dataset: nc.Dataset, file_to_subset: str) -> nc.Dataset:
    """
    Transform a netCDF4 Dataset that has groups to an xarray compatible
    dataset. xarray does not work with groups, so this transformation
    will flatten the variables in the dataset and use the group path as
    the new variable name. For example, data_01 > km > sst would become
    'data_01__km__sst', where GROUP_DELIM is __.

    This same pattern is applied to dimensions, which are located under
    the appropriate group. They are renamed and placed in the root
    group.

    Parameters
    ----------
    nc_dataset : nc.Dataset
        netCDF4 Dataset that contains groups
    file_to_subset : str

    Returns
    -------
    nc.Dataset
        netCDF4 Dataset that does not contain groups and that has been
        flattened.
    """

    # Close the existing read-only dataset and reopen in append mode
    nc_dataset.close()
    nc_dataset = nc.Dataset(file_to_subset, 'r+')

    dimensions = {}

    def walk(group_node, path):
        for key, item in group_node.items():
            group_path = f'{path}{GROUP_DELIM}{key}'

            # If there are variables in this group, copy to root group
            # and then delete from current group
            if item.variables:
                # Copy variables to root group with new name
                for var_name, var in item.variables.items():
                    var_group_name = f'{group_path}{GROUP_DELIM}{var_name}'
                    nc_dataset.variables[var_group_name] = var
                # Delete variables
                var_names = list(item.variables.keys())
                for var_name in var_names:
                    del item.variables[var_name]

            if item.dimensions:
                dims = list(item.dimensions.keys())
                for dim_name in dims:
                    new_dim_name = f'{group_path.replace("/", GROUP_DELIM)}{GROUP_DELIM}{dim_name}'
                    item.dimensions[new_dim_name] = item.dimensions[dim_name]
                    dimensions[new_dim_name] = item.dimensions[dim_name]
                    item.renameDimension(dim_name, new_dim_name)

            # If there are subgroups in this group, call this function
            # again on that group.
            if item.groups:
                walk(item.groups, group_path)

        # Delete non-root groups
        group_names = list(group_node.keys())
        for group_name in group_names:
            del group_node[group_name]

    for var_name in list(nc_dataset.variables.keys()):
        new_var_name = f'{GROUP_DELIM}{var_name}'
        nc_dataset.variables[new_var_name] = nc_dataset.variables[var_name]
        del nc_dataset.variables[var_name]

    walk(nc_dataset.groups, '')

    # Update the dimensions of the dataset in the root group
    nc_dataset.dimensions.update(dimensions)

    return nc_dataset


def recombine_grouped_datasets(datasets: List[xr.Dataset], output_file: str, start_date, time_vars) -> None:  # pylint: disable=too-many-branches
    """
    Given a list of xarray datasets, combine those datasets into a
    single netCDF4 Dataset and write to the disk. Each dataset has been
    transformed using its group path and needs to be un-transformed and
    placed in the appropriate group.

    Parameters
    ----------
    datasets : list (xr.Dataset)
        List of xarray datasets to be combined
    output_file : str
        Name of the output file to write the resulting NetCDF file to.
    TODO: add docstring and type hint for `start_date` parameter.
    """

    base_dataset = nc.Dataset(output_file, mode='w')

    for dataset in datasets:
        group_lst = []
        for var_name in dataset.variables.keys():  # need logic if there is data in the top level not in a group
            group_lst.append('/'.join(var_name.split(GROUP_DELIM)[:-1]))
        group_lst = ['/' if group == '' else group for group in group_lst]
        groups = set(group_lst)
        for group in groups:
            base_dataset.createGroup(group)

        for dim_name in list(dataset.dims.keys()):
            new_dim_name = dim_name.split(GROUP_DELIM)[-1]
            dim_group = _get_nested_group(base_dataset, dim_name)
            dim_group.createDimension(new_dim_name, dataset.dims[dim_name])

        # Rename variables
        _rename_variables(dataset, base_dataset, start_date, time_vars)

    # Remove group vars from base dataset
    for var_name in list(base_dataset.variables.keys()):
        if GROUP_DELIM in var_name:
            del base_dataset.variables[var_name]

    # Remove group dims from base dataset
    for dim_name in list(base_dataset.dimensions.keys()):
        if GROUP_DELIM in dim_name:
            del base_dataset.dimensions[dim_name]

    # Copy global attributes
    base_dataset.setncatts(datasets[0].attrs)
    # Write and close
    base_dataset.close()


def _get_nested_group(dataset: nc.Dataset, group_path: str) -> nc.Group:
    nested_group = dataset
    for group in group_path.strip(GROUP_DELIM).split(GROUP_DELIM)[:-1]:
        nested_group = nested_group.groups[group]
    return nested_group


def _rename_variables(dataset: xr.Dataset, base_dataset: nc.Dataset, start_date, time_vars) -> None:
    for var_name in list(dataset.variables.keys()):
        new_var_name = var_name.split(GROUP_DELIM)[-1]
        var_group = _get_nested_group(base_dataset, var_name)
        variable = dataset.variables[var_name]
        var_dims = [x.split(GROUP_DELIM)[-1] for x in dataset.variables[var_name].dims]
        if np.issubdtype(
                dataset.variables[var_name].dtype, np.dtype(np.datetime64)
        ) or np.issubdtype(
            dataset.variables[var_name].dtype, np.dtype(np.timedelta64)
        ) and var_name in time_vars:  # check that time changes are done to a time variable
            if start_date:
                dataset.variables[var_name].values = (dataset.variables[var_name].values - np.datetime64(start_date))/np.timedelta64(1, 's')
                variable = dataset.variables[var_name]
            else:
                cf_dt_coder = xr.coding.times.CFDatetimeCoder()
                encoded_var = cf_dt_coder.encode(dataset.variables[var_name])
                variable = encoded_var

        var_attrs = {}
        for key, value in variable.attrs.items():
            new_key = key.replace("/", "_") if isinstance(key, str) else key
            var_attrs[new_key] = value

        fill_value = var_attrs.get('_FillValue')
        var_attrs.pop('_FillValue', None)
        comp_args = {"zlib": True, "complevel": 1}

        var_data = variable.data
        if variable.dtype == object:
            comp_args = {"zlib": False, "complevel": 1}
            var_group.createVariable(new_var_name, 'S4', var_dims, fill_value=fill_value, **comp_args)
            var_data = np.array(variable.data)
        elif variable.dtype == 'timedelta64[ns]':
            var_group.createVariable(new_var_name, 'i4', var_dims, fill_value=fill_value, **comp_args)
        elif variable.dtype in ['|S1', '|S2']:
            var_group.createVariable(new_var_name, variable.dtype, var_dims, fill_value=fill_value)
        else:
            var_group.createVariable(new_var_name, variable.dtype, var_dims, fill_value=fill_value, **comp_args)

        # Copy attributes
        var_group.variables[new_var_name].setncatts(var_attrs)

        # Copy data
        var_group.variables[new_var_name].set_auto_maskandscale(False)
        if variable.dtype in ['|S1', '|S2']:
            var_group.variables[new_var_name][:] = variable.values
        else:
            var_group.variables[new_var_name][:] = var_data


def h5file_transform(finput: str) -> Tuple[nc.Dataset, bool]:
    """
    Transform a h5py  Dataset that has groups to an xarray compatible
    dataset. xarray does not work with groups, so this transformation
    will flatten the variables in the dataset and use the group path as
    the new variable name. For example, data_01 > km > sst would become
    'data_01__km__sst', where GROUP_DELIM is __.

    Returns
    -------
    nc.Dataset
        netCDF4 Dataset that does not contain groups and that has been
        flattened.
    bool
        Whether this dataset contains groups
    """
    data_new = h5py.File(finput, 'r+')
    del_group_list = list(data_new.keys())
    has_groups = bool(data_new['/'])

    def walk_h5py(data_new, group):
        # flattens h5py file
        for key, item in data_new[group].items():
            group_path = f'{group}{key}'
            if isinstance(item, h5py.Dataset):
                new_var_name = group_path.replace('/', '__')

                data_new[new_var_name] = data_new[group_path]
                del data_new[group_path]

            elif isinstance(item, h5py.Group):
                if len(list(item.keys())) == 0:
                    new_group_name = group_path.replace('/', '__')
                    data_new[new_group_name] = data_new[group_path]

                walk_h5py(data_new, data_new[group_path].name + '/')

    walk_h5py(data_new, data_new.name)

    # Get the instrument name from the file attributes

    additional_file_attributes = data_new.get('__HDFEOS__ADDITIONAL__FILE_ATTRIBUTES')
    instrument = ""

    if additional_file_attributes:
        instrument = additional_file_attributes.attrs['InstrumentName'].decode("utf-8")
    if 'OMI' in instrument:
        hdf_type = 'OMI'
    elif 'MLS' in instrument:
        hdf_type = 'MLS'
    else:
        hdf_type = None

    for del_group in del_group_list:
        del data_new[del_group]

    finputnc = '.'.join(finput.split('.')[:-1]) + '.nc'

    data_new.close()  # close the h5py dataset
    copy(finput, finputnc)  # copy to a nc file

    nc_dataset = nc.Dataset(finputnc, mode='r')

    return nc_dataset, has_groups, hdf_type
