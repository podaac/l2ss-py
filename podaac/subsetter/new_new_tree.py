import xarray as xr
import numpy as np
from typing import Union, Dict, Tuple, List
from xarray import DataTree
from typing import Union, Dict, Tuple, List, Any
import cf_xarray as cfxr

import logging
import xarray as xr
import re

import xarray as xr
import re
from typing import List, Optional
from typing import Optional, Set

GROUP_DELIM = "/"  # Adjust based on actual dataset structure

def get_indexers_from_1d(cond: xr.Dataset) -> dict:
    """
    Get indexers from a dataset with 1 dimension.

    Parameters
    ----------
    cond : xarray.Dataset
        Contains the result of the initial lat lon condition.

    Returns
    -------
    dict
        Indexer dictionary for the provided condition.
    """
    cols = cond.values

    if not cols.any():
        logging.info("No data within the given bounding box.")

    indexers = {
        cond.dims[0]: np.where(cols)[0]
    }
    return indexers


def get_indexers_from_nd(cond: xr.Dataset, cut: bool) -> dict:
    """
    Get indexers from a dataset with more than one dimension.

    Parameters
    ----------
    cond : xarray.Dataset
        Contains the result of the initial lat lon condition.
    cut : bool
        True if the scanline should be cut.

    Returns
    -------
    dict
        Indexer dictionary for the provided condition.
    """
    # check if the lat/lon coordinate numpy array has 2 or more dimensions
    transpose = dim_grid = False
    ndim = cond.values.squeeze().ndim

    # Determine axes and flags
    if ndim == 2:
        x_axis, y_axis = 1, 0
    else:
        if 'xtrack' in cond.dims and 'atrack' in cond.dims:
            x_axis, y_axis = cond.dims.index('xtrack'), cond.dims.index('atrack')
            transpose = True
        elif 'xdim_grid' in cond.dims and 'ydim_grid' in cond.dims:
            x_axis, y_axis = cond.dims.index('xdim_grid'), cond.dims.index('ydim_grid')
            dim_grid = x_axis == 1 and y_axis == 0
        else:
            x_axis, y_axis = 2, 1

    # Compute rows and columns
    squeezed_values = cond.values.squeeze()
    rows = np.any(squeezed_values, axis=x_axis)
    cols = np.any(squeezed_values, axis=y_axis) if cut else np.ones(len(squeezed_values[0]))

    # Log information about subsetted area
    if np.all(rows) and np.all(cols):
        logging.info("Subsetted area equal to the original granule.")
    if not np.any(rows) or not np.any(cols):
        logging.info("No data within the given bounding box.")

    # Determine dimensions and clean them up
    cond_dims = list(cond.dims)
    cond_shape = list(cond.shape)
    cond_dims = [dim for dim, size in zip(cond_dims, cond_shape) if size > 1]

    # Adjust for 3D data
    if rows.ndim > 1:
        if transpose:
            rows, cols = rows.transpose()[0], cols.transpose()[0]
        elif not dim_grid:
            rows, cols = rows[0], cols[0]

    # Generate indexers
    indexers = {
        cond_dims[y_axis]: np.where(rows)[0],
        cond_dims[x_axis]: np.where(cols)[0]
    }

    return indexers

def create_subset_dataset(
    dataset: xr.Dataset,
    variables: List[str],
    indexers: Dict,
    cond: Union[xr.Dataset, xr.DataArray]
) -> xr.Dataset:
    """
    Create a dataset for a subset of variables with proper masking,
    strictly maintaining each variable's original dimensions and order.
    """
    # Keep track of original variable order
    original_var_order = list(dataset.variables)
    original_dims = {var: dataset[var].dims for var in variables if var in dataset}
    
    # Create a full-size mask matching the original data size
    full_mask = np.zeros(dataset.dims['num_lines'], dtype=bool)
    full_mask[indexers['num_lines']] = True
    
    # Process variables maintaining order
    subset_dict = {}
    for var_name in original_var_order:
        if var_name in variables and var_name in dataset:
            var = dataset[var_name]
            orig_dims = original_dims[var_name]
            
            # Create a condition matching the original variable's shape
            if set(orig_dims).intersection(cond.dims):
                subset_dict[var_name] = var.where(full_mask)
            else:
                subset_dict[var_name] = var.copy()
            
            # Verify dimensions haven't changed
            assert subset_dict[var_name].dims == orig_dims, (
                f"Dimensions changed for {var_name}. "
                f"Expected {orig_dims}, got {subset_dict[var_name].dims}"
            )
    
    # Create dataset with original structure and order
    result = xr.Dataset()
    
    # Add variables in original order
    for var_name in original_var_order:
        if var_name in subset_dict:
            result[var_name] = subset_dict[var_name]
    
    # Copy coordinates in original order
    for coord_name in original_var_order:
        if coord_name in dataset.coords and coord_name not in result.coords:
            result.coords[coord_name] = dataset.coords[coord_name].copy()
    
    # Add provenance information
    result.attrs.update(dataset.attrs)
    
    return result

def _create_subset_dataset(
    dataset: xr.Dataset,
    variables: List[str],
    indexers: Dict,
    cond: Union[xr.Dataset, xr.DataArray]
) -> xr.Dataset:
    """
    Create a dataset for a subset of variables with proper masking,
    maintaining original dimension sets.
    """
    subset_dict = {}
    
    for var_name in variables:
        if var_name in dataset:
            var = dataset[var_name]
            # Only use indexers that apply to this variable's dimensions
            relevant_indexers = {
                dim: idx for dim, idx in indexers.items()
                if dim in var.dims
            }
            
            if relevant_indexers:
                indexed_var = var.isel(**relevant_indexers)
                if set(relevant_indexers.keys()).intersection(var.dims):
                    # Only apply condition if the variable shares dimensions with it
                    relevant_cond = cond.isel(**{k: v for k, v in relevant_indexers.items()
                                               if k in cond.dims})
                    subset_dict[var_name] = indexed_var.where(relevant_cond)
                else:
                    subset_dict[var_name] = indexed_var
            else:
                # If no relevant indexers, keep the variable as is
                subset_dict[var_name] = var.copy()
            
            # Ensure original dtype is preserved
            subset_dict[var_name] = subset_dict[var_name].astype(var.dtype)
    
    # Create dataset maintaining original structure
    result = xr.Dataset(subset_dict)
    
    # Copy only relevant coordinates
    used_coords = set()
    for var in result.data_vars.values():
        used_coords.update(var.dims)
    
    for coord_name in used_coords:
        if coord_name in dataset.coords and coord_name not in result.coords:
            result.coords[coord_name] = dataset.coords[coord_name]
    
    return DataTree(name='root', dataset=result)

def where_tree(tree: DataTree, cond: Union[xr.Dataset, xr.DataArray], cut: bool) -> DataTree:
    """
    Return a DataTree which meets the given condition, processing all nodes in the tree.
    
    Parameters
    ----------
    tree : xarray.DataTree
        The input DataTree to filter
    cond : DataArray or Dataset with boolean dtype
        Locations at which to preserve this object's values
    cut : boolean
        True if the scanline should be cut, False if not
        
    Returns
    -------
    xarray.DataTree
        The filtered DataTree with all nodes processed
    """
    def process_node(node: DataTree) -> Tuple[xr.Dataset, Dict[str, DataTree]]:
        """
        Process a single node and its children in the tree.
        
        Parameters
        ----------
        node : DataTree
            The node to process
            
        Returns
        -------
        Tuple[xr.Dataset, Dict[str, DataTree]]
            Processed dataset and dictionary of processed child nodes
        """
        # Get the dataset directly from the node
        dataset = node.ds
        
        original_dtypes = {var: dataset[var].dtype for var in dataset.variables}

        # Process current node's dataset
        if len(dataset.variables) > 0:  # Only process if node has data
            # Create indexers from condition
            if cond.values.ndim == 1:
                indexers = get_indexers_from_1d(cond)
            else:
                indexers = get_indexers_from_nd(cond, cut)

            if not all(len(value) > 0 for value in indexers.values()):
                return copy_empty_dataset(dataset), {}

            print(indexers)
            # Check for partial dimension overlap
            partial_dim_in_vars = check_partial_dim_overlap_node(dataset, indexers)

            # Apply indexing to condition and dataset
            indexed_cond = cond.isel(**indexers)
            try:
                indexed_ds = dataset.isel(**indexers)
            except Exception as ex:
                indexed_ds = dataset

            # Get variables with and without indexers
            subset_vars, non_subset_vars = get_variables_with_indexers(dataset, indexers)
            
            # Process variables
            sub_ds = create_subset_dataset(dataset, subset_vars, indexers, cond)
            non_sub_ds = create_subset_dataset(dataset, non_subset_vars, indexers, cond)
            
            # Merge the datasets
            merged_ds = xr.merge([non_sub_ds, sub_ds])
            
            processed_ds = merged_ds
            processed_ds.attrs.update(dataset.attrs)

            # Restore original data types
            for var, dtype in original_dtypes.items():
                if var in merged_ds:
                    merged_ds[var] = merged_ds[var].astype(dtype)

            # Cast variables and handle fill values
            processed_ds = cast_variables_to_original_types(
                merged_ds,
                indexed_ds,
                dataset,
                indexers,
                partial_dim_in_vars,
                cond
            )
        else:
            processed_ds = dataset.copy()
            processed_ds.attrs.update(dataset.attrs)

        # Process child nodes
        processed_children = {}
        for child_name, child_node in node.children.items():
            # Process the child node
            child_ds, child_children = process_node(child_node)
            
            # Create new DataTree for the processed child
            child_tree = DataTree(name=child_name, dataset=child_ds)
            
            # Add all processed grandchildren to the child tree
            for grandchild_name, grandchild_tree in child_children.items():
                child_tree[grandchild_name] = grandchild_tree
                
            processed_children[child_name] = child_tree

        return processed_ds, processed_children

    # Start processing from root
    root_ds, children = process_node(tree)
    
    # Create new root tree preserving the original name and attributes
    result_tree = DataTree(name=tree.name, dataset=root_ds)
    
    # Add processed children to the result tree
    for child_name, child_tree in children.items():
        result_tree[child_name] = child_tree

    # Copy over root attributes
    result_tree.attrs.update(tree.attrs)

    return result_tree

"""
def where_tree(tree: DataTree, cond: Union[xr.Dataset, xr.DataArray], cut: bool) -> DataTree:

    Return a DataTree which meets the given condition, processing all nodes in the tree.
    
    Parameters
    ----------
    tree : xarray.DataTree
        The input DataTree to filter
    cond : DataArray or Dataset with boolean dtype
        Locations at which to preserve this object's values
    cut : boolean
        True if the scanline should be cut, False if not
        
    Returns
    -------
    xarray.DataTree
        The filtered DataTree with all nodes processed
    def process_node(node: DataTree, path: str) -> Tuple[xr.Dataset, Dict[str, DataTree]]:
        
        Process a single node and its children in the tree.
        
        Parameters
        ----------
        node : DataTree
            The node to process
        path : str
            Current path in the tree
            
        Returns
        -------
        Tuple[xr.Dataset, Dict[str, DataTree]]
            Processed dataset and dictionary of processed child nodes
        
        dataset = node[path]
        
        original_dtypes = {var: dataset[var].dtype for var in dataset.variables}

        # Process current node's dataset
        if len(dataset.variables) > 0:  # Only process if node has data
            # Create indexers from condition

            if cond.values.ndim == 1:
                indexers = get_indexers_from_1d(cond)
            else:
                indexers = get_indexers_from_nd(cond, cut)

            if not all(len(value) > 0 for value in indexers.values()):
                return copy_empty_dataset(dataset), {}

            # Check for partial dimension overlap
            partial_dim_in_vars = check_partial_dim_overlap_node(dataset, indexers)

            # Apply indexing to condition and dataset
            indexed_cond = cond.isel(**indexers)
            try:
                indexed_ds = dataset.isel(**indexers)
            except Exception as ex:
                indexed_ds = dataset

            # Get variables with and without indexers
            subset_vars, non_subset_vars = get_variables_with_indexers(dataset, indexers)
            
            # Process variables
            sub_ds = create_subset_dataset(dataset, subset_vars, indexers, cond)
            non_sub_ds = create_subset_dataset(dataset, non_subset_vars, indexers, cond)
            
            # Merge the datasets
            merged_ds = xr.merge([non_sub_ds, sub_ds])
            
            processed_ds = merged_ds
            processed_ds.attrs.update(dataset.attrs)

            for var, dtype in original_dtypes.items():
                if var in merged_ds:
                    merged_ds[var] = merged_ds[var].astype(dtype)

            # Cast variables and handle fill values
            
            processed_ds = cast_variables_to_original_types(
                merged_ds,
                indexed_ds,
                dataset,
                indexers,
                partial_dim_in_vars,
                cond
            )
        else:
            processed_ds = dataset
            processed_ds.attrs.update(dataset.attrs)

        # Process child nodes
        processed_children = {}
        for child_name, child_node in node.children.items():
            child_path = f"{path}/{child_name}" if path != '/' else f"/{child_name}"
            processed_child_ds, processed_child_children = process_node(child_node, child_path)
            
            # Create new DataTree for the processed child
            child_tree = DataTree(name='', dataset=processed_child_ds)
            for grandchild_name, grandchild_tree in processed_child_children.items():
                child_tree[grandchild_name] = grandchild_tree
                
            processed_children[child_name] = child_tree

        return processed_ds, processed_children

    # Start processing from root
    root_ds, children = process_node(tree, '/')
    
    # Create new root tree
    result_tree = DataTree(name='', dataset=root_ds)
    
    # Add processed children to the result tree
    for child_name, child_tree in children.items():
        result_tree[child_name] = child_tree

    return result_tree
"""

def check_partial_dim_overlap_node(dataset: xr.Dataset, indexers: Dict) -> bool:
    """
    Check if any variables in the dataset have partial dimension overlap with indexers.
    """
    for var_name, var in dataset.variables.items():
        overlap_dims = set(indexers.keys()).intersection(var.dims)
        missing_dims = set(indexers.keys()) - set(var.dims)
        
        if len(overlap_dims) > 0 and len(missing_dims) > 0:
            return True
    return False

def create_subset_dataset(
    dataset: xr.Dataset,
    variables: List[str],
    indexers: Dict,
    cond: Union[xr.Dataset, xr.DataArray]
) -> xr.Dataset:
    """
    Create a dataset for a subset of variables with proper masking.
    """
    subset_dict = {}
    
    for var_name in variables:
        if var_name in dataset:
            if set(indexers.keys()).intersection(dataset[var_name].dims):
                var_indexers = {
                    dim: idx for dim, idx in indexers.items()
                    if dim in dataset[var_name].dims
                }
                indexed_var = dataset[var_name].isel(**var_indexers)
                if var_indexers:
                    var_cond = cond.isel(**var_indexers)
                    subset_dict[var_name] = indexed_var.where(var_cond)
                else:
                    subset_dict[var_name] = indexed_var
            else:
                subset_dict[var_name] = dataset[var_name]
    
    return xr.Dataset(subset_dict)

def get_variables_with_indexers(dataset, indexers):
    """
    returns a list of variables with bounding box dimensions and variables that
    don't have bounding box dimensions
    """
    index_list = list(indexers.keys())

    subset_vars = []
    no_subset_vars = []



    for i in list(dataset.variables.keys()):
        variable_dims = list(dataset[i].dims)
        if any(item in index_list for item in variable_dims):
            subset_vars.append(i)
        else:
            no_subset_vars.append(i)

    return subset_vars, no_subset_vars

def get_fill_value(var: xr.DataArray) -> Union[np.datetime64, np.timedelta64, Any]:
    """
    Get appropriate fill value based on variable type.
    
    Parameters
    ----------
    var : xr.DataArray
        The variable to get fill value for
        
    Returns
    -------
    Union[np.datetime64, np.timedelta64, Any]
        The appropriate fill value for the variable type
    """
    fill_value = var.attrs.get('_FillValue')
    
    if np.issubdtype(var.dtype, np.dtype(np.datetime64)):
        return np.datetime64('nat')
    elif np.issubdtype(var.dtype, np.dtype(np.timedelta64)):
        return np.timedelta64('nat')
    return fill_value

def cast_type(data: xr.DataArray, dtype_str: str) -> xr.DataArray:
    """
    Cast data to the specified dtype.
    
    Parameters
    ----------
    data : xr.DataArray
        The data to cast
    dtype_str : str
        The target dtype as a string
        
    Returns
    -------
    xr.DataArray
        The data cast to the new dtype
    """
    return data.astype(dtype_str)


def copy_empty_dataset(dataset: xr.Dataset) -> xr.Dataset:
    """
    Create an empty copy of a dataset while preserving its structure, coordinates, 
    attributes and dtypes.
    
    Parameters
    ----------
    dataset : xr.Dataset
        The dataset to create an empty copy from
        
    Returns
    -------
    xr.Dataset
        A new dataset with the same structure but empty data arrays
    """
    empty_vars = {}
    
    # Copy each data variable with empty data but preserve attributes and encoding
    for name, var in dataset.data_vars.items():
        # Create empty array with same shape and dtype
        empty_data = np.full(var.shape, fill_value=np.nan, dtype=var.dtype)
        
        # Create new DataArray with empty data but preserve attributes and encoding
        empty_vars[name] = xr.DataArray(
            data=empty_data,
            dims=var.dims,
            coords=var.coords,
            attrs=var.attrs.copy(),
        )
        # Preserve encoding separately as it's not part of attrs
        empty_vars[name].encoding.update(var.encoding)
        
    # Create new dataset with empty variables
    empty_ds = xr.Dataset(empty_vars)
    
    # Copy coordinates that aren't already included
    for coord_name, coord in dataset.coords.items():
        if coord_name not in empty_ds.coords:
            empty_ds[coord_name] = coord.copy()
    
    # Copy dataset attributes
    empty_ds.attrs.update(dataset.attrs)
    
    # Copy dataset encoding
    empty_ds.encoding.update(dataset.encoding)
    
    return empty_ds


def cast_variables_to_original_types(new_dataset, indexed_ds, dataset, indexers, partial_dim_in_in_vars, cond=None):
    """
    Cast variables in a dataset to their original types while handling fill values and dimension indexing.
    
    Parameters
    ----------
    new_dataset : xarray.Dataset
        The target dataset where variables will be modified
    indexed_ds : xarray.Dataset
        The dataset containing the original data types
    dataset : xarray.Dataset
        The source dataset
    indexers : dict
        Dictionary of dimension names and their corresponding index values
    partial_dim_in_in_vars : bool
        Flag indicating if there are partial dimensions in input variables
    cond : xarray.DataArray, optional
        Conditional mask for filtering data
        
    Returns
    -------
    xarray.Dataset
        Modified dataset with variables cast to their original types
    """
    def cast_type(data, dtype_str):
        """Helper function to cast data to specified type"""
        return data.astype(dtype_str)
    
    for variable_name, variable in new_dataset.data_vars.items():
        original_type = indexed_ds[variable_name].dtype
        new_type = variable.dtype
        indexed_var = indexed_ds[variable_name]

        # Handle partial dimension indexing
        if partial_dim_in_in_vars and (indexers.keys() - dataset[variable_name].dims) and set(
                indexers.keys()).intersection(dataset[variable_name].dims):

            missing_dim = (indexers.keys() - dataset[variable_name].dims).pop()  # Assume only 1
            var_indexers = {
                dim_name: dim_value for dim_name, dim_value in indexers.items()
                if dim_name in dataset[variable_name].dims
            }

            var_cond = cond.any(axis=cond.dims.index(missing_dim)).isel(**var_indexers)
            indexed_var = dataset[variable_name].isel(**var_indexers)
            new_dataset[variable_name] = indexed_var.where(var_cond)
            variable = new_dataset[variable_name]
            
        elif partial_dim_in_in_vars and (indexers.keys() - dataset[variable_name].dims) and set(
                indexers.keys()).intersection(new_dataset[variable_name].dims):
            new_dataset[variable_name] = indexed_var
            new_dataset[variable_name].attrs = indexed_var.attrs
            variable.attrs = indexed_var.attrs

        # Handle variables without _FillValue or scalar variables
        if '_FillValue' not in variable.attrs or len(indexed_var.shape) == 0:
            if original_type != new_type:
                new_dataset[variable_name] = xr.apply_ufunc(
                    cast_type, 
                    variable,
                    str(original_type), 
                    dask='allowed',
                    keep_attrs=True
                )
            new_dataset[variable_name] = indexed_var
            new_dataset[variable_name].attrs = indexed_var.attrs
            variable.attrs = indexed_var.attrs
            new_dataset[variable_name].encoding['_FillValue'] = None
            variable.encoding['_FillValue'] = None

        else:
            # Handle variables with _FillValue
            fill_value = new_dataset[variable_name].attrs.get('_FillValue')
            
            # Special handling for datetime and timedelta types
            if np.issubdtype(new_dataset[variable_name].dtype, np.dtype(np.datetime64)):
                fill_value = np.datetime64('nat')
            if np.issubdtype(new_dataset[variable_name].dtype, np.dtype(np.timedelta64)):
                fill_value = np.timedelta64('nat')
                
            new_dataset[variable_name] = new_dataset[variable_name].fillna(fill_value)
            
            if original_type != new_type:
                new_dataset[variable_name] = xr.apply_ufunc(
                    cast_type, 
                    new_dataset[variable_name],
                    str(original_type), 
                    dask='allowed',
                    keep_attrs=True
                )
    
    return new_dataset


def compute_coordinate_variable_names_from_tree(tree) -> Tuple[List[str], List[str]]:
    """
    Recursively search for latitude and longitude coordinate variables in a DataTree
    and return their full paths.

    Parameters
    ----------
    tree : DataTree
        The DataTree to search.

    Returns
    -------
    tuple
        - List of latitude coordinate names prefixed with their full path.
        - List of longitude coordinate names prefixed with their full path.
    """
    lat_coord_names = []
    lon_coord_names = []

    def find_coords_in_dataset(dataset: xr.Dataset, path: str):
        """Find latitude and longitude variable names in a single dataset and track full paths."""
        
        possible_lat_coord_names = {'lat', 'latitude', 'y'}
        possible_lon_coord_names = {'lon', 'longitude', 'x'}

        current_lat_coord_names = []
        current_lon_coord_names = []

        custom_criteria = {
            "latitude": {"standard_name": "latitude|projection_y_coordinate"},
            "longitude": {"standard_name": "longitude|projection_x_coordinate"},
        }

        dataset = xr.decode_cf(dataset)

        def append_coords_matching(names_set, coords_list):
            """Append matching coordinates to the list."""
            for var_name in dataset.variables:
                if var_name.lower() in names_set:
                    coords_list.append(f"{path}/{var_name}")

        # First check for direct matches
        append_coords_matching(possible_lat_coord_names, current_lat_coord_names)
        append_coords_matching(possible_lon_coord_names, current_lon_coord_names)

        # Fallback: check metadata for coordinates if not found
        if not current_lat_coord_names or not current_lon_coord_names:
            lat_matches = find_matching_coords(dataset, possible_lat_coord_names)
            lon_matches = find_matching_coords(dataset, possible_lon_coord_names)

            current_lat_coord_names.extend(f"{path}/{lat}" for lat in lat_matches)
            current_lon_coord_names.extend(f"{path}/{lon}" for lon in lon_matches)

            if not current_lat_coord_names or not current_lon_coord_names:
                with cfxr.set_options(custom_criteria=custom_criteria):
                    possible_lat_coords = dataset.cf.coordinates.get('latitude', [])
                    possible_lon_coords = dataset.cf.coordinates.get('longitude', [])
                    if possible_lat_coords:
                        current_lat_coord_names.append(f"{path}/{possible_lat_coords[0]}")
                    if possible_lon_coords:
                        current_lon_coord_names.append(f"{path}/{possible_lon_coords[0]}")

                if not current_lat_coord_names or not current_lon_coord_names:
                    try:
                        if "latitude" in dataset.cf and "longitude" in dataset.cf:
                            current_lat_coord_names.append(f"{path}/{dataset.cf['latitude'].name}")
                            current_lon_coord_names.append(f"{path}/{dataset.cf['longitude'].name}")
                    except KeyError:
                        pass

        return current_lon_coord_names, current_lat_coord_names

    def traverse_tree(node, path):
        """Recursively search through the tree for latitude and longitude coordinates."""
        if node.ds is not None:
            return_lon, return_lat = find_coords_in_dataset(node.ds, path)
            lon_coord_names.extend(return_lon)
            lat_coord_names.extend(return_lat)

        for child_name, child_node in node.children.items():
            new_path = f"{path}/{child_name}" if path else child_name
            traverse_tree(child_node, new_path)

    # Start recursive tree traversal
    traverse_tree(tree, "")

    if not lat_coord_names or not lon_coord_names:
        raise ValueError("Could not determine coordinate variables in the DataTree")

    return lon_coord_names, lat_coord_names

def find_matching_coords(dataset: xr.Dataset, match_list: List[str]) -> List[str]:
    """
    As a backup for finding a coordinate var, look at the 'coordinates'
    metadata attribute of all data vars in the granule. Return any
    coordinate vars that have name matches with the provided
    'match_list'

    Parameters
    ----------
    dataset : xr.Dataset
        Dataset to search data variable coordinate metadata attribute
    match_list : list (str)
        List of possible matches to search for. For example,
        ['lat', 'latitude'] would search for variables in the
        'coordinates' metadata attribute containing either 'lat'
        or 'latitude'

    Returns
    -------
    list (str)
        List of matching coordinate variables names
    """
    coord_attrs = [
        var.attrs['coordinates'] for var_name, var in dataset.data_vars.items()
        if 'coordinates' in var.attrs
    ]
    coord_attrs = list(set(coord_attrs))
    match_coord_vars = []
    for coord_attr in coord_attrs:
        coords = coord_attr.split(' ')
        match_vars = [
            coord for coord in coords
            if any(coord_cand in coord for coord_cand in match_list)
        ]
        if match_vars and match_vars[0] in dataset:
            # Check if the var actually exists in the dataset
            match_coord_vars.append(match_vars[0])
    return match_coord_vars




def compute_time_variable_name_tree(tree, lat_var, total_time_vars):

    time_coord_name = []

    def find_time_in_dataset(dataset: xr.Dataset, lat_var: xr.Variable, path: str, total_time_vars: Set[str]) -> Optional[str]:
        """
        Find the time variable name in a dataset using various criteria.

        The search is performed in order of reliability:
        1. Coordinates with 'time' in name
        2. Variables with explicit time metadata
        3. Variables with 'time' in name matching lat dimensions
        4. Variables with specific time-related names

        Parameters
        ----------
        dataset : xr.Dataset
            xarray dataset to search
        lat_var : xr.Variable
            Latitude variable for dimension matching
        total_time_vars : Set[str]
            Set of previously found time variables to exclude

        Returns
        -------
        Optional[str]
            Name of found time variable or None if not found
        """
        lat_dims = lat_var.squeeze().dims
        
        # Cache squeezed dimensions to avoid repeated computations
        dim_cache = {}
        def get_squeezed_dims(var_name: str) -> tuple:
            if var_name not in dim_cache:
                dim_cache[var_name] = dataset[var_name].squeeze().dims
            return dim_cache[var_name]

        # Check coordinates first (most likely location)
        for coord_name in dataset.coords:
            if 'time' in coord_name.lower() and coord_name not in total_time_vars:
                if get_squeezed_dims(coord_name) == lat_dims:
                    return coord_name

        # Compile regex pattern once
        time_units_pattern = re.compile(
            r"(days?|hours?|hr|minutes?|min|seconds?|sec|s) since \d{4}-\d{2}-\d{2}( \d{2}:\d{2}:\d{2})?",
            re.IGNORECASE
        )

        # Check variables with time-related metadata
        for var_name, var in dataset.variables.items():
            if var_name in total_time_vars:
                continue
                
            # Check metadata indicators
            if any([
                var.attrs.get('standard_name') == 'time',
                var.attrs.get('axis') == 'T',
                ('units' in var.attrs and time_units_pattern.match(var.attrs['units']))
            ]):
                return var_name

        # Check variables with 'time' in name and matching dimensions
        for var_name in dataset.variables:
            if var_name in total_time_vars:
                continue
                
            dims = get_squeezed_dims(var_name)
            if not dims:  # Skip dimensionless variables
                continue
                
            var_basename = var_name.strip(GROUP_DELIM).split(GROUP_DELIM)[-1].lower()
            
            # Check for exact time variable names first
            if var_basename in {'time', 'timemidscan'} and dims[0] in lat_dims:
                return var_name
                
            # Then check for 'time' in name
            if 'time' in var_basename and dims[0] in lat_dims:
                return var_name

        return None

    def traverse_tree(node, path):
        print(node)
        """Recursively search through the tree for latitude and longitude coordinates."""
        if node.ds is not None:
            return_time = find_time_in_dataset(node.ds, lat_var, path, total_time_vars)
            if return_time:
                print("FOUND TIME")
                time_var = f"{path}/{return_time}"
                return time_var
                #print(time_var)
                #time_coord_name.append(time_var)

        for child_name, child_node in node.children.items():
            new_path = f"{path}/{child_name}" if path else child_name
            traverse_tree(child_node, new_path)

    # Start recursive tree traversal
    return traverse_tree(tree, "")

def compute_lon_lat_time_variable_name_tree(tree, total_time_vars):

    lat_coord_names = []
    lon_coord_names = []
    time_coord_names = []

    def find_coords_in_dataset(dataset: xr.Dataset, path: str):
        """Find latitude and longitude variable names in a single dataset and track full paths."""
        
        possible_lat_coord_names = {'lat', 'latitude', 'y'}
        possible_lon_coord_names = {'lon', 'longitude', 'x'}

        current_lat_coord_names = []
        current_lon_coord_names = []

        custom_criteria = {
            "latitude": {"standard_name": "latitude|projection_y_coordinate"},
            "longitude": {"standard_name": "longitude|projection_x_coordinate"},
        }

        dataset = xr.decode_cf(dataset)

        def append_coords_matching(names_set, coords_list):
            """Append matching coordinates to the list."""
            for var_name in dataset.variables:
                if var_name.lower() in names_set:
                    coords_list.append(f"{path}/{var_name}")

        # First check for direct matches
        append_coords_matching(possible_lat_coord_names, current_lat_coord_names)
        append_coords_matching(possible_lon_coord_names, current_lon_coord_names)

        # Fallback: check metadata for coordinates if not found
        if not current_lat_coord_names or not current_lon_coord_names:
            lat_matches = find_matching_coords(dataset, possible_lat_coord_names)
            lon_matches = find_matching_coords(dataset, possible_lon_coord_names)

            current_lat_coord_names.extend(f"{path}/{lat}" for lat in lat_matches)
            current_lon_coord_names.extend(f"{path}/{lon}" for lon in lon_matches)

            if not current_lat_coord_names or not current_lon_coord_names:
                with cfxr.set_options(custom_criteria=custom_criteria):
                    possible_lat_coords = dataset.cf.coordinates.get('latitude', [])
                    possible_lon_coords = dataset.cf.coordinates.get('longitude', [])
                    if possible_lat_coords:
                        current_lat_coord_names.append(f"{path}/{possible_lat_coords[0]}")
                    if possible_lon_coords:
                        current_lon_coord_names.append(f"{path}/{possible_lon_coords[0]}")

                if not current_lat_coord_names or not current_lon_coord_names:
                    try:
                        if "latitude" in dataset.cf and "longitude" in dataset.cf:
                            current_lat_coord_names.append(f"{path}/{dataset.cf['latitude'].name}")
                            current_lon_coord_names.append(f"{path}/{dataset.cf['longitude'].name}")
                    except KeyError:
                        pass

        return current_lon_coord_names, current_lat_coord_names

    def find_time_in_dataset(dataset: xr.Dataset, lat_var: xr.Variable, path: str, total_time_vars: Set[str]) -> Optional[str]:
        """
        Find the time variable name in a dataset using various criteria.

        The search is performed in order of reliability:
        1. Coordinates with 'time' in name
        2. Variables with explicit time metadata
        3. Variables with 'time' in name matching lat dimensions
        4. Variables with specific time-related names

        Parameters
        ----------
        dataset : xr.Dataset
            xarray dataset to search
        lat_var : xr.Variable
            Latitude variable for dimension matching
        total_time_vars : Set[str]
            Set of previously found time variables to exclude

        Returns
        -------
        Optional[str]
            Name of found time variable or None if not found
        """
        lat_dims = lat_var.squeeze().dims
        
        # Cache squeezed dimensions to avoid repeated computations
        dim_cache = {}
        def get_squeezed_dims(var_name: str) -> tuple:
            if var_name not in dim_cache:
                dim_cache[var_name] = dataset[var_name].squeeze().dims
            return dim_cache[var_name]

        # Check coordinates first (most likely location)
        for coord_name in dataset.coords:
            if 'time' in coord_name.lower() and coord_name not in total_time_vars:
                if get_squeezed_dims(coord_name) == lat_dims:
                    return coord_name

        # Compile regex pattern once
        time_units_pattern = re.compile(
            r"(days?|hours?|hr|minutes?|min|seconds?|sec|s) since \d{4}-\d{2}-\d{2}( \d{2}:\d{2}:\d{2})?",
            re.IGNORECASE
        )

        # Check variables with time-related metadata
        for var_name, var in dataset.variables.items():
            if var_name in total_time_vars:
                continue
                
            # Check metadata indicators
            if any([
                var.attrs.get('standard_name') == 'time',
                var.attrs.get('axis') == 'T',
                ('units' in var.attrs and time_units_pattern.match(var.attrs['units']))
            ]):
                return var_name

        # Check variables with 'time' in name and matching dimensions
        for var_name in dataset.variables:
            if var_name in total_time_vars:
                continue
                
            dims = get_squeezed_dims(var_name)
            if not dims:  # Skip dimensionless variables
                continue
                
            var_basename = var_name.strip(GROUP_DELIM).split(GROUP_DELIM)[-1].lower()
            
            # Check for exact time variable names first
            if var_basename in {'time', 'timemidscan'} and dims[0] in lat_dims:
                return var_name
                
            # Then check for 'time' in name
            if 'time' in var_basename and dims[0] in lat_dims:
                return var_name

        return None

    def traverse_tree(node, path):
        """Recursively search through the tree for latitude and longitude coordinates."""
        if node.ds is not None:
            return_lon, return_lat = find_coords_in_dataset(node.ds, path)
            lon_coord_names.extend(return_lon)
            lat_coord_names.extend(return_lat)

            if return_lat:
                return_time = find_time_in_dataset(node.ds, return_lat, path, total_time_vars)
                if return_time:
                    time_var = f"{path}/{return_time}"
                    time_coord_name.append(time_var)


        for child_name, child_node in node.children.items():
            new_path = f"{path}/{child_name}" if path else child_name
            traverse_tree(child_node, new_path)

    # Start recursive tree traversal
    traverse_tree(tree, "")

