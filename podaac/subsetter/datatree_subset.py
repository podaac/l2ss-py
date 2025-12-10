"""script to help with subsetting xarray datatree objects"""

# pylint: disable=inconsistent-return-statements
import datetime
import logging
import re
from typing import Dict, List, Set, Tuple, Union

import cf_xarray as cfxr
import numpy as np
import xarray as xr
from xarray import DataTree
from netCDF4 import date2num  # pylint: disable=no-name-in-module
from podaac.subsetter import dimension_cleanup as dc
from podaac.subsetter.utils import mask_utils
try:
    from harmony_service_lib.exceptions import NoDataException
except ImportError:
    class NoDataException(Exception):
        """Fallback exception if harmony_service_lib is not available."""


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

    indexers = {
        cond_dims[y_axis]: np.where(rows)[0],
        cond_dims[x_axis]: np.where(cols)[0]
    }

    return indexers


def get_sibling_or_parent_condition(condition_dict, path):
    """
    Retrieve a condition from a dictionary based on a given path, prioritizing
    parent paths first, then sibling paths with the same immediate parent,
    then other siblings, and finally the root condition.

    The function first attempts to find the closest parent match by walking up
    the directory-like hierarchy. If no parent is found, it looks for a sibling
    at the same depth that shares the same immediate parent. If neither is found,
    it returns the condition for the root ("/") if available.

    Args:
        condition_dict (dict): A dictionary mapping paths (keys) to conditions (values).
        path (str): The path for which to find a matching condition.

    Returns:
        Any: The condition found in the dictionary, or None if no match exists.
    """
    # Normalize the path by removing trailing slashes
    path = path.rstrip('/')

    # First try to find parent match by walking up the tree
    current_path = path
    while current_path:
        if current_path in condition_dict:
            return condition_dict[current_path]
        current_path = "/".join(current_path.split("/")[:-1])
        if not current_path or current_path == "":
            break

    # If no parent found, look for sibling match with same immediate parent
    path_parts = path.split("/")
    for potential_path in condition_dict:
        potential_path_clean = potential_path.rstrip('/')
        potential_parts = potential_path_clean.split("/")
        if len(path_parts) == len(potential_parts):
            # Check if immediate parent matches
            if path_parts[:-1] == potential_parts[:-1] and potential_path_clean != path:
                return condition_dict[potential_path]

    # If no such sibling found, fall back to original sibling logic (common grandparent)
    for potential_path in condition_dict:
        potential_path_clean = potential_path.rstrip('/')
        potential_parts = potential_path_clean.split("/")
        if len(path_parts) == len(potential_parts):
            # Check if they share the same structure up to grandparent
            if all(p1 == p2 for p1, p2 in zip(path_parts[:-2], potential_parts[:-2])) and potential_path_clean != path:
                return condition_dict[potential_path]

    # If no parent or sibling found, return root condition if it exists
    return condition_dict.get("/", None)


def is_empty(dt, check_attrs=False):
    """
    Check if a DataTree node is empty.
    If check_attrs is True, only require data_vars, ds.attrs, and dt.attrs to be empty.
    If check_attrs is False, require both data_vars and coords to be empty.
    """
    ds = dt.ds
    if ds is None:
        return True
    if check_attrs:
        return len(ds.data_vars) == 0 and len(ds.attrs) == 0 and len(dt.attrs) == 0
    return len(ds.data_vars) == 0 and len(ds.coords) == 0


def subtree_is_empty(dt, check_attrs=False):
    """
    Check if a DataTree entire tree is empty.
    """
    if not is_empty(dt, check_attrs):
        return False
    return all(subtree_is_empty(child, check_attrs) for child in dt.children.values())


def find_fully_empty_paths(dt: xr.DataTree):
    """
    Returns a list of paths in the DataTree where the node and all descendants are empty.
    """
    results = []
    if subtree_is_empty(dt):
        results.append(dt.path)  # dt.path is typically a tuple like ("root", "child1", ...)
    else:
        for child in dt.children.values():
            results.extend(find_fully_empty_paths(child))
    return results


def where_tree(tree: DataTree, condition_dict, cut: bool, pixel_subset=False) -> DataTree:
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
    pixel_subset : boolean
        Cut the lon lat based on the rows and columns within the bounding box,
        but could result with lon lats that are outside the bounding box

    Returns
    -------
    xarray.DataTree
        The filtered DataTree with all nodes processed
    """
    def process_node(node: DataTree, path: str, empty_paths) -> Tuple[xr.Dataset, Dict[str, DataTree]]:  # pylint: disable=too-many-branches
        """
        Process a single node and its children in the tree.

        Parameters
        ----------
        node : DataTree
            The node to process
        path : str
            The current path of the node

        Returns
        -------
        Tuple[xr.Dataset, Dict[str, DataTree]]
            Processed dataset and dictionary of processed child nodes
        """
        cond = get_sibling_or_parent_condition(condition_dict, path)

        # if only one condition in dictionary then get the one condition
        if cond is None:
            if len(condition_dict) == 1:
                _, cond = next(iter(condition_dict.items()))

        # Get the dataset directly from the node
        dataset = node.ds
        dataset = dc.remove_duplicate_dims_xarray(dataset)

        indexers = None

        if dataset.variables and cond is not None:  # Only process if node has data
            # Create indexers from condition
            cond = mask_utils.align_dims_cond_only(dataset, cond)

            if cond.values.ndim == 1:
                indexers = get_indexers_from_1d(cond)
            else:
                indexers = get_indexers_from_nd(cond, cut)
            if not all(len(value) > 0 for value in indexers.values()):
                raise NoDataException("No data in subsetted granule.")

            # Check for partial dimension overlap
            partial_dim_in_vars = check_partial_dim_overlap_node(dataset, indexers)
            partial_dim_in_in_vars = partial_dim_in_vars

            indexed_cond = cond.isel(**indexers)
            indexed_ds = dataset.isel(**indexers, missing_dims='ignore')

            if pixel_subset:
                new_dataset = indexed_ds
            else:
                # Get variables with and without indexers
                subset_vars, non_subset_vars = get_variables_with_indexers(dataset, indexers)

                new_dataset_sub = indexed_ds[subset_vars].where(indexed_cond)

                # data with variables that shouldn't be subsetted
                new_dataset_non_sub = indexed_ds[non_subset_vars]

                # Merge the datasets
                new_dataset = xr.merge([new_dataset_non_sub, new_dataset_sub])

            new_dataset.attrs.update(dataset.attrs)

            # Cast all variables to their original type
            for variable_name, variable in new_dataset.data_vars.items():
                original_type = indexed_ds[variable_name].dtype
                new_type = variable.dtype
                indexed_var = indexed_ds[variable_name]

                if partial_dim_in_in_vars and (indexers.keys() - dataset[variable_name].dims) and set(
                        indexers.keys()).intersection(dataset[variable_name].dims):

                    missing_dim = sorted(indexers.keys() - dataset[variable_name].dims)[0]
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
                # Check if variable has no _FillValue. If so, use original data
                if '_FillValue' not in variable.attrs or len(indexed_var.shape) == 0:
                    if original_type != new_type:
                        new_dataset[variable_name] = xr.apply_ufunc(cast_type, variable,
                                                                    str(original_type), dask='allowed',
                                                                    keep_attrs=True)

                    # Replace nans with values from original dataset. If the
                    # variable has more than one dimension, copy the entire
                    # variable over, otherwise use a NaN mask to copy over the
                    # relevant values.
                    new_dataset[variable_name] = indexed_var
                    new_dataset[variable_name].attrs = indexed_var.attrs
                    variable.attrs = indexed_var.attrs
                    new_dataset[variable_name].encoding['_FillValue'] = None
                    variable.encoding['_FillValue'] = None

                else:
                    # Manually replace nans with FillValue
                    # If variable represents time, cast _FillValue to datetime
                    fill_value = new_dataset[variable_name].attrs.get('_FillValue')

                    if np.issubdtype(new_dataset[variable_name].dtype, np.dtype(np.datetime64)):
                        fill_value = np.datetime64('nat')
                    if np.issubdtype(new_dataset[variable_name].dtype, np.dtype(np.timedelta64)):
                        fill_value = np.timedelta64('nat')
                    new_dataset[variable_name] = new_dataset[variable_name].fillna(fill_value)
                    if original_type != new_type:
                        new_dataset[variable_name] = xr.apply_ufunc(cast_type, new_dataset[variable_name],
                                                                    str(original_type), dask='allowed',
                                                                    keep_attrs=True)
            processed_ds = new_dataset
            dc.sync_dims_inplace(dataset, processed_ds)
        else:
            processed_ds = dataset.copy()
            processed_ds.attrs.update(dataset.attrs)

        processed_children = {}
        for child_name, child_node in node.children.items():
            # Process the child node
            current_path = f"{path}/{child_name}"
            if current_path in empty_paths:
                processed_children[child_name] = child_node
            else:
                child_ds, child_children, child_indexers = process_node(child_node, current_path, empty_paths)

                # --- Align parent and child datasets before attaching child ---
                if indexers is None and child_indexers:
                    indexers = child_indexers
                    processed_ds = processed_ds.isel(**child_indexers, missing_dims='ignore')

                # Create new DataTree for the processed child
                child_tree = DataTree(name=child_name, dataset=child_ds)

                # Add all processed grandchildren to the child tree
                for grandchild_name, grandchild_tree in child_children.items():
                    child_tree[grandchild_name] = grandchild_tree

                child_tree_empty = subtree_is_empty(child_tree, check_attrs=True)
                # trees that have no data and attributes after processing are empty so we don't want to attach them
                if child_tree_empty is False:
                    processed_children[child_name] = child_tree

        return processed_ds, processed_children, indexers

    empty_paths = find_fully_empty_paths(tree)
    # Start processing from root
    root_ds, children, _ = process_node(tree, '', empty_paths)

    # Create new root tree preserving the original name and attributes
    result_tree = DataTree(name=tree.name, dataset=root_ds)

    # Add processed children to the result tree
    for child_name, child_tree in children.items():
        result_tree[child_name] = child_tree

    # Copy over root attributes
    result_tree.attrs.update(tree.attrs)

    return result_tree


def check_partial_dim_overlap_node(dataset: xr.Dataset, indexers: Dict) -> bool:
    """
    Check if any variables in the dataset have partial dimension overlap with indexers.
    """
    for _, var in dataset.variables.items():
        overlap_dims = set(indexers.keys()).intersection(var.dims)
        missing_dims = set(indexers.keys()) - set(var.dims)

        if len(overlap_dims) > 0 and len(missing_dims) > 0:
            return True
    return False


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

        possible_pairs = [
            ('lat', 'lon'),
            ('latitude', 'longitude'),
            ('y', 'x')
        ]

        current_lat_coord_names = []
        current_lon_coord_names = []

        dataset = xr.decode_cf(dataset)

        def append_coords_pair(pair):
            lat_name, lon_name = pair
            lat_found = None
            lon_found = None
            for var_name in dataset.variables:
                lname = var_name.lower()
                if lname == lat_name:
                    lat_found = f"{path}/{var_name}"
                if lname == lon_name:
                    lon_found = f"{path}/{var_name}"
            if lat_found and lon_found:
                current_lat_coord_names.append(lat_found)
                current_lon_coord_names.append(lon_found)

        for pair in possible_pairs:
            append_coords_pair(pair)

        custom_criteria = {
            "latitude": {"standard_name": "latitude|projection_y_coordinate"},
            "longitude": {"standard_name": "longitude|projection_x_coordinate"},
        }

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
    """Attempt to get the time variable for a datatree by applying all method 1 first, then 2, etc."""

    def get_all_datasets(node, path=""):
        """Recursively collect all datasets in the tree with their paths."""
        datasets = []
        if node.ds is not None:
            datasets.append((path, node.ds))
        for child_name, child_node in node.children.items():
            child_path = f"{path}/{child_name}" if path else child_name
            datasets.extend(get_all_datasets(child_node, child_path))
        return datasets

    def method_1(path, ds):
        for coord_name in ds.coords:
            if 'time' == coord_name.lower() and coord_name not in total_time_vars:
                if ds[coord_name].squeeze().dims == lat_var.squeeze().dims:
                    return f"{path}/{coord_name}"
        return None

    def method_2(path, ds):
        pattern = re.compile(
            r"(days?|hours?|hr|minutes?|min|seconds?|sec|s) since \d{4}-\d{2}-\d{2}( \d{2}:\d{2}:\d{2})?",
            re.IGNORECASE
        )
        for var_name, var in ds.variables.items():
            if var_name in total_time_vars:
                continue
            if any([
                var.attrs.get("standard_name") == "time",
                var.attrs.get("axis") == "T",
                ("units" in var.attrs and pattern.match(var.attrs["units"]))
            ]):
                if var.size > 1:
                    return f"{path}/{var_name}"
        return None

    def method_3(path, ds):
        # Only proceed if both 'time' and 'sst_dtime' exist in the dataset variables
        if 'time' in ds.variables and 'sst_dtime' in ds.variables:
            time_var = ds['time']
            # Check the long_name attribute for the 'time' variable
            if time_var.attrs.get('long_name', None) == "reference time of sst file":
                return f"{path}/time"
        return None

    def method_4(path, ds):
        lat_dims = lat_var.squeeze().dims
        for var_name in ds.variables:
            if var_name in total_time_vars:
                continue
            dims = ds[var_name].squeeze().dims
            if not dims:
                continue
            if 'time' == var_name.lower() and dims[0] in lat_dims:
                return f"{path}/{var_name}"
        return None

    def method_5(path, ds):
        lat_dims = lat_var.squeeze().dims
        for var_name in ds.variables:
            if var_name in total_time_vars:
                continue
            dims = ds[var_name].squeeze().dims
            if not dims:
                continue
            if 'time' in var_name.lower() and dims[0] in lat_dims:
                return f"{path}/{var_name}"
        return None

    def method_6(path, ds):
        lat_dims = lat_var.squeeze().dims
        for var_name in ds.variables:
            if var_name in total_time_vars:
                continue
            dims = ds[var_name].squeeze().dims
            if not dims:
                continue
            var_basename = var_name.strip(GROUP_DELIM).split(GROUP_DELIM)[-1].lower()
            if var_basename in {'time', 'timemidscan'} and dims[0] in lat_dims:
                return f"{path}/{var_name}"
        return None

    # Start
    if isinstance(tree, xr.Dataset):
        tree = DataTree(dataset=tree)

    all_datasets = get_all_datasets(tree)

    # Try each method across the entire tree, in priority order
    for method in [method_1, method_2, method_3, method_4, method_5, method_6]:
        for path, ds in all_datasets:
            result = method(path, ds)
            if result:
                return result
    return None


def remove_scale_offset(value: float, scale: float, offset: float) -> float:
    """Remove scale and offset from the given value"""
    return (value * scale) - offset


def tree_get_spatial_bounds(datatree: xr.Dataset, lat_var_names: List[str], lon_var_names: List[str]) -> Union[np.ndarray, None]:
    """
    Get the spatial bounds for this dataset tree. These values are masked and scaled.

    Parameters
    ----------
    datatree : xr.Dataset
        Dataset tree to retrieve spatial bounds for
    lat_var_names : List[str]
        List of paths to latitude variables
    lon_var_names : List[str]
        List of paths to longitude variables

    Returns
    -------
    np.array
        [[lon min, lon max], [lat min, lat max]]
    """
    if len(lat_var_names) != len(lon_var_names):
        raise ValueError("Number of latitude and longitude paths must match")

    min_lats, max_lats, min_lons, max_lons = [], [], [], []

    for lat_var_name, lon_var_name in zip(lat_var_names, lon_var_names):
        try:
            # Get variables from paths
            lat_data = datatree[lat_var_name]
            lon_data = datatree[lon_var_name]

            # Get metadata attributes efficiently
            lat_attrs = lat_data.attrs
            lon_attrs = lon_data.attrs

            # Extract metadata with defaults
            lat_scale = lat_attrs.get('scale_factor', 1.0)
            lon_scale = lon_attrs.get('scale_factor', 1.0)
            lat_offset = lat_attrs.get('add_offset', 0.0)
            lon_offset = lon_attrs.get('add_offset', 0.0)
            lon_valid_min = lon_attrs.get('valid_min', None)

            # Flatten and mask data
            lats = lat_data.values.flatten()
            lons = lon_data.values.flatten()

            # Apply fill value masks if present
            lat_fill = lat_attrs.get('_FillValue')
            lon_fill = lon_attrs.get('_FillValue')

            if lat_fill is not None:
                lats = lats[lats != lat_fill]
            if lon_fill is not None:
                lons = lons[lons != lon_fill]

            if len(lats) == 0 or len(lons) == 0:
                continue

            original_min_lat = remove_scale_offset(np.nanmin(lats), lat_scale, lat_offset)
            original_max_lat = remove_scale_offset(np.nanmax(lats), lat_scale, lat_offset)
            original_min_lon = remove_scale_offset(np.nanmin(lons), lon_scale, lon_offset)
            original_max_lon = remove_scale_offset(np.nanmax(lons), lon_scale, lon_offset)

            min_lat = round(original_min_lat, 5)
            max_lat = round(original_max_lat, 5)
            min_lon = round(original_min_lon, 1)
            max_lon = round(original_max_lon, 1)

            # Convert longitude to [-180,180] format
            if lon_valid_min == 0 or 0 <= min_lon <= max_lon <= 360:
                min_lon = (min_lon - 360) if min_lon > 180 else min_lon
                max_lon = (max_lon - 360) if max_lon > 180 else max_lon
                if min_lon == max_lon:
                    min_lon, max_lon = -180, 180

            # After rounding to 1 if not at the edges then round to 5
            if min_lon != -180:
                min_lon = round((original_min_lon - 360) if original_min_lon > 180 else original_min_lon, 5)

            if max_lon != 180:
                max_lon = round((original_max_lon - 360) if original_max_lon > 180 else original_max_lon, 5)

            min_lats.append(min_lat)
            max_lats.append(max_lat)
            min_lons.append(min_lon)
            max_lons.append(max_lon)

        except (KeyError, AttributeError):
            continue

    if not min_lats:  # If no valid bounds were found
        return None

    # Calculate overall bounds using numpy operations
    return np.array([
        [min(min_lons), max(max_lons)],
        [min(min_lats), max(max_lats)]
    ])


def get_vars_with_paths(tree: DataTree) -> List[str]:
    """
    Get all variables and coordinates with their full paths from a DataTree

    Parameters
    ----------
    tree : DataTree
        The input DataTree

    Returns
    -------
    List[str]
        List of variable paths in format '/group/var' or '/var' for root level,
        including coordinate variables at root level

    Examples
    --------
    >>> ds = xr.Dataset({'var1': [1], 'var2': [2], 'time': ('time', [0])})
    >>> tree = DataTree(data=ds)
    >>> tree['group1'] = DataTree(data=ds.copy())
    >>> paths = get_vars_with_paths(tree)
    >>> print(paths)
    ['/time', '/var1', '/var2', '/group1/var1', '/group1/var2']
    """
    paths = []

    def collect_vars(node: DataTree, current_path: str = '') -> None:
        # Add data variables from current node
        for var_name in node.ds.data_vars:
            paths.append(f'{current_path}/{var_name}')

        # Recursively process child nodes
        for child_name in node.children:
            new_path = f'{current_path}/{child_name}' if current_path else f'/{child_name}'
            collect_vars(node[child_name], new_path)

    collect_vars(tree)
    return sorted(paths)  # Sort for consistent ordering


def drop_vars_by_path(tree: DataTree, var_paths: Union[str, List[str]]) -> DataTree:
    """
    Drop variables from a DataTree using paths in the format '/group/var' or '/var' for root level

    Parameters
    ----------
    tree : DataTree
        The input DataTree
    var_paths : str or List[str]
        Paths to variables to drop in format '/group/var' or '/var' for root level
        Examples:
            - '/var1'  # root level variable
            - '/group1/var1'  # variable in group1
            - '/group1/subgroup/var1'  # variable in nested group

    Returns
    -------
    DataTree
        Modified DataTree with variables dropped
    """
    if isinstance(var_paths, str):
        var_paths = [var_paths]

    for path in var_paths:
        # Split the path into group path and variable name
        parts = path.strip('/').split('/')

        if len(parts) == 1:
            # Root level variable
            var_name = parts[0]
            # Modify the dataset in-place using xarray's drop_vars
            tree.ds = tree.ds.drop_vars([var_name], errors='ignore')
        else:
            # Group variable
            group_path = '/'.join(parts[:-1])
            var_name = parts[-1]
            try:
                node = tree[group_path]
                node.ds = node.ds.drop_vars([var_name], errors='ignore')
            except KeyError:
                pass

    return tree


def prepare_basic_encoding(datasets: DataTree, time_encoding) -> dict:
    """
    Prepare basic encoding dictionary for DataTree organized by groups.
    Only applies zlib and complevel for float32, float64, int32, uint16 datatypes.
    All paths start with '/' for root and nested groups.

    Args:
        datasets: xarray DataTree
    Returns:
        dict: Dictionary structure {'/group': {var: encoding, ...}, ...}
    """
    group_encodings = {}

    # Types that should have compression
    compress_types = {
        'float32', 'float64',  # Floating point
        'int8', 'int16', 'int32', 'int64',  # Signed integers
        'uint8', 'uint16', 'uint32', 'uint64'  # Unsigned integers
    }

    def process_node(node: DataTree, group_path: str):
        # Initialize encoding dict for this group
        var_encodings = {}

        # Process only data variables in this group
        for var_name in node.ds.data_vars:
            var = node.ds[var_name]

            encoding = {
                "_FillValue": var.encoding.get('_FillValue')
            }

            # Add compression only for specific dtypes
            if var.dtype.name in compress_types:
                encoding['zlib'] = True
                encoding['complevel'] = 5

            # Only add to var_encodings if we have any encoding settings
            if encoding:
                var_encodings[var_name] = encoding

        # Add this group's encodings to the main dict
        if var_encodings:  # only add if there are variables with encoding
            group_encodings[group_path] = var_encodings

        # Process child groups
        for child_name, child in node.children.items():
            child_path = f"{group_path}/{child_name}" if group_path != '/' else f"/{child_name}"
            process_node(child, child_path)

    # Start processing from root with '/'
    process_node(datasets, '/')

    def deep_merge(dict1, dict2):
        merged = dict1.copy()
        for key, value in dict2.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = deep_merge(merged[key], value)
            else:
                merged[key] = value
        return merged

    if time_encoding:
        group_encodings = deep_merge(group_encodings, time_encoding)

    return group_encodings


def clean_inherited_coords(dt: DataTree) -> DataTree:
    """
    Clean coordinates that are inherited from parent nodes throughout the tree.
    Each node will only keep coordinates that are unique to it and not present in any ancestor.

    Parameters
    ----------
    dt : DataTree
        The input DataTree to clean

    Returns
    -------
    DataTree
        The cleaned DataTree where each node only has its unique coordinates
    """
    def get_ancestor_coords(node: DataTree) -> Set[str]:
        """Get all coordinate names from ancestor nodes."""
        ancestor_coords = set()
        current = node.parent
        while current is not None:
            if current.ds is not None:
                ancestor_coords.update(current.ds.coords)
            current = current.parent
        return ancestor_coords

    # Process each node in the tree except root
    for node in dt.subtree:
        if node is not dt:  # Skip root
            if node.ds is not None:
                # Get coordinates from all ancestors
                ancestor_coords = get_ancestor_coords(node)

                # Find coordinates to drop (ones that exist in ancestors)
                coords_to_drop = ancestor_coords.intersection(node.ds.coords)

                # Drop inherited coordinates if any exist
                # Try to drop the dimensions are different could throw an exception when dropping
                if coords_to_drop:
                    try:
                        node.ds = node.ds.drop_vars(coords_to_drop, errors='ignore')
                    except Exception:  # pylint: disable=broad-exception-caught
                        pass
    return dt


def update_dataset_with_time(og_ds, time_name="timeMidScan", group_path=None):
    """
    Update dataset dimensions based on 'DimensionNames' attributes and compute a time variable
    if not present, using values from the dataset.
    """
    ds = og_ds.copy()

    def convert_to_int(value, unit_type):
        if isinstance(value, np.timedelta64):
            ns_per_unit = {'day': 24 * 60 * 60 * 1e9, 'hour': 60 * 60 * 1e9, 'minute': 60 * 1e9}
            return int(value.astype('int64') / ns_per_unit[unit_type])
        return int(value)

    if not any(time_name in var for var in ds.variables):
        if "ScanTime" in (group_path or ""):
            time_unit_out = "seconds since 1980-01-06 00:00:00"
            new_time_list = []
            new_time_list_dt = []

            for i, _ in enumerate(ds["Year"].values):
                ms = int(ds["MilliSecond"].values[i])
                if not 0 <= ms < 1000:
                    raise ValueError(f"Milliseconds out of range: {ms} at index {i}")
                microsecond = ms * 1000
                dt = datetime.datetime(
                    int(ds["Year"].values[i]),
                    int(ds["Month"].values[i]),
                    convert_to_int(ds["DayOfMonth"].values[i], 'day'),
                    hour=convert_to_int(ds["Hour"].values[i], 'hour'),
                    minute=convert_to_int(ds["Minute"].values[i], 'minute'),
                    second=int(ds["Second"].values[i]),
                    microsecond=microsecond
                )
                new_time_list.append(date2num(dt, time_unit_out))
                new_time_list_dt.append(dt)  # keep actual datetime

            ds[time_name] = (ds["Year"].dims, np.array(new_time_list))
            ds[time_name].attrs["unit"] = time_unit_out

            ds[time_name + "_datetime"] = (ds["Year"].dims, np.array(new_time_list_dt))

    return ds
