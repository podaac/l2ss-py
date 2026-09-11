"""Subset a DataTree spatially using condition masks and index slicing."""

# pylint: disable=duplicate-code
import numpy as np
import xarray as xr
from xarray import DataTree

from podaac.subsetter import dimension_cleanup as dc
from podaac.subsetter.datatree_subset import (
    _get_fill_value_for_var,
    apply_indexers_to_tree,
    cast_type,
    find_fully_empty_paths,
    get_indexers_from_1d,
    get_indexers_from_nd,
    get_sibling_or_parent_condition,
    subtree_is_empty,
)
from podaac.subsetter.utils import mask_utils

try:
    from harmony_service_lib.exceptions import NoDataException
except ImportError:
    class NoDataException(Exception):
        """Fallback exception when harmony_service_lib is not installed."""


def _build_indexers(cond, cut):
    """Build indexers from a boolean condition array."""
    if cond.values.ndim == 1:
        return get_indexers_from_1d(cond)
    return get_indexers_from_nd(cond, cut)


def _subset_dataset(dataset, cond, cut, pixel_subset):
    """Align condition to dataset, build indexers, apply isel and optional masking.

    Returns (processed_ds, indexers).
    """
    cond = mask_utils.align_dims_cond_only(dataset, cond)
    indexers = _build_indexers(cond, cut)

    if not all(len(v) > 0 for v in indexers.values()):
        raise NoDataException("No data in subsetted granule.")

    indexed_ds = dataset.isel(**indexers, missing_dims="ignore")

    if pixel_subset:
        processed_ds = indexed_ds
    else:
        indexed_cond = cond.isel(**indexers)
        processed_ds = _apply_masking(indexed_ds, indexed_cond, pre_aligned=True)

    processed_ds.attrs.update(dataset.attrs)
    return processed_ds, indexers


def subset_tree(tree: DataTree, condition_dict, cut: bool, pixel_subset=False) -> DataTree:
    """
    Subset a DataTree by applying spatial condition masks.

    When all groups share the same dimension sizes, a single tree-wide isel is
    used. When groups have conflicting sizes on the same dimension name (e.g.,
    different temporal resolutions), each condition is applied to its matching
    subtree independently.
    """
    if not condition_dict:
        return tree

    # Single condition: apply uniformly to the whole tree
    if len(condition_dict) == 1:
        cond = next(iter(condition_dict.values()))
        return _apply_single_condition(tree, cond, cut, pixel_subset)

    # Multiple conditions: check if they're compatible (same shape)
    conditions = list(condition_dict.values())
    shapes_match = all(c.shape == conditions[0].shape for c in conditions)

    if shapes_match:
        # Same shape: combine with OR for indexers (keep any row/col where
        # ANY group has data), but apply each group's own condition for masking
        combined = conditions[0]
        for c in conditions[1:]:
            combined = combined | c
        return _apply_single_condition(tree, combined, cut, pixel_subset, condition_dict)

    # Different shapes: apply each condition to its subtree independently
    return _apply_per_group(tree, condition_dict, cut, pixel_subset)


def _apply_single_condition(tree, cond, cut, pixel_subset, per_group_conditions=None):
    """Apply a single condition uniformly to the whole tree.

    Parameters
    ----------
    per_group_conditions : dict, optional
        When provided, maps group paths to their individual conditions.
        Each group uses its own condition for masking instead of the
        combined condition, so that pixels valid in one group aren't
        incorrectly masked by another group's slightly different grid.
    """
    ref_ds = _find_reference_dataset(tree, cond)
    cond = mask_utils.align_dims_cond_only(ref_ds, cond)
    indexers = _build_indexers(cond, cut)

    if not all(len(value) > 0 for value in indexers.values()):
        raise NoDataException("No data in subsetted granule.")

    # Tree-wide isel via map_over_datasets (handles empty nodes gracefully).
    result = tree.map_over_datasets(
        lambda ds: dc.remove_duplicate_dims_xarray(ds).isel(**indexers, missing_dims="ignore")
    )

    if not pixel_subset:
        if per_group_conditions:
            indexed_per_group = {}
            for path, grp_cond in per_group_conditions.items():
                aligned = mask_utils.align_dims_cond_only(ref_ds, grp_cond)
                indexed_per_group[path] = aligned.isel(**indexers)

            def _mask_with_per_group(ds, node_path):
                grp_indexed_cond = indexed_per_group.get(node_path)
                if grp_indexed_cond is not None:
                    return _apply_masking(ds, grp_indexed_cond, pre_aligned=True)
                sibling_cond = get_sibling_or_parent_condition(indexed_per_group, node_path)
                if sibling_cond is not None:
                    return _apply_masking(ds, sibling_cond, pre_aligned=True)
                return ds

            def _mask_tree_recursive(node, path):
                masked_ds = _mask_with_per_group(node.ds, path)
                new_node = DataTree(name=node.name, dataset=masked_ds)
                new_node.attrs.update(node.attrs)
                for child_name, child_node in node.children.items():
                    child_path = f"{path}/{child_name}" if path != "/" else f"/{child_name}"
                    new_node[child_name] = _mask_tree_recursive(child_node, child_path)
                return new_node

            result = _mask_tree_recursive(result, "/")
        else:
            indexed_cond = cond.isel(**indexers)
            result = result.map_over_datasets(
                lambda ds: _apply_masking(ds, indexed_cond)
            )

    return _prune_empty(result)


def _align_to_parent(child_ds, parent_ds):
    """Align a child dataset to a parent's subsetted coordinate values."""
    sel_kwargs = {}
    for dim in list(child_ds.dims):
        if (dim in parent_ds.dims
                and dim in child_ds.coords
                and dim in parent_ds.coords):
            parent_values = parent_ds[dim].values
            child_values = child_ds[dim].values
            if len(parent_values) != len(child_values) or not np.array_equal(parent_values, child_values):
                common = np.intersect1d(parent_values, child_values)
                if len(common) > 0:
                    sel_kwargs[dim] = common
    if sel_kwargs:
        return child_ds.sel(**sel_kwargs)
    return child_ds


def _apply_per_group(tree, condition_dict, cut, pixel_subset):
    """Apply different conditions to different subtrees when dimensions conflict.

    Each node finds its matching condition via path lookup, builds its own
    indexers, applies isel + masking independently, then the tree is
    reassembled.
    """
    empty_paths = find_fully_empty_paths(tree)

    def _process_node(node, path, parent_processed_ds=None):
        cond = get_sibling_or_parent_condition(condition_dict, path)
        if cond is None and len(condition_dict) == 1:
            _, cond = next(iter(condition_dict.items()))

        dataset = dc.remove_duplicate_dims_xarray(node.ds)
        indexers = None

        if dataset.data_vars and cond is not None:
            processed_ds, indexers = _subset_dataset(dataset, cond, cut, pixel_subset)
            dc.sync_dims_inplace(dataset, processed_ds)
        else:
            processed_ds = dataset.copy()
            processed_ds.attrs.update(dataset.attrs)
            if parent_processed_ds is not None:
                processed_ds = _align_to_parent(processed_ds, parent_processed_ds)

        processed_children = {}
        for child_name, child_node in node.children.items():
            current_path = f"{path}/{child_name}"
            if current_path in empty_paths:
                if indexers is not None:
                    child_node = apply_indexers_to_tree(child_node, indexers, processed_ds)
                processed_children[child_name] = child_node
            else:
                child_ds, child_children, child_indexers = _process_node(
                    child_node, current_path, processed_ds
                )
                if indexers is None and child_indexers:
                    indexers = child_indexers
                    processed_ds = processed_ds.isel(**child_indexers, missing_dims="ignore")

                child_tree = DataTree(name=child_name, dataset=child_ds)
                for gc_name, gc_tree in child_children.items():
                    child_tree[gc_name] = gc_tree

                if not subtree_is_empty(child_tree, check_attrs=True):
                    processed_children[child_name] = child_tree

        return processed_ds, processed_children, indexers

    root_ds, children, _ = _process_node(tree, "")
    result = DataTree(name=tree.name, dataset=root_ds)
    for child_name, child_tree in children.items():
        result[child_name] = child_tree
    result.attrs.update(tree.attrs)
    return result


def _find_reference_dataset(tree, cond):
    """Find a dataset in the tree that shares dimensions with the condition.
    Falls back to the root dataset if no match is found."""
    cond_dims = set(cond.dims)
    root_ds = dc.remove_duplicate_dims_xarray(tree.ds)
    if cond_dims.intersection(set(root_ds.dims)):
        return root_ds
    for node in tree.subtree:
        ds = node.ds
        if ds is not None and ds.dims:
            node_ds = dc.remove_duplicate_dims_xarray(ds)
            if cond_dims.intersection(set(node_ds.dims)):
                return node_ds
    return root_ds


def _apply_masking(ds, indexed_cond, pre_aligned=False):
    """Apply .where() NaN masking and fill-value/type-casting logic per dataset.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset already cut to the bounding region via isel.
    indexed_cond : xr.DataArray
        Boolean condition, already isel'd to match ds dimensions.
    pre_aligned : bool
        If True, skip the align_dims_cond_only call (caller already aligned).
    """
    if not ds.data_vars:
        return ds

    cond_dims = set(indexed_cond.dims)
    ds_dims = set(ds.dims)
    if not cond_dims.intersection(ds_dims):
        return ds

    aligned_cond = indexed_cond if pre_aligned else mask_utils.align_dims_cond_only(ds, indexed_cond)

    new_dataset = ds.copy()
    new_dataset.attrs.update(ds.attrs)

    for variable_name in list(ds.data_vars):
        var = ds[variable_name]
        var_dims = set(var.dims)
        cond_var_dims = set(aligned_cond.dims)

        if cond_var_dims.issubset(var_dims):
            var_cond = aligned_cond
        elif cond_var_dims.intersection(var_dims):
            extra_dims = cond_var_dims - var_dims
            var_cond = aligned_cond
            for dim in extra_dims:
                var_cond = var_cond.any(dim=dim)
        else:
            continue

        if len(var.shape) == 0:
            new_dataset[variable_name] = var
            continue

        fv = _get_fill_value_for_var(var)
        new_dataset[variable_name] = var.where(var_cond, other=fv)

        if "_FillValue" in var.attrs:
            fill_value = var.attrs.get("_FillValue")
            if np.issubdtype(var.dtype, np.dtype(np.datetime64)):
                fill_value = np.datetime64("nat")
            if np.issubdtype(var.dtype, np.dtype(np.timedelta64)):
                fill_value = np.timedelta64("nat")
            new_dataset[variable_name] = new_dataset[variable_name].fillna(fill_value)

        if new_dataset[variable_name].dtype != var.dtype:
            new_dataset[variable_name] = xr.apply_ufunc(
                cast_type, new_dataset[variable_name], str(var.dtype),
                dask="allowed", keep_attrs=True
            )

    return new_dataset


def _prune_empty(tree: DataTree) -> DataTree:
    """Remove fully empty subtrees from the result."""
    result = DataTree(name=tree.name, dataset=tree.ds)
    result.attrs.update(tree.attrs)
    for child_name, child_node in tree.children.items():
        if not subtree_is_empty(child_node, check_attrs=True):
            result[child_name] = _prune_empty(child_node)
    return result
