"""Simplified where_tree using DataTree.isel for tree-wide subsetting."""

# pylint: disable=duplicate-code
import numpy as np
import xarray as xr
from xarray import DataTree

from podaac.subsetter import dimension_cleanup as dc
from podaac.subsetter.datatree_subset import (
    _get_fill_value_for_var,
    cast_type,
    get_indexers_from_1d,
    get_indexers_from_nd,
    get_sibling_or_parent_condition,
    subtree_is_empty,
    where_tree,
)
from podaac.subsetter.utils import mask_utils

try:
    from harmony_service_lib.exceptions import NoDataException
except ImportError:
    class NoDataException(Exception):
        """Fallback exception when harmony_service_lib is not installed."""


def where_tree_v2(tree: DataTree, condition_dict, cut: bool, pixel_subset=False) -> DataTree:
    """
    Simplified where_tree using DataTree operations for tree-wide subsetting.

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

    if cond.values.ndim == 1:
        indexers = get_indexers_from_1d(cond)
    else:
        indexers = get_indexers_from_nd(cond, cut)

    if not all(len(value) > 0 for value in indexers.values()):
        raise NoDataException("No data in subsetted granule.")

    # Tree-wide isel via map_over_datasets (handles empty nodes gracefully).
    # Also fix duplicate dims before indexing.
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
                    return _apply_masking(ds, grp_indexed_cond)
                sibling_cond = get_sibling_or_parent_condition(indexed_per_group, node_path)
                if sibling_cond is not None:
                    return _apply_masking(ds, sibling_cond)
                return ds

            new_children = {}
            for child_name, child_node in result.children.items():
                child_path = f"/{child_name}"
                new_child_ds = _mask_with_per_group(child_node.ds, child_path)
                new_child = DataTree(name=child_name, dataset=new_child_ds)
                for gc_name, gc_node in child_node.children.items():
                    new_child[gc_name] = gc_node
                new_children[child_name] = new_child
            result = DataTree(name=result.name, dataset=result.ds)
            result.attrs.update(tree.attrs)
            for child_name, child_tree in new_children.items():
                result[child_name] = child_tree
        else:
            indexed_cond = cond.isel(**indexers)
            result = result.map_over_datasets(
                lambda ds: _apply_masking(ds, indexed_cond)
            )

    return _prune_empty(result)


def _apply_per_group(tree, condition_dict, cut, pixel_subset):
    """Apply different conditions to different subtrees when dimensions conflict."""
    return where_tree(tree, condition_dict, cut, pixel_subset)


def _find_reference_dataset(tree, cond):
    """Find a dataset in the tree that shares dimensions with the condition.
    Falls back to the root dataset if no match is found."""
    cond_dims = set(cond.dims)
    # Check root first
    root_ds = dc.remove_duplicate_dims_xarray(tree.ds)
    if cond_dims.intersection(set(root_ds.dims)):
        return root_ds
    # Search children for a dataset with matching dims
    for node in tree.subtree:
        ds = node.ds
        if ds is not None and ds.dims:
            node_ds = dc.remove_duplicate_dims_xarray(ds)
            if cond_dims.intersection(set(node_ds.dims)):
                return node_ds
    return root_ds


def _resolve_condition(tree, condition_dict):
    """Resolve a single unified condition to apply to the whole tree.

    For single-condition dicts, return it directly. For multi-condition dicts,
    find the condition whose dimensions best match the tree's primary dimensions.
    Multiple conditions typically represent the same spatial bbox applied to
    different groups — we pick the one that aligns with the most nodes.
    """
    if not condition_dict:
        return None
    if len(condition_dict) == 1:
        return next(iter(condition_dict.values()))
    # Multiple conditions with potentially different sizes (e.g., different
    # groups with different time dimension lengths). Find the condition
    # whose size matches the most common dimension size in the tree.
    dim_sizes = {}
    for node in tree.subtree:
        for dim, size in node.ds.sizes.items():
            dim_sizes.setdefault(dim, []).append(size)

    # Pick condition whose dims best match the tree
    best_cond = None
    best_score = -1
    for _, cond in condition_dict.items():
        score = 0
        for dim in cond.dims:
            if dim in dim_sizes:
                cond_size = cond.sizes[dim]
                if cond_size in dim_sizes[dim]:
                    score += dim_sizes[dim].count(cond_size)
        if score > best_score:
            best_score = score
            best_cond = cond
    return best_cond


def _apply_masking(ds, indexed_cond):
    """Apply .where() NaN masking and fill-value/type-casting logic per dataset.

    At this point, `ds` has already been isel'd (cut to the bounding region).
    We apply .where() to NaN-mask values that are inside the bounding box of
    indices but outside the actual spatial condition.
    """
    if not ds.data_vars:
        return ds

    # Check if condition dims overlap with this dataset's dims
    cond_dims = set(indexed_cond.dims)
    ds_dims = set(ds.dims)
    if not cond_dims.intersection(ds_dims):
        return ds

    # Align condition to this dataset's dims
    aligned_cond = mask_utils.align_dims_cond_only(ds, indexed_cond)

    # For variables with partial dim overlap (e.g., 1D time var when cond is 2D),
    # collapse the missing dims in the condition before applying .where()
    new_dataset = ds.copy()
    new_dataset.attrs.update(ds.attrs)

    for variable_name in list(ds.data_vars):
        var = ds[variable_name]
        var_dims = set(var.dims)
        cond_var_dims = set(aligned_cond.dims)

        # Determine the appropriate condition for this variable
        if cond_var_dims.issubset(var_dims):
            # Full overlap: apply condition directly
            var_cond = aligned_cond
        elif cond_var_dims.intersection(var_dims):
            # Partial overlap: collapse dims not in the variable
            extra_dims = cond_var_dims - var_dims
            var_cond = aligned_cond
            for dim in extra_dims:
                var_cond = var_cond.any(dim=dim)
        else:
            # No overlap: don't mask this variable
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
