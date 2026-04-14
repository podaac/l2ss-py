"""
===============
hdf_utils.py
===============

Utility functions for handling HDF files, and dimension recovery
"""

import re
import warnings

import numpy as np
import xarray as xr
from xarray import DataTree

_PHONY_RE = re.compile(r"^phony_dim_(\d+)$")


def _is_phony(dim: str) -> bool:
    return _PHONY_RE.match(dim) is not None


def rename_phony_dims(dt: DataTree) -> DataTree:
    """
    Return a new DataTree with all phony_dim_N dimensions renamed to
    meaningful names. First check if `DimensionNames` is present as
    attributes for each variable otherwise check for `StructMeta.0`
    and assuming an HDFEOS structure. the first one that produces a
    non-empty mapping is used.

    Parameters
    ----------
    dt : DataTree
        Tree loaded from an HDF/HDF-EOS5 file

    Returns
    -------
    DataTree
        A new tree with renamed dimensions.
    """
    mapping = _mapping_from_dimension_names(dt)
    if mapping:
        dt = _apply_mapping(dt, mapping)

    struct_text = _find_struct_metadata(dt)
    if struct_text:
        scope_field_dimlists = _parse_odl_field_dimlists(struct_text)
        if scope_field_dimlists:
            dt = _apply_from_field_dimlists(dt, scope_field_dimlists)

    # null condition
    return dt


def _mapping_from_dimension_names(dt: DataTree) -> dict[str, str]:
    """
    Iterate every variable in every group.  If a variable carries a
    DimensionNames attribute (comma-separated list of real dim names),
    zip those names against the variable's current (phony) dim names to
    build the global mapping.

    A variable with dims (phony_dim_0, phony_dim_1) and attribute
    DimensionNames = "nTimes,nXtrack" results in {phony_dim_0: nTimes,
    phony_dim_1: nXtrack}
    """
    mapping: dict[str, str] = {}

    for node in dt.subtree:
        ds = node.ds
        if ds is None:
            continue
        for _, da in ds.items():
            dim_names_attr = da.attrs.get("DimensionNames")
            if dim_names_attr is None:
                continue

            real_names = [n.strip() for n in dim_names_attr.split(",")]

            # if attribute shapes mismatch, then continue
            if len(real_names) != len(da.dims):
                continue

            for phony, real in zip(da.dims, real_names):
                # guard for incorrectly specified HDFEOS data
                # which might not have phony_dims being populated
                if not _is_phony(phony):
                    continue
                existing = mapping.get(phony)
                if existing is not None and existing != real:
                    # if two variables disagree on what this phony dim
                    # should be called then keep the first assignment and warn.
                    warnings.warn(f"Conflicting DimensionNames for {phony!r}: " f"{existing!r} vs {real!r}.  Keeping {existing!r}.")
                else:
                    mapping[phony] = real

    return mapping


def _find_struct_metadata(dt: DataTree) -> str | None:
    """Return the decoded text of StructMetadata.0 if present anywhere
    in the tree.
    """
    for node in dt.subtree:
        ds = node.ds
        if ds is None:
            continue
        for var_name in ds.data_vars:
            if "StructMetadata" in var_name:
                raw = ds[var_name].values
                if isinstance(raw, (bytes, np.bytes_)):
                    return raw.decode("utf-8", errors="replace")
                if hasattr(raw, "item"):
                    val = raw.item()
                    if isinstance(val, (bytes, np.bytes_)):
                        return val.decode("utf-8", errors="replace")
                    return str(val)
                return str(raw)
    return None


def _find_scope_roots(dt: DataTree) -> list[DataTree]:
    """
    Return the immediate children of /HDFEOS/SWATHS or /HDFEOS/GRIDS.
    These represent individual swaths / grids and are the correct scopes
    within which phony dims should be unified.
    """
    candidates = []
    for node in dt.subtree:
        path = node.path.rstrip("/").casefold()
        if path in (
            "/hdfeos/swaths",
            "/hdfeos/grids",
        ):
            candidates.extend(node.children.values())
    return candidates


def _parse_odl_field_dimlists(odl_text: str) -> dict[str, dict[str, list[str]]]:
    """
    Parse the ODL block and return a per-scope, per-field dim list:
        {scope_name: {field_name: [dim_name, ...]}}

    Handles both SwathStructure (SWATH_N / SwathName) and GridStructure
    (GRID_N / GridName) blocks. The DimList order matches the axis
    order of the variable as stored in the file, so it can be zipped
    directly against the variable's actual (phony) dims.
    """
    result: dict[str, dict[str, list[str]]] = {}

    scope_blocks = re.split(r"END_GROUP\s*=\s*(?:SWATH|GRID)_\d+", odl_text)

    for block in scope_blocks:
        name_match = re.search(r'(?:SwathName|GridName)\s*=\s*"([^"]+)"', block)
        if not name_match:
            continue
        scope_name = name_match.group(1).strip()
        field_dimlists: dict[str, list[str]] = {}

        # GeoField and DataField blocks both use the same OBJECT structure
        for field_block in re.split(r"END_OBJECT\s*=\s*(?:Geo|Data)Field_\d+", block):
            field_name_match = re.search(r'(?:GeoFieldName|DataFieldName)\s*=\s*"([^"]+)"', field_block)
            dimlist_match = re.search(r"DimList\s*=\s*\(([^)]+)\)", field_block)
            if not field_name_match or not dimlist_match:
                continue

            field_name = field_name_match.group(1).strip()
            dim_names = [d.strip().strip('"') for d in dimlist_match.group(1).split(",")]
            field_dimlists[field_name] = dim_names

        if field_dimlists:
            result[scope_name] = field_dimlists

    return result


def _apply_mapping(dt: DataTree, mapping: dict[str, str]) -> DataTree:
    """
    Walk the tree, apply rename_dims to every group dataset that contains
    any of the phony dims in mapping, then reconstruct the entire tree
    from a flat path -> dataset dict using DataTree.from_dict.

    This avoids any direct node copying and is the idiomatic way to rebuild
    a modified DataTree.
    """
    path_to_ds: dict[str, xr.Dataset] = {}

    for node in dt.subtree:
        ds = node.ds if node.ds is not None else xr.Dataset()

        local_mapping = {k: v for k, v in mapping.items() if k in ds.dims}
        if local_mapping:
            ds = ds.rename_dims(local_mapping)

        path_to_ds[node.path] = ds

    return DataTree.from_dict(path_to_ds)


def _find_scope_for_node(node: DataTree, scope_roots: list[DataTree]) -> str | None:
    """
    Return the scope name (last path component of the scope root) that this
    node lives under, or None if the node is not under any known scope root.
    """
    for scope_root in scope_roots:
        scope_path = scope_root.path.rstrip("/")
        if node.path == scope_path or node.path.startswith(scope_path + "/"):
            return scope_path.split("/")[-1]
    return None


def _apply_from_field_dimlists(
    dt: DataTree,
    scope_field_dimlists: dict[str, dict[str, list[str]]],
) -> DataTree:
    """
    Walk every node in the tree. For each variable in the node, look up its
    real dim names by scope name + variable name directly from the ODL field
    dimlists, then rename only that variable's dims within that node.

    This never builds a global phony to real mapping, so two variables with the
    same name in different scopes are always resolved against their own scope's
    DimList entries independently.
    """
    scope_roots = _find_scope_roots(dt)
    path_to_ds: dict[str, xr.Dataset] = {}

    for node in dt.subtree:
        ds = node.ds if node.ds is not None else xr.Dataset()

        scope_name = _find_scope_for_node(node, scope_roots)
        field_dimlists = scope_field_dimlists.get(scope_name, {}) if scope_name else {}

        if field_dimlists:
            local_mapping: dict[str, str] = {}

            for var_name, da in ds.items():
                odl_dims = field_dimlists.get(var_name)
                if odl_dims is None or len(odl_dims) != len(da.dims):
                    continue
                for current_dim, real_name in zip(da.dims, odl_dims):
                    # guard for incorrectly specified HDFEOS data
                    # which might not have phony_dims being populated
                    if not _is_phony(current_dim):
                        continue
                    existing = local_mapping.get(current_dim)
                    if existing is not None and existing != real_name:
                        warnings.warn(f"Conflicting ODL DimList for {current_dim!r} in " f"{node.path!r}: {existing!r} vs {real_name!r}. " f"Keeping {existing!r}.")
                    else:
                        local_mapping[current_dim] = real_name

            if local_mapping:
                ds = ds.rename_dims(local_mapping)

        path_to_ds[node.path] = ds

    return DataTree.from_dict(path_to_ds)
