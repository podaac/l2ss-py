"""
===============
hdf_utils.py
===============

Utility functions for handling HDF files, and dimension recovery
"""

import re
import warnings
from collections import defaultdict

import numpy as np
import xarray as xr
from xarray import DataTree

_PHONY_RE = re.compile(r"^phony_dim_(\d+)$")


def _is_phony(dim: str) -> bool:
    return _PHONY_RE.match(dim) is not None


def _phony_index(dim: str) -> int:
    m = _PHONY_RE.match(dim)
    return int(m.group(1)) if m else -1


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

    if not mapping:
        mapping = _mapping_from_struct_metadata(dt)

    if not mapping:
        return dt

    return _apply_mapping(dt, mapping)


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
                if not _is_phony(phony):
                    continue
                existing = mapping.get(phony)
                if existing is not None and existing != real:
                    # if two variables disagree on what this phony dim
                    # should be called then keep the first assignment and warn.
                    warnings.warn(
                        f"Conflicting DimensionNames for {phony!r}: "
                        f"{existing!r} vs {real!r}.  Keeping {existing!r}."
                    )
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
                # may be bytes, numpy bytes_, or already a string
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
        path = node.path.rstrip("/")
        if path in (
            "/HDFEOS/SWATHS",
            "/HDFEOS/GRIDS",
            "/HDFEOS/Swaths",
            "/HDFEOS/Grids",
        ):
            candidates.extend(node.children.values())
    return candidates


def _parse_odl_scope_dimensions(odl_text: str) -> dict[str, dict[str, int]]:
    """
    Parse the ODL block and return a per-scope dimension mapping:
        {scope_name: {dim_name: size}}

    Handles both SwathStructure (SWATH_N / SwathName) and GridStructure
    (GRID_N / GridName) blocks. Each scope is parsed independently so that
    two scopes sharing a dimension name but with different sizes never
    overwrite each other.
    """
    result: dict[str, dict[str, int]] = {}

    # split on any SWATH_N or GRID_N end-group boundary
    scope_blocks = re.split(r"END_GROUP\s*=\s*(?:SWATH|GRID)_\d+", odl_text)

    for block in scope_blocks:
        name_match = re.search(r'(?:SwathName|GridName)\s*=\s*"([^"]+)"', block)
        if not name_match:
            continue
        scope_name = name_match.group(1).strip()

        dim_group_match = re.search(
            r"GROUP\s*=\s*Dimension(.+?)END_GROUP\s*=\s*Dimension",
            block,
            re.DOTALL,
        )
        if not dim_group_match:
            continue

        named_dims: dict[str, int] = {}
        for obj_block in re.split(
            r"END_OBJECT\s*=\s*Dimension_\d+", dim_group_match.group(1)
        ):
            dim_name_match = re.search(r'DimensionName\s*=\s*"([^"]+)"', obj_block)
            size_match = re.search(r"\bSize\s*=\s*(\d+)", obj_block)
            if dim_name_match and size_match:
                named_dims[dim_name_match.group(1).strip()] = int(size_match.group(1))

        if named_dims:
            result[scope_name] = named_dims

    return result


def _mapping_from_struct_metadata(dt: DataTree) -> dict[str, str]:
    """
    Locate the StructMetadata.0 variable (typically at /HDFEOS
    INFORMATION), parse its ODL (Object Description Language) content
    to extract every named dimension and its size, then match those
    against the phony dims in the tree by size.

    Returns an empty dict if no StructMetadata.0 is found or parsing
    fails.
    """
    struct_text = _find_struct_metadata(dt)
    if not struct_text:
        return {}

    swath_dims = _parse_odl_scope_dimensions(struct_text)
    if not swath_dims:
        return {}

    mapping: dict[str, str] = {}

    scope_roots = _find_scope_roots(dt)
    if not scope_roots:
        scope_roots = [dt]

    for scope_root in scope_roots:
        # extract the swath name from the path, e.g.
        # "/HDFEOS/SWATHS/OMI Ground Pixel Corners UV-1" -> "OMI Ground Pixel Corners UV-1"
        swath_name = scope_root.path.rstrip("/").split("/")[-1]
        named_dims = swath_dims.get(swath_name)

        if not named_dims:
            # swath name not found in ODL
            continue

        size_to_names: dict[int, list[str]] = defaultdict(list)
        for name, size in named_dims.items():
            size_to_names[size].append(name)

        scope_mapping = _match_phony_to_named(scope_root, size_to_names, named_dims)
        mapping.update(scope_mapping)

    return mapping


def _match_phony_to_named(
    dt: DataTree,
    size_to_names: dict[int, list[str]],
    named_dims: dict[str, int],
) -> dict[str, str]:
    """
    For every phony dim in the tree, look up its size in ``size_to_names``.

    - If only one real name has that size, the mapping is unambiguous.
    - If multiple real names share the same size, we use the *position* of
      the phony dim within its group's sorted dimension list to pick among
      the candidates (ordered as they appear in the ODL block).
    """
    # ordered list of all named dims so we can use index as a tiebreaker
    ordered_named = list(named_dims.keys())

    mapping: dict[str, str] = {}

    for node in dt.subtree:
        ds = node.ds
        if ds is None:
            continue

        # phony dims in this group sorted by their index (N in
        # phony_dim_N)
        local_phonies = sorted([d for d in ds.dims if _is_phony(d)], key=_phony_index)

        for phony in local_phonies:
            size = ds.dims[phony]
            candidates = size_to_names.get(size, [])
            if not candidates:
                continue
            if len(candidates) == 1:
                mapping[phony] = candidates[0]
            else:
                # use the position of this phony dim among same-size dims in
                # this group to index into the ordered candidate list.
                same_size_in_group = [p for p in local_phonies if ds.dims[p] == size]
                pos = same_size_in_group.index(phony)
                # sort candidates by their order in the odl block
                ordered_candidates = sorted(candidates, key=ordered_named.index)
                if pos < len(ordered_candidates):
                    mapping[phony] = ordered_candidates[pos]

    return mapping


def _apply_mapping(dt: DataTree, mapping: dict[str, str]) -> DataTree:
    """
    Walk the tree, apply rename_dims to every group dataset that contains
    any of the phony dims in ``mapping``, then reconstruct the entire tree
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
