"""
===============
variables_utils.py
===============

Utility functions to get variables and normalize variables for a granule files.
"""
from typing import List
import xarray as xr


def get_all_variable_names_from_dtree(dtree: xr.DataTree) -> List[str]:
    """
    Recursively extract all variable names (with full paths) from an xarray DataTree.

    Parameters
    ----------
    dtree : xr.DataTree
        The root of the DataTree.

    Returns
    -------
    List[str]
        A list of variable full paths (e.g. '/group1/var').
    """
    var_names = []

    def recurse(node: xr.DataTree):
        group_path = node.path
        for var_name in node.data_vars:
            if group_path in ("", "/"):
                full_path = f"/{var_name}"
            else:
                full_path = f"{group_path}/{var_name}"
            var_names.append(full_path)
        for child in node.children.values():
            recurse(child)

    recurse(dtree)
    return var_names


def _normalize_for_matching(path: str) -> str:
    """
    Normalize path for matching:
    - Remove spaces and underscores
    - Lowercase
    - Strip leading slash
    """
    return path.lstrip("/").replace(" ", "").replace("_", "").lower()


def normalize_candidate_paths_against_dtree(
    candidates: List[str], all_vars: List[str]
) -> List[str]:
    """
    Normalize and match candidate variable paths to actual variable paths from a DataTree.

    - Normalization ignores differences between underscores and spaces.
    - Matching is case-insensitive.
    - If a match is found, the actual variable path from the DataTree is returned.
    - If no match is found, the original candidate path is returned as-is.

    Parameters
    ----------
    candidates : List[str]
        List of candidate variable paths (e.g., from user input or spreadsheets).

    all_vars : List[str]
        List of actual variable paths from the DataTree, typically from
        get_all_variable_names_from_dtree(dtree).

    Returns
    -------
    List[str]
        List of resolved variable paths:
        - Matched paths are returned using their canonical DataTree form.
        - Unmatched candidates are returned unchanged.
    """
    # Build normalized lookup: no slashes, underscores/spaces ignored
    norm_to_real = {
        _normalize_for_matching(real_path): real_path for real_path in all_vars
    }

    resolved = []
    for cand in candidates:
        norm_cand = _normalize_for_matching(cand)
        match = norm_to_real.get(norm_cand)

        if match:
            # Ensure only one leading slash
            resolved_path = "/" + match.lstrip("/")
        else:
            # Keep the original candidate exactly as given
            resolved_path = cand

        resolved.append(resolved_path)

    return resolved
