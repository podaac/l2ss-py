"""
===============
variables_utils.py
===============

Utility functions to get variables and normalize variables for a granule files.
"""

import logging

import xarray as xr

logger = logging.getLogger(__name__)


def get_vars_with_paths(tree: xr.DataTree) -> set[str]:
    """
    Get all variables and coordinates with their full paths from a DataTree

    Parameters
    ----------
    tree : DataTree
        The input DataTree

    Returns
    -------
    set[str]
        Unordered set of variable and coordinate paths in format
        '/group/var' or '/var' for root level.

    Examples
    --------
    >>> ds = xr.Dataset({'var1': [1], 'var2': [2], 'time': ('time', [0])})
    >>> tree = DataTree(data=ds)
    >>> tree['group1'] = DataTree(data=ds.copy())
    >>> paths = get_vars_with_paths(tree)
    >>> print(paths)
    {'/time', '/var1', '/var2', '/group1/var1', '/group1/var2'}
    """
    paths: set[str] = set()
    for node in tree.subtree:
        prefix = node.path.rstrip("/") + "/"
        for name in set(node.data_vars) | set(node.to_dataset(inherit=False).coords):
            paths.add(f"{prefix}{name}")
    return paths


def drop_vars_by_path(tree: xr.DataTree, var_paths: str | list[str] | set[str]) -> None:
    """
    Drop variables *in place* from a DataTree using paths in the
    format '/group/var' or '/var' for root level.

    Parameters
    ----------
    tree : DataTree
        The input DataTree
    var_paths : str or list[str] or set[str]
        Paths to variables to drop in format '/group/var' or '/var' for root level
        Examples:
            - '/var1'  # root level variable
            - '/group1/var1'  # variable in group1
            - '/group1/subgroup/var1'  # variable in nested group

    """
    # guard for single string being passed
    drop: set[str] = {var_paths} if isinstance(var_paths, str) else set(var_paths)

    for node in tree.subtree:
        prefix = node.path.rstrip("/") + "/"
        to_drop = [name for name in node.variables if f"{prefix}{name}" in drop]
        if to_drop:
            node.dataset = node.dataset.drop_vars(to_drop, errors="ignore")


def _normalize_for_matching(path: str) -> str:
    """
    Normalize path for matching:
    - Remove spaces and underscores
    - Lowercase
    - Strip leading slash
    """
    return path.lstrip("/").replace(" ", "").replace("_", "").lower()


def normalize_candidate_paths_against_dtree(candidates: list[str], all_vars: list[str]) -> list[str]:
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
    norm_to_real = {_normalize_for_matching(real_path): real_path for real_path in all_vars}

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


def get_varinfo_required_variables(
    file_to_subset: str,
    requested_variables: list[str] | set[str],
    all_vars: set[str],
    short_name: str | None = None,
) -> set[str]:
    """
    Use ``earthdata-varinfo`` to expand a set of requested variables with all
    variables they require (coordinates, bounds, grid mapping, ancillary
    variables, cell measures, etc.) by parsing the granule's CF attributes.

    This augments the hand-rolled resolution in
    ``coordinate_utils.collect_coordinate_variables`` which does not parse the
    CF ``bounds``, ``grid_mapping``, ``ancillary_variables`` or ``cell_measures``
    attributes.

    The call is additive and fails soft: any error reading the granule or
    resolving dependencies results in an empty set so that subsetting can
    continue using the existing coordinate resolution.

    Parameters
    ----------
    file_to_subset : str
        Path to the granule file to inspect.
    requested_variables : list[str] or set[str]
        The variables requested by the user (full DataTree paths, e.g.
        '/group/var'), including any lat/lon/time variables already added.
    all_vars : set[str]
        All variable/coordinate paths present in the DataTree, used to
        reconcile varinfo's path convention with l2ss-py's and to ensure only
        existing variables are returned.
    short_name : str, optional
        Collection short name, passed through to varinfo.

    Returns
    -------
    set[str]
        Required variable paths in canonical DataTree form ('/group/var'),
        intersected with ``all_vars``. Empty if varinfo is unavailable or fails.
    """
    try:
        # Imported lazily so a varinfo import error never breaks subsetting.
        from varinfo import VarInfoFromNetCDF4  # pylint: disable=import-outside-toplevel

        var_info = VarInfoFromNetCDF4(file_to_subset, short_name=short_name)
        required = var_info.get_required_variables(set(requested_variables))
    except Exception as err:  # pylint: disable=broad-except
        logger.warning("earthdata-varinfo required-variable resolution failed: %s", err)
        return set()

    # Reconcile varinfo's leading-slash full paths with the DataTree convention
    # and keep only variables that actually exist in the tree.
    normalized = normalize_candidate_paths_against_dtree(list(required), list(all_vars))
    return {path for path in normalized if path in all_vars}
