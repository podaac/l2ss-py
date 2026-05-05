import xarray as xr
from xarray import DataTree

from podaac.subsetter.datatree_subset import find_fully_empty_paths, is_empty, subtree_is_empty


def make_simple_tree():
    ds = xr.Dataset({"a": ("x", [1, 2]), "b": ("x", [3, 4])})
    ds.attrs["foo bar"] = "baz"
    tree = DataTree(ds)
    tree["child"] = DataTree(xr.Dataset({"c": ("y", [5, 6])}))
    return tree

def test_is_empty_and_subtree_is_empty():
    empty_ds = xr.Dataset()
    empty_tree = DataTree(empty_ds)
    assert is_empty(empty_tree)
    assert subtree_is_empty(empty_tree)
    tree = make_simple_tree()
    assert not is_empty(tree)
    assert not subtree_is_empty(tree)
    # Make child empty
    tree["child"].ds = xr.Dataset()
    assert not subtree_is_empty(tree)
    # Make all empty
    tree.ds = xr.Dataset()
    assert subtree_is_empty(tree)

def test_find_fully_empty_paths():
    tree = make_simple_tree()
    tree.ds = xr.Dataset()
    tree["child"].ds = xr.Dataset()
    paths = find_fully_empty_paths(tree)
    assert paths == [tree.path]
