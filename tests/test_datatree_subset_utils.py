import pytest
import xarray as xr
from xarray import DataTree
import numpy as np
from podaac.subsetter.datatree_subset import (
    is_empty,
    subtree_is_empty,
    find_fully_empty_paths,
    safe_name,
    safe_attr_value,
    collect_subtree_attrs_flat,
    sanitize_node_attrs,
    sanitize_node_variables,
    sanitize_datatree,
    set_subtree_attrs_on_node,
)

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

def test_safe_name():
    assert safe_name("foo bar") == "foo_bar"
    assert safe_name("1abc") == "_1abc"
    assert safe_name("a-b.c") == "a_b_c"

def test_safe_attr_value():
    assert safe_attr_value(np.int32(5)) == 5
    assert safe_attr_value(np.float64(2.5)) == 2.5
    assert safe_attr_value([np.int32(1), np.float64(2.2)]) == [1, 2.2]
    assert safe_attr_value("hello") == "hello"
    assert safe_attr_value({"x": 1}) == "{'x': 1}"

def test_collect_subtree_attrs_flat():
    tree = make_simple_tree()
    flat = collect_subtree_attrs_flat(tree)
    assert any("foo_bar" in k for k in flat)
    assert any("baz" == v for v in flat.values())

def test_sanitize_node_attrs():
    tree = make_simple_tree()
    sanitize_node_attrs(tree)
    for k in tree.attrs:
        assert " " not in k
        assert not k[0].isdigit()

def test_sanitize_node_variables():
    tree = make_simple_tree()
    tree.ds = tree.ds.rename({"a": "1a"})
    sanitize_node_variables(tree)
    assert "_1a" in tree.ds.data_vars

def test_sanitize_datatree():
    tree = make_simple_tree()
    tree.ds = tree.ds.rename({"a": "1a"})
    sanitize_datatree(tree)
    assert "_1a" in tree.ds.data_vars
    for k in tree.attrs:
        assert " " not in k
        assert not k[0].isdigit()

def test_set_subtree_attrs_on_node():
    tree = make_simple_tree()
    attrs = {"foo": "bar", "baz": 123}
    set_subtree_attrs_on_node(tree, attrs)
    for k, v in attrs.items():
        assert tree.attrs[k] == v
