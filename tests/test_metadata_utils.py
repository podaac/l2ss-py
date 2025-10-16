import pytest
import xarray as xr
from xarray import DataTree
from podaac.subsetter.utils.metadata_utils import (
    legalize_attr_name,
    check_illegal_datatree_attrs,
    fix_illegal_datatree_attrs,
)

def make_datatree_with_attrs(global_attrs, var_attrs):
    ds = xr.Dataset()
    ds.attrs = global_attrs
    ds["var"] = xr.DataArray([1, 2, 3])
    for k, v in var_attrs.items():
        ds["var"].attrs[k] = v
    return DataTree(ds)

def test_check_illegal_datatree_attrs_detects_illegal():
    dt = make_datatree_with_attrs(
        {"1bad": "x", "good": "y"},
        {"bad attr": "z", "good_attr": "w"}
    )
    assert check_illegal_datatree_attrs(dt) is True

def test_check_illegal_datatree_attrs_no_illegal():
    dt = make_datatree_with_attrs(
        {"good": "x", "also_good": "y"},
        {"good_attr": "z"}
    )
    assert check_illegal_datatree_attrs(dt) is False

def test_fix_illegal_datatree_attrs_changes_names():
    dt = make_datatree_with_attrs(
        {"bad attr": "x", "good": "y"},
        {"bad attr": "z", "good_attr": "w"}
    )
    fix_illegal_datatree_attrs(dt)
    root_attrs = dict(dt.ds.attrs)
    assert "bad_attr" in root_attrs
    assert "good" in root_attrs
    var_attrs = dict(dt.ds["var"].attrs)
    assert "bad_attr" in var_attrs
    assert "good_attr" in var_attrs
    assert not check_illegal_datatree_attrs(dt)
