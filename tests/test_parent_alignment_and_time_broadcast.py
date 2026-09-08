"""Tests covering new lines introduced in issue/441 branch."""

import numpy as np
import xarray as xr
from xarray import DataTree

from podaac.subsetter.datatree_subset import apply_indexers_to_tree


class TestApplyIndexersToTreeWithParentDs:
    """Tests for apply_indexers_to_tree when parent_ds is provided."""

    def test_shared_dim_aligned_by_value(self):
        """When parent_ds is provided and child has a superset of the parent's
        coordinate values, the child should be trimmed to the intersection."""
        parent_ds = xr.Dataset(
            {"temp": ("x", [10, 20, 30])},
            coords={"x": [0, 1, 2]},
        )
        child_ds = xr.Dataset(
            {"salinity": ("x", [100, 200, 300, 400, 500])},
            coords={"x": [0, 1, 2, 3, 4]},
        )
        child_node = DataTree(name="child", dataset=child_ds)
        indexers = {"x": slice(0, 1)}

        result = apply_indexers_to_tree(child_node, indexers, parent_ds=parent_ds)

        # Child should be sel'd to x=[0,1,2] (common with parent), not isel'd
        np.testing.assert_array_equal(result.ds.coords["x"].values, [0, 1, 2])
        np.testing.assert_array_equal(result.ds["salinity"].values, [100, 200, 300])

    def test_no_shared_dim_falls_back_to_isel(self):
        """When parent_ds has no dimensions in common with child, isel is used."""
        parent_ds = xr.Dataset(
            {"temp": ("y", [10, 20, 30])},
            coords={"y": [0, 1, 2]},
        )
        child_ds = xr.Dataset(
            {"salinity": ("x", [100, 200, 300, 400, 500])},
            coords={"x": [0, 1, 2, 3, 4]},
        )
        child_node = DataTree(name="child", dataset=child_ds)
        indexers = {"x": slice(0, 3)}

        result = apply_indexers_to_tree(child_node, indexers, parent_ds=parent_ds)

        # Falls back to isel since no shared dims: slice(0,3) -> indices 0,1,2
        np.testing.assert_array_equal(result.ds.coords["x"].values, [0, 1, 2])

    def test_no_parent_ds_uses_isel(self):
        """When parent_ds is None, plain isel is used (original behavior)."""
        child_ds = xr.Dataset(
            {"salinity": ("x", [100, 200, 300, 400, 500])},
            coords={"x": [0, 1, 2, 3, 4]},
        )
        child_node = DataTree(name="child", dataset=child_ds)
        indexers = {"x": slice(1, 4)}

        result = apply_indexers_to_tree(child_node, indexers, parent_ds=None)

        # isel with slice(1,4) -> indices 1,2,3
        np.testing.assert_array_equal(result.ds.coords["x"].values, [1, 2, 3])

    def test_child_already_aligned_with_parent_no_sel_needed(self):
        """When child coord values are a subset of parent (common == child),
        no trimming happens and isel is used as fallback."""
        parent_ds = xr.Dataset(
            {"temp": ("x", [10, 20, 30, 40, 50])},
            coords={"x": [0, 1, 2, 3, 4]},
        )
        # Child already has same values as parent
        child_ds = xr.Dataset(
            {"salinity": ("x", [100, 200, 300, 400, 500])},
            coords={"x": [0, 1, 2, 3, 4]},
        )
        child_node = DataTree(name="child", dataset=child_ds)
        indexers = {"x": slice(0, 3)}

        result = apply_indexers_to_tree(child_node, indexers, parent_ds=parent_ds)

        # common = [0,1,2,3,4], len(common)=5 == len(child_values)=5, so no sel needed
        # falls back to isel: slice(0,3) -> indices 0,1,2
        np.testing.assert_array_equal(result.ds.coords["x"].values, [0, 1, 2])

    def test_recursive_application_to_grandchild(self):
        """parent_ds is passed recursively to all descendants, and they align
        shared dims against it."""
        parent_ds = xr.Dataset(
            {"temp": ("x", [10, 20, 30])},
            coords={"x": [0, 1, 2]},
        )
        child_ds = xr.Dataset(
            {"salinity": ("x", [100, 200, 300, 400])},
            coords={"x": [0, 1, 2, 3]},
        )
        # Grandchild has z (unique) but also inherits x from parent DataTree node
        grandchild_ds = xr.Dataset(
            {"pressure": ("z", [1, 2, 3, 4, 5])},
            coords={"z": [0, 1, 2, 3, 4]},
        )
        child_node = DataTree(name="child", dataset=child_ds)
        child_node["grandchild"] = DataTree(name="grandchild", dataset=grandchild_ds)

        indexers = {"x": slice(0, 2), "z": slice(0, 2)}

        result = apply_indexers_to_tree(child_node, indexers, parent_ds=parent_ds)

        # Child should be aligned to parent's x=[0,1,2] (intersect [0,1,2,3] & [0,1,2])
        np.testing.assert_array_equal(result.ds.coords["x"].values, [0, 1, 2])
        # Grandchild inherits 'x' from DataTree parent, so it also aligns via sel
        # on the x dimension from parent_ds. z is untouched since sel is used (not isel).
        np.testing.assert_array_equal(
            result["grandchild"].ds.coords["z"].values, [0, 1, 2, 3, 4]
        )

    def test_node_with_none_dataset(self):
        """Nodes with ds=None are handled gracefully."""
        parent_ds = xr.Dataset(
            {"temp": ("x", [10, 20])},
            coords={"x": [0, 1]},
        )
        child_node = DataTree(name="empty_child", dataset=None)
        indexers = {"x": slice(0, 1)}

        result = apply_indexers_to_tree(child_node, indexers, parent_ds=parent_ds)

        assert result.ds.sizes == {}

    def test_partial_overlap_selects_common_values(self):
        """When child coords partially overlap parent coords, only common values
        are kept via sel."""
        parent_ds = xr.Dataset(
            {"temp": ("x", [10, 20, 30])},
            coords={"x": [2, 3, 4]},
        )
        child_ds = xr.Dataset(
            {"salinity": ("x", [100, 200, 300, 400, 500])},
            coords={"x": [0, 1, 2, 3, 4]},
        )
        child_node = DataTree(name="child", dataset=child_ds)
        indexers = {"x": slice(0, 2)}

        result = apply_indexers_to_tree(child_node, indexers, parent_ds=parent_ds)

        # common = intersect([2,3,4], [0,1,2,3,4]) = [2,3,4]; len(common)=3 < len(child)=5
        np.testing.assert_array_equal(result.ds.coords["x"].values, [2, 3, 4])
        np.testing.assert_array_equal(result.ds["salinity"].values, [300, 400, 500])


class TestParentProcessedDsAlignment:
    """Tests for the parent_processed_ds alignment in the else branch of process_node.

    This tests the behavior indirectly through subset_tree since process_node is
    a nested function.
    """

    def test_child_aligned_to_parent_when_subsetted(self):
        """A child node that shares a coordinate dimension with a subsetted
        parent should be trimmed to match the parent's subsetted range."""
        from podaac.subsetter.subset_tree import subset_tree

        # Parent and child share x with same size (DataTree allows this)
        parent_ds = xr.Dataset(
            {"temp": ("x", np.arange(5, dtype=float))},
            coords={"x": np.arange(5)},
        )
        child_ds = xr.Dataset(
            {"salinity": ("x", np.arange(5, dtype=float) * 10)},
            coords={"x": np.arange(5)},
        )

        tree = DataTree(name="root", dataset=parent_ds)
        tree["child"] = DataTree(name="child", dataset=child_ds)

        # Condition on root that subsets to x=[1,2,3]
        cond = xr.DataArray(
            [False, True, True, True, False], dims=["x"], coords={"x": np.arange(5)}
        )
        condition_dict = {"/": cond}

        result = subset_tree(tree, condition_dict, cut=True)

        # Parent should be subsetted to x=[1,2,3]
        np.testing.assert_array_equal(result.ds.coords["x"].values, [1, 2, 3])
        # Child should also be aligned to x=[1,2,3]
        np.testing.assert_array_equal(
            result["child"].ds.coords["x"].values, [1, 2, 3]
        )
        np.testing.assert_array_equal(
            result["child"].ds["salinity"].values, [10.0, 20.0, 30.0]
        )

    def test_child_not_modified_when_all_kept(self):
        """A child with the same coordinate range as the parent is unchanged
        when all values pass the condition."""
        from podaac.subsetter.subset_tree import subset_tree

        coords = np.arange(5)
        parent_ds = xr.Dataset(
            {"temp": ("x", np.arange(5, dtype=float))},
            coords={"x": coords},
        )
        child_ds = xr.Dataset(
            {"salinity": ("x", np.arange(5, dtype=float) * 10)},
            coords={"x": coords},
        )

        tree = DataTree(name="root", dataset=parent_ds)
        tree["child"] = DataTree(name="child", dataset=child_ds)

        # Condition that keeps all values
        cond = xr.DataArray(
            [True, True, True, True, True], dims=["x"], coords={"x": coords}
        )
        condition_dict = {"/": cond}

        result = subset_tree(tree, condition_dict, cut=True)

        np.testing.assert_array_equal(
            result["child"].ds.coords["x"].values, coords
        )

    def test_child_with_no_shared_dim_unchanged(self):
        """A child with different dims from parent is not affected by parent subsetting."""
        from podaac.subsetter.subset_tree import subset_tree

        parent_ds = xr.Dataset(
            {"temp": ("x", np.arange(5, dtype=float))},
            coords={"x": np.arange(5)},
        )
        child_ds = xr.Dataset(
            {"salinity": ("y", np.arange(7, dtype=float))},
            coords={"y": np.arange(7)},
        )

        tree = DataTree(name="root", dataset=parent_ds)
        tree["child"] = DataTree(name="child", dataset=child_ds)

        # Condition that subsets root
        cond = xr.DataArray(
            [False, True, True, False, False], dims=["x"], coords={"x": np.arange(5)}
        )
        condition_dict = {"/": cond}

        result = subset_tree(tree, condition_dict, cut=True)

        # Child has dim 'y' - should be unchanged
        np.testing.assert_array_equal(
            result["child"].ds.coords["y"].values, np.arange(7)
        )


    def test_sibling_aligned_after_first_child_subsets_parent(self):
        """When the first child returns indexers that subset the parent's
        processed_ds, the second child (which has no condition) gets aligned
        to the updated parent coordinates."""
        from podaac.subsetter.subset_tree import subset_tree

        # Root has x-dim data. Two children under 'child1' have conditions at depth 2.
        # 'child2' at depth 1 has NO condition match -> goes to else branch.
        # After child1's sub-nodes return indexers, root's processed_ds is updated,
        # and child2 sees the subsetted parent_processed_ds.
        root_ds = xr.Dataset(
            {"temp": ("x", np.arange(10, dtype=float))},
            coords={"x": np.arange(10)},
        )
        sub1_ds = xr.Dataset(
            {"lat": ("x", np.linspace(-90, 90, 10))},
            coords={"x": np.arange(10)},
        )
        sub2_ds = xr.Dataset(
            {"lon": ("x", np.linspace(-180, 180, 10))},
            coords={"x": np.arange(10)},
        )
        child2_ds = xr.Dataset(
            {"nav": ("x", np.arange(10, dtype=float) * 2)},
            coords={"x": np.arange(10)},
        )

        tree = DataTree(name="root", dataset=root_ds)
        tree["child1"] = DataTree(
            name="child1",
            dataset=xr.Dataset(
                {"flag": ("x", np.ones(10))}, coords={"x": np.arange(10)}
            ),
        )
        tree["child1/sub1"] = DataTree(name="sub1", dataset=sub1_ds)
        tree["child1/sub2"] = DataTree(name="sub2", dataset=sub2_ds)
        tree["child2"] = DataTree(name="child2", dataset=child2_ds)

        # Condition at depth-2 paths only (child2 at depth-1 won't match)
        cond = xr.DataArray(
            [False, False, True, True, True, True, True, False, False, False],
            dims=["x"],
            coords={"x": np.arange(10)},
        )
        condition_dict = {"/child1/sub1": cond, "/child1/sub2": cond}

        result = subset_tree(tree, condition_dict, cut=True)

        # child2 should be aligned to the subsetted parent x=[2,3,4,5,6]
        np.testing.assert_array_equal(
            result["child2"].ds.coords["x"].values, [2, 3, 4, 5, 6]
        )
        np.testing.assert_array_equal(
            result["child2"].ds["nav"].values, [4.0, 6.0, 8.0, 10.0, 12.0]
        )

    def test_empty_subtree_gets_indexers_applied(self):
        """When a child is in empty_paths and the parent has indexers,
        apply_indexers_to_tree is called with the parent's processed_ds."""
        from podaac.subsetter.subset_tree import subset_tree

        # Root with phony_dim (no coords, so children don't inherit)
        root_ds = xr.Dataset(
            {"temp": (("phony_dim_0",), np.arange(10, dtype=float))}
        )
        tree = DataTree(name="root", dataset=root_ds)
        tree["empty_sub"] = DataTree(name="empty_sub", dataset=xr.Dataset())
        tree["empty_sub/also_empty"] = DataTree(
            name="also_empty", dataset=xr.Dataset()
        )

        cond = xr.DataArray(
            [False, False, True, True, True, True, True, False, False, False],
            dims=["phony_dim_0"],
        )
        condition_dict = {"/": cond}

        result = subset_tree(tree, condition_dict, cut=True)

        # Root should be subsetted (5 values kept from 10)
        assert result.ds.sizes["phony_dim_0"] == 5


class TestSubsetWithBboxSingleTimeVar:
    """Test that a single time variable is broadcast to all lat/lon pairs."""

    def test_single_time_var_replicated_for_multiple_lat_lon(self):
        """When there's 1 time var but multiple lat/lon vars, the time var
        should be replicated to pair with each lat/lon via zip."""

        lat_var_names = ["/group1/lat", "/group2/lat"]
        lon_var_names = ["/group1/lon", "/group2/lon"]
        time_var_names = ["/time"]

        # Verify the iterator logic directly: old behavior was zip() which
        # truncates to shortest, losing the second lat/lon pair.
        # New behavior: time_var_names * len(lat_var_names) replicates time.
        if len(time_var_names) == 1 and len(lat_var_names) > 1:
            iterator = zip(lat_var_names, lon_var_names, time_var_names * len(lat_var_names))
        else:
            iterator = zip(lat_var_names, lon_var_names, time_var_names)

        pairs = list(iterator)
        assert len(pairs) == 2
        assert pairs[0] == ("/group1/lat", "/group1/lon", "/time")
        assert pairs[1] == ("/group2/lat", "/group2/lon", "/time")

    def test_multiple_time_vars_not_replicated(self):
        """When there are equal numbers of time and lat/lon vars, normal zip is used."""
        lat_var_names = ["/group1/lat", "/group2/lat"]
        lon_var_names = ["/group1/lon", "/group2/lon"]
        time_var_names = ["/group1/time", "/group2/time"]

        if len(time_var_names) == 1 and len(lat_var_names) > 1:
            iterator = zip(lat_var_names, lon_var_names, time_var_names * len(lat_var_names))
        else:
            iterator = zip(lat_var_names, lon_var_names, time_var_names)

        pairs = list(iterator)
        assert len(pairs) == 2
        assert pairs[0] == ("/group1/lat", "/group1/lon", "/group1/time")
        assert pairs[1] == ("/group2/lat", "/group2/lon", "/group2/time")

    def test_no_time_vars_uses_zip_longest(self):
        """When time_var_names is empty, zip_longest fills with None."""
        from itertools import zip_longest

        lat_var_names = ["/group1/lat", "/group2/lat"]
        lon_var_names = ["/group1/lon", "/group2/lon"]
        time_var_names = []

        if not time_var_names:
            iterator = zip_longest(lat_var_names, lon_var_names, [])
        elif len(time_var_names) == 1 and len(lat_var_names) > 1:
            iterator = zip(lat_var_names, lon_var_names, time_var_names * len(lat_var_names))
        else:
            iterator = zip(lat_var_names, lon_var_names, time_var_names)

        pairs = list(iterator)
        assert len(pairs) == 2
        assert pairs[0] == ("/group1/lat", "/group1/lon", None)
        assert pairs[1] == ("/group2/lat", "/group2/lon", None)


class TestSubsetTreeEmptyConditionDict:
    """Test subset_tree with empty condition dict returns tree unchanged."""

    def test_empty_condition_dict_returns_tree(self):
        from podaac.subsetter.subset_tree import subset_tree

        ds = xr.Dataset({"temp": (("x", "y"), np.arange(12).reshape(3, 4))})
        tree = DataTree(name="root", dataset=ds)
        result = subset_tree(tree, {}, cut=True)
        assert result is tree


class TestSubsetTreeMultiConditionSameShape:
    """Test subset_tree with multiple conditions that have the same shape."""

    def test_same_shape_conditions_combined_with_or(self):
        from podaac.subsetter.subset_tree import subset_tree

        root_ds = xr.Dataset({"temp": (("x", "y"), np.arange(20, dtype=np.float64).reshape(4, 5))})
        root_ds["temp"].attrs["_FillValue"] = -999.0
        child_ds = xr.Dataset({"salt": (("x", "y"), np.arange(20, dtype=np.float64).reshape(4, 5) * 10)})
        child_ds["salt"].attrs["_FillValue"] = -999.0

        tree = DataTree(name="root", dataset=root_ds)
        tree["child"] = DataTree(name="child", dataset=child_ds)

        cond1 = xr.DataArray(
            np.array([[True, True, False, False, False]] * 4),
            dims=("x", "y"),
        )
        cond2 = xr.DataArray(
            np.array([[False, False, False, True, True]] * 4),
            dims=("x", "y"),
        )
        condition_dict = {"/": cond1, "/child": cond2}
        result = subset_tree(tree, condition_dict, cut=True)

        assert "child" in result.children
        assert result.ds.sizes["y"] == 4
        assert result["child"].ds.sizes["y"] == 4


class TestSubsetTreePerGroupDifferentShapes:
    """Test _apply_per_group with conditions of different shapes."""

    def test_different_shape_conditions(self):
        from podaac.subsetter.subset_tree import subset_tree

        root_ds = xr.Dataset()
        child1_ds = xr.Dataset({
            "temp": (("x",), np.arange(10, dtype=np.float64)),
        })
        child1_ds["temp"].attrs["_FillValue"] = -999.0
        child2_ds = xr.Dataset({
            "salt": (("y",), np.arange(5, dtype=np.float64)),
        })
        child2_ds["salt"].attrs["_FillValue"] = -999.0

        tree = DataTree(name="root", dataset=root_ds)
        tree["child1"] = DataTree(name="child1", dataset=child1_ds)
        tree["child2"] = DataTree(name="child2", dataset=child2_ds)

        cond1 = xr.DataArray(np.array([True] * 5 + [False] * 5), dims=("x",))
        cond2 = xr.DataArray(np.array([True, True, False, False, False]), dims=("y",))

        condition_dict = {"/child1": cond1, "/child2": cond2}
        result = subset_tree(tree, condition_dict, cut=True)

        assert result["child1"].ds.sizes["x"] == 5
        assert result["child2"].ds.sizes["y"] == 2


class TestApplyPerGroupPixelSubset:
    """Test _apply_per_group with pixel_subset=True."""

    def test_pixel_subset_skips_masking(self):
        from podaac.subsetter.subset_tree import subset_tree

        root_ds = xr.Dataset()
        child_ds = xr.Dataset({
            "temp": (("x",), np.arange(10, dtype=np.float64)),
        })
        child_ds["temp"].attrs["_FillValue"] = -999.0

        tree = DataTree(name="root", dataset=root_ds)
        tree["child1"] = DataTree(name="child1", dataset=child_ds)

        cond1 = xr.DataArray(np.array([True] * 5 + [False] * 5), dims=("x",))
        cond2 = xr.DataArray(np.array([True, True, False]), dims=("z",))
        condition_dict = {"/child1": cond1, "/other": cond2}

        result = subset_tree(tree, condition_dict, cut=True, pixel_subset=True)
        assert result["child1"].ds.sizes["x"] == 5


class TestApplyPerGroupEmptyPaths:
    """Test _apply_per_group when child is in empty_paths."""

    def test_empty_child_gets_indexers_applied(self):
        from podaac.subsetter.subset_tree import subset_tree

        root_ds = xr.Dataset()
        child_ds = xr.Dataset({
            "temp": (("x",), np.arange(10, dtype=np.float64)),
        })
        child_ds["temp"].attrs["_FillValue"] = -999.0
        empty_child_ds = xr.Dataset()

        tree = DataTree(name="root", dataset=root_ds)
        tree["child"] = DataTree(name="child", dataset=child_ds)
        tree["child"]["empty_grandchild"] = DataTree(name="empty_grandchild", dataset=empty_child_ds)

        cond = xr.DataArray(np.array([True] * 5 + [False] * 5), dims=("x",))
        cond2 = xr.DataArray(np.array([True, True, False]), dims=("z",))
        condition_dict = {"/child": cond, "/other": cond2}

        result = subset_tree(tree, condition_dict, cut=True)
        assert result["child"].ds.sizes["x"] == 5


class TestApplyPerGroupChildIndexersPropagateToParent:
    """Test that child indexers propagate up to parent when parent has no condition."""

    def test_child_indexers_propagate(self):
        from podaac.subsetter.subset_tree import subset_tree

        root_ds = xr.Dataset({
            "root_var": (("x",), np.arange(10, dtype=np.float64)),
        })
        child_ds = xr.Dataset({
            "temp": (("x",), np.arange(10, dtype=np.float64)),
        })
        child_ds["temp"].attrs["_FillValue"] = -999.0

        tree = DataTree(name="root", dataset=root_ds)
        tree["child"] = DataTree(name="child", dataset=child_ds)

        cond1 = xr.DataArray(np.array([True] * 5 + [False] * 5), dims=("x",))
        cond2 = xr.DataArray(np.array([True, True, False]), dims=("z",))
        condition_dict = {"/child": cond1, "/other": cond2}

        result = subset_tree(tree, condition_dict, cut=True)
        assert result.ds.sizes["x"] == 5


class TestAlignToParent:
    """Test _align_to_parent directly."""

    def test_aligns_child_to_parent_coords(self):
        from podaac.subsetter.subset_tree import _align_to_parent

        parent_ds = xr.Dataset(
            {"temp": ("x", [10, 20, 30])},
            coords={"x": [0, 1, 2]},
        )
        child_ds = xr.Dataset(
            {"salt": ("x", [100, 200, 300, 400, 500])},
            coords={"x": [0, 1, 2, 3, 4]},
        )
        result = _align_to_parent(child_ds, parent_ds)
        np.testing.assert_array_equal(result.coords["x"].values, [0, 1, 2])

    def test_no_change_when_already_aligned(self):
        from podaac.subsetter.subset_tree import _align_to_parent

        parent_ds = xr.Dataset(
            {"temp": ("x", [10, 20, 30])},
            coords={"x": [0, 1, 2]},
        )
        child_ds = xr.Dataset(
            {"salt": ("x", [100, 200, 300])},
            coords={"x": [0, 1, 2]},
        )
        result = _align_to_parent(child_ds, parent_ds)
        np.testing.assert_array_equal(result.coords["x"].values, [0, 1, 2])


class TestFindReferenceDataset:
    """Test _find_reference_dataset."""

    def test_finds_child_when_root_has_no_matching_dims(self):
        from podaac.subsetter.subset_tree import _find_reference_dataset

        root_ds = xr.Dataset({"meta": (("z",), [1, 2])})
        child_ds = xr.Dataset({"temp": (("x", "y"), np.zeros((3, 4)))})
        tree = DataTree(name="root", dataset=root_ds)
        tree["child"] = DataTree(name="child", dataset=child_ds)

        cond = xr.DataArray(np.ones((3, 4), dtype=bool), dims=("x", "y"))
        ref = _find_reference_dataset(tree, cond)
        assert set(ref.dims) == {"x", "y"}

    def test_falls_back_to_root(self):
        from podaac.subsetter.subset_tree import _find_reference_dataset

        root_ds = xr.Dataset({"meta": (("z",), [1, 2])})
        tree = DataTree(name="root", dataset=root_ds)

        cond = xr.DataArray(np.ones(3, dtype=bool), dims=("x",))
        ref = _find_reference_dataset(tree, cond)
        assert "z" in ref.dims


class TestResolveCondition:
    """Test _resolve_condition."""

    def test_single_condition(self):
        from podaac.subsetter.subset_tree import _resolve_condition

        tree = DataTree(name="root", dataset=xr.Dataset({"temp": (("x",), [1, 2, 3])}))
        cond = xr.DataArray(np.ones(3, dtype=bool), dims=("x",))
        result = _resolve_condition(tree, {"path": cond})
        assert result is cond

    def test_empty_dict(self):
        from podaac.subsetter.subset_tree import _resolve_condition

        tree = DataTree(name="root", dataset=xr.Dataset({"temp": (("x",), [1, 2, 3])}))
        result = _resolve_condition(tree, {})
        assert result is None

    def test_picks_best_matching_condition(self):
        from podaac.subsetter.subset_tree import _resolve_condition

        child_ds = xr.Dataset({"temp": (("x",), np.arange(5))})
        root_ds = xr.Dataset({"meta": (("x",), np.arange(5))})
        tree = DataTree(name="root", dataset=root_ds)
        tree["child"] = DataTree(name="child", dataset=child_ds)

        cond_good = xr.DataArray(np.ones(5, dtype=bool), dims=("x",))
        cond_bad = xr.DataArray(np.ones(3, dtype=bool), dims=("y",))
        result = _resolve_condition(tree, {"/child": cond_good, "/other": cond_bad})
        assert result is cond_good


class TestApplyMaskingEdgeCases:
    """Test _apply_masking edge cases."""

    def test_no_data_vars_returns_unchanged(self):
        from podaac.subsetter.subset_tree import _apply_masking

        ds = xr.Dataset()
        cond = xr.DataArray(np.ones(3, dtype=bool), dims=("x",))
        result = _apply_masking(ds, cond)
        assert len(result.data_vars) == 0

    def test_no_dim_overlap_returns_unchanged(self):
        from podaac.subsetter.subset_tree import _apply_masking

        ds = xr.Dataset({"temp": (("z",), [1.0, 2.0, 3.0])})
        cond = xr.DataArray(np.ones(3, dtype=bool), dims=("x",))
        result = _apply_masking(ds, cond)
        np.testing.assert_array_equal(result["temp"].values, [1.0, 2.0, 3.0])

    def test_partial_dim_overlap_collapses_extra_dims(self):
        from podaac.subsetter.subset_tree import _apply_masking

        ds = xr.Dataset({
            "temp_1d": (("x",), np.arange(3, dtype=np.float64)),
        })
        ds["temp_1d"].attrs["_FillValue"] = -999.0
        cond = xr.DataArray(
            np.array([[True, False], [True, True], [False, False]]),
            dims=("x", "y"),
        )
        result = _apply_masking(ds, cond)
        assert result["temp_1d"].values[0] == 0.0
        assert result["temp_1d"].values[1] == 1.0
        assert result["temp_1d"].values[2] == -999.0

    def test_scalar_variable_preserved(self):
        from podaac.subsetter.subset_tree import _apply_masking

        ds = xr.Dataset({
            "temp": (("x",), [1.0, 2.0, 3.0]),
            "scalar_val": ((), 42.0),
        })
        ds["scalar_val"].attrs["_FillValue"] = -999.0
        cond = xr.DataArray(np.array([True, False, True]), dims=("x",))
        result = _apply_masking(ds, cond)
        assert float(result["scalar_val"].values) == 42.0

    def test_fillna_and_dtype_casting(self):
        from podaac.subsetter.subset_tree import _apply_masking

        ds = xr.Dataset({
            "temp": (("x",), np.array([1, 2, 3], dtype=np.int32)),
        })
        ds["temp"].attrs["_FillValue"] = -999
        cond = xr.DataArray(np.array([True, False, True]), dims=("x",))
        result = _apply_masking(ds, cond)
        assert result["temp"].dtype == np.int32
        assert result["temp"].values[1] == -999


class TestPruneEmpty:
    """Test _prune_empty."""

    def test_removes_fully_empty_subtrees(self):
        from podaac.subsetter.subset_tree import _prune_empty

        root_ds = xr.Dataset({"temp": (("x",), [1, 2, 3])})
        tree = DataTree(name="root", dataset=root_ds)
        tree["empty_child"] = DataTree(name="empty_child", dataset=xr.Dataset())
        tree["data_child"] = DataTree(
            name="data_child",
            dataset=xr.Dataset({"salt": (("x",), [4, 5, 6])}),
        )

        result = _prune_empty(tree)
        assert "data_child" in result.children
        assert "empty_child" not in result.children
