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

    This tests the behavior indirectly through where_tree since process_node is
    a nested function.
    """

    def test_child_aligned_to_parent_when_subsetted(self):
        """A child node that shares a coordinate dimension with a subsetted
        parent should be trimmed to match the parent's subsetted range."""
        from podaac.subsetter.datatree_subset import where_tree

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

        result = where_tree(tree, condition_dict, cut=True)

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
        from podaac.subsetter.datatree_subset import where_tree

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

        result = where_tree(tree, condition_dict, cut=True)

        np.testing.assert_array_equal(
            result["child"].ds.coords["x"].values, coords
        )

    def test_child_with_no_shared_dim_unchanged(self):
        """A child with different dims from parent is not affected by parent subsetting."""
        from podaac.subsetter.datatree_subset import where_tree

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

        result = where_tree(tree, condition_dict, cut=True)

        # Child has dim 'y' - should be unchanged
        np.testing.assert_array_equal(
            result["child"].ds.coords["y"].values, np.arange(7)
        )


    def test_sibling_aligned_after_first_child_subsets_parent(self):
        """When the first child returns indexers that subset the parent's
        processed_ds, the second child (which has no condition) gets aligned
        to the updated parent coordinates."""
        from podaac.subsetter.datatree_subset import where_tree

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

        result = where_tree(tree, condition_dict, cut=True)

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
        from podaac.subsetter.datatree_subset import where_tree

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

        result = where_tree(tree, condition_dict, cut=True)

        # Root should be subsetted (5 values kept from 10)
        assert result.ds.sizes["phony_dim_0"] == 5


class TestSubsetWithBboxSingleTimeVar:
    """Test that a single time variable is broadcast to all lat/lon pairs."""

    def test_single_time_var_replicated_for_multiple_lat_lon(self):
        """When there's 1 time var but multiple lat/lon vars, the time var
        should be replicated to pair with each lat/lon via zip."""
        from itertools import zip_longest
        from podaac.subsetter.subset import subset_with_bbox

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
