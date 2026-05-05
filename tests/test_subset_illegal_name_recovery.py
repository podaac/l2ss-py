import numpy as np
import pytest
from unittest.mock import Mock

from podaac.subsetter import subset


class _DummyTree:
    groups = []


class _DummyOpenDataTree:
    def __init__(self, tree):
        self._tree = tree

    def __enter__(self):
        return self._tree

    def __exit__(self, exc_type, exc, tb):
        return False


class _RecoverableToNetcdfDataset:
    def __init__(self):
        self.calls = 0

    def to_netcdf(self, *_args, **_kwargs):
        self.calls += 1
        if self.calls == 1:
            raise AttributeError("NetCDF: Name contains illegal characters")


class _UnrecoverableToNetcdfDataset:
    def to_netcdf(self, *_args, **_kwargs):
        raise AttributeError("some other attribute error")


def _patch_subset_dependencies(monkeypatch, subsetted_dataset, spatial_bounds):
    dummy_tree = _DummyTree()

    monkeypatch.setattr(subset.xr, "open_datatree", lambda *_args, **_kwargs: _DummyOpenDataTree(dummy_tree))
    monkeypatch.setattr(subset.file_utils, "override_decode_cf_datetime", lambda: None)
    monkeypatch.setattr(subset.file_utils, "has_scantime", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(subset.coordinate_utils, "get_coordinate_variable_names", lambda **_kwargs: (["/lat"], ["/lon"], []))
    monkeypatch.setattr(subset.file_utils, "calculate_chunks", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(subset.variables_utils, "get_all_variable_names_from_dtree", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(
        subset.variables_utils, "normalize_candidate_paths_against_dtree", lambda paths, *_args, **_kwargs: paths
    )
    monkeypatch.setattr(subset, "subset_with_bbox", lambda **_kwargs: subsetted_dataset)
    monkeypatch.setattr(subset.metadata_utils, "set_version_history", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(subset.metadata_utils, "set_json_history", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(subset.datatree_subset, "clean_inherited_coords", lambda ds: ds)
    monkeypatch.setattr(subset.datatree_subset, "prepare_basic_encoding", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        subset.datatree_subset, "tree_get_spatial_bounds", lambda *_args, **_kwargs: spatial_bounds
    )
    monkeypatch.setattr(subset.metadata_utils, "update_netcdf_attrs", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(subset.metadata_utils, "ensure_time_units", lambda *_args, **_kwargs: None)


def test_subset_recovers_when_netcdf_has_illegal_names(monkeypatch, tmp_path):
    spatial_bounds = np.array([[1.0, 2.0], [3.0, 4.0]])
    recoverable_dataset = _RecoverableToNetcdfDataset()
    fix_illegal_attrs = Mock()

    _patch_subset_dependencies(monkeypatch, recoverable_dataset, spatial_bounds)
    monkeypatch.setattr(subset.metadata_utils, "fix_illegal_datatree_attrs", fix_illegal_attrs)

    result = subset.subset(
        file_to_subset=str(tmp_path / "input.nc"),
        bbox=np.array([[-180.0, 180.0], [-90.0, 90.0]]),
        output_file=str(tmp_path / "output.nc"),
    )

    assert recoverable_dataset.calls == 2
    fix_illegal_attrs.assert_called_once_with(recoverable_dataset)
    np.testing.assert_array_equal(result, spatial_bounds)


def test_subset_reraises_unexpected_attribute_error(monkeypatch, tmp_path):
    spatial_bounds = np.array([[1.0, 2.0], [3.0, 4.0]])
    unrecoverable_dataset = _UnrecoverableToNetcdfDataset()
    fix_illegal_attrs = Mock()

    _patch_subset_dependencies(monkeypatch, unrecoverable_dataset, spatial_bounds)
    monkeypatch.setattr(subset.metadata_utils, "fix_illegal_datatree_attrs", fix_illegal_attrs)

    with pytest.raises(AttributeError, match="some other attribute error"):
        subset.subset(
            file_to_subset=str(tmp_path / "input.nc"),
            bbox=np.array([[-180.0, 180.0], [-90.0, 90.0]]),
            output_file=str(tmp_path / "output.nc"),
        )

    fix_illegal_attrs.assert_not_called()
