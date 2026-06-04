import numpy as np
import pytest
import xarray as xr
from podaac.subsetter.utils.file_utils import calculate_chunks, chunk_datatree


@pytest.fixture
def small_dataset() -> xr.Dataset:
    return xr.Dataset({"var": (["x", "y"], np.zeros((100, 100)))})


@pytest.fixture
def large_dataset() -> xr.Dataset:
    return xr.Dataset({"var": (["x", "y"], np.zeros((5000, 5000)))})


def test_calculate_chunks_returns_empty_for_small_dims(small_dataset: xr.Dataset) -> None:
    # a small enough dataset is not chunked
    result = calculate_chunks(small_dataset)
    assert result == {}


def test_calculate_chunks_returns_chunks_for_large_dims(large_dataset: xr.Dataset) -> None:
    result = calculate_chunks(large_dataset)
    assert result == {"x": 4000, "y": 4000}


def test_chunk_datatree_chunks_leaf_nodes(large_dataset: xr.Dataset) -> None:
    tree = xr.DataTree.from_dict({"/data": large_dataset})
    result = chunk_datatree(tree)
    chunks = result["/data"].ds["var"].chunks
    assert chunks is not None
    assert chunks[0] == (4000, 1000)  # 5000 split into 4000 + 1000
    assert chunks[1] == (4000, 1000)


def test_chunk_datatree_skips_empty_nodes(small_dataset: xr.Dataset) -> None:
    tree = xr.DataTree.from_dict({"/": None, "/data": small_dataset})
    result = chunk_datatree(tree)
    assert result["/data"].ds["var"].chunks is None


def test_chunk_datatree_skips_root_with_no_dims(large_dataset: xr.Dataset) -> None:
    tree = xr.DataTree.from_dict({"/": xr.Dataset(), "/leaf": large_dataset})

    # given a node with no dims, the the chunk calculation should return an empty dict
    assert calculate_chunks(tree.root) == {}

    # but when applying chunk_datatree, child nodes should get chunked
    result = chunk_datatree(tree)
    assert result["/leaf"].ds["var"].chunks is not None
    assert result["/"].ds.sizes == {}


def test_chunk_datatree_ignores_inherited_unused_dimensions() -> None:
    # reproduces a oco-3 lite sif 10.r / offset failure:
    # root carries a large sounding_dim so calculate_chunks returns
    #   {'sounding_dim': 4000} for any node that inherits it
    # /Offset has only small, unrelated dims (signalbin_dim, footprint_dim)

    offset_ds = xr.Dataset(
        {"histogram": (["signalbin_dim", "footprint_dim"], np.zeros((227, 8)))},
        coords={"sounding_dim": np.arange(5000)},
    )

    tree = xr.DataTree.from_dict({"/Offset": offset_ds})

    # precondition to confirm the inheritance is actually present
    assert "sounding_dim" in tree["/Offset"].ds.dims
    assert "sounding_dim" not in tree["/Offset"].ds["histogram"].dims

    # must not raise ValueError
    result = chunk_datatree(tree)

    # /Offset dims are below the chunking threshold so no chunks are applied,
    # but critically no error should be occuring
    assert result["/Offset"].ds["histogram"].chunks is None
