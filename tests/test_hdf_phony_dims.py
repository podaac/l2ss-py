import xarray as xr
from podaac.subsetter.utils.hdf_utils import rename_phony_dims
from xarray import DataTree


def make_tree(groups: dict[str, xr.Dataset]) -> DataTree:
    return DataTree.from_dict(groups)


def phony(n: int) -> str:
    return f"phony_dim_{n}"


def test_dimension_names_single_group():
    da = xr.DataArray(
        data=[[1, 2], [3, 4], [5, 6]],
        dims=[phony(0), phony(1)],
        attrs={"DimensionNames": "nTimes,nXtrack"},
    )
    dt = make_tree({"/": xr.Dataset({"var": da})})
    result = rename_phony_dims(dt)
    assert "nTimes" in result.ds.dims
    assert "nXtrack" in result.ds.dims


def test_dimension_names_match_across_groups():
    da_data = xr.DataArray(
        [[1.0, 2.0]],
        dims=[phony(0), phony(1)],
        attrs={"DimensionNames": "nTimes,nXtrack"},
    )
    da_geo = xr.DataArray(
        [[3.0, 4.0]],
        dims=[phony(2), phony(3)],
        attrs={"DimensionNames": "nTimes,nXtrack"},
    )
    dt = make_tree(
        {
            "/Data Fields": xr.Dataset({"ColumnAmount": da_data}),
            "/Geolocation Fields": xr.Dataset({"Latitude": da_geo}),
        }
    )
    result = rename_phony_dims(dt)
    data_dims = result["/Data Fields"].ds["ColumnAmount"].dims
    geo_dims = result["/Geolocation Fields"].ds["Latitude"].dims
    assert data_dims == geo_dims == ("nTimes", "nXtrack")


def test_dimension_names_no_phony_dims_unchanged():
    da = xr.DataArray([[1.0]], dims=["nTimes", "nXtrack"])
    dt = make_tree({"/": xr.Dataset({"var": da})})
    result = rename_phony_dims(dt)
    assert list(result.ds.dims) == ["nTimes", "nXtrack"]


def test_size_matching_different_sizes_get_different_names():
    da = xr.DataArray(
        [[[1.0] * 4] * 60] * 1496,
        dims=[phony(0), phony(1), phony(2)],
    )
    dt = make_tree({"/HDFEOS/SWATHS/S/Data Fields": xr.Dataset({"a": da})})
    result = rename_phony_dims(dt)
    dims = result["/HDFEOS/SWATHS/S/Data Fields"].ds["a"].dims
    assert len(set(dims)) == 3


def test_dimension_names_independent_swaths_do_not_share_dims():
    da1 = xr.DataArray(
        [[1.0, 2.0]],
        dims=[phony(0), phony(1)],
        attrs={"DimensionNames": "nTimesA,nXtrackA"},
    )
    da2 = xr.DataArray(
        [[3.0, 4.0]],
        dims=[phony(2), phony(3)],
        attrs={"DimensionNames": "nTimesB,nXtrackB"},
    )
    dt = make_tree(
        {
            "/HDFEOS/SWATHS/SwathA/Data Fields": xr.Dataset({"a": da1}),
            "/HDFEOS/SWATHS/SwathB/Data Fields": xr.Dataset({"b": da2}),
        }
    )
    result = rename_phony_dims(dt)
    dims_a = result["/HDFEOS/SWATHS/SwathA/Data Fields"].ds["a"].dims
    dims_b = result["/HDFEOS/SWATHS/SwathB/Data Fields"].ds["b"].dims
    assert dims_a == ("nTimesA", "nXtrackA")
    assert dims_b == ("nTimesB", "nXtrackB")
    assert set(dims_a).isdisjoint(set(dims_b))


def test_struct_metadata_renames_dims():
    odl = (
        "GROUP=SwathStructure\n"
        "GROUP=SWATH_1\n"
        'SwathName="MySWATH\n"'
        "GROUP=Dimension\n"
        "OBJECT=Dimension_1\n"
        'DimensionName="nTimes"\n'
        "Size=1496\n"
        "END_OBJECT=Dimension_1\n"
        "OBJECT=Dimension_2\n"
        'DimensionName="nXtrack"\n'
        "Size=60\n"
        "END_OBJECT=Dimension_2\n"
        "END_GROUP=Dimension\n"
        "GROUP=DimensionMap\n"
        "END_GROUP=DimensionMap\n"
        "GROUP=DataField\n"
        "OBJECT=DataField_1\n"
        'DataFieldName="ColumnAmount"\n'
        'DimList=("nTimes","nXtrack")\n'
        "END_OBJECT=DataField_1\n"
        "END_GROUP=DataField\n"
        "END_GROUP=SWATH_1\n"
        "END_GROUP=SwathStructure\n"
    )

    struct_da = xr.DataArray(odl)
    da = xr.DataArray(
        [[1.0] * 60] * 1496,
        dims=[phony(0), phony(1)],
    )
    dt = make_tree(
        {
            "/HDFEOS INFORMATION": xr.Dataset({"StructMetadata.0": struct_da}),
            "/HDFEOS/SWATHS/MySWATH/Data Fields": xr.Dataset({"ColumnAmount": da}),
        }
    )
    result = rename_phony_dims(dt)
    dims = result["/HDFEOS/SWATHS/MySWATH/Data Fields"].ds["ColumnAmount"].dims
    assert dims == ("nTimes", "nXtrack")


def test_struct_metadata_renames_dims_multi_swath():
    # note(ocs): this type of multi swatch structure occurs in collections like OMPIXCOR_003
    odl = (
        "GROUP=SwathStructure\n"
        "GROUP=SWATH_1\n"
        'SwathName="MySWATH\n"'
        "GROUP=Dimension\n"
        "OBJECT=Dimension_1\n"
        'DimensionName="nTimes"\n'
        "Size=1496\n"
        "END_OBJECT=Dimension_1\n"
        "OBJECT=Dimension_2\n"
        'DimensionName="nXtrack"\n'
        "Size=60\n"
        "END_OBJECT=Dimension_2\n"
        "END_GROUP=Dimension\n"
        "GROUP=DimensionMap\n"
        "END_GROUP=DimensionMap\n"
        "GROUP=DataField\n"
        "OBJECT=DataField_1\n"
        'DataFieldName="ColumnAmount"\n'
        'DimList=("nTimes","nXtrack")\n'
        "END_OBJECT=DataField_1\n"
        "END_GROUP=DataField\n"
        "END_GROUP=SWATH_1\n"
        # start of swath 2
        "GROUP=SWATH_2\n"
        'SwathName="MySWATH2\n"'
        "GROUP=Dimension\n"
        "OBJECT=Dimension_1\n"
        'DimensionName="nTimes"\n'
        "Size=1496\n"
        "END_OBJECT=Dimension_1\n"
        "OBJECT=Dimension_2\n"
        'DimensionName="nXtrack"\n'
        # this nXtrack size is different in swath 2
        "Size=30\n"
        "END_OBJECT=Dimension_2\n"
        "END_GROUP=Dimension\n"
        "GROUP=DataField\n"
        "OBJECT=DataField_1\n"
        'DataFieldName="ColumnAmount2"\n'
        'DimList=("nTimes","nXtrack")\n'
        "END_OBJECT=DataField_1\n"
        "END_GROUP=DataField\n"
        "END_GROUP=SWATH_2\n"
        "END_GROUP=SwathStructure\n"
    )
    struct_da = xr.DataArray(odl)
    da = xr.DataArray(
        [[1.0] * 60] * 1496,
        dims=[phony(0), phony(1)],
    )

    da2 = xr.DataArray(
        [[1.0] * 30] * 1496,
        dims=[phony(2), phony(3)],
    )
    dt = make_tree(
        {
            "/HDFEOS INFORMATION": xr.Dataset({"StructMetadata.0": struct_da}),
            "/HDFEOS/SWATHS/MySWATH/Data Fields": xr.Dataset({"ColumnAmount": da}),
            "/HDFEOS/SWATHS/MySWATH2/Data Fields": xr.Dataset({"ColumnAmount2": da2}),
        }
    )
    result = rename_phony_dims(dt)
    swath1_dims = result["/HDFEOS/SWATHS/MySWATH/Data Fields"].ds["ColumnAmount"].dims
    swath2_dims = result["/HDFEOS/SWATHS/MySWATH2/Data Fields"].ds["ColumnAmount2"].dims

    # have the names been recovered?
    assert swath1_dims == ("nTimes", "nXtrack")
    assert swath2_dims == ("nTimes", "nXtrack")

    swath1_dim_shape = (
        result["/HDFEOS/SWATHS/MySWATH/Data Fields"].ds["ColumnAmount"].shape
    )
    swath2_dim_shape = (
        result["/HDFEOS/SWATHS/MySWATH2/Data Fields"].ds["ColumnAmount2"].shape
    )

    # are dimensions in differnet swaths, with different sizes but
    # same name matched correctly?
    assert swath1_dim_shape == (1496, 60)
    assert swath2_dim_shape == (1496, 30)


def test_struct_grid_metadata_renames_dims():
    odl = (
        "GROUP=GridStructure\n"
        "GROUP=GRID_1\n"
        'SwathName="MyGRID\n"'
        "GROUP=Dimension\n"
        "OBJECT=Dimension_1\n"
        'DimensionName="nTimes"\n'
        "Size=1496\n"
        "END_OBJECT=Dimension_1\n"
        "OBJECT=Dimension_2\n"
        'DimensionName="nXtrack"\n'
        "Size=60\n"
        "END_OBJECT=Dimension_2\n"
        "END_GROUP=Dimension\n"
        "GROUP=DimensionMap\n"
        "END_GROUP=DimensionMap\n"
        "GROUP=DataField\n"
        "OBJECT=DataField_1\n"
        'DataFieldName="ColumnAmount"\n'
        'DimList=("nTimes","nXtrack")\n'
        "END_OBJECT=DataField_1\n"
        "END_GROUP=DataField\n"
        "END_GROUP=GRID_1\n"
        "END_GROUP=GridStructure\n"
    )
    struct_da = xr.DataArray(odl)
    da = xr.DataArray(
        [[1.0] * 60] * 1496,
        dims=[phony(0), phony(1)],
    )
    dt = make_tree(
        {
            "/HDFEOS INFORMATION": xr.Dataset({"StructMetadata.0": struct_da}),
            "/HDFEOS/GRIDS/MyGRID/Data Fields": xr.Dataset({"ColumnAmount": da}),
        }
    )
    result = rename_phony_dims(dt)
    dims = result["/HDFEOS/GRIDS/MyGRID/Data Fields"].ds["ColumnAmount"].dims
    assert dims == ("nTimes", "nXtrack")


def test_no_phony_dims_returns_unchanged():
    da = xr.DataArray([[1.0]], dims=["x", "y"])
    dt = make_tree({"/": xr.Dataset({"var": da})})
    result = rename_phony_dims(dt)
    assert list(result.ds["var"].dims) == ["x", "y"]


def test_empty_tree_does_not_raise():
    dt = make_tree({"/": xr.Dataset()})
    result = rename_phony_dims(dt)
    assert result is not None
