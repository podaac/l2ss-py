import h5py
import numpy as np
import pytest


@pytest.fixture(scope="session")
def fake_omi_pixcor_file(tmp_path_factory):
    """
    Generated from OMI-Aura_L2-OMPIXCOR_2004m1001t1452-o01141_v003-2018m0228t201200.he5
    """
    filepath = (
        tmp_path_factory.mktemp("data")
        / "OMI-Aura_L2-OMPIXCOR_2004m1001t1452-o01141_v003-2018m0228t201200.he5"
    )

    dim_0_1643 = 18
    dim_0_30 = 10
    dim_0_4 = 2
    dim_0_60 = 14
    dim_1_1643 = 18
    dim_1_27 = 6
    dim_1_30 = 10
    dim_1_60 = 14
    dim_2_30 = 10
    dim_2_60 = 14

    with h5py.File(filepath, "w") as f:
        # create groups
        hdfeos = f.require_group("HDFEOS")
        hdfeos_additional = f.require_group("HDFEOS/ADDITIONAL")
        hdfeos_additional_file_attributes = f.require_group(
            "HDFEOS/ADDITIONAL/FILE_ATTRIBUTES"
        )
        hdfeos_swaths = f.require_group("HDFEOS/SWATHS")
        hdfeos_swaths_omi_ground_pixel_corners_uv_1 = f.require_group(
            "HDFEOS/SWATHS/OMI Ground Pixel Corners UV-1"
        )
        hdfeos_swaths_omi_ground_pixel_corners_uv_1_data_fields = f.require_group(
            "HDFEOS/SWATHS/OMI Ground Pixel Corners UV-1/Data Fields"
        )
        hdfeos_swaths_omi_ground_pixel_corners_uv_1_geolocation_fields = (
            f.require_group(
                "HDFEOS/SWATHS/OMI Ground Pixel Corners UV-1/Geolocation Fields"
            )
        )
        hdfeos_swaths_omi_ground_pixel_corners_uv_2 = f.require_group(
            "HDFEOS/SWATHS/OMI Ground Pixel Corners UV-2"
        )
        hdfeos_swaths_omi_ground_pixel_corners_uv_2_data_fields = f.require_group(
            "HDFEOS/SWATHS/OMI Ground Pixel Corners UV-2/Data Fields"
        )
        hdfeos_swaths_omi_ground_pixel_corners_uv_2_geolocation_fields = (
            f.require_group(
                "HDFEOS/SWATHS/OMI Ground Pixel Corners UV-2/Geolocation Fields"
            )
        )
        hdfeos_swaths_omi_ground_pixel_corners_vis = f.require_group(
            "HDFEOS/SWATHS/OMI Ground Pixel Corners VIS"
        )
        hdfeos_swaths_omi_ground_pixel_corners_vis_data_fields = f.require_group(
            "HDFEOS/SWATHS/OMI Ground Pixel Corners VIS/Data Fields"
        )
        hdfeos_swaths_omi_ground_pixel_corners_vis_geolocation_fields = f.require_group(
            "HDFEOS/SWATHS/OMI Ground Pixel Corners VIS/Geolocation Fields"
        )
        hdfeos_information = f.require_group("HDFEOS INFORMATION")

        # attributes for HDFEOS/ADDITIONAL/FILE_ATTRIBUTES
        hdfeos_additional_file_attributes.attrs["AuthorAffiliation"] = (
            "Smithsonian Astrophysical Observatory"
        )
        hdfeos_additional_file_attributes.attrs["AuthorName"] = "Thomas P. Kurosu"
        hdfeos_additional_file_attributes.attrs["ESDTDescriptorRevision"] = "2.0.05"
        hdfeos_additional_file_attributes.attrs["GranuleDay"] = np.array(
            [np.int32(np.int32(1))], dtype=np.int32
        )
        hdfeos_additional_file_attributes.attrs["GranuleMonth"] = np.array(
            [np.int32(np.int32(10))], dtype=np.int32
        )
        hdfeos_additional_file_attributes.attrs["GranuleYear"] = np.array(
            [np.int32(np.int32(2004))], dtype=np.int32
        )
        hdfeos_additional_file_attributes.attrs["InputVersions"] = (
            "OML1BRUG:1.0.00 OML1BRVG:1.0.00"
        )
        hdfeos_additional_file_attributes.attrs["InstrumentName"] = "OMI"
        hdfeos_additional_file_attributes.attrs["LongName"] = (
            "OMI/Aura Global Ground Pixel Corners 1-Orbit L2 Swath 13x24km"
        )
        hdfeos_additional_file_attributes.attrs["OPERATIONMODE"] = "Test"
        hdfeos_additional_file_attributes.attrs["OrbitData"] = "DEFINITIVE"
        hdfeos_additional_file_attributes.attrs["PGEVERSION"] = "1.2.7"
        hdfeos_additional_file_attributes.attrs["ProcessLevel"] = "2"
        hdfeos_additional_file_attributes.attrs["ProcessingCenter"] = "OMI SIPS"
        hdfeos_additional_file_attributes.attrs["ProcessingHost"] = (
            "Linux minion7090 3.10.0-693.11.6.el7.x86_64 x86_64"
        )
        hdfeos_additional_file_attributes.attrs["SpaceCraftMaxAltitude"] = np.array(
            [np.float64(np.float64(732023.1))], dtype=np.float64
        )
        hdfeos_additional_file_attributes.attrs["SpaceCraftMinAltitude"] = np.array(
            [np.float64(np.float64(704162.3))], dtype=np.float64
        )
        hdfeos_additional_file_attributes.attrs["TAI93At0zOfGranule"] = np.array(
            [np.float64(np.float64(370742405.0))], dtype=np.float64
        )

        # attributes for HDFEOS/SWATHS/OMI Ground Pixel Corners UV-1
        hdfeos_swaths_omi_ground_pixel_corners_uv_1.attrs["EarthSunDistance"] = (
            np.array([np.float32(np.float32(1.4973326e11))], dtype=np.float32)
        )

        # attributes for HDFEOS/SWATHS/OMI Ground Pixel Corners UV-2
        hdfeos_swaths_omi_ground_pixel_corners_uv_2.attrs["EarthSunDistance"] = (
            np.array([np.float32(np.float32(1.4973326e11))], dtype=np.float32)
        )

        # attributes for HDFEOS/SWATHS/OMI Ground Pixel Corners VIS
        hdfeos_swaths_omi_ground_pixel_corners_vis.attrs["EarthSunDistance"] = np.array(
            [np.float32(np.float32(1.4973326e11))], dtype=np.float32
        )

        # attributes for HDFEOS INFORMATION
        hdfeos_information.attrs["HDFEOSVersion"] = "HDFEOS_5.1.15"

        # HDFEOS/SWATHS/OMI Ground Pixel Corners UV-1/Data Fields/FoV75Area
        hdfeos_swaths_omi_ground_pixel_corners_uv_1_data_fields_fov75area = (
            hdfeos_swaths_omi_ground_pixel_corners_uv_1_data_fields.create_dataset(
                "FoV75Area",
                data=np.arange(dim_0_30, dtype=np.float32),
            )
        )
        hdfeos_swaths_omi_ground_pixel_corners_uv_1_data_fields_fov75area.attrs[
            "MissingValue"
        ] = np.array([np.float32(np.float32(-1e30))], dtype=np.float32)
        hdfeos_swaths_omi_ground_pixel_corners_uv_1_data_fields_fov75area.attrs[
            "Offset"
        ] = np.array([np.float64(np.float64(0.0))], dtype=np.float64)
        hdfeos_swaths_omi_ground_pixel_corners_uv_1_data_fields_fov75area.attrs[
            "ScaleFactor"
        ] = np.array([np.float64(np.float64(1.0))], dtype=np.float64)
        hdfeos_swaths_omi_ground_pixel_corners_uv_1_data_fields_fov75area.attrs[
            "Title"
        ] = "Mean Area for 75% Field of View Pixels on the WGS-85 Ellipsoid"
        hdfeos_swaths_omi_ground_pixel_corners_uv_1_data_fields_fov75area.attrs[
            "UniqueFieldDefinition"
        ] = "OMI-Specific"
        hdfeos_swaths_omi_ground_pixel_corners_uv_1_data_fields_fov75area.attrs[
            "Units"
        ] = "km^2"
        hdfeos_swaths_omi_ground_pixel_corners_uv_1_data_fields_fov75area.attrs[
            "ValidRange"
        ] = np.array(
            [np.float32(np.float32(0.0)), np.float32(np.float32(5.10072e08))],
            dtype=np.float32,
        )
        hdfeos_swaths_omi_ground_pixel_corners_uv_1_data_fields_fov75area.attrs[
            "_FillValue"
        ] = np.array([np.float32(np.float32(-1e30))], dtype=np.float32)

        # HDFEOS/SWATHS/OMI Ground Pixel Corners UV-1/Data Fields/FoV75CornerLatitude
        hdfeos_swaths_omi_ground_pixel_corners_uv_1_data_fields_fov75cornerlatitude = (
            hdfeos_swaths_omi_ground_pixel_corners_uv_1_data_fields.create_dataset(
                "FoV75CornerLatitude",
                data=np.arange(
                    dim_0_4 * dim_0_1643 * dim_0_30, dtype=np.float32
                ).reshape(
                    (
                        dim_0_4,
                        dim_0_1643,
                        dim_0_30,
                    )
                ),
            )
        )
        hdfeos_swaths_omi_ground_pixel_corners_uv_1_data_fields_fov75cornerlatitude.attrs[
            "MissingValue"
        ] = np.array([np.float32(np.float32(-1e30))], dtype=np.float32)
        hdfeos_swaths_omi_ground_pixel_corners_uv_1_data_fields_fov75cornerlatitude.attrs[
            "Offset"
        ] = np.array([np.float64(np.float64(0.0))], dtype=np.float64)
        hdfeos_swaths_omi_ground_pixel_corners_uv_1_data_fields_fov75cornerlatitude.attrs[
            "ScaleFactor"
        ] = np.array([np.float64(np.float64(1.0))], dtype=np.float64)
        hdfeos_swaths_omi_ground_pixel_corners_uv_1_data_fields_fov75cornerlatitude.attrs[
            "Title"
        ] = "Corner Latitudes for 75% Field of View Pixels on the WGS-85 Ellipsoid (CCW relative to flight direction: LL,LR,UR,UL)"
        hdfeos_swaths_omi_ground_pixel_corners_uv_1_data_fields_fov75cornerlatitude.attrs[
            "UniqueFieldDefinition"
        ] = "OMI-Specific"
        hdfeos_swaths_omi_ground_pixel_corners_uv_1_data_fields_fov75cornerlatitude.attrs[
            "Units"
        ] = "deg"
        hdfeos_swaths_omi_ground_pixel_corners_uv_1_data_fields_fov75cornerlatitude.attrs[
            "ValidRange"
        ] = np.array(
            [np.float32(np.float32(-90.0)), np.float32(np.float32(90.0))],
            dtype=np.float32,
        )
        hdfeos_swaths_omi_ground_pixel_corners_uv_1_data_fields_fov75cornerlatitude.attrs[
            "_FillValue"
        ] = np.array([np.float32(np.float32(-1e30))], dtype=np.float32)

        # HDFEOS/SWATHS/OMI Ground Pixel Corners UV-1/Data Fields/FoV75CornerLongitude
        hdfeos_swaths_omi_ground_pixel_corners_uv_1_data_fields_fov75cornerlongitude = (
            hdfeos_swaths_omi_ground_pixel_corners_uv_1_data_fields.create_dataset(
                "FoV75CornerLongitude",
                data=np.arange(
                    dim_0_4 * dim_0_1643 * dim_0_30, dtype=np.float32
                ).reshape(
                    (
                        dim_0_4,
                        dim_0_1643,
                        dim_0_30,
                    )
                ),
            )
        )
        hdfeos_swaths_omi_ground_pixel_corners_uv_1_data_fields_fov75cornerlongitude.attrs[
            "MissingValue"
        ] = np.array([np.float32(np.float32(-1e30))], dtype=np.float32)
        hdfeos_swaths_omi_ground_pixel_corners_uv_1_data_fields_fov75cornerlongitude.attrs[
            "Offset"
        ] = np.array([np.float64(np.float64(0.0))], dtype=np.float64)
        hdfeos_swaths_omi_ground_pixel_corners_uv_1_data_fields_fov75cornerlongitude.attrs[
            "ScaleFactor"
        ] = np.array([np.float64(np.float64(1.0))], dtype=np.float64)
        hdfeos_swaths_omi_ground_pixel_corners_uv_1_data_fields_fov75cornerlongitude.attrs[
            "Title"
        ] = "Corner Longitudes for 75% Field of View Pixels on the WGS-85 Ellipsoid (CCW relative to flight direction: LL,LR,UR,UL)"
        hdfeos_swaths_omi_ground_pixel_corners_uv_1_data_fields_fov75cornerlongitude.attrs[
            "UniqueFieldDefinition"
        ] = "OMI-Specific"
        hdfeos_swaths_omi_ground_pixel_corners_uv_1_data_fields_fov75cornerlongitude.attrs[
            "Units"
        ] = "deg"
        hdfeos_swaths_omi_ground_pixel_corners_uv_1_data_fields_fov75cornerlongitude.attrs[
            "ValidRange"
        ] = np.array(
            [np.float32(np.float32(-180.0)), np.float32(np.float32(180.0))],
            dtype=np.float32,
        )
        hdfeos_swaths_omi_ground_pixel_corners_uv_1_data_fields_fov75cornerlongitude.attrs[
            "_FillValue"
        ] = np.array([np.float32(np.float32(-1e30))], dtype=np.float32)

        # HDFEOS/SWATHS/OMI Ground Pixel Corners UV-1/Data Fields/TiledArea
        hdfeos_swaths_omi_ground_pixel_corners_uv_1_data_fields_tiledarea = (
            hdfeos_swaths_omi_ground_pixel_corners_uv_1_data_fields.create_dataset(
                "TiledArea",
                data=np.arange(dim_0_30, dtype=np.float32),
            )
        )
        hdfeos_swaths_omi_ground_pixel_corners_uv_1_data_fields_tiledarea.attrs[
            "MissingValue"
        ] = np.array([np.float32(np.float32(-1e30))], dtype=np.float32)
        hdfeos_swaths_omi_ground_pixel_corners_uv_1_data_fields_tiledarea.attrs[
            "Offset"
        ] = np.array([np.float64(np.float64(0.0))], dtype=np.float64)
        hdfeos_swaths_omi_ground_pixel_corners_uv_1_data_fields_tiledarea.attrs[
            "ScaleFactor"
        ] = np.array([np.float64(np.float64(1.0))], dtype=np.float64)
        hdfeos_swaths_omi_ground_pixel_corners_uv_1_data_fields_tiledarea.attrs[
            "Title"
        ] = "Mean Area for Tiled Pixels on the WGS-85 Ellipsoid"
        hdfeos_swaths_omi_ground_pixel_corners_uv_1_data_fields_tiledarea.attrs[
            "UniqueFieldDefinition"
        ] = "OMI-Specific"
        hdfeos_swaths_omi_ground_pixel_corners_uv_1_data_fields_tiledarea.attrs[
            "Units"
        ] = "km^2"
        hdfeos_swaths_omi_ground_pixel_corners_uv_1_data_fields_tiledarea.attrs[
            "ValidRange"
        ] = np.array(
            [np.float32(np.float32(0.0)), np.float32(np.float32(5.10072e08))],
            dtype=np.float32,
        )
        hdfeos_swaths_omi_ground_pixel_corners_uv_1_data_fields_tiledarea.attrs[
            "_FillValue"
        ] = np.array([np.float32(np.float32(-1e30))], dtype=np.float32)

        # HDFEOS/SWATHS/OMI Ground Pixel Corners UV-1/Data Fields/TiledCornerLatitude
        hdfeos_swaths_omi_ground_pixel_corners_uv_1_data_fields_tiledcornerlatitude = (
            hdfeos_swaths_omi_ground_pixel_corners_uv_1_data_fields.create_dataset(
                "TiledCornerLatitude",
                data=np.arange(
                    dim_0_4 * dim_0_1643 * dim_0_30, dtype=np.float32
                ).reshape(
                    (
                        dim_0_4,
                        dim_0_1643,
                        dim_0_30,
                    )
                ),
            )
        )
        hdfeos_swaths_omi_ground_pixel_corners_uv_1_data_fields_tiledcornerlatitude.attrs[
            "MissingValue"
        ] = np.array([np.float32(np.float32(-1e30))], dtype=np.float32)
        hdfeos_swaths_omi_ground_pixel_corners_uv_1_data_fields_tiledcornerlatitude.attrs[
            "Offset"
        ] = np.array([np.float64(np.float64(0.0))], dtype=np.float64)
        hdfeos_swaths_omi_ground_pixel_corners_uv_1_data_fields_tiledcornerlatitude.attrs[
            "ScaleFactor"
        ] = np.array([np.float64(np.float64(1.0))], dtype=np.float64)
        hdfeos_swaths_omi_ground_pixel_corners_uv_1_data_fields_tiledcornerlatitude.attrs[
            "Title"
        ] = "Corner Latitudes for Tiled Pixels on the WGS-85 Ellipsoid (CCW relative to flight direction: LL,LR,UR,UL)"
        hdfeos_swaths_omi_ground_pixel_corners_uv_1_data_fields_tiledcornerlatitude.attrs[
            "UniqueFieldDefinition"
        ] = "OMI-Specific"
        hdfeos_swaths_omi_ground_pixel_corners_uv_1_data_fields_tiledcornerlatitude.attrs[
            "Units"
        ] = "deg"
        hdfeos_swaths_omi_ground_pixel_corners_uv_1_data_fields_tiledcornerlatitude.attrs[
            "ValidRange"
        ] = np.array(
            [np.float32(np.float32(-90.0)), np.float32(np.float32(90.0))],
            dtype=np.float32,
        )
        hdfeos_swaths_omi_ground_pixel_corners_uv_1_data_fields_tiledcornerlatitude.attrs[
            "_FillValue"
        ] = np.array([np.float32(np.float32(-1e30))], dtype=np.float32)

        # HDFEOS/SWATHS/OMI Ground Pixel Corners UV-1/Data Fields/TiledCornerLongitude
        hdfeos_swaths_omi_ground_pixel_corners_uv_1_data_fields_tiledcornerlongitude = (
            hdfeos_swaths_omi_ground_pixel_corners_uv_1_data_fields.create_dataset(
                "TiledCornerLongitude",
                data=np.arange(
                    dim_0_4 * dim_0_1643 * dim_0_30, dtype=np.float32
                ).reshape(
                    (
                        dim_0_4,
                        dim_0_1643,
                        dim_0_30,
                    )
                ),
            )
        )
        hdfeos_swaths_omi_ground_pixel_corners_uv_1_data_fields_tiledcornerlongitude.attrs[
            "MissingValue"
        ] = np.array([np.float32(np.float32(-1e30))], dtype=np.float32)
        hdfeos_swaths_omi_ground_pixel_corners_uv_1_data_fields_tiledcornerlongitude.attrs[
            "Offset"
        ] = np.array([np.float64(np.float64(0.0))], dtype=np.float64)
        hdfeos_swaths_omi_ground_pixel_corners_uv_1_data_fields_tiledcornerlongitude.attrs[
            "ScaleFactor"
        ] = np.array([np.float64(np.float64(1.0))], dtype=np.float64)
        hdfeos_swaths_omi_ground_pixel_corners_uv_1_data_fields_tiledcornerlongitude.attrs[
            "Title"
        ] = "Corner Longitudes for Tiled Pixels on the WGS-85 Ellipsoid (CCW relative to flight direction: LL,LR,UR,UL)"
        hdfeos_swaths_omi_ground_pixel_corners_uv_1_data_fields_tiledcornerlongitude.attrs[
            "UniqueFieldDefinition"
        ] = "OMI-Specific"
        hdfeos_swaths_omi_ground_pixel_corners_uv_1_data_fields_tiledcornerlongitude.attrs[
            "Units"
        ] = "deg"
        hdfeos_swaths_omi_ground_pixel_corners_uv_1_data_fields_tiledcornerlongitude.attrs[
            "ValidRange"
        ] = np.array(
            [np.float32(np.float32(-180.0)), np.float32(np.float32(180.0))],
            dtype=np.float32,
        )
        hdfeos_swaths_omi_ground_pixel_corners_uv_1_data_fields_tiledcornerlongitude.attrs[
            "_FillValue"
        ] = np.array([np.float32(np.float32(-1e30))], dtype=np.float32)

        # HDFEOS/SWATHS/OMI Ground Pixel Corners UV-1/Geolocation Fields/Latitude
        hdfeos_swaths_omi_ground_pixel_corners_uv_1_geolocation_fields_latitude = hdfeos_swaths_omi_ground_pixel_corners_uv_1_geolocation_fields.create_dataset(
            "Latitude",
            data=np.linspace(-90.0, 90.0, dim_0_1643 * dim_0_30)
            .reshape(
                (
                    dim_0_1643,
                    dim_0_30,
                )
            )
            .astype(np.float32),
        )
        hdfeos_swaths_omi_ground_pixel_corners_uv_1_geolocation_fields_latitude.attrs[
            "MissingValue"
        ] = np.array([np.float32(np.float32(-1e30))], dtype=np.float32)
        hdfeos_swaths_omi_ground_pixel_corners_uv_1_geolocation_fields_latitude.attrs[
            "Offset"
        ] = np.array([np.float64(np.float64(0.0))], dtype=np.float64)
        hdfeos_swaths_omi_ground_pixel_corners_uv_1_geolocation_fields_latitude.attrs[
            "ScaleFactor"
        ] = np.array([np.float64(np.float64(1.0))], dtype=np.float64)
        hdfeos_swaths_omi_ground_pixel_corners_uv_1_geolocation_fields_latitude.attrs[
            "Title"
        ] = "Geodetic Latitude"
        hdfeos_swaths_omi_ground_pixel_corners_uv_1_geolocation_fields_latitude.attrs[
            "UniqueFieldDefinition"
        ] = "Aura-Shared"
        hdfeos_swaths_omi_ground_pixel_corners_uv_1_geolocation_fields_latitude.attrs[
            "Units"
        ] = "deg"
        hdfeos_swaths_omi_ground_pixel_corners_uv_1_geolocation_fields_latitude.attrs[
            "ValidRange"
        ] = np.array(
            [np.float32(np.float32(-90.0)), np.float32(np.float32(90.0))],
            dtype=np.float32,
        )
        hdfeos_swaths_omi_ground_pixel_corners_uv_1_geolocation_fields_latitude.attrs[
            "_FillValue"
        ] = np.array([np.float32(np.float32(-1e30))], dtype=np.float32)

        # HDFEOS/SWATHS/OMI Ground Pixel Corners UV-1/Geolocation Fields/Longitude
        hdfeos_swaths_omi_ground_pixel_corners_uv_1_geolocation_fields_longitude = hdfeos_swaths_omi_ground_pixel_corners_uv_1_geolocation_fields.create_dataset(
            "Longitude",
            data=np.linspace(-180.0, 180.0, dim_0_1643 * dim_0_30)
            .reshape(
                (
                    dim_0_1643,
                    dim_0_30,
                )
            )
            .astype(np.float32),
        )
        hdfeos_swaths_omi_ground_pixel_corners_uv_1_geolocation_fields_longitude.attrs[
            "MissingValue"
        ] = np.array([np.float32(np.float32(-1e30))], dtype=np.float32)
        hdfeos_swaths_omi_ground_pixel_corners_uv_1_geolocation_fields_longitude.attrs[
            "Offset"
        ] = np.array([np.float64(np.float64(0.0))], dtype=np.float64)
        hdfeos_swaths_omi_ground_pixel_corners_uv_1_geolocation_fields_longitude.attrs[
            "ScaleFactor"
        ] = np.array([np.float64(np.float64(1.0))], dtype=np.float64)
        hdfeos_swaths_omi_ground_pixel_corners_uv_1_geolocation_fields_longitude.attrs[
            "Title"
        ] = "Geodetic Longitude"
        hdfeos_swaths_omi_ground_pixel_corners_uv_1_geolocation_fields_longitude.attrs[
            "UniqueFieldDefinition"
        ] = "Aura-Shared"
        hdfeos_swaths_omi_ground_pixel_corners_uv_1_geolocation_fields_longitude.attrs[
            "Units"
        ] = "deg"
        hdfeos_swaths_omi_ground_pixel_corners_uv_1_geolocation_fields_longitude.attrs[
            "ValidRange"
        ] = np.array(
            [np.float32(np.float32(-180.0)), np.float32(np.float32(180.0))],
            dtype=np.float32,
        )
        hdfeos_swaths_omi_ground_pixel_corners_uv_1_geolocation_fields_longitude.attrs[
            "_FillValue"
        ] = np.array([np.float32(np.float32(-1e30))], dtype=np.float32)

        # HDFEOS/SWATHS/OMI Ground Pixel Corners UV-1/Geolocation Fields/SpacecraftAltitude
        hdfeos_swaths_omi_ground_pixel_corners_uv_1_geolocation_fields_spacecraftaltitude = hdfeos_swaths_omi_ground_pixel_corners_uv_1_geolocation_fields.create_dataset(
            "SpacecraftAltitude",
            data=np.arange(dim_0_1643, dtype=np.float32),
        )
        hdfeos_swaths_omi_ground_pixel_corners_uv_1_geolocation_fields_spacecraftaltitude.attrs[
            "MissingValue"
        ] = np.array([np.float32(np.float32(-1e30))], dtype=np.float32)
        hdfeos_swaths_omi_ground_pixel_corners_uv_1_geolocation_fields_spacecraftaltitude.attrs[
            "Offset"
        ] = np.array([np.float64(np.float64(0.0))], dtype=np.float64)
        hdfeos_swaths_omi_ground_pixel_corners_uv_1_geolocation_fields_spacecraftaltitude.attrs[
            "ScaleFactor"
        ] = np.array([np.float64(np.float64(1.0))], dtype=np.float64)
        hdfeos_swaths_omi_ground_pixel_corners_uv_1_geolocation_fields_spacecraftaltitude.attrs[
            "Title"
        ] = "Altitude of EOS-Aura Spacecraft"
        hdfeos_swaths_omi_ground_pixel_corners_uv_1_geolocation_fields_spacecraftaltitude.attrs[
            "UniqueFieldDefinition"
        ] = "Aura-Shared"
        hdfeos_swaths_omi_ground_pixel_corners_uv_1_geolocation_fields_spacecraftaltitude.attrs[
            "Units"
        ] = "m"
        hdfeos_swaths_omi_ground_pixel_corners_uv_1_geolocation_fields_spacecraftaltitude.attrs[
            "ValidRange"
        ] = np.array(
            [np.float32(np.float32(0.0)), np.float32(np.float32(1e30))],
            dtype=np.float32,
        )
        hdfeos_swaths_omi_ground_pixel_corners_uv_1_geolocation_fields_spacecraftaltitude.attrs[
            "_FillValue"
        ] = np.array([np.float32(np.float32(-1e30))], dtype=np.float32)

        # HDFEOS/SWATHS/OMI Ground Pixel Corners UV-1/Geolocation Fields/SpacecraftLatitude
        hdfeos_swaths_omi_ground_pixel_corners_uv_1_geolocation_fields_spacecraftlatitude = hdfeos_swaths_omi_ground_pixel_corners_uv_1_geolocation_fields.create_dataset(
            "SpacecraftLatitude",
            data=np.arange(dim_0_1643, dtype=np.float32),
        )
        hdfeos_swaths_omi_ground_pixel_corners_uv_1_geolocation_fields_spacecraftlatitude.attrs[
            "MissingValue"
        ] = np.array([np.float32(np.float32(-1e30))], dtype=np.float32)
        hdfeos_swaths_omi_ground_pixel_corners_uv_1_geolocation_fields_spacecraftlatitude.attrs[
            "Offset"
        ] = np.array([np.float64(np.float64(0.0))], dtype=np.float64)
        hdfeos_swaths_omi_ground_pixel_corners_uv_1_geolocation_fields_spacecraftlatitude.attrs[
            "ScaleFactor"
        ] = np.array([np.float64(np.float64(1.0))], dtype=np.float64)
        hdfeos_swaths_omi_ground_pixel_corners_uv_1_geolocation_fields_spacecraftlatitude.attrs[
            "Title"
        ] = "Latitude of EOS-Aura Spacecraft"
        hdfeos_swaths_omi_ground_pixel_corners_uv_1_geolocation_fields_spacecraftlatitude.attrs[
            "UniqueFieldDefinition"
        ] = "Aura-Shared"
        hdfeos_swaths_omi_ground_pixel_corners_uv_1_geolocation_fields_spacecraftlatitude.attrs[
            "Units"
        ] = "deg"
        hdfeos_swaths_omi_ground_pixel_corners_uv_1_geolocation_fields_spacecraftlatitude.attrs[
            "ValidRange"
        ] = np.array(
            [np.float32(np.float32(-90.0)), np.float32(np.float32(90.0))],
            dtype=np.float32,
        )
        hdfeos_swaths_omi_ground_pixel_corners_uv_1_geolocation_fields_spacecraftlatitude.attrs[
            "_FillValue"
        ] = np.array([np.float32(np.float32(-1e30))], dtype=np.float32)

        # HDFEOS/SWATHS/OMI Ground Pixel Corners UV-1/Geolocation Fields/SpacecraftLongitude
        hdfeos_swaths_omi_ground_pixel_corners_uv_1_geolocation_fields_spacecraftlongitude = hdfeos_swaths_omi_ground_pixel_corners_uv_1_geolocation_fields.create_dataset(
            "SpacecraftLongitude",
            data=np.arange(dim_0_1643, dtype=np.float32),
        )
        hdfeos_swaths_omi_ground_pixel_corners_uv_1_geolocation_fields_spacecraftlongitude.attrs[
            "MissingValue"
        ] = np.array([np.float32(np.float32(-1e30))], dtype=np.float32)
        hdfeos_swaths_omi_ground_pixel_corners_uv_1_geolocation_fields_spacecraftlongitude.attrs[
            "Offset"
        ] = np.array([np.float64(np.float64(0.0))], dtype=np.float64)
        hdfeos_swaths_omi_ground_pixel_corners_uv_1_geolocation_fields_spacecraftlongitude.attrs[
            "ScaleFactor"
        ] = np.array([np.float64(np.float64(1.0))], dtype=np.float64)
        hdfeos_swaths_omi_ground_pixel_corners_uv_1_geolocation_fields_spacecraftlongitude.attrs[
            "Title"
        ] = "Longitude of EOS-Aura Spacecraft"
        hdfeos_swaths_omi_ground_pixel_corners_uv_1_geolocation_fields_spacecraftlongitude.attrs[
            "UniqueFieldDefinition"
        ] = "Aura-Shared"
        hdfeos_swaths_omi_ground_pixel_corners_uv_1_geolocation_fields_spacecraftlongitude.attrs[
            "Units"
        ] = "deg"
        hdfeos_swaths_omi_ground_pixel_corners_uv_1_geolocation_fields_spacecraftlongitude.attrs[
            "ValidRange"
        ] = np.array(
            [np.float32(np.float32(-180.0)), np.float32(np.float32(180.0))],
            dtype=np.float32,
        )
        hdfeos_swaths_omi_ground_pixel_corners_uv_1_geolocation_fields_spacecraftlongitude.attrs[
            "_FillValue"
        ] = np.array([np.float32(np.float32(-1e30))], dtype=np.float32)

        # HDFEOS/SWATHS/OMI Ground Pixel Corners UV-1/Geolocation Fields/Time
        hdfeos_swaths_omi_ground_pixel_corners_uv_1_geolocation_fields_time = hdfeos_swaths_omi_ground_pixel_corners_uv_1_geolocation_fields.create_dataset(
            "Time",
            data=np.linspace(0.0, 1000000000.0, dim_0_1643).astype(np.float64),
        )
        hdfeos_swaths_omi_ground_pixel_corners_uv_1_geolocation_fields_time.attrs[
            "MissingValue"
        ] = np.array([np.float64(np.float64(-1.0000000150474662e30))], dtype=np.float64)
        hdfeos_swaths_omi_ground_pixel_corners_uv_1_geolocation_fields_time.attrs[
            "Offset"
        ] = np.array([np.float64(np.float64(0.0))], dtype=np.float64)
        hdfeos_swaths_omi_ground_pixel_corners_uv_1_geolocation_fields_time.attrs[
            "ScaleFactor"
        ] = np.array([np.float64(np.float64(1.0))], dtype=np.float64)
        hdfeos_swaths_omi_ground_pixel_corners_uv_1_geolocation_fields_time.attrs[
            "Title"
        ] = "Time at Start of Swath Line (s, TAI93)"
        hdfeos_swaths_omi_ground_pixel_corners_uv_1_geolocation_fields_time.attrs[
            "UniqueFieldDefinition"
        ] = "Aura-Shared"
        hdfeos_swaths_omi_ground_pixel_corners_uv_1_geolocation_fields_time.attrs[
            "Units"
        ] = "s"
        hdfeos_swaths_omi_ground_pixel_corners_uv_1_geolocation_fields_time.attrs[
            "ValidRange"
        ] = np.array(
            [np.float64(np.float64(0.0)), np.float64(np.float64(10000000000.0))],
            dtype=np.float64,
        )
        hdfeos_swaths_omi_ground_pixel_corners_uv_1_geolocation_fields_time.attrs[
            "_FillValue"
        ] = np.array([np.float64(np.float64(-1e30))], dtype=np.float64)

        # HDFEOS/SWATHS/OMI Ground Pixel Corners UV-1/Geolocation Fields/TimeUTC
        hdfeos_swaths_omi_ground_pixel_corners_uv_1_geolocation_fields_timeutc = hdfeos_swaths_omi_ground_pixel_corners_uv_1_geolocation_fields.create_dataset(
            "TimeUTC",
            data=np.full(
                (
                    dim_0_1643,
                    dim_1_27,
                ),
                b"",
                dtype=np.dtype("|S1"),
            ),
        )
        hdfeos_swaths_omi_ground_pixel_corners_uv_1_geolocation_fields_timeutc.attrs[
            "MissingValue"
        ] = "undefined"
        hdfeos_swaths_omi_ground_pixel_corners_uv_1_geolocation_fields_timeutc.attrs[
            "Offset"
        ] = np.array([np.float64(np.float64(0.0))], dtype=np.float64)
        hdfeos_swaths_omi_ground_pixel_corners_uv_1_geolocation_fields_timeutc.attrs[
            "ScaleFactor"
        ] = np.array([np.float64(np.float64(1.0))], dtype=np.float64)
        hdfeos_swaths_omi_ground_pixel_corners_uv_1_geolocation_fields_timeutc.attrs[
            "Title"
        ] = "Coordinated Universal Time"
        hdfeos_swaths_omi_ground_pixel_corners_uv_1_geolocation_fields_timeutc.attrs[
            "UniqueFieldDefinition"
        ] = "OMI-Specific"
        hdfeos_swaths_omi_ground_pixel_corners_uv_1_geolocation_fields_timeutc.attrs[
            "Units"
        ] = "NoUnits"

        # HDFEOS/SWATHS/OMI Ground Pixel Corners UV-2/Data Fields/FoV75Area
        hdfeos_swaths_omi_ground_pixel_corners_uv_2_data_fields_fov75area = (
            hdfeos_swaths_omi_ground_pixel_corners_uv_2_data_fields.create_dataset(
                "FoV75Area",
                data=np.arange(dim_0_60, dtype=np.float32),
            )
        )
        hdfeos_swaths_omi_ground_pixel_corners_uv_2_data_fields_fov75area.attrs[
            "MissingValue"
        ] = np.array([np.float32(np.float32(-1e30))], dtype=np.float32)
        hdfeos_swaths_omi_ground_pixel_corners_uv_2_data_fields_fov75area.attrs[
            "Offset"
        ] = np.array([np.float64(np.float64(0.0))], dtype=np.float64)
        hdfeos_swaths_omi_ground_pixel_corners_uv_2_data_fields_fov75area.attrs[
            "ScaleFactor"
        ] = np.array([np.float64(np.float64(1.0))], dtype=np.float64)
        hdfeos_swaths_omi_ground_pixel_corners_uv_2_data_fields_fov75area.attrs[
            "Title"
        ] = "Mean Area for 75% Field of View Pixels on the WGS-85 Ellipsoid"
        hdfeos_swaths_omi_ground_pixel_corners_uv_2_data_fields_fov75area.attrs[
            "UniqueFieldDefinition"
        ] = "OMI-Specific"
        hdfeos_swaths_omi_ground_pixel_corners_uv_2_data_fields_fov75area.attrs[
            "Units"
        ] = "km^2"
        hdfeos_swaths_omi_ground_pixel_corners_uv_2_data_fields_fov75area.attrs[
            "ValidRange"
        ] = np.array(
            [np.float32(np.float32(0.0)), np.float32(np.float32(5.10072e08))],
            dtype=np.float32,
        )
        hdfeos_swaths_omi_ground_pixel_corners_uv_2_data_fields_fov75area.attrs[
            "_FillValue"
        ] = np.array([np.float32(np.float32(-1e30))], dtype=np.float32)

        # HDFEOS/SWATHS/OMI Ground Pixel Corners UV-2/Data Fields/FoV75CornerLatitude
        hdfeos_swaths_omi_ground_pixel_corners_uv_2_data_fields_fov75cornerlatitude = (
            hdfeos_swaths_omi_ground_pixel_corners_uv_2_data_fields.create_dataset(
                "FoV75CornerLatitude",
                data=np.arange(
                    dim_0_4 * dim_0_1643 * dim_0_60, dtype=np.float32
                ).reshape(
                    (
                        dim_0_4,
                        dim_0_1643,
                        dim_0_60,
                    )
                ),
            )
        )
        hdfeos_swaths_omi_ground_pixel_corners_uv_2_data_fields_fov75cornerlatitude.attrs[
            "MissingValue"
        ] = np.array([np.float32(np.float32(-1e30))], dtype=np.float32)
        hdfeos_swaths_omi_ground_pixel_corners_uv_2_data_fields_fov75cornerlatitude.attrs[
            "Offset"
        ] = np.array([np.float64(np.float64(0.0))], dtype=np.float64)
        hdfeos_swaths_omi_ground_pixel_corners_uv_2_data_fields_fov75cornerlatitude.attrs[
            "ScaleFactor"
        ] = np.array([np.float64(np.float64(1.0))], dtype=np.float64)
        hdfeos_swaths_omi_ground_pixel_corners_uv_2_data_fields_fov75cornerlatitude.attrs[
            "Title"
        ] = "Corner Latitudes for 75% Field of View Pixels on the WGS-85 Ellipsoid (CCW relative to flight direction: LL,LR,UR,UL)"
        hdfeos_swaths_omi_ground_pixel_corners_uv_2_data_fields_fov75cornerlatitude.attrs[
            "UniqueFieldDefinition"
        ] = "OMI-Specific"
        hdfeos_swaths_omi_ground_pixel_corners_uv_2_data_fields_fov75cornerlatitude.attrs[
            "Units"
        ] = "deg"
        hdfeos_swaths_omi_ground_pixel_corners_uv_2_data_fields_fov75cornerlatitude.attrs[
            "ValidRange"
        ] = np.array(
            [np.float32(np.float32(-90.0)), np.float32(np.float32(90.0))],
            dtype=np.float32,
        )
        hdfeos_swaths_omi_ground_pixel_corners_uv_2_data_fields_fov75cornerlatitude.attrs[
            "_FillValue"
        ] = np.array([np.float32(np.float32(-1e30))], dtype=np.float32)

        # HDFEOS/SWATHS/OMI Ground Pixel Corners UV-2/Data Fields/FoV75CornerLongitude
        hdfeos_swaths_omi_ground_pixel_corners_uv_2_data_fields_fov75cornerlongitude = (
            hdfeos_swaths_omi_ground_pixel_corners_uv_2_data_fields.create_dataset(
                "FoV75CornerLongitude",
                data=np.arange(
                    dim_0_4 * dim_0_1643 * dim_0_60, dtype=np.float32
                ).reshape(
                    (
                        dim_0_4,
                        dim_0_1643,
                        dim_0_60,
                    )
                ),
            )
        )
        hdfeos_swaths_omi_ground_pixel_corners_uv_2_data_fields_fov75cornerlongitude.attrs[
            "MissingValue"
        ] = np.array([np.float32(np.float32(-1e30))], dtype=np.float32)
        hdfeos_swaths_omi_ground_pixel_corners_uv_2_data_fields_fov75cornerlongitude.attrs[
            "Offset"
        ] = np.array([np.float64(np.float64(0.0))], dtype=np.float64)
        hdfeos_swaths_omi_ground_pixel_corners_uv_2_data_fields_fov75cornerlongitude.attrs[
            "ScaleFactor"
        ] = np.array([np.float64(np.float64(1.0))], dtype=np.float64)
        hdfeos_swaths_omi_ground_pixel_corners_uv_2_data_fields_fov75cornerlongitude.attrs[
            "Title"
        ] = "Corner Longitudes for 75% Field of View Pixels on the WGS-85 Ellipsoid (CCW relative to flight direction: LL,LR,UR,UL)"
        hdfeos_swaths_omi_ground_pixel_corners_uv_2_data_fields_fov75cornerlongitude.attrs[
            "UniqueFieldDefinition"
        ] = "OMI-Specific"
        hdfeos_swaths_omi_ground_pixel_corners_uv_2_data_fields_fov75cornerlongitude.attrs[
            "Units"
        ] = "deg"
        hdfeos_swaths_omi_ground_pixel_corners_uv_2_data_fields_fov75cornerlongitude.attrs[
            "ValidRange"
        ] = np.array(
            [np.float32(np.float32(-180.0)), np.float32(np.float32(180.0))],
            dtype=np.float32,
        )
        hdfeos_swaths_omi_ground_pixel_corners_uv_2_data_fields_fov75cornerlongitude.attrs[
            "_FillValue"
        ] = np.array([np.float32(np.float32(-1e30))], dtype=np.float32)

        # HDFEOS/SWATHS/OMI Ground Pixel Corners UV-2/Data Fields/TiledArea
        hdfeos_swaths_omi_ground_pixel_corners_uv_2_data_fields_tiledarea = (
            hdfeos_swaths_omi_ground_pixel_corners_uv_2_data_fields.create_dataset(
                "TiledArea",
                data=np.arange(dim_0_60, dtype=np.float32),
            )
        )
        hdfeos_swaths_omi_ground_pixel_corners_uv_2_data_fields_tiledarea.attrs[
            "MissingValue"
        ] = np.array([np.float32(np.float32(-1e30))], dtype=np.float32)
        hdfeos_swaths_omi_ground_pixel_corners_uv_2_data_fields_tiledarea.attrs[
            "Offset"
        ] = np.array([np.float64(np.float64(0.0))], dtype=np.float64)
        hdfeos_swaths_omi_ground_pixel_corners_uv_2_data_fields_tiledarea.attrs[
            "ScaleFactor"
        ] = np.array([np.float64(np.float64(1.0))], dtype=np.float64)
        hdfeos_swaths_omi_ground_pixel_corners_uv_2_data_fields_tiledarea.attrs[
            "Title"
        ] = "Mean Area for Tiled Pixels on the WGS-85 Ellipsoid"
        hdfeos_swaths_omi_ground_pixel_corners_uv_2_data_fields_tiledarea.attrs[
            "UniqueFieldDefinition"
        ] = "OMI-Specific"
        hdfeos_swaths_omi_ground_pixel_corners_uv_2_data_fields_tiledarea.attrs[
            "Units"
        ] = "km^2"
        hdfeos_swaths_omi_ground_pixel_corners_uv_2_data_fields_tiledarea.attrs[
            "ValidRange"
        ] = np.array(
            [np.float32(np.float32(0.0)), np.float32(np.float32(5.10072e08))],
            dtype=np.float32,
        )
        hdfeos_swaths_omi_ground_pixel_corners_uv_2_data_fields_tiledarea.attrs[
            "_FillValue"
        ] = np.array([np.float32(np.float32(-1e30))], dtype=np.float32)

        # HDFEOS/SWATHS/OMI Ground Pixel Corners UV-2/Data Fields/TiledCornerLatitude
        hdfeos_swaths_omi_ground_pixel_corners_uv_2_data_fields_tiledcornerlatitude = (
            hdfeos_swaths_omi_ground_pixel_corners_uv_2_data_fields.create_dataset(
                "TiledCornerLatitude",
                data=np.arange(
                    dim_0_4 * dim_0_1643 * dim_0_60, dtype=np.float32
                ).reshape(
                    (
                        dim_0_4,
                        dim_0_1643,
                        dim_0_60,
                    )
                ),
            )
        )
        hdfeos_swaths_omi_ground_pixel_corners_uv_2_data_fields_tiledcornerlatitude.attrs[
            "MissingValue"
        ] = np.array([np.float32(np.float32(-1e30))], dtype=np.float32)
        hdfeos_swaths_omi_ground_pixel_corners_uv_2_data_fields_tiledcornerlatitude.attrs[
            "Offset"
        ] = np.array([np.float64(np.float64(0.0))], dtype=np.float64)
        hdfeos_swaths_omi_ground_pixel_corners_uv_2_data_fields_tiledcornerlatitude.attrs[
            "ScaleFactor"
        ] = np.array([np.float64(np.float64(1.0))], dtype=np.float64)
        hdfeos_swaths_omi_ground_pixel_corners_uv_2_data_fields_tiledcornerlatitude.attrs[
            "Title"
        ] = "Corner Latitudes for Tiled Pixels on the WGS-85 Ellipsoid (CCW relative to flight direction: LL,LR,UR,UL)"
        hdfeos_swaths_omi_ground_pixel_corners_uv_2_data_fields_tiledcornerlatitude.attrs[
            "UniqueFieldDefinition"
        ] = "OMI-Specific"
        hdfeos_swaths_omi_ground_pixel_corners_uv_2_data_fields_tiledcornerlatitude.attrs[
            "Units"
        ] = "deg"
        hdfeos_swaths_omi_ground_pixel_corners_uv_2_data_fields_tiledcornerlatitude.attrs[
            "ValidRange"
        ] = np.array(
            [np.float32(np.float32(-90.0)), np.float32(np.float32(90.0))],
            dtype=np.float32,
        )
        hdfeos_swaths_omi_ground_pixel_corners_uv_2_data_fields_tiledcornerlatitude.attrs[
            "_FillValue"
        ] = np.array([np.float32(np.float32(-1e30))], dtype=np.float32)

        # HDFEOS/SWATHS/OMI Ground Pixel Corners UV-2/Data Fields/TiledCornerLongitude
        hdfeos_swaths_omi_ground_pixel_corners_uv_2_data_fields_tiledcornerlongitude = (
            hdfeos_swaths_omi_ground_pixel_corners_uv_2_data_fields.create_dataset(
                "TiledCornerLongitude",
                data=np.arange(
                    dim_0_4 * dim_0_1643 * dim_0_60, dtype=np.float32
                ).reshape(
                    (
                        dim_0_4,
                        dim_0_1643,
                        dim_0_60,
                    )
                ),
            )
        )
        hdfeos_swaths_omi_ground_pixel_corners_uv_2_data_fields_tiledcornerlongitude.attrs[
            "MissingValue"
        ] = np.array([np.float32(np.float32(-1e30))], dtype=np.float32)
        hdfeos_swaths_omi_ground_pixel_corners_uv_2_data_fields_tiledcornerlongitude.attrs[
            "Offset"
        ] = np.array([np.float64(np.float64(0.0))], dtype=np.float64)
        hdfeos_swaths_omi_ground_pixel_corners_uv_2_data_fields_tiledcornerlongitude.attrs[
            "ScaleFactor"
        ] = np.array([np.float64(np.float64(1.0))], dtype=np.float64)
        hdfeos_swaths_omi_ground_pixel_corners_uv_2_data_fields_tiledcornerlongitude.attrs[
            "Title"
        ] = "Corner Longitudes for Tiled Pixels on the WGS-85 Ellipsoid (CCW relative to flight direction: LL,LR,UR,UL)"
        hdfeos_swaths_omi_ground_pixel_corners_uv_2_data_fields_tiledcornerlongitude.attrs[
            "UniqueFieldDefinition"
        ] = "OMI-Specific"
        hdfeos_swaths_omi_ground_pixel_corners_uv_2_data_fields_tiledcornerlongitude.attrs[
            "Units"
        ] = "deg"
        hdfeos_swaths_omi_ground_pixel_corners_uv_2_data_fields_tiledcornerlongitude.attrs[
            "ValidRange"
        ] = np.array(
            [np.float32(np.float32(-180.0)), np.float32(np.float32(180.0))],
            dtype=np.float32,
        )
        hdfeos_swaths_omi_ground_pixel_corners_uv_2_data_fields_tiledcornerlongitude.attrs[
            "_FillValue"
        ] = np.array([np.float32(np.float32(-1e30))], dtype=np.float32)

        # HDFEOS/SWATHS/OMI Ground Pixel Corners UV-2/Geolocation Fields/Latitude
        hdfeos_swaths_omi_ground_pixel_corners_uv_2_geolocation_fields_latitude = hdfeos_swaths_omi_ground_pixel_corners_uv_2_geolocation_fields.create_dataset(
            "Latitude",
            data=np.linspace(-90.0, 90.0, dim_0_1643 * dim_0_60)
            .reshape(
                (
                    dim_0_1643,
                    dim_0_60,
                )
            )
            .astype(np.float32),
        )
        hdfeos_swaths_omi_ground_pixel_corners_uv_2_geolocation_fields_latitude.attrs[
            "MissingValue"
        ] = np.array([np.float32(np.float32(-1e30))], dtype=np.float32)
        hdfeos_swaths_omi_ground_pixel_corners_uv_2_geolocation_fields_latitude.attrs[
            "Offset"
        ] = np.array([np.float64(np.float64(0.0))], dtype=np.float64)
        hdfeos_swaths_omi_ground_pixel_corners_uv_2_geolocation_fields_latitude.attrs[
            "ScaleFactor"
        ] = np.array([np.float64(np.float64(1.0))], dtype=np.float64)
        hdfeos_swaths_omi_ground_pixel_corners_uv_2_geolocation_fields_latitude.attrs[
            "Title"
        ] = "Geodetic Latitude"
        hdfeos_swaths_omi_ground_pixel_corners_uv_2_geolocation_fields_latitude.attrs[
            "UniqueFieldDefinition"
        ] = "Aura-Shared"
        hdfeos_swaths_omi_ground_pixel_corners_uv_2_geolocation_fields_latitude.attrs[
            "Units"
        ] = "deg"
        hdfeos_swaths_omi_ground_pixel_corners_uv_2_geolocation_fields_latitude.attrs[
            "ValidRange"
        ] = np.array(
            [np.float32(np.float32(-90.0)), np.float32(np.float32(90.0))],
            dtype=np.float32,
        )
        hdfeos_swaths_omi_ground_pixel_corners_uv_2_geolocation_fields_latitude.attrs[
            "_FillValue"
        ] = np.array([np.float32(np.float32(-1e30))], dtype=np.float32)

        # HDFEOS/SWATHS/OMI Ground Pixel Corners UV-2/Geolocation Fields/Longitude
        hdfeos_swaths_omi_ground_pixel_corners_uv_2_geolocation_fields_longitude = hdfeos_swaths_omi_ground_pixel_corners_uv_2_geolocation_fields.create_dataset(
            "Longitude",
            data=np.linspace(-180.0, 180.0, dim_0_1643 * dim_0_60)
            .reshape(
                (
                    dim_0_1643,
                    dim_0_60,
                )
            )
            .astype(np.float32),
        )
        hdfeos_swaths_omi_ground_pixel_corners_uv_2_geolocation_fields_longitude.attrs[
            "MissingValue"
        ] = np.array([np.float32(np.float32(-1e30))], dtype=np.float32)
        hdfeos_swaths_omi_ground_pixel_corners_uv_2_geolocation_fields_longitude.attrs[
            "Offset"
        ] = np.array([np.float64(np.float64(0.0))], dtype=np.float64)
        hdfeos_swaths_omi_ground_pixel_corners_uv_2_geolocation_fields_longitude.attrs[
            "ScaleFactor"
        ] = np.array([np.float64(np.float64(1.0))], dtype=np.float64)
        hdfeos_swaths_omi_ground_pixel_corners_uv_2_geolocation_fields_longitude.attrs[
            "Title"
        ] = "Geodetic Longitude"
        hdfeos_swaths_omi_ground_pixel_corners_uv_2_geolocation_fields_longitude.attrs[
            "UniqueFieldDefinition"
        ] = "Aura-Shared"
        hdfeos_swaths_omi_ground_pixel_corners_uv_2_geolocation_fields_longitude.attrs[
            "Units"
        ] = "deg"
        hdfeos_swaths_omi_ground_pixel_corners_uv_2_geolocation_fields_longitude.attrs[
            "ValidRange"
        ] = np.array(
            [np.float32(np.float32(-180.0)), np.float32(np.float32(180.0))],
            dtype=np.float32,
        )
        hdfeos_swaths_omi_ground_pixel_corners_uv_2_geolocation_fields_longitude.attrs[
            "_FillValue"
        ] = np.array([np.float32(np.float32(-1e30))], dtype=np.float32)

        # HDFEOS/SWATHS/OMI Ground Pixel Corners UV-2/Geolocation Fields/SpacecraftAltitude
        hdfeos_swaths_omi_ground_pixel_corners_uv_2_geolocation_fields_spacecraftaltitude = hdfeos_swaths_omi_ground_pixel_corners_uv_2_geolocation_fields.create_dataset(
            "SpacecraftAltitude",
            data=np.arange(dim_0_1643, dtype=np.float32),
        )
        hdfeos_swaths_omi_ground_pixel_corners_uv_2_geolocation_fields_spacecraftaltitude.attrs[
            "MissingValue"
        ] = np.array([np.float32(np.float32(-1e30))], dtype=np.float32)
        hdfeos_swaths_omi_ground_pixel_corners_uv_2_geolocation_fields_spacecraftaltitude.attrs[
            "Offset"
        ] = np.array([np.float64(np.float64(0.0))], dtype=np.float64)
        hdfeos_swaths_omi_ground_pixel_corners_uv_2_geolocation_fields_spacecraftaltitude.attrs[
            "ScaleFactor"
        ] = np.array([np.float64(np.float64(1.0))], dtype=np.float64)
        hdfeos_swaths_omi_ground_pixel_corners_uv_2_geolocation_fields_spacecraftaltitude.attrs[
            "Title"
        ] = "Altitude of EOS-Aura Spacecraft"
        hdfeos_swaths_omi_ground_pixel_corners_uv_2_geolocation_fields_spacecraftaltitude.attrs[
            "UniqueFieldDefinition"
        ] = "Aura-Shared"
        hdfeos_swaths_omi_ground_pixel_corners_uv_2_geolocation_fields_spacecraftaltitude.attrs[
            "Units"
        ] = "m"
        hdfeos_swaths_omi_ground_pixel_corners_uv_2_geolocation_fields_spacecraftaltitude.attrs[
            "ValidRange"
        ] = np.array(
            [np.float32(np.float32(0.0)), np.float32(np.float32(1e30))],
            dtype=np.float32,
        )
        hdfeos_swaths_omi_ground_pixel_corners_uv_2_geolocation_fields_spacecraftaltitude.attrs[
            "_FillValue"
        ] = np.array([np.float32(np.float32(-1e30))], dtype=np.float32)

        # HDFEOS/SWATHS/OMI Ground Pixel Corners UV-2/Geolocation Fields/SpacecraftLatitude
        hdfeos_swaths_omi_ground_pixel_corners_uv_2_geolocation_fields_spacecraftlatitude = hdfeos_swaths_omi_ground_pixel_corners_uv_2_geolocation_fields.create_dataset(
            "SpacecraftLatitude",
            data=np.arange(dim_0_1643, dtype=np.float32),
        )
        hdfeos_swaths_omi_ground_pixel_corners_uv_2_geolocation_fields_spacecraftlatitude.attrs[
            "MissingValue"
        ] = np.array([np.float32(np.float32(-1e30))], dtype=np.float32)
        hdfeos_swaths_omi_ground_pixel_corners_uv_2_geolocation_fields_spacecraftlatitude.attrs[
            "Offset"
        ] = np.array([np.float64(np.float64(0.0))], dtype=np.float64)
        hdfeos_swaths_omi_ground_pixel_corners_uv_2_geolocation_fields_spacecraftlatitude.attrs[
            "ScaleFactor"
        ] = np.array([np.float64(np.float64(1.0))], dtype=np.float64)
        hdfeos_swaths_omi_ground_pixel_corners_uv_2_geolocation_fields_spacecraftlatitude.attrs[
            "Title"
        ] = "Latitude of EOS-Aura Spacecraft"
        hdfeos_swaths_omi_ground_pixel_corners_uv_2_geolocation_fields_spacecraftlatitude.attrs[
            "UniqueFieldDefinition"
        ] = "Aura-Shared"
        hdfeos_swaths_omi_ground_pixel_corners_uv_2_geolocation_fields_spacecraftlatitude.attrs[
            "Units"
        ] = "deg"
        hdfeos_swaths_omi_ground_pixel_corners_uv_2_geolocation_fields_spacecraftlatitude.attrs[
            "ValidRange"
        ] = np.array(
            [np.float32(np.float32(-90.0)), np.float32(np.float32(90.0))],
            dtype=np.float32,
        )
        hdfeos_swaths_omi_ground_pixel_corners_uv_2_geolocation_fields_spacecraftlatitude.attrs[
            "_FillValue"
        ] = np.array([np.float32(np.float32(-1e30))], dtype=np.float32)

        # HDFEOS/SWATHS/OMI Ground Pixel Corners UV-2/Geolocation Fields/SpacecraftLongitude
        hdfeos_swaths_omi_ground_pixel_corners_uv_2_geolocation_fields_spacecraftlongitude = hdfeos_swaths_omi_ground_pixel_corners_uv_2_geolocation_fields.create_dataset(
            "SpacecraftLongitude",
            data=np.arange(dim_0_1643, dtype=np.float32),
        )
        hdfeos_swaths_omi_ground_pixel_corners_uv_2_geolocation_fields_spacecraftlongitude.attrs[
            "MissingValue"
        ] = np.array([np.float32(np.float32(-1e30))], dtype=np.float32)
        hdfeos_swaths_omi_ground_pixel_corners_uv_2_geolocation_fields_spacecraftlongitude.attrs[
            "Offset"
        ] = np.array([np.float64(np.float64(0.0))], dtype=np.float64)
        hdfeos_swaths_omi_ground_pixel_corners_uv_2_geolocation_fields_spacecraftlongitude.attrs[
            "ScaleFactor"
        ] = np.array([np.float64(np.float64(1.0))], dtype=np.float64)
        hdfeos_swaths_omi_ground_pixel_corners_uv_2_geolocation_fields_spacecraftlongitude.attrs[
            "Title"
        ] = "Longitude of EOS-Aura Spacecraft"
        hdfeos_swaths_omi_ground_pixel_corners_uv_2_geolocation_fields_spacecraftlongitude.attrs[
            "UniqueFieldDefinition"
        ] = "Aura-Shared"
        hdfeos_swaths_omi_ground_pixel_corners_uv_2_geolocation_fields_spacecraftlongitude.attrs[
            "Units"
        ] = "deg"
        hdfeos_swaths_omi_ground_pixel_corners_uv_2_geolocation_fields_spacecraftlongitude.attrs[
            "ValidRange"
        ] = np.array(
            [np.float32(np.float32(-180.0)), np.float32(np.float32(180.0))],
            dtype=np.float32,
        )
        hdfeos_swaths_omi_ground_pixel_corners_uv_2_geolocation_fields_spacecraftlongitude.attrs[
            "_FillValue"
        ] = np.array([np.float32(np.float32(-1e30))], dtype=np.float32)

        # HDFEOS/SWATHS/OMI Ground Pixel Corners UV-2/Geolocation Fields/Time
        hdfeos_swaths_omi_ground_pixel_corners_uv_2_geolocation_fields_time = hdfeos_swaths_omi_ground_pixel_corners_uv_2_geolocation_fields.create_dataset(
            "Time",
            data=np.linspace(0.0, 1000000000.0, dim_0_1643).astype(np.float64),
        )
        hdfeos_swaths_omi_ground_pixel_corners_uv_2_geolocation_fields_time.attrs[
            "MissingValue"
        ] = np.array([np.float64(np.float64(-1.0000000150474662e30))], dtype=np.float64)
        hdfeos_swaths_omi_ground_pixel_corners_uv_2_geolocation_fields_time.attrs[
            "Offset"
        ] = np.array([np.float64(np.float64(0.0))], dtype=np.float64)
        hdfeos_swaths_omi_ground_pixel_corners_uv_2_geolocation_fields_time.attrs[
            "ScaleFactor"
        ] = np.array([np.float64(np.float64(1.0))], dtype=np.float64)
        hdfeos_swaths_omi_ground_pixel_corners_uv_2_geolocation_fields_time.attrs[
            "Title"
        ] = "Time at Start of Swath Line (s, TAI93)"
        hdfeos_swaths_omi_ground_pixel_corners_uv_2_geolocation_fields_time.attrs[
            "UniqueFieldDefinition"
        ] = "Aura-Shared"
        hdfeos_swaths_omi_ground_pixel_corners_uv_2_geolocation_fields_time.attrs[
            "Units"
        ] = "s"
        hdfeos_swaths_omi_ground_pixel_corners_uv_2_geolocation_fields_time.attrs[
            "ValidRange"
        ] = np.array(
            [np.float64(np.float64(0.0)), np.float64(np.float64(10000000000.0))],
            dtype=np.float64,
        )
        hdfeos_swaths_omi_ground_pixel_corners_uv_2_geolocation_fields_time.attrs[
            "_FillValue"
        ] = np.array([np.float64(np.float64(-1e30))], dtype=np.float64)

        # HDFEOS/SWATHS/OMI Ground Pixel Corners UV-2/Geolocation Fields/TimeUTC
        hdfeos_swaths_omi_ground_pixel_corners_uv_2_geolocation_fields_timeutc = hdfeos_swaths_omi_ground_pixel_corners_uv_2_geolocation_fields.create_dataset(
            "TimeUTC",
            data=np.full(
                (
                    dim_0_1643,
                    dim_1_27,
                ),
                b"",
                dtype=np.dtype("|S1"),
            ),
        )
        hdfeos_swaths_omi_ground_pixel_corners_uv_2_geolocation_fields_timeutc.attrs[
            "MissingValue"
        ] = "undefined"
        hdfeos_swaths_omi_ground_pixel_corners_uv_2_geolocation_fields_timeutc.attrs[
            "Offset"
        ] = np.array([np.float64(np.float64(0.0))], dtype=np.float64)
        hdfeos_swaths_omi_ground_pixel_corners_uv_2_geolocation_fields_timeutc.attrs[
            "ScaleFactor"
        ] = np.array([np.float64(np.float64(1.0))], dtype=np.float64)
        hdfeos_swaths_omi_ground_pixel_corners_uv_2_geolocation_fields_timeutc.attrs[
            "Title"
        ] = "Coordinated Universal Time"
        hdfeos_swaths_omi_ground_pixel_corners_uv_2_geolocation_fields_timeutc.attrs[
            "UniqueFieldDefinition"
        ] = "OMI-Specific"
        hdfeos_swaths_omi_ground_pixel_corners_uv_2_geolocation_fields_timeutc.attrs[
            "Units"
        ] = "NoUnits"

        # HDFEOS/SWATHS/OMI Ground Pixel Corners VIS/Data Fields/FoV75Area
        hdfeos_swaths_omi_ground_pixel_corners_vis_data_fields_fov75area = (
            hdfeos_swaths_omi_ground_pixel_corners_vis_data_fields.create_dataset(
                "FoV75Area",
                data=np.arange(dim_0_60, dtype=np.float32),
            )
        )
        hdfeos_swaths_omi_ground_pixel_corners_vis_data_fields_fov75area.attrs[
            "MissingValue"
        ] = np.array([np.float32(np.float32(-1e30))], dtype=np.float32)
        hdfeos_swaths_omi_ground_pixel_corners_vis_data_fields_fov75area.attrs[
            "Offset"
        ] = np.array([np.float64(np.float64(0.0))], dtype=np.float64)
        hdfeos_swaths_omi_ground_pixel_corners_vis_data_fields_fov75area.attrs[
            "ScaleFactor"
        ] = np.array([np.float64(np.float64(1.0))], dtype=np.float64)
        hdfeos_swaths_omi_ground_pixel_corners_vis_data_fields_fov75area.attrs[
            "Title"
        ] = "Mean Area for 75% Field of View Pixels on the WGS-85 Ellipsoid"
        hdfeos_swaths_omi_ground_pixel_corners_vis_data_fields_fov75area.attrs[
            "UniqueFieldDefinition"
        ] = "OMI-Specific"
        hdfeos_swaths_omi_ground_pixel_corners_vis_data_fields_fov75area.attrs[
            "Units"
        ] = "km^2"
        hdfeos_swaths_omi_ground_pixel_corners_vis_data_fields_fov75area.attrs[
            "ValidRange"
        ] = np.array(
            [np.float32(np.float32(0.0)), np.float32(np.float32(5.10072e08))],
            dtype=np.float32,
        )
        hdfeos_swaths_omi_ground_pixel_corners_vis_data_fields_fov75area.attrs[
            "_FillValue"
        ] = np.array([np.float32(np.float32(-1e30))], dtype=np.float32)

        # HDFEOS/SWATHS/OMI Ground Pixel Corners VIS/Data Fields/FoV75CornerLatitude
        hdfeos_swaths_omi_ground_pixel_corners_vis_data_fields_fov75cornerlatitude = (
            hdfeos_swaths_omi_ground_pixel_corners_vis_data_fields.create_dataset(
                "FoV75CornerLatitude",
                data=np.arange(
                    dim_0_4 * dim_0_1643 * dim_0_60, dtype=np.float32
                ).reshape(
                    (
                        dim_0_4,
                        dim_0_1643,
                        dim_0_60,
                    )
                ),
            )
        )
        hdfeos_swaths_omi_ground_pixel_corners_vis_data_fields_fov75cornerlatitude.attrs[
            "MissingValue"
        ] = np.array([np.float32(np.float32(-1e30))], dtype=np.float32)
        hdfeos_swaths_omi_ground_pixel_corners_vis_data_fields_fov75cornerlatitude.attrs[
            "Offset"
        ] = np.array([np.float64(np.float64(0.0))], dtype=np.float64)
        hdfeos_swaths_omi_ground_pixel_corners_vis_data_fields_fov75cornerlatitude.attrs[
            "ScaleFactor"
        ] = np.array([np.float64(np.float64(1.0))], dtype=np.float64)
        hdfeos_swaths_omi_ground_pixel_corners_vis_data_fields_fov75cornerlatitude.attrs[
            "Title"
        ] = "Corner Latitudes for 75% Field of View Pixels on the WGS-85 Ellipsoid (CCW relative to flight direction: LL,LR,UR,UL)"
        hdfeos_swaths_omi_ground_pixel_corners_vis_data_fields_fov75cornerlatitude.attrs[
            "UniqueFieldDefinition"
        ] = "OMI-Specific"
        hdfeos_swaths_omi_ground_pixel_corners_vis_data_fields_fov75cornerlatitude.attrs[
            "Units"
        ] = "deg"
        hdfeos_swaths_omi_ground_pixel_corners_vis_data_fields_fov75cornerlatitude.attrs[
            "ValidRange"
        ] = np.array(
            [np.float32(np.float32(-90.0)), np.float32(np.float32(90.0))],
            dtype=np.float32,
        )
        hdfeos_swaths_omi_ground_pixel_corners_vis_data_fields_fov75cornerlatitude.attrs[
            "_FillValue"
        ] = np.array([np.float32(np.float32(-1e30))], dtype=np.float32)

        # HDFEOS/SWATHS/OMI Ground Pixel Corners VIS/Data Fields/FoV75CornerLongitude
        hdfeos_swaths_omi_ground_pixel_corners_vis_data_fields_fov75cornerlongitude = (
            hdfeos_swaths_omi_ground_pixel_corners_vis_data_fields.create_dataset(
                "FoV75CornerLongitude",
                data=np.arange(
                    dim_0_4 * dim_0_1643 * dim_0_60, dtype=np.float32
                ).reshape(
                    (
                        dim_0_4,
                        dim_0_1643,
                        dim_0_60,
                    )
                ),
            )
        )
        hdfeos_swaths_omi_ground_pixel_corners_vis_data_fields_fov75cornerlongitude.attrs[
            "MissingValue"
        ] = np.array([np.float32(np.float32(-1e30))], dtype=np.float32)
        hdfeos_swaths_omi_ground_pixel_corners_vis_data_fields_fov75cornerlongitude.attrs[
            "Offset"
        ] = np.array([np.float64(np.float64(0.0))], dtype=np.float64)
        hdfeos_swaths_omi_ground_pixel_corners_vis_data_fields_fov75cornerlongitude.attrs[
            "ScaleFactor"
        ] = np.array([np.float64(np.float64(1.0))], dtype=np.float64)
        hdfeos_swaths_omi_ground_pixel_corners_vis_data_fields_fov75cornerlongitude.attrs[
            "Title"
        ] = "Corner Longitudes for 75% Field of View Pixels on the WGS-85 Ellipsoid (CCW relative to flight direction: LL,LR,UR,UL)"
        hdfeos_swaths_omi_ground_pixel_corners_vis_data_fields_fov75cornerlongitude.attrs[
            "UniqueFieldDefinition"
        ] = "OMI-Specific"
        hdfeos_swaths_omi_ground_pixel_corners_vis_data_fields_fov75cornerlongitude.attrs[
            "Units"
        ] = "deg"
        hdfeos_swaths_omi_ground_pixel_corners_vis_data_fields_fov75cornerlongitude.attrs[
            "ValidRange"
        ] = np.array(
            [np.float32(np.float32(-180.0)), np.float32(np.float32(180.0))],
            dtype=np.float32,
        )
        hdfeos_swaths_omi_ground_pixel_corners_vis_data_fields_fov75cornerlongitude.attrs[
            "_FillValue"
        ] = np.array([np.float32(np.float32(-1e30))], dtype=np.float32)

        # HDFEOS/SWATHS/OMI Ground Pixel Corners VIS/Data Fields/TiledArea
        hdfeos_swaths_omi_ground_pixel_corners_vis_data_fields_tiledarea = (
            hdfeos_swaths_omi_ground_pixel_corners_vis_data_fields.create_dataset(
                "TiledArea",
                data=np.arange(dim_0_60, dtype=np.float32),
            )
        )
        hdfeos_swaths_omi_ground_pixel_corners_vis_data_fields_tiledarea.attrs[
            "MissingValue"
        ] = np.array([np.float32(np.float32(-1e30))], dtype=np.float32)
        hdfeos_swaths_omi_ground_pixel_corners_vis_data_fields_tiledarea.attrs[
            "Offset"
        ] = np.array([np.float64(np.float64(0.0))], dtype=np.float64)
        hdfeos_swaths_omi_ground_pixel_corners_vis_data_fields_tiledarea.attrs[
            "ScaleFactor"
        ] = np.array([np.float64(np.float64(1.0))], dtype=np.float64)
        hdfeos_swaths_omi_ground_pixel_corners_vis_data_fields_tiledarea.attrs[
            "Title"
        ] = "Mean Area for Tiled Pixels on the WGS-85 Ellipsoid"
        hdfeos_swaths_omi_ground_pixel_corners_vis_data_fields_tiledarea.attrs[
            "UniqueFieldDefinition"
        ] = "OMI-Specific"
        hdfeos_swaths_omi_ground_pixel_corners_vis_data_fields_tiledarea.attrs[
            "Units"
        ] = "km^2"
        hdfeos_swaths_omi_ground_pixel_corners_vis_data_fields_tiledarea.attrs[
            "ValidRange"
        ] = np.array(
            [np.float32(np.float32(0.0)), np.float32(np.float32(5.10072e08))],
            dtype=np.float32,
        )
        hdfeos_swaths_omi_ground_pixel_corners_vis_data_fields_tiledarea.attrs[
            "_FillValue"
        ] = np.array([np.float32(np.float32(-1e30))], dtype=np.float32)

        # HDFEOS/SWATHS/OMI Ground Pixel Corners VIS/Data Fields/TiledCornerLatitude
        hdfeos_swaths_omi_ground_pixel_corners_vis_data_fields_tiledcornerlatitude = (
            hdfeos_swaths_omi_ground_pixel_corners_vis_data_fields.create_dataset(
                "TiledCornerLatitude",
                data=np.arange(
                    dim_0_4 * dim_0_1643 * dim_0_60, dtype=np.float32
                ).reshape(
                    (
                        dim_0_4,
                        dim_0_1643,
                        dim_0_60,
                    )
                ),
            )
        )
        hdfeos_swaths_omi_ground_pixel_corners_vis_data_fields_tiledcornerlatitude.attrs[
            "MissingValue"
        ] = np.array([np.float32(np.float32(-1e30))], dtype=np.float32)
        hdfeos_swaths_omi_ground_pixel_corners_vis_data_fields_tiledcornerlatitude.attrs[
            "Offset"
        ] = np.array([np.float64(np.float64(0.0))], dtype=np.float64)
        hdfeos_swaths_omi_ground_pixel_corners_vis_data_fields_tiledcornerlatitude.attrs[
            "ScaleFactor"
        ] = np.array([np.float64(np.float64(1.0))], dtype=np.float64)
        hdfeos_swaths_omi_ground_pixel_corners_vis_data_fields_tiledcornerlatitude.attrs[
            "Title"
        ] = "Corner Latitudes for Tiled Pixels on the WGS-85 Ellipsoid (CCW relative to flight direction: LL,LR,UR,UL)"
        hdfeos_swaths_omi_ground_pixel_corners_vis_data_fields_tiledcornerlatitude.attrs[
            "UniqueFieldDefinition"
        ] = "OMI-Specific"
        hdfeos_swaths_omi_ground_pixel_corners_vis_data_fields_tiledcornerlatitude.attrs[
            "Units"
        ] = "deg"
        hdfeos_swaths_omi_ground_pixel_corners_vis_data_fields_tiledcornerlatitude.attrs[
            "ValidRange"
        ] = np.array(
            [np.float32(np.float32(-90.0)), np.float32(np.float32(90.0))],
            dtype=np.float32,
        )
        hdfeos_swaths_omi_ground_pixel_corners_vis_data_fields_tiledcornerlatitude.attrs[
            "_FillValue"
        ] = np.array([np.float32(np.float32(-1e30))], dtype=np.float32)

        # HDFEOS/SWATHS/OMI Ground Pixel Corners VIS/Data Fields/TiledCornerLongitude
        hdfeos_swaths_omi_ground_pixel_corners_vis_data_fields_tiledcornerlongitude = (
            hdfeos_swaths_omi_ground_pixel_corners_vis_data_fields.create_dataset(
                "TiledCornerLongitude",
                data=np.arange(
                    dim_0_4 * dim_0_1643 * dim_0_60, dtype=np.float32
                ).reshape(
                    (
                        dim_0_4,
                        dim_0_1643,
                        dim_0_60,
                    )
                ),
            )
        )
        hdfeos_swaths_omi_ground_pixel_corners_vis_data_fields_tiledcornerlongitude.attrs[
            "MissingValue"
        ] = np.array([np.float32(np.float32(-1e30))], dtype=np.float32)
        hdfeos_swaths_omi_ground_pixel_corners_vis_data_fields_tiledcornerlongitude.attrs[
            "Offset"
        ] = np.array([np.float64(np.float64(0.0))], dtype=np.float64)
        hdfeos_swaths_omi_ground_pixel_corners_vis_data_fields_tiledcornerlongitude.attrs[
            "ScaleFactor"
        ] = np.array([np.float64(np.float64(1.0))], dtype=np.float64)
        hdfeos_swaths_omi_ground_pixel_corners_vis_data_fields_tiledcornerlongitude.attrs[
            "Title"
        ] = "Corner Longitudes for Tiled Pixels on the WGS-85 Ellipsoid (CCW relative to flight direction: LL,LR,UR,UL)"
        hdfeos_swaths_omi_ground_pixel_corners_vis_data_fields_tiledcornerlongitude.attrs[
            "UniqueFieldDefinition"
        ] = "OMI-Specific"
        hdfeos_swaths_omi_ground_pixel_corners_vis_data_fields_tiledcornerlongitude.attrs[
            "Units"
        ] = "deg"
        hdfeos_swaths_omi_ground_pixel_corners_vis_data_fields_tiledcornerlongitude.attrs[
            "ValidRange"
        ] = np.array(
            [np.float32(np.float32(-180.0)), np.float32(np.float32(180.0))],
            dtype=np.float32,
        )
        hdfeos_swaths_omi_ground_pixel_corners_vis_data_fields_tiledcornerlongitude.attrs[
            "_FillValue"
        ] = np.array([np.float32(np.float32(-1e30))], dtype=np.float32)

        # HDFEOS/SWATHS/OMI Ground Pixel Corners VIS/Geolocation Fields/Latitude
        hdfeos_swaths_omi_ground_pixel_corners_vis_geolocation_fields_latitude = hdfeos_swaths_omi_ground_pixel_corners_vis_geolocation_fields.create_dataset(
            "Latitude",
            data=np.linspace(-90.0, 90.0, dim_0_1643 * dim_0_60)
            .reshape(
                (
                    dim_0_1643,
                    dim_0_60,
                )
            )
            .astype(np.float32),
        )
        hdfeos_swaths_omi_ground_pixel_corners_vis_geolocation_fields_latitude.attrs[
            "MissingValue"
        ] = np.array([np.float32(np.float32(-1e30))], dtype=np.float32)
        hdfeos_swaths_omi_ground_pixel_corners_vis_geolocation_fields_latitude.attrs[
            "Offset"
        ] = np.array([np.float64(np.float64(0.0))], dtype=np.float64)
        hdfeos_swaths_omi_ground_pixel_corners_vis_geolocation_fields_latitude.attrs[
            "ScaleFactor"
        ] = np.array([np.float64(np.float64(1.0))], dtype=np.float64)
        hdfeos_swaths_omi_ground_pixel_corners_vis_geolocation_fields_latitude.attrs[
            "Title"
        ] = "Geodetic Latitude"
        hdfeos_swaths_omi_ground_pixel_corners_vis_geolocation_fields_latitude.attrs[
            "UniqueFieldDefinition"
        ] = "Aura-Shared"
        hdfeos_swaths_omi_ground_pixel_corners_vis_geolocation_fields_latitude.attrs[
            "Units"
        ] = "deg"
        hdfeos_swaths_omi_ground_pixel_corners_vis_geolocation_fields_latitude.attrs[
            "ValidRange"
        ] = np.array(
            [np.float32(np.float32(-90.0)), np.float32(np.float32(90.0))],
            dtype=np.float32,
        )
        hdfeos_swaths_omi_ground_pixel_corners_vis_geolocation_fields_latitude.attrs[
            "_FillValue"
        ] = np.array([np.float32(np.float32(-1e30))], dtype=np.float32)

        # HDFEOS/SWATHS/OMI Ground Pixel Corners VIS/Geolocation Fields/Longitude
        hdfeos_swaths_omi_ground_pixel_corners_vis_geolocation_fields_longitude = hdfeos_swaths_omi_ground_pixel_corners_vis_geolocation_fields.create_dataset(
            "Longitude",
            data=np.linspace(-180.0, 180.0, dim_0_1643 * dim_0_60)
            .reshape(
                (
                    dim_0_1643,
                    dim_0_60,
                )
            )
            .astype(np.float32),
        )
        hdfeos_swaths_omi_ground_pixel_corners_vis_geolocation_fields_longitude.attrs[
            "MissingValue"
        ] = np.array([np.float32(np.float32(-1e30))], dtype=np.float32)
        hdfeos_swaths_omi_ground_pixel_corners_vis_geolocation_fields_longitude.attrs[
            "Offset"
        ] = np.array([np.float64(np.float64(0.0))], dtype=np.float64)
        hdfeos_swaths_omi_ground_pixel_corners_vis_geolocation_fields_longitude.attrs[
            "ScaleFactor"
        ] = np.array([np.float64(np.float64(1.0))], dtype=np.float64)
        hdfeos_swaths_omi_ground_pixel_corners_vis_geolocation_fields_longitude.attrs[
            "Title"
        ] = "Geodetic Longitude"
        hdfeos_swaths_omi_ground_pixel_corners_vis_geolocation_fields_longitude.attrs[
            "UniqueFieldDefinition"
        ] = "Aura-Shared"
        hdfeos_swaths_omi_ground_pixel_corners_vis_geolocation_fields_longitude.attrs[
            "Units"
        ] = "deg"
        hdfeos_swaths_omi_ground_pixel_corners_vis_geolocation_fields_longitude.attrs[
            "ValidRange"
        ] = np.array(
            [np.float32(np.float32(-180.0)), np.float32(np.float32(180.0))],
            dtype=np.float32,
        )
        hdfeos_swaths_omi_ground_pixel_corners_vis_geolocation_fields_longitude.attrs[
            "_FillValue"
        ] = np.array([np.float32(np.float32(-1e30))], dtype=np.float32)

        # HDFEOS/SWATHS/OMI Ground Pixel Corners VIS/Geolocation Fields/SpacecraftAltitude
        hdfeos_swaths_omi_ground_pixel_corners_vis_geolocation_fields_spacecraftaltitude = hdfeos_swaths_omi_ground_pixel_corners_vis_geolocation_fields.create_dataset(
            "SpacecraftAltitude",
            data=np.arange(dim_0_1643, dtype=np.float32),
        )
        hdfeos_swaths_omi_ground_pixel_corners_vis_geolocation_fields_spacecraftaltitude.attrs[
            "MissingValue"
        ] = np.array([np.float32(np.float32(-1e30))], dtype=np.float32)
        hdfeos_swaths_omi_ground_pixel_corners_vis_geolocation_fields_spacecraftaltitude.attrs[
            "Offset"
        ] = np.array([np.float64(np.float64(0.0))], dtype=np.float64)
        hdfeos_swaths_omi_ground_pixel_corners_vis_geolocation_fields_spacecraftaltitude.attrs[
            "ScaleFactor"
        ] = np.array([np.float64(np.float64(1.0))], dtype=np.float64)
        hdfeos_swaths_omi_ground_pixel_corners_vis_geolocation_fields_spacecraftaltitude.attrs[
            "Title"
        ] = "Altitude of EOS-Aura Spacecraft"
        hdfeos_swaths_omi_ground_pixel_corners_vis_geolocation_fields_spacecraftaltitude.attrs[
            "UniqueFieldDefinition"
        ] = "Aura-Shared"
        hdfeos_swaths_omi_ground_pixel_corners_vis_geolocation_fields_spacecraftaltitude.attrs[
            "Units"
        ] = "m"
        hdfeos_swaths_omi_ground_pixel_corners_vis_geolocation_fields_spacecraftaltitude.attrs[
            "ValidRange"
        ] = np.array(
            [np.float32(np.float32(0.0)), np.float32(np.float32(1e30))],
            dtype=np.float32,
        )
        hdfeos_swaths_omi_ground_pixel_corners_vis_geolocation_fields_spacecraftaltitude.attrs[
            "_FillValue"
        ] = np.array([np.float32(np.float32(-1e30))], dtype=np.float32)

        # HDFEOS/SWATHS/OMI Ground Pixel Corners VIS/Geolocation Fields/SpacecraftLatitude
        hdfeos_swaths_omi_ground_pixel_corners_vis_geolocation_fields_spacecraftlatitude = hdfeos_swaths_omi_ground_pixel_corners_vis_geolocation_fields.create_dataset(
            "SpacecraftLatitude",
            data=np.arange(dim_0_1643, dtype=np.float32),
        )
        hdfeos_swaths_omi_ground_pixel_corners_vis_geolocation_fields_spacecraftlatitude.attrs[
            "MissingValue"
        ] = np.array([np.float32(np.float32(-1e30))], dtype=np.float32)
        hdfeos_swaths_omi_ground_pixel_corners_vis_geolocation_fields_spacecraftlatitude.attrs[
            "Offset"
        ] = np.array([np.float64(np.float64(0.0))], dtype=np.float64)
        hdfeos_swaths_omi_ground_pixel_corners_vis_geolocation_fields_spacecraftlatitude.attrs[
            "ScaleFactor"
        ] = np.array([np.float64(np.float64(1.0))], dtype=np.float64)
        hdfeos_swaths_omi_ground_pixel_corners_vis_geolocation_fields_spacecraftlatitude.attrs[
            "Title"
        ] = "Latitude of EOS-Aura Spacecraft"
        hdfeos_swaths_omi_ground_pixel_corners_vis_geolocation_fields_spacecraftlatitude.attrs[
            "UniqueFieldDefinition"
        ] = "Aura-Shared"
        hdfeos_swaths_omi_ground_pixel_corners_vis_geolocation_fields_spacecraftlatitude.attrs[
            "Units"
        ] = "deg"
        hdfeos_swaths_omi_ground_pixel_corners_vis_geolocation_fields_spacecraftlatitude.attrs[
            "ValidRange"
        ] = np.array(
            [np.float32(np.float32(-90.0)), np.float32(np.float32(90.0))],
            dtype=np.float32,
        )
        hdfeos_swaths_omi_ground_pixel_corners_vis_geolocation_fields_spacecraftlatitude.attrs[
            "_FillValue"
        ] = np.array([np.float32(np.float32(-1e30))], dtype=np.float32)

        # HDFEOS/SWATHS/OMI Ground Pixel Corners VIS/Geolocation Fields/SpacecraftLongitude
        hdfeos_swaths_omi_ground_pixel_corners_vis_geolocation_fields_spacecraftlongitude = hdfeos_swaths_omi_ground_pixel_corners_vis_geolocation_fields.create_dataset(
            "SpacecraftLongitude",
            data=np.arange(dim_0_1643, dtype=np.float32),
        )
        hdfeos_swaths_omi_ground_pixel_corners_vis_geolocation_fields_spacecraftlongitude.attrs[
            "MissingValue"
        ] = np.array([np.float32(np.float32(-1e30))], dtype=np.float32)
        hdfeos_swaths_omi_ground_pixel_corners_vis_geolocation_fields_spacecraftlongitude.attrs[
            "Offset"
        ] = np.array([np.float64(np.float64(0.0))], dtype=np.float64)
        hdfeos_swaths_omi_ground_pixel_corners_vis_geolocation_fields_spacecraftlongitude.attrs[
            "ScaleFactor"
        ] = np.array([np.float64(np.float64(1.0))], dtype=np.float64)
        hdfeos_swaths_omi_ground_pixel_corners_vis_geolocation_fields_spacecraftlongitude.attrs[
            "Title"
        ] = "Longitude of EOS-Aura Spacecraft"
        hdfeos_swaths_omi_ground_pixel_corners_vis_geolocation_fields_spacecraftlongitude.attrs[
            "UniqueFieldDefinition"
        ] = "Aura-Shared"
        hdfeos_swaths_omi_ground_pixel_corners_vis_geolocation_fields_spacecraftlongitude.attrs[
            "Units"
        ] = "deg"
        hdfeos_swaths_omi_ground_pixel_corners_vis_geolocation_fields_spacecraftlongitude.attrs[
            "ValidRange"
        ] = np.array(
            [np.float32(np.float32(-180.0)), np.float32(np.float32(180.0))],
            dtype=np.float32,
        )
        hdfeos_swaths_omi_ground_pixel_corners_vis_geolocation_fields_spacecraftlongitude.attrs[
            "_FillValue"
        ] = np.array([np.float32(np.float32(-1e30))], dtype=np.float32)

        # HDFEOS/SWATHS/OMI Ground Pixel Corners VIS/Geolocation Fields/Time
        hdfeos_swaths_omi_ground_pixel_corners_vis_geolocation_fields_time = hdfeos_swaths_omi_ground_pixel_corners_vis_geolocation_fields.create_dataset(
            "Time",
            data=np.linspace(0.0, 1000000000.0, dim_0_1643).astype(np.float64),
        )
        hdfeos_swaths_omi_ground_pixel_corners_vis_geolocation_fields_time.attrs[
            "MissingValue"
        ] = np.array([np.float64(np.float64(-1.0000000150474662e30))], dtype=np.float64)
        hdfeos_swaths_omi_ground_pixel_corners_vis_geolocation_fields_time.attrs[
            "Offset"
        ] = np.array([np.float64(np.float64(0.0))], dtype=np.float64)
        hdfeos_swaths_omi_ground_pixel_corners_vis_geolocation_fields_time.attrs[
            "ScaleFactor"
        ] = np.array([np.float64(np.float64(1.0))], dtype=np.float64)
        hdfeos_swaths_omi_ground_pixel_corners_vis_geolocation_fields_time.attrs[
            "Title"
        ] = "Time at Start of Swath Line (s, TAI93)"
        hdfeos_swaths_omi_ground_pixel_corners_vis_geolocation_fields_time.attrs[
            "UniqueFieldDefinition"
        ] = "Aura-Shared"
        hdfeos_swaths_omi_ground_pixel_corners_vis_geolocation_fields_time.attrs[
            "Units"
        ] = "s"
        hdfeos_swaths_omi_ground_pixel_corners_vis_geolocation_fields_time.attrs[
            "ValidRange"
        ] = np.array(
            [np.float64(np.float64(0.0)), np.float64(np.float64(10000000000.0))],
            dtype=np.float64,
        )
        hdfeos_swaths_omi_ground_pixel_corners_vis_geolocation_fields_time.attrs[
            "_FillValue"
        ] = np.array([np.float64(np.float64(-1e30))], dtype=np.float64)

        # HDFEOS/SWATHS/OMI Ground Pixel Corners VIS/Geolocation Fields/TimeUTC
        hdfeos_swaths_omi_ground_pixel_corners_vis_geolocation_fields_timeutc = hdfeos_swaths_omi_ground_pixel_corners_vis_geolocation_fields.create_dataset(
            "TimeUTC",
            data=np.full(
                (
                    dim_0_1643,
                    dim_1_27,
                ),
                b"",
                dtype=np.dtype("|S1"),
            ),
        )
        hdfeos_swaths_omi_ground_pixel_corners_vis_geolocation_fields_timeutc.attrs[
            "MissingValue"
        ] = "undefined"
        hdfeos_swaths_omi_ground_pixel_corners_vis_geolocation_fields_timeutc.attrs[
            "Offset"
        ] = np.array([np.float64(np.float64(0.0))], dtype=np.float64)
        hdfeos_swaths_omi_ground_pixel_corners_vis_geolocation_fields_timeutc.attrs[
            "ScaleFactor"
        ] = np.array([np.float64(np.float64(1.0))], dtype=np.float64)
        hdfeos_swaths_omi_ground_pixel_corners_vis_geolocation_fields_timeutc.attrs[
            "Title"
        ] = "Coordinated Universal Time"
        hdfeos_swaths_omi_ground_pixel_corners_vis_geolocation_fields_timeutc.attrs[
            "UniqueFieldDefinition"
        ] = "OMI-Specific"
        hdfeos_swaths_omi_ground_pixel_corners_vis_geolocation_fields_timeutc.attrs[
            "Units"
        ] = "NoUnits"

        # HDFEOS INFORMATION/ArchiveMetadata
        hdfeos_information_archivemetadata_text = (
            "\n"
            "GROUP                  = ARCHIVEDMETADATA\n"
            "  GROUPTYPE            = MASTERGROUP\n"
            "\n"
            "  OBJECT                 = LONGNAME\n"
            "    NUM_VAL              = 1\n"
            '    VALUE                = "OMI/Aura Global Ground Pixel Corners 1-Orbit L2 Swath 13x24km"\n'
            "  END_OBJECT             = LONGNAME\n"
            "\n"
            "  OBJECT                 = ESDTDESCRIPTORREVISION\n"
            "    NUM_VAL              = 1\n"
            '    VALUE                = "2.0.05"\n'
            "  END_OBJECT             = ESDTDESCRIPTORREVISION\n"
            "\n"
            "END_GROUP              = ARCHIVEDMETADATA\n"
            "\n"
            "END\n"
        )
        hdfeos_information.create_dataset(
            "ArchiveMetadata",
            data=np.bytes_(hdfeos_information_archivemetadata_text),
            dtype=np.dtype("|S65535"),
        )

        # HDFEOS INFORMATION/CoreMetadata
        hdfeos_information_coremetadata_text = (
            "\n"
            "GROUP                  = INVENTORYMETADATA\n"
            "  GROUPTYPE            = MASTERGROUP\n"
            "\n"
            "  GROUP                  = ECSDATAGRANULE\n"
            "\n"
            "    OBJECT                 = LOCALGRANULEID\n"
            "      NUM_VAL              = 1\n"
            '      VALUE                = "OMI-Aura_L2-OMPIXCOR_2004m1001t1452-o01141_v003-2018m0228t201200.he5"\n'
            "    END_OBJECT             = LOCALGRANULEID\n"
            "\n"
            "    OBJECT                 = PRODUCTIONDATETIME\n"
            "      NUM_VAL              = 1\n"
            '      VALUE                = "2018-02-28T20:12:00.000Z"\n'
            "    END_OBJECT             = PRODUCTIONDATETIME\n"
            "\n"
            "    OBJECT                 = DAYNIGHTFLAG\n"
            "      NUM_VAL              = 1\n"
            '      VALUE                = "Day"\n'
            "    END_OBJECT             = DAYNIGHTFLAG\n"
            "\n"
            "    OBJECT                 = REPROCESSINGACTUAL\n"
            "      NUM_VAL              = 1\n"
            '      VALUE                = ("processed 1 time")\n'
            "    END_OBJECT             = REPROCESSINGACTUAL\n"
            "\n"
            "    OBJECT                 = LOCALVERSIONID\n"
            "      NUM_VAL              = 1\n"
            '      VALUE                = ("RFC1321 MD5 = not yet calculated")\n'
            "    END_OBJECT             = LOCALVERSIONID\n"
            "\n"
            "    OBJECT                 = REPROCESSINGPLANNED\n"
            "      NUM_VAL              = 1\n"
            '      VALUE                = "further update is anticipated"\n'
            "    END_OBJECT             = REPROCESSINGPLANNED\n"
            "\n"
            "  END_GROUP              = ECSDATAGRANULE\n"
            "\n"
            "  GROUP                  = MEASUREDPARAMETER\n"
            "\n"
            "    OBJECT                 = MEASUREDPARAMETERCONTAINER\n"
            '      CLASS                = "1"\n'
            "\n"
            "      GROUP                  = QAFLAGS\n"
            '        CLASS                = "1"\n'
            "\n"
            "        OBJECT                 = SCIENCEQUALITYFLAG\n"
            "          NUM_VAL              = 1\n"
            '          VALUE                = "Not Investigated"\n'
            '          CLASS                = "1"\n'
            "        END_OBJECT             = SCIENCEQUALITYFLAG\n"
            "\n"
            "        OBJECT                 = AUTOMATICQUALITYFLAGEXPLANATION\n"
            "          NUM_VAL              = 1\n"
            '          CLASS                = "1"\n'
            '          VALUE                = "An updated automatic quality flag and explanation is put in the product .met file when a granule has been evaluated. The flag value in this file, Not Investigated, is an automatic default that is put into every granule during production."\n'
            "        END_OBJECT             = AUTOMATICQUALITYFLAGEXPLANATION\n"
            "\n"
            "        OBJECT                 = AUTOMATICQUALITYFLAG\n"
            "          NUM_VAL              = 1\n"
            '          CLASS                = "1"\n'
            '          VALUE                = "Not Investigated"\n'
            "        END_OBJECT             = AUTOMATICQUALITYFLAG\n"
            "\n"
            "        OBJECT                 = OPERATIONALQUALITYFLAGEXPLANATION\n"
            "          NUM_VAL              = 1\n"
            '          VALUE                = "This granule passed operational tests that were administered by the OMI SIPS.  QA metadata was extracted and the file was successfully read using standard HDF-EOS utilities."\n'
            '          CLASS                = "1"\n'
            "        END_OBJECT             = OPERATIONALQUALITYFLAGEXPLANATION\n"
            "\n"
            "        OBJECT                 = OPERATIONALQUALITYFLAG\n"
            "          NUM_VAL              = 1\n"
            '          VALUE                = "Passed"\n'
            '          CLASS                = "1"\n'
            "        END_OBJECT             = OPERATIONALQUALITYFLAG\n"
            "\n"
            "        OBJECT                 = SCIENCEQUALITYFLAGEXPLANATION\n"
            "          NUM_VAL              = 1\n"
            '          VALUE                = "An updated science quality flag and explanation is put in the product .met file when a granule has been evaluated.  The flag value in this file, Not Investigated, is an automatic default that is put into every granule during production."\n'
            '          CLASS                = "1"\n'
            "        END_OBJECT             = SCIENCEQUALITYFLAGEXPLANATION\n"
            "\n"
            "      END_GROUP              = QAFLAGS\n"
            "\n"
            "      GROUP                  = QASTATS\n"
            '        CLASS                = "1"\n'
            "\n"
            "        OBJECT                 = QAPERCENTMISSINGDATA\n"
            "          NUM_VAL              = 1\n"
            '          CLASS                = "1"\n'
            "          VALUE                = 0\n"
            "        END_OBJECT             = QAPERCENTMISSINGDATA\n"
            "\n"
            "        OBJECT                 = QAPERCENTOUTOFBOUNDSDATA\n"
            "          NUM_VAL              = 1\n"
            '          CLASS                = "1"\n'
            "          VALUE                = 0\n"
            "        END_OBJECT             = QAPERCENTOUTOFBOUNDSDATA\n"
            "\n"
            "      END_GROUP              = QASTATS\n"
            "\n"
            "      OBJECT                 = PARAMETERNAME\n"
            '        CLASS                = "1"\n'
            "        NUM_VAL              = 1\n"
            '        VALUE                = "OMI Groundpixel Corner Coordinates UV-1"\n'
            "      END_OBJECT             = PARAMETERNAME\n"
            "\n"
            "    END_OBJECT             = MEASUREDPARAMETERCONTAINER\n"
            "\n"
            "    OBJECT                 = MEASUREDPARAMETERCONTAINER\n"
            '      CLASS                = "2"\n'
            "\n"
            "      GROUP                  = QAFLAGS\n"
            '        CLASS                = "2"\n'
            "\n"
            "        OBJECT                 = SCIENCEQUALITYFLAG\n"
            "          NUM_VAL              = 1\n"
            '          VALUE                = "Not Investigated"\n'
            '          CLASS                = "2"\n'
            "        END_OBJECT             = SCIENCEQUALITYFLAG\n"
            "\n"
            "        OBJECT                 = AUTOMATICQUALITYFLAGEXPLANATION\n"
            "          NUM_VAL              = 1\n"
            '          CLASS                = "2"\n'
            '          VALUE                = "An updated automatic quality flag and explanation is put in the product .met file when a granule has been evaluated. The flag value in this file, Not Investigated, is an automatic default that is put into every granule during production."\n'
            "        END_OBJECT             = AUTOMATICQUALITYFLAGEXPLANATION\n"
            "\n"
            "        OBJECT                 = AUTOMATICQUALITYFLAG\n"
            "          NUM_VAL              = 1\n"
            '          CLASS                = "2"\n'
            '          VALUE                = "Not Investigated"\n'
            "        END_OBJECT             = AUTOMATICQUALITYFLAG\n"
            "\n"
            "        OBJECT                 = OPERATIONALQUALITYFLAGEXPLANATION\n"
            "          NUM_VAL              = 1\n"
            '          VALUE                = "This granule passed operational tests that were administered by the OMI SIPS.  QA metadata was extracted and the file was successfully read using standard HDF-EOS utilities."\n'
            '          CLASS                = "2"\n'
            "        END_OBJECT             = OPERATIONALQUALITYFLAGEXPLANATION\n"
            "\n"
            "        OBJECT                 = OPERATIONALQUALITYFLAG\n"
            "          NUM_VAL              = 1\n"
            '          VALUE                = "Passed"\n'
            '          CLASS                = "2"\n'
            "        END_OBJECT             = OPERATIONALQUALITYFLAG\n"
            "\n"
            "        OBJECT                 = SCIENCEQUALITYFLAGEXPLANATION\n"
            "          NUM_VAL              = 1\n"
            '          VALUE                = "An updated science quality flag and explanation is put in the product .met file when a granule has been evaluated.  The flag value in this file, Not Investigated, is an automatic default that is put into every granule during production."\n'
            '          CLASS                = "2"\n'
            "        END_OBJECT             = SCIENCEQUALITYFLAGEXPLANATION\n"
            "\n"
            "      END_GROUP              = QAFLAGS\n"
            "\n"
            "      GROUP                  = QASTATS\n"
            '        CLASS                = "2"\n'
            "\n"
            "        OBJECT                 = QAPERCENTMISSINGDATA\n"
            "          NUM_VAL              = 1\n"
            '          CLASS                = "2"\n'
            "          VALUE                = 0\n"
            "        END_OBJECT             = QAPERCENTMISSINGDATA\n"
            "\n"
            "        OBJECT                 = QAPERCENTOUTOFBOUNDSDATA\n"
            "          NUM_VAL              = 1\n"
            '          CLASS                = "2"\n'
            "          VALUE                = 0\n"
            "        END_OBJECT             = QAPERCENTOUTOFBOUNDSDATA\n"
            "\n"
            "      END_GROUP              = QASTATS\n"
            "\n"
            "      OBJECT                 = PARAMETERNAME\n"
            '        CLASS                = "2"\n'
            "        NUM_VAL              = 1\n"
            '        VALUE                = "OMI Groundpixel Corner Coordinates UV-2"\n'
            "      END_OBJECT             = PARAMETERNAME\n"
            "\n"
            "    END_OBJECT             = MEASUREDPARAMETERCONTAINER\n"
            "\n"
            "    OBJECT                 = MEASUREDPARAMETERCONTAINER\n"
            '      CLASS                = "3"\n'
            "\n"
            "      GROUP                  = QAFLAGS\n"
            '        CLASS                = "3"\n'
            "\n"
            "        OBJECT                 = SCIENCEQUALITYFLAG\n"
            "          NUM_VAL              = 1\n"
            '          VALUE                = "Not Investigated"\n'
            '          CLASS                = "3"\n'
            "        END_OBJECT             = SCIENCEQUALITYFLAG\n"
            "\n"
            "        OBJECT                 = AUTOMATICQUALITYFLAGEXPLANATION\n"
            "          NUM_VAL              = 1\n"
            '          CLASS                = "3"\n'
            '          VALUE                = "An updated automatic quality flag and explanation is put in the product .met file when a granule has been evaluated. The flag value in this file, Not Investigated, is an automatic default that is put into every granule during production."\n'
            "        END_OBJECT             = AUTOMATICQUALITYFLAGEXPLANATION\n"
            "\n"
            "        OBJECT                 = AUTOMATICQUALITYFLAG\n"
            "          NUM_VAL              = 1\n"
            '          CLASS                = "3"\n'
            '          VALUE                = "Not Investigated"\n'
            "        END_OBJECT             = AUTOMATICQUALITYFLAG\n"
            "\n"
            "        OBJECT                 = OPERATIONALQUALITYFLAGEXPLANATION\n"
            "          NUM_VAL              = 1\n"
            '          VALUE                = "This granule passed operational tests that were administered by the OMI SIPS.  QA metadata was extracted and the file was successfully read using standard HDF-EOS utilities."\n'
            '          CLASS                = "3"\n'
            "        END_OBJECT             = OPERATIONALQUALITYFLAGEXPLANATION\n"
            "\n"
            "        OBJECT                 = OPERATIONALQUALITYFLAG\n"
            "          NUM_VAL              = 1\n"
            '          VALUE                = "Passed"\n'
            '          CLASS                = "3"\n'
            "        END_OBJECT             = OPERATIONALQUALITYFLAG\n"
            "\n"
            "        OBJECT                 = SCIENCEQUALITYFLAGEXPLANATION\n"
            "          NUM_VAL              = 1\n"
            '          VALUE                = "An updated science quality flag and explanation is put in the product .met file when a granule has been evaluated.  The flag value in this file, Not Investigated, is an automatic default that is put into every granule during production."\n'
            '          CLASS                = "3"\n'
            "        END_OBJECT             = SCIENCEQUALITYFLAGEXPLANATION\n"
            "\n"
            "      END_GROUP              = QAFLAGS\n"
            "\n"
            "      GROUP                  = QASTATS\n"
            '        CLASS                = "3"\n'
            "\n"
            "        OBJECT                 = QAPERCENTMISSINGDATA\n"
            "          NUM_VAL              = 1\n"
            '          CLASS                = "3"\n'
            "          VALUE                = 0\n"
            "        END_OBJECT             = QAPERCENTMISSINGDATA\n"
            "\n"
            "        OBJECT                 = QAPERCENTOUTOFBOUNDSDATA\n"
            "          NUM_VAL              = 1\n"
            '          CLASS                = "3"\n'
            "          VALUE                = 0\n"
            "        END_OBJECT             = QAPERCENTOUTOFBOUNDSDATA\n"
            "\n"
            "      END_GROUP              = QASTATS\n"
            "\n"
            "      OBJECT                 = PARAMETERNAME\n"
            '        CLASS                = "3"\n'
            "        NUM_VAL              = 1\n"
            '        VALUE                = "OMI Groundpixel Corner Coordinates VIS"\n'
            "      END_OBJECT             = PARAMETERNAME\n"
            "\n"
            "    END_OBJECT             = MEASUREDPARAMETERCONTAINER\n"
            "\n"
            "  END_GROUP              = MEASUREDPARAMETER\n"
            "\n"
            "  GROUP                  = ORBITCALCULATEDSPATIALDOMAIN\n"
            "\n"
            "    OBJECT                 = ORBITCALCULATEDSPATIALDOMAINCONTAINER\n"
            '      CLASS                = "1"\n'
            "\n"
            "      OBJECT                 = EQUATORCROSSINGDATE\n"
            '        CLASS                = "1"\n'
            "        NUM_VAL              = 1\n"
            '        VALUE                = "2004-10-01"\n'
            "      END_OBJECT             = EQUATORCROSSINGDATE\n"
            "\n"
            "      OBJECT                 = EQUATORCROSSINGTIME\n"
            '        CLASS                = "1"\n'
            "        NUM_VAL              = 1\n"
            '        VALUE                = "15:42:19.000000"\n'
            "      END_OBJECT             = EQUATORCROSSINGTIME\n"
            "\n"
            "      OBJECT                 = ORBITNUMBER\n"
            '        CLASS                = "1"\n'
            "        NUM_VAL              = 1\n"
            "        VALUE                = 1141\n"
            "      END_OBJECT             = ORBITNUMBER\n"
            "\n"
            "      OBJECT                 = EQUATORCROSSINGLONGITUDE\n"
            '        CLASS                = "1"\n'
            "        NUM_VAL              = 1\n"
            "        VALUE                = -30.06\n"
            "      END_OBJECT             = EQUATORCROSSINGLONGITUDE\n"
            "\n"
            "    END_OBJECT             = ORBITCALCULATEDSPATIALDOMAINCONTAINER\n"
            "\n"
            "  END_GROUP              = ORBITCALCULATEDSPATIALDOMAIN\n"
            "\n"
            "  GROUP                  = COLLECTIONDESCRIPTIONCLASS\n"
            "\n"
            "    OBJECT                 = VERSIONID\n"
            "      NUM_VAL              = 1\n"
            "      VALUE                = (3)\n"
            "    END_OBJECT             = VERSIONID\n"
            "\n"
            "    OBJECT                 = SHORTNAME\n"
            "      NUM_VAL              = 1\n"
            '      VALUE                = "OMPIXCOR"\n'
            "    END_OBJECT             = SHORTNAME\n"
            "\n"
            "  END_GROUP              = COLLECTIONDESCRIPTIONCLASS\n"
            "\n"
            "  GROUP                  = INPUTGRANULE\n"
            "\n"
            "    OBJECT                 = INPUTPOINTER\n"
            "      NUM_VAL              = 20\n"
            '      VALUE                = ("OMI-Aura_L1-OML1BRUG_2004m1001t1452-o01141_v003-2011m0110t121517-p1.he4", "OMI-Aura_L1-OML1BRVG_2004m1001t1452-o01141_v003-2011m0110t121709-p1.he4")\n'
            "    END_OBJECT             = INPUTPOINTER\n"
            "\n"
            "  END_GROUP              = INPUTGRANULE\n"
            "\n"
            "  GROUP                  = RANGEDATETIME\n"
            "\n"
            "    OBJECT                 = RANGEENDINGDATE\n"
            "      NUM_VAL              = 1\n"
            '      VALUE                = "2004-10-01"\n'
            "    END_OBJECT             = RANGEENDINGDATE\n"
            "\n"
            "    OBJECT                 = RANGEENDINGTIME\n"
            "      NUM_VAL              = 1\n"
            '      VALUE                = "16:31:50.000000"\n'
            "    END_OBJECT             = RANGEENDINGTIME\n"
            "\n"
            "    OBJECT                 = RANGEBEGINNINGDATE\n"
            "      NUM_VAL              = 1\n"
            '      VALUE                = "2004-10-01"\n'
            "    END_OBJECT             = RANGEBEGINNINGDATE\n"
            "\n"
            "    OBJECT                 = RANGEBEGINNINGTIME\n"
            "      NUM_VAL              = 1\n"
            '      VALUE                = "14:52:58.000000"\n'
            "    END_OBJECT             = RANGEBEGINNINGTIME\n"
            "\n"
            "  END_GROUP              = RANGEDATETIME\n"
            "\n"
            "  GROUP                  = PGEVERSIONCLASS\n"
            "\n"
            "    OBJECT                 = PGEVERSION\n"
            "      NUM_VAL              = 1\n"
            '      VALUE                = ("1.2.7")\n'
            "    END_OBJECT             = PGEVERSION\n"
            "\n"
            "  END_GROUP              = PGEVERSIONCLASS\n"
            "\n"
            "  GROUP                  = ASSOCIATEDPLATFORMINSTRUMENTSENSOR\n"
            "\n"
            "    OBJECT                 = ASSOCIATEDPLATFORMINSTRUMENTSENSORCONTAINER\n"
            '      CLASS                = "1"\n'
            "\n"
            "      OBJECT                 = ASSOCIATEDSENSORSHORTNAME\n"
            '        CLASS                = "1"\n'
            "        NUM_VAL              = 1\n"
            '        VALUE                = "CCD Ultra Violet,CCD Visible"\n'
            "      END_OBJECT             = ASSOCIATEDSENSORSHORTNAME\n"
            "\n"
            "      OBJECT                 = ASSOCIATEDPLATFORMSHORTNAME\n"
            '        CLASS                = "1"\n'
            "        NUM_VAL              = 1\n"
            '        VALUE                = "Aura"\n'
            "      END_OBJECT             = ASSOCIATEDPLATFORMSHORTNAME\n"
            "\n"
            "      OBJECT                 = OPERATIONMODE\n"
            '        CLASS                = "1"\n'
            "        NUM_VAL              = 1\n"
            '        VALUE                = ("Test")\n'
            "      END_OBJECT             = OPERATIONMODE\n"
            "\n"
            "      OBJECT                 = ASSOCIATEDINSTRUMENTSHORTNAME\n"
            '        CLASS                = "1"\n'
            "        NUM_VAL              = 1\n"
            '        VALUE                = "OMI"\n'
            "      END_OBJECT             = ASSOCIATEDINSTRUMENTSHORTNAME\n"
            "\n"
            "    END_OBJECT             = ASSOCIATEDPLATFORMINSTRUMENTSENSORCONTAINER\n"
            "\n"
            "  END_GROUP              = ASSOCIATEDPLATFORMINSTRUMENTSENSOR\n"
            "\n"
            "  GROUP                  = ADDITIONALATTRIBUTES\n"
            "\n"
            "    OBJECT                 = ADDITIONALATTRIBUTESCONTAINER\n"
            '      CLASS                = "1"\n'
            "\n"
            "      OBJECT                 = ADDITIONALATTRIBUTENAME\n"
            '        CLASS                = "1"\n'
            "        NUM_VAL              = 1\n"
            '        VALUE                = "NrMeasurements"\n'
            "      END_OBJECT             = ADDITIONALATTRIBUTENAME\n"
            "\n"
            "      GROUP                  = INFORMATIONCONTENT\n"
            '        CLASS                = "1"\n'
            "\n"
            "        OBJECT                 = PARAMETERVALUE\n"
            "          NUM_VAL              = 1\n"
            '          CLASS                = "1"\n'
            '          VALUE                = "1643"\n'
            "        END_OBJECT             = PARAMETERVALUE\n"
            "\n"
            "      END_GROUP              = INFORMATIONCONTENT\n"
            "\n"
            "    END_OBJECT             = ADDITIONALATTRIBUTESCONTAINER\n"
            "\n"
            "    OBJECT                 = ADDITIONALATTRIBUTESCONTAINER\n"
            '      CLASS                = "2"\n'
            "\n"
            "      OBJECT                 = ADDITIONALATTRIBUTENAME\n"
            '        CLASS                = "2"\n'
            "        NUM_VAL              = 1\n"
            '        VALUE                = "NrZoom"\n'
            "      END_OBJECT             = ADDITIONALATTRIBUTENAME\n"
            "\n"
            "      GROUP                  = INFORMATIONCONTENT\n"
            '        CLASS                = "2"\n'
            "\n"
            "        OBJECT                 = PARAMETERVALUE\n"
            "          NUM_VAL              = 1\n"
            '          CLASS                = "2"\n'
            '          VALUE                = "0"\n'
            "        END_OBJECT             = PARAMETERVALUE\n"
            "\n"
            "      END_GROUP              = INFORMATIONCONTENT\n"
            "\n"
            "    END_OBJECT             = ADDITIONALATTRIBUTESCONTAINER\n"
            "\n"
            "    OBJECT                 = ADDITIONALATTRIBUTESCONTAINER\n"
            '      CLASS                = "3"\n'
            "\n"
            "      OBJECT                 = ADDITIONALATTRIBUTENAME\n"
            '        CLASS                = "3"\n'
            "        NUM_VAL              = 1\n"
            '        VALUE                = "NrSpatialZoom"\n'
            "      END_OBJECT             = ADDITIONALATTRIBUTENAME\n"
            "\n"
            "      GROUP                  = INFORMATIONCONTENT\n"
            '        CLASS                = "3"\n'
            "\n"
            "        OBJECT                 = PARAMETERVALUE\n"
            "          NUM_VAL              = 1\n"
            '          CLASS                = "3"\n'
            '          VALUE                = "0"\n'
            "        END_OBJECT             = PARAMETERVALUE\n"
            "\n"
            "      END_GROUP              = INFORMATIONCONTENT\n"
            "\n"
            "    END_OBJECT             = ADDITIONALATTRIBUTESCONTAINER\n"
            "\n"
            "    OBJECT                 = ADDITIONALATTRIBUTESCONTAINER\n"
            '      CLASS                = "4"\n'
            "\n"
            "      OBJECT                 = ADDITIONALATTRIBUTENAME\n"
            '        CLASS                = "4"\n'
            "        NUM_VAL              = 1\n"
            '        VALUE                = "NrSpectralZoom"\n'
            "      END_OBJECT             = ADDITIONALATTRIBUTENAME\n"
            "\n"
            "      GROUP                  = INFORMATIONCONTENT\n"
            '        CLASS                = "4"\n'
            "\n"
            "        OBJECT                 = PARAMETERVALUE\n"
            "          NUM_VAL              = 1\n"
            '          CLASS                = "4"\n'
            '          VALUE                = "0"\n'
            "        END_OBJECT             = PARAMETERVALUE\n"
            "\n"
            "      END_GROUP              = INFORMATIONCONTENT\n"
            "\n"
            "    END_OBJECT             = ADDITIONALATTRIBUTESCONTAINER\n"
            "\n"
            "    OBJECT                 = ADDITIONALATTRIBUTESCONTAINER\n"
            '      CLASS                = "5"\n'
            "\n"
            "      OBJECT                 = ADDITIONALATTRIBUTENAME\n"
            '        CLASS                = "5"\n'
            "        NUM_VAL              = 1\n"
            '        VALUE                = "ExpeditedData"\n'
            "      END_OBJECT             = ADDITIONALATTRIBUTENAME\n"
            "\n"
            "      GROUP                  = INFORMATIONCONTENT\n"
            '        CLASS                = "5"\n'
            "\n"
            "        OBJECT                 = PARAMETERVALUE\n"
            "          NUM_VAL              = 1\n"
            '          CLASS                = "5"\n'
            '          VALUE                = "FALSE"\n'
            "        END_OBJECT             = PARAMETERVALUE\n"
            "\n"
            "      END_GROUP              = INFORMATIONCONTENT\n"
            "\n"
            "    END_OBJECT             = ADDITIONALATTRIBUTESCONTAINER\n"
            "\n"
            "    OBJECT                 = ADDITIONALATTRIBUTESCONTAINER\n"
            '      CLASS                = "6"\n'
            "\n"
            "      OBJECT                 = ADDITIONALATTRIBUTENAME\n"
            '        CLASS                = "6"\n'
            "        NUM_VAL              = 1\n"
            '        VALUE                = "SouthAtlanticAnomalyCrossing"\n'
            "      END_OBJECT             = ADDITIONALATTRIBUTENAME\n"
            "\n"
            "      GROUP                  = INFORMATIONCONTENT\n"
            '        CLASS                = "6"\n'
            "\n"
            "        OBJECT                 = PARAMETERVALUE\n"
            "          NUM_VAL              = 1\n"
            '          CLASS                = "6"\n'
            '          VALUE                = "TRUE"\n'
            "        END_OBJECT             = PARAMETERVALUE\n"
            "\n"
            "      END_GROUP              = INFORMATIONCONTENT\n"
            "\n"
            "    END_OBJECT             = ADDITIONALATTRIBUTESCONTAINER\n"
            "\n"
            "    OBJECT                 = ADDITIONALATTRIBUTESCONTAINER\n"
            '      CLASS                = "7"\n'
            "\n"
            "      OBJECT                 = ADDITIONALATTRIBUTENAME\n"
            '        CLASS                = "7"\n'
            "        NUM_VAL              = 1\n"
            '        VALUE                = "SpacecraftManeuverFlag"\n'
            "      END_OBJECT             = ADDITIONALATTRIBUTENAME\n"
            "\n"
            "      GROUP                  = INFORMATIONCONTENT\n"
            '        CLASS                = "7"\n'
            "\n"
            "        OBJECT                 = PARAMETERVALUE\n"
            "          NUM_VAL              = 1\n"
            '          CLASS                = "7"\n'
            '          VALUE                = "FALSE"\n'
            "        END_OBJECT             = PARAMETERVALUE\n"
            "\n"
            "      END_GROUP              = INFORMATIONCONTENT\n"
            "\n"
            "    END_OBJECT             = ADDITIONALATTRIBUTESCONTAINER\n"
            "\n"
            "    OBJECT                 = ADDITIONALATTRIBUTESCONTAINER\n"
            '      CLASS                = "8"\n'
            "\n"
            "      OBJECT                 = ADDITIONALATTRIBUTENAME\n"
            '        CLASS                = "8"\n'
            "        NUM_VAL              = 1\n"
            '        VALUE                = "SolarEclipse"\n'
            "      END_OBJECT             = ADDITIONALATTRIBUTENAME\n"
            "\n"
            "      GROUP                  = INFORMATIONCONTENT\n"
            '        CLASS                = "8"\n'
            "\n"
            "        OBJECT                 = PARAMETERVALUE\n"
            "          NUM_VAL              = 1\n"
            '          CLASS                = "8"\n'
            '          VALUE                = "FALSE"\n'
            "        END_OBJECT             = PARAMETERVALUE\n"
            "\n"
            "      END_GROUP              = INFORMATIONCONTENT\n"
            "\n"
            "    END_OBJECT             = ADDITIONALATTRIBUTESCONTAINER\n"
            "\n"
            "    OBJECT                 = ADDITIONALATTRIBUTESCONTAINER\n"
            '      CLASS                = "9"\n'
            "\n"
            "      OBJECT                 = ADDITIONALATTRIBUTENAME\n"
            '        CLASS                = "9"\n'
            "        NUM_VAL              = 1\n"
            '        VALUE                = "InstrumentConfigurationIDs"\n'
            "      END_OBJECT             = ADDITIONALATTRIBUTENAME\n"
            "\n"
            "      GROUP                  = INFORMATIONCONTENT\n"
            '        CLASS                = "9"\n'
            "\n"
            "        OBJECT                 = PARAMETERVALUE\n"
            "          NUM_VAL              = 4\n"
            '          CLASS                = "9"\n'
            '          VALUE                = ("0", "1", "2", "7")\n'
            "        END_OBJECT             = PARAMETERVALUE\n"
            "\n"
            "      END_GROUP              = INFORMATIONCONTENT\n"
            "\n"
            "    END_OBJECT             = ADDITIONALATTRIBUTESCONTAINER\n"
            "\n"
            "    OBJECT                 = ADDITIONALATTRIBUTESCONTAINER\n"
            '      CLASS                = "10"\n'
            "\n"
            "      OBJECT                 = ADDITIONALATTRIBUTENAME\n"
            '        CLASS                = "10"\n'
            "        NUM_VAL              = 1\n"
            '        VALUE                = "MasterClockPeriods"\n'
            "      END_OBJECT             = ADDITIONALATTRIBUTENAME\n"
            "\n"
            "      GROUP                  = INFORMATIONCONTENT\n"
            '        CLASS                = "10"\n'
            "\n"
            "        OBJECT                 = PARAMETERVALUE\n"
            "          NUM_VAL              = 1\n"
            '          CLASS                = "10"\n'
            '          VALUE                = "2.000400"\n'
            "        END_OBJECT             = PARAMETERVALUE\n"
            "\n"
            "      END_GROUP              = INFORMATIONCONTENT\n"
            "\n"
            "    END_OBJECT             = ADDITIONALATTRIBUTESCONTAINER\n"
            "\n"
            "    OBJECT                 = ADDITIONALATTRIBUTESCONTAINER\n"
            '      CLASS                = "11"\n'
            "\n"
            "      OBJECT                 = ADDITIONALATTRIBUTENAME\n"
            '        CLASS                = "11"\n'
            "        NUM_VAL              = 1\n"
            '        VALUE                = "ExposureTimes"\n'
            "      END_OBJECT             = ADDITIONALATTRIBUTENAME\n"
            "\n"
            "      GROUP                  = INFORMATIONCONTENT\n"
            '        CLASS                = "11"\n'
            "\n"
            "        OBJECT                 = PARAMETERVALUE\n"
            "          NUM_VAL              = 3\n"
            '          CLASS                = "11"\n'
            '          VALUE                = ("1.000200", "0.500100", "0.400080")\n'
            "        END_OBJECT             = PARAMETERVALUE\n"
            "\n"
            "      END_GROUP              = INFORMATIONCONTENT\n"
            "\n"
            "    END_OBJECT             = ADDITIONALATTRIBUTESCONTAINER\n"
            "\n"
            "    OBJECT                 = ADDITIONALATTRIBUTESCONTAINER\n"
            '      CLASS                = "12"\n'
            "\n"
            "      OBJECT                 = ADDITIONALATTRIBUTENAME\n"
            '        CLASS                = "12"\n'
            "        NUM_VAL              = 1\n"
            '        VALUE                = "PathNr"\n'
            "      END_OBJECT             = ADDITIONALATTRIBUTENAME\n"
            "\n"
            "      GROUP                  = INFORMATIONCONTENT\n"
            '        CLASS                = "12"\n'
            "\n"
            "        OBJECT                 = PARAMETERVALUE\n"
            "          NUM_VAL              = 1\n"
            '          CLASS                = "12"\n'
            '          VALUE                = "213"\n'
            "        END_OBJECT             = PARAMETERVALUE\n"
            "\n"
            "      END_GROUP              = INFORMATIONCONTENT\n"
            "\n"
            "    END_OBJECT             = ADDITIONALATTRIBUTESCONTAINER\n"
            "\n"
            "    OBJECT                 = ADDITIONALATTRIBUTESCONTAINER\n"
            '      CLASS                = "13"\n'
            "\n"
            "      OBJECT                 = ADDITIONALATTRIBUTENAME\n"
            '        CLASS                = "13"\n'
            "        NUM_VAL              = 1\n"
            '        VALUE                = "StartBlockNr"\n'
            "      END_OBJECT             = ADDITIONALATTRIBUTENAME\n"
            "\n"
            "      GROUP                  = INFORMATIONCONTENT\n"
            '        CLASS                = "13"\n'
            "\n"
            "        OBJECT                 = PARAMETERVALUE\n"
            "          NUM_VAL              = 1\n"
            '          CLASS                = "13"\n'
            '          VALUE                = "12"\n'
            "        END_OBJECT             = PARAMETERVALUE\n"
            "\n"
            "      END_GROUP              = INFORMATIONCONTENT\n"
            "\n"
            "    END_OBJECT             = ADDITIONALATTRIBUTESCONTAINER\n"
            "\n"
            "    OBJECT                 = ADDITIONALATTRIBUTESCONTAINER\n"
            '      CLASS                = "14"\n'
            "\n"
            "      OBJECT                 = ADDITIONALATTRIBUTENAME\n"
            '        CLASS                = "14"\n'
            "        NUM_VAL              = 1\n"
            '        VALUE                = "EndBlockNr"\n'
            "      END_OBJECT             = ADDITIONALATTRIBUTENAME\n"
            "\n"
            "      GROUP                  = INFORMATIONCONTENT\n"
            '        CLASS                = "14"\n'
            "\n"
            "        OBJECT                 = PARAMETERVALUE\n"
            "          NUM_VAL              = 1\n"
            '          CLASS                = "14"\n'
            '          VALUE                = "39"\n'
            "        END_OBJECT             = PARAMETERVALUE\n"
            "\n"
            "      END_GROUP              = INFORMATIONCONTENT\n"
            "\n"
            "    END_OBJECT             = ADDITIONALATTRIBUTESCONTAINER\n"
            "\n"
            "  END_GROUP              = ADDITIONALATTRIBUTES\n"
            "\n"
            "END_GROUP              = INVENTORYMETADATA\n"
            "\n"
            "END\n"
        )
        hdfeos_information.create_dataset(
            "CoreMetadata",
            data=np.bytes_(hdfeos_information_coremetadata_text),
            dtype=np.dtype("|S65535"),
        )

        # HDFEOS INFORMATION/StructMetadata.0
        hdfeos_information_structmetadata_0_text = (
            "GROUP=SwathStructure\n"
            "	GROUP=SWATH_1\n"
            '		SwathName="OMI Ground Pixel Corners UV-1"\n'
            "		GROUP=Dimension\n"
            "			OBJECT=Dimension_1\n"
            '				DimensionName="nTimes"\n'
            "				Size=1643\n"
            "			END_OBJECT=Dimension_1\n"
            "			OBJECT=Dimension_2\n"
            '				DimensionName="nXtrack"\n'
            "				Size=30\n"
            "			END_OBJECT=Dimension_2\n"
            "			OBJECT=Dimension_3\n"
            '				DimensionName="nCorners"\n'
            "				Size=4\n"
            "			END_OBJECT=Dimension_3\n"
            "			OBJECT=Dimension_4\n"
            '				DimensionName="nUTC"\n'
            "				Size=27\n"
            "			END_OBJECT=Dimension_4\n"
            "		END_GROUP=Dimension\n"
            "		GROUP=DimensionMap\n"
            "		END_GROUP=DimensionMap\n"
            "		GROUP=IndexDimensionMap\n"
            "		END_GROUP=IndexDimensionMap\n"
            "		GROUP=GeoField\n"
            "			OBJECT=GeoField_1\n"
            '				GeoFieldName="Latitude"\n'
            "				DataType=H5T_NATIVE_FLOAT\n"
            '				DimList=("nTimes","nXtrack")\n'
            '				MaxdimList=("nTimes","nXtrack")\n'
            "				CompressionType=HE5_HDFE_COMP_SHUF_DEFLATE\n"
            "				DeflateLevel=9\n"
            "			END_OBJECT=GeoField_1\n"
            "			OBJECT=GeoField_2\n"
            '				GeoFieldName="Longitude"\n'
            "				DataType=H5T_NATIVE_FLOAT\n"
            '				DimList=("nTimes","nXtrack")\n'
            '				MaxdimList=("nTimes","nXtrack")\n'
            "				CompressionType=HE5_HDFE_COMP_SHUF_DEFLATE\n"
            "				DeflateLevel=9\n"
            "			END_OBJECT=GeoField_2\n"
            "			OBJECT=GeoField_3\n"
            '				GeoFieldName="SpacecraftAltitude"\n'
            "				DataType=H5T_NATIVE_FLOAT\n"
            '				DimList=("nTimes")\n'
            '				MaxdimList=("nTimes")\n'
            "			END_OBJECT=GeoField_3\n"
            "			OBJECT=GeoField_4\n"
            '				GeoFieldName="SpacecraftLatitude"\n'
            "				DataType=H5T_NATIVE_FLOAT\n"
            '				DimList=("nTimes")\n'
            '				MaxdimList=("nTimes")\n'
            "			END_OBJECT=GeoField_4\n"
            "			OBJECT=GeoField_5\n"
            '				GeoFieldName="SpacecraftLongitude"\n'
            "				DataType=H5T_NATIVE_FLOAT\n"
            '				DimList=("nTimes")\n'
            '				MaxdimList=("nTimes")\n'
            "			END_OBJECT=GeoField_5\n"
            "			OBJECT=GeoField_6\n"
            '				GeoFieldName="Time"\n'
            "				DataType=H5T_NATIVE_DOUBLE\n"
            '				DimList=("nTimes")\n'
            '				MaxdimList=("nTimes")\n'
            "			END_OBJECT=GeoField_6\n"
            "			OBJECT=GeoField_7\n"
            '				GeoFieldName="TimeUTC"\n'
            "				DataType=H5T_NATIVE_SCHAR\n"
            '				DimList=("nTimes","nUTC")\n'
            '				MaxdimList=("nTimes","nUTC")\n'
            "			END_OBJECT=GeoField_7\n"
            "		END_GROUP=GeoField\n"
            "		GROUP=DataField\n"
            "			OBJECT=DataField_1\n"
            '				DataFieldName="FoV75Area"\n'
            "				DataType=H5T_NATIVE_FLOAT\n"
            '				DimList=("nXtrack")\n'
            '				MaxdimList=("nXtrack")\n'
            "			END_OBJECT=DataField_1\n"
            "			OBJECT=DataField_2\n"
            '				DataFieldName="FoV75CornerLatitude"\n'
            "				DataType=H5T_NATIVE_FLOAT\n"
            '				DimList=("nCorners","nTimes","nXtrack")\n'
            '				MaxdimList=("nCorners","nTimes","nXtrack")\n'
            "				CompressionType=HE5_HDFE_COMP_SHUF_DEFLATE\n"
            "				DeflateLevel=9\n"
            "			END_OBJECT=DataField_2\n"
            "			OBJECT=DataField_3\n"
            '				DataFieldName="FoV75CornerLongitude"\n'
            "				DataType=H5T_NATIVE_FLOAT\n"
            '				DimList=("nCorners","nTimes","nXtrack")\n'
            '				MaxdimList=("nCorners","nTimes","nXtrack")\n'
            "				CompressionType=HE5_HDFE_COMP_SHUF_DEFLATE\n"
            "				DeflateLevel=9\n"
            "			END_OBJECT=DataField_3\n"
            "			OBJECT=DataField_4\n"
            '				DataFieldName="TiledArea"\n'
            "				DataType=H5T_NATIVE_FLOAT\n"
            '				DimList=("nXtrack")\n'
            '				MaxdimList=("nXtrack")\n'
            "			END_OBJECT=DataField_4\n"
            "			OBJECT=DataField_5\n"
            '				DataFieldName="TiledCornerLatitude"\n'
            "				DataType=H5T_NATIVE_FLOAT\n"
            '				DimList=("nCorners","nTimes","nXtrack")\n'
            '				MaxdimList=("nCorners","nTimes","nXtrack")\n'
            "				CompressionType=HE5_HDFE_COMP_SHUF_DEFLATE\n"
            "				DeflateLevel=9\n"
            "			END_OBJECT=DataField_5\n"
            "			OBJECT=DataField_6\n"
            '				DataFieldName="TiledCornerLongitude"\n'
            "				DataType=H5T_NATIVE_FLOAT\n"
            '				DimList=("nCorners","nTimes","nXtrack")\n'
            '				MaxdimList=("nCorners","nTimes","nXtrack")\n'
            "				CompressionType=HE5_HDFE_COMP_SHUF_DEFLATE\n"
            "				DeflateLevel=9\n"
            "			END_OBJECT=DataField_6\n"
            "		END_GROUP=DataField\n"
            "		GROUP=ProfileField\n"
            "		END_GROUP=ProfileField\n"
            "		GROUP=MergedFields\n"
            "		END_GROUP=MergedFields\n"
            "	END_GROUP=SWATH_1\n"
            "	GROUP=SWATH_2\n"
            '		SwathName="OMI Ground Pixel Corners UV-2"\n'
            "		GROUP=Dimension\n"
            "			OBJECT=Dimension_1\n"
            '				DimensionName="nTimes"\n'
            "				Size=1643\n"
            "			END_OBJECT=Dimension_1\n"
            "			OBJECT=Dimension_2\n"
            '				DimensionName="nXtrack"\n'
            "				Size=60\n"
            "			END_OBJECT=Dimension_2\n"
            "			OBJECT=Dimension_3\n"
            '				DimensionName="nCorners"\n'
            "				Size=4\n"
            "			END_OBJECT=Dimension_3\n"
            "			OBJECT=Dimension_4\n"
            '				DimensionName="nUTC"\n'
            "				Size=27\n"
            "			END_OBJECT=Dimension_4\n"
            "		END_GROUP=Dimension\n"
            "		GROUP=DimensionMap\n"
            "		END_GROUP=DimensionMap\n"
            "		GROUP=IndexDimensionMap\n"
            "		END_GROUP=IndexDimensionMap\n"
            "		GROUP=GeoField\n"
            "			OBJECT=GeoField_1\n"
            '				GeoFieldName="Latitude"\n'
            "				DataType=H5T_NATIVE_FLOAT\n"
            '				DimList=("nTimes","nXtrack")\n'
            '				MaxdimList=("nTimes","nXtrack")\n'
            "				CompressionType=HE5_HDFE_COMP_SHUF_DEFLATE\n"
            "				DeflateLevel=9\n"
            "			END_OBJECT=GeoField_1\n"
            "			OBJECT=GeoField_2\n"
            '				GeoFieldName="Longitude"\n'
            "				DataType=H5T_NATIVE_FLOAT\n"
            '				DimList=("nTimes","nXtrack")\n'
            '				MaxdimList=("nTimes","nXtrack")\n'
            "				CompressionType=HE5_HDFE_COMP_SHUF_DEFLATE\n"
            "				DeflateLevel=9\n"
            "			END_OBJECT=GeoField_2\n"
            "			OBJECT=GeoField_3\n"
            '				GeoFieldName="SpacecraftAltitude"\n'
            "				DataType=H5T_NATIVE_FLOAT\n"
            '				DimList=("nTimes")\n'
            '				MaxdimList=("nTimes")\n'
            "			END_OBJECT=GeoField_3\n"
            "			OBJECT=GeoField_4\n"
            '				GeoFieldName="SpacecraftLatitude"\n'
            "				DataType=H5T_NATIVE_FLOAT\n"
            '				DimList=("nTimes")\n'
            '				MaxdimList=("nTimes")\n'
            "			END_OBJECT=GeoField_4\n"
            "			OBJECT=GeoField_5\n"
            '				GeoFieldName="SpacecraftLongitude"\n'
            "				DataType=H5T_NATIVE_FLOAT\n"
            '				DimList=("nTimes")\n'
            '				MaxdimList=("nTimes")\n'
            "			END_OBJECT=GeoField_5\n"
            "			OBJECT=GeoField_6\n"
            '				GeoFieldName="Time"\n'
            "				DataType=H5T_NATIVE_DOUBLE\n"
            '				DimList=("nTimes")\n'
            '				MaxdimList=("nTimes")\n'
            "			END_OBJECT=GeoField_6\n"
            "			OBJECT=GeoField_7\n"
            '				GeoFieldName="TimeUTC"\n'
            "				DataType=H5T_NATIVE_SCHAR\n"
            '				DimList=("nTimes","nUTC")\n'
            '				MaxdimList=("nTimes","nUTC")\n'
            "			END_OBJECT=GeoField_7\n"
            "		END_GROUP=GeoField\n"
            "		GROUP=DataField\n"
            "			OBJECT=DataField_1\n"
            '				DataFieldName="FoV75Area"\n'
            "				DataType=H5T_NATIVE_FLOAT\n"
            '				DimList=("nXtrack")\n'
            '				MaxdimList=("nXtrack")\n'
            "			END_OBJECT=DataField_1\n"
            "			OBJECT=DataField_2\n"
            '				DataFieldName="FoV75CornerLatitude"\n'
            "				DataType=H5T_NATIVE_FLOAT\n"
            '				DimList=("nCorners","nTimes","nXtrack")\n'
            '				MaxdimList=("nCorners","nTimes","nXtrack")\n'
            "				CompressionType=HE5_HDFE_COMP_SHUF_DEFLATE\n"
            "				DeflateLevel=9\n"
            "			END_OBJECT=DataField_2\n"
            "			OBJECT=DataField_3\n"
            '				DataFieldName="FoV75CornerLongitude"\n'
            "				DataType=H5T_NATIVE_FLOAT\n"
            '				DimList=("nCorners","nTimes","nXtrack")\n'
            '				MaxdimList=("nCorners","nTimes","nXtrack")\n'
            "				CompressionType=HE5_HDFE_COMP_SHUF_DEFLATE\n"
            "				DeflateLevel=9\n"
            "			END_OBJECT=DataField_3\n"
            "			OBJECT=DataField_4\n"
            '				DataFieldName="TiledArea"\n'
            "				DataType=H5T_NATIVE_FLOAT\n"
            '				DimList=("nXtrack")\n'
            '				MaxdimList=("nXtrack")\n'
            "			END_OBJECT=DataField_4\n"
            "			OBJECT=DataField_5\n"
            '				DataFieldName="TiledCornerLatitude"\n'
            "				DataType=H5T_NATIVE_FLOAT\n"
            '				DimList=("nCorners","nTimes","nXtrack")\n'
            '				MaxdimList=("nCorners","nTimes","nXtrack")\n'
            "				CompressionType=HE5_HDFE_COMP_SHUF_DEFLATE\n"
            "				DeflateLevel=9\n"
            "			END_OBJECT=DataField_5\n"
            "			OBJECT=DataField_6\n"
            '				DataFieldName="TiledCornerLongitude"\n'
            "				DataType=H5T_NATIVE_FLOAT\n"
            '				DimList=("nCorners","nTimes","nXtrack")\n'
            '				MaxdimList=("nCorners","nTimes","nXtrack")\n'
            "				CompressionType=HE5_HDFE_COMP_SHUF_DEFLATE\n"
            "				DeflateLevel=9\n"
            "			END_OBJECT=DataField_6\n"
            "		END_GROUP=DataField\n"
            "		GROUP=ProfileField\n"
            "		END_GROUP=ProfileField\n"
            "		GROUP=MergedFields\n"
            "		END_GROUP=MergedFields\n"
            "	END_GROUP=SWATH_2\n"
            "	GROUP=SWATH_3\n"
            '		SwathName="OMI Ground Pixel Corners VIS"\n'
            "		GROUP=Dimension\n"
            "			OBJECT=Dimension_1\n"
            '				DimensionName="nTimes"\n'
            "				Size=1643\n"
            "			END_OBJECT=Dimension_1\n"
            "			OBJECT=Dimension_2\n"
            '				DimensionName="nXtrack"\n'
            "				Size=60\n"
            "			END_OBJECT=Dimension_2\n"
            "			OBJECT=Dimension_3\n"
            '				DimensionName="nCorners"\n'
            "				Size=4\n"
            "			END_OBJECT=Dimension_3\n"
            "			OBJECT=Dimension_4\n"
            '				DimensionName="nUTC"\n'
            "				Size=27\n"
            "			END_OBJECT=Dimension_4\n"
            "		END_GROUP=Dimension\n"
            "		GROUP=DimensionMap\n"
            "		END_GROUP=DimensionMap\n"
            "		GROUP=IndexDimensionMap\n"
            "		END_GROUP=IndexDimensionMap\n"
            "		GROUP=GeoField\n"
            "			OBJECT=GeoField_1\n"
            '				GeoFieldName="Latitude"\n'
            "				DataType=H5T_NATIVE_FLOAT\n"
            '				DimList=("nTimes","nXtrack")\n'
            '				MaxdimList=("nTimes","nXtrack")\n'
            "				CompressionType=HE5_HDFE_COMP_SHUF_DEFLATE\n"
            "				DeflateLevel=9\n"
            "			END_OBJECT=GeoField_1\n"
            "			OBJECT=GeoField_2\n"
            '				GeoFieldName="Longitude"\n'
            "				DataType=H5T_NATIVE_FLOAT\n"
            '				DimList=("nTimes","nXtrack")\n'
            '				MaxdimList=("nTimes","nXtrack")\n'
            "				CompressionType=HE5_HDFE_COMP_SHUF_DEFLATE\n"
            "				DeflateLevel=9\n"
            "			END_OBJECT=GeoField_2\n"
            "			OBJECT=GeoField_3\n"
            '				GeoFieldName="SpacecraftAltitude"\n'
            "				DataType=H5T_NATIVE_FLOAT\n"
            '				DimList=("nTimes")\n'
            '				MaxdimList=("nTimes")\n'
            "			END_OBJECT=GeoField_3\n"
            "			OBJECT=GeoField_4\n"
            '				GeoFieldName="SpacecraftLatitude"\n'
            "				DataType=H5T_NATIVE_FLOAT\n"
            '				DimList=("nTimes")\n'
            '				MaxdimList=("nTimes")\n'
            "			END_OBJECT=GeoField_4\n"
            "			OBJECT=GeoField_5\n"
            '				GeoFieldName="SpacecraftLongitude"\n'
            "				DataType=H5T_NATIVE_FLOAT\n"
            '				DimList=("nTimes")\n'
            '				MaxdimList=("nTimes")\n'
            "			END_OBJECT=GeoField_5\n"
            "			OBJECT=GeoField_6\n"
            '				GeoFieldName="Time"\n'
            "				DataType=H5T_NATIVE_DOUBLE\n"
            '				DimList=("nTimes")\n'
            '				MaxdimList=("nTimes")\n'
            "			END_OBJECT=GeoField_6\n"
            "			OBJECT=GeoField_7\n"
            '				GeoFieldName="TimeUTC"\n'
            "				DataType=H5T_NATIVE_SCHAR\n"
            '				DimList=("nTimes","nUTC")\n'
            '				MaxdimList=("nTimes","nUTC")\n'
            "			END_OBJECT=GeoField_7\n"
            "		END_GROUP=GeoField\n"
            "		GROUP=DataField\n"
            "			OBJECT=DataField_1\n"
            '				DataFieldName="FoV75Area"\n'
            "				DataType=H5T_NATIVE_FLOAT\n"
            '				DimList=("nXtrack")\n'
            '				MaxdimList=("nXtrack")\n'
            "			END_OBJECT=DataField_1\n"
            "			OBJECT=DataField_2\n"
            '				DataFieldName="FoV75CornerLatitude"\n'
            "				DataType=H5T_NATIVE_FLOAT\n"
            '				DimList=("nCorners","nTimes","nXtrack")\n'
            '				MaxdimList=("nCorners","nTimes","nXtrack")\n'
            "				CompressionType=HE5_HDFE_COMP_SHUF_DEFLATE\n"
            "				DeflateLevel=9\n"
            "			END_OBJECT=DataField_2\n"
            "			OBJECT=DataField_3\n"
            '				DataFieldName="FoV75CornerLongitude"\n'
            "				DataType=H5T_NATIVE_FLOAT\n"
            '				DimList=("nCorners","nTimes","nXtrack")\n'
            '				MaxdimList=("nCorners","nTimes","nXtrack")\n'
            "				CompressionType=HE5_HDFE_COMP_SHUF_DEFLATE\n"
            "				DeflateLevel=9\n"
            "			END_OBJECT=DataField_3\n"
            "			OBJECT=DataField_4\n"
            '				DataFieldName="TiledArea"\n'
            "				DataType=H5T_NATIVE_FLOAT\n"
            '				DimList=("nXtrack")\n'
            '				MaxdimList=("nXtrack")\n'
            "			END_OBJECT=DataField_4\n"
            "			OBJECT=DataField_5\n"
            '				DataFieldName="TiledCornerLatitude"\n'
            "				DataType=H5T_NATIVE_FLOAT\n"
            '				DimList=("nCorners","nTimes","nXtrack")\n'
            '				MaxdimList=("nCorners","nTimes","nXtrack")\n'
            "				CompressionType=HE5_HDFE_COMP_SHUF_DEFLATE\n"
            "				DeflateLevel=9\n"
            "			END_OBJECT=DataField_5\n"
            "			OBJECT=DataField_6\n"
            '				DataFieldName="TiledCornerLongitude"\n'
            "				DataType=H5T_NATIVE_FLOAT\n"
            '				DimList=("nCorners","nTimes","nXtrack")\n'
            '				MaxdimList=("nCorners","nTimes","nXtrack")\n'
            "				CompressionType=HE5_HDFE_COMP_SHUF_DEFLATE\n"
            "				DeflateLevel=9\n"
            "			END_OBJECT=DataField_6\n"
            "		END_GROUP=DataField\n"
            "		GROUP=ProfileField\n"
            "		END_GROUP=ProfileField\n"
            "		GROUP=MergedFields\n"
            "		END_GROUP=MergedFields\n"
            "	END_GROUP=SWATH_3\n"
            "END_GROUP=SwathStructure\n"
            "GROUP=GridStructure\n"
            "END_GROUP=GridStructure\n"
            "GROUP=PointStructure\n"
            "END_GROUP=PointStructure\n"
            "GROUP=ZaStructure\n"
            "END_GROUP=ZaStructure\n"
            "END\n"
        )
        hdfeos_information.create_dataset(
            "StructMetadata.0",
            data=np.bytes_(hdfeos_information_structmetadata_0_text),
            dtype=np.dtype("|S32000"),
        )

    return filepath
