import h5py
import numpy as np
import pytest


@pytest.fixture(scope="session")
def fake_omi_bro_file(tmp_path_factory):
    """
    Generated from OMI-Aura_L2-OMBRO_2020m0116t1207-o82471_v003-2020m0116t182003.he5
    """
    filepath = (
        tmp_path_factory.mktemp("data")
        / "OMI-Aura_L2-OMBRO_2020m0116t1207-o82471_v003-2020m0116t182003.he5"
    )

    dim_0_1 = 2
    dim_0_1997 = 22
    dim_0_1998 = 26
    dim_0_4 = 6
    dim_0_60 = 14
    dim_1_6 = 10
    dim_1_60 = 14
    dim_1_61 = 18

    with h5py.File(filepath, "w") as f:
        # create groups
        hdfeos = f.require_group("HDFEOS")
        hdfeos_additional = f.require_group("HDFEOS/ADDITIONAL")
        hdfeos_additional_file_attributes = f.require_group(
            "HDFEOS/ADDITIONAL/FILE_ATTRIBUTES"
        )
        hdfeos_swaths = f.require_group("HDFEOS/SWATHS")
        hdfeos_swaths_omi_total_column_amount_bro = f.require_group(
            "HDFEOS/SWATHS/OMI Total Column Amount BrO"
        )
        hdfeos_swaths_omi_total_column_amount_bro_data_fields = f.require_group(
            "HDFEOS/SWATHS/OMI Total Column Amount BrO/Data Fields"
        )
        hdfeos_swaths_omi_total_column_amount_bro_geolocation_fields = f.require_group(
            "HDFEOS/SWATHS/OMI Total Column Amount BrO/Geolocation Fields"
        )
        hdfeos_information = f.require_group("HDFEOS INFORMATION")

        # attributes for HDFEOS/ADDITIONAL/FILE_ATTRIBUTES
        hdfeos_additional_file_attributes.attrs["AuthorAffiliation"] = (
            "Smithsonian Astrophysical Observatory"
        )
        hdfeos_additional_file_attributes.attrs["AuthorName"] = "Thomas P. Kurosu"
        hdfeos_additional_file_attributes.attrs["ESDTDescriptorRevision"] = "3.0.4"
        hdfeos_additional_file_attributes.attrs["FittingWindowLimits"] = np.array(
            [
                np.float32(np.float32(317.0)),
                np.float32(np.float32(319.0)),
                np.float32(np.float32(347.5)),
                np.float32(np.float32(349.5)),
                np.float32(np.float32(0.0)),
                np.float32(np.float32(0.0)),
            ],
            dtype=np.float32,
        )
        hdfeos_additional_file_attributes.attrs["GranuleDay"] = np.array(
            [np.int32(np.int32(16))], dtype=np.int32
        )
        hdfeos_additional_file_attributes.attrs["GranuleMonth"] = np.array(
            [np.int32(np.int32(1))], dtype=np.int32
        )
        hdfeos_additional_file_attributes.attrs["GranuleYear"] = np.array(
            [np.int32(np.int32(2020))], dtype=np.int32
        )
        hdfeos_additional_file_attributes.attrs["InputVersions"] = "OML1BRUG:1.1.3.0.3"
        hdfeos_additional_file_attributes.attrs["InstrumentName"] = "OMI"
        hdfeos_additional_file_attributes.attrs["LongName"] = (
            "OMI/Aura Bromine Monoxide (BrO) Total Column 1-Orbit L2 Swath 13x24km"
        )
        hdfeos_additional_file_attributes.attrs["NumberOfBadOutputSamples"] = np.array(
            [np.int32(np.int32(16))], dtype=np.int32
        )
        hdfeos_additional_file_attributes.attrs["NumberOfConvergedSamples"] = np.array(
            [np.int32(np.int32(88780))], dtype=np.int32
        )
        hdfeos_additional_file_attributes.attrs["NumberOfCrossTrackPixels"] = np.array(
            [np.int32(np.int32(60))], dtype=np.int32
        )
        hdfeos_additional_file_attributes.attrs["NumberOfExceededIterationsSamples"] = (
            np.array([np.int32(np.int32(0))], dtype=np.int32)
        )
        hdfeos_additional_file_attributes.attrs["NumberOfFailedConvergenceSamples"] = (
            np.array([np.int32(np.int32(5))], dtype=np.int32)
        )
        hdfeos_additional_file_attributes.attrs["NumberOfGoodInputSamples"] = np.array(
            [np.int32(np.int32(89058))], dtype=np.int32
        )
        hdfeos_additional_file_attributes.attrs["NumberOfGoodOutputSamples"] = np.array(
            [np.int32(np.int32(88780))], dtype=np.int32
        )
        hdfeos_additional_file_attributes.attrs["NumberOfInputSamples"] = np.array(
            [np.int32(np.int32(119820))], dtype=np.int32
        )
        hdfeos_additional_file_attributes.attrs["NumberOfOutOfBoundsSamples"] = (
            np.array([np.int32(np.int32(11))], dtype=np.int32)
        )
        hdfeos_additional_file_attributes.attrs["NumberOfScanLines"] = np.array(
            [np.int32(np.int32(1997))], dtype=np.int32
        )
        hdfeos_additional_file_attributes.attrs["NumberOfSuspectOutputSamples"] = (
            np.array([np.int32(np.int32(262))], dtype=np.int32)
        )
        hdfeos_additional_file_attributes.attrs["OPERATIONMODE"] = "Test"
        hdfeos_additional_file_attributes.attrs["OrbitData"] = "PREDICTED"
        hdfeos_additional_file_attributes.attrs["PGEVERSION"] = "3.0.9"
        hdfeos_additional_file_attributes.attrs["PercentBadOutputSamples"] = np.array(
            [np.float32(np.float32(0.01796582))], dtype=np.float32
        )
        hdfeos_additional_file_attributes.attrs["PercentGoodOutputSamples"] = np.array(
            [np.float32(np.float32(99.68784))], dtype=np.float32
        )
        hdfeos_additional_file_attributes.attrs["PercentSuspectOutputSamples"] = (
            np.array([np.float32(np.float32(0.29419032))], dtype=np.float32)
        )
        hdfeos_additional_file_attributes.attrs["ProcessLevel"] = "2"
        hdfeos_additional_file_attributes.attrs["ProcessingCenter"] = "OMI SIPS"
        hdfeos_additional_file_attributes.attrs["ProcessingHost"] = (
            "Linux minion7079 3.10.0-1062.9.1.el7.x86_64 x86_64"
        )
        hdfeos_additional_file_attributes.attrs["SpaceCraftMaxAltitude"] = np.array(
            [np.float64(np.float64(731780.6))], dtype=np.float64
        )
        hdfeos_additional_file_attributes.attrs["SpaceCraftMinAltitude"] = np.array(
            [np.float64(np.float64(704337.7))], dtype=np.float64
        )
        hdfeos_additional_file_attributes.attrs["TAI93At0zOfGranule"] = np.array(
            [np.float64(np.float64(853286410.0))], dtype=np.float64
        )

        # attributes for HDFEOS/SWATHS/OMI Total Column Amount BrO
        hdfeos_swaths_omi_total_column_amount_bro.attrs["EarthSunDistance"] = np.array(
            [np.float32(np.float32(1.471492e11))], dtype=np.float32
        )
        hdfeos_swaths_omi_total_column_amount_bro.attrs["VerticalCoordinate"] = (
            "Total Column"
        )

        # attributes for HDFEOS INFORMATION
        hdfeos_information.attrs["HDFEOSVersion"] = "HDFEOS_5.1.11"

        # HDFEOS/SWATHS/OMI Total Column Amount BrO/Data Fields/AMFCloudFraction
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_amfcloudfraction = (
            hdfeos_swaths_omi_total_column_amount_bro_data_fields.create_dataset(
                "AMFCloudFraction",
                data=np.arange(dim_0_1997 * dim_0_60, dtype=np.float32).reshape(
                    (
                        dim_0_1997,
                        dim_0_60,
                    )
                ),
            )
        )
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_amfcloudfraction.attrs[
            "MissingValue"
        ] = np.array([np.float32(np.float32(-1e30))], dtype=np.float32)
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_amfcloudfraction.attrs[
            "Offset"
        ] = np.array([np.float64(np.float64(0.0))], dtype=np.float64)
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_amfcloudfraction.attrs[
            "ScaleFactor"
        ] = np.array([np.float64(np.float64(1.0))], dtype=np.float64)
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_amfcloudfraction.attrs[
            "Title"
        ] = "Adjusted Cloud Fraction for AMF Computation"
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_amfcloudfraction.attrs[
            "UniqueFieldDefinition"
        ] = "OMI-Specific"
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_amfcloudfraction.attrs[
            "Units"
        ] = "NoUnits"
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_amfcloudfraction.attrs[
            "ValidRange"
        ] = np.array(
            [np.float32(np.float32(0.0)), np.float32(np.float32(1.0))], dtype=np.float32
        )
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_amfcloudfraction.attrs[
            "_FillValue"
        ] = np.array([np.float32(np.float32(-1e30))], dtype=np.float32)

        # HDFEOS/SWATHS/OMI Total Column Amount BrO/Data Fields/AMFCloudPressure
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_amfcloudpressure = (
            hdfeos_swaths_omi_total_column_amount_bro_data_fields.create_dataset(
                "AMFCloudPressure",
                data=np.arange(dim_0_1997 * dim_0_60, dtype=np.float32).reshape(
                    (
                        dim_0_1997,
                        dim_0_60,
                    )
                ),
            )
        )
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_amfcloudpressure.attrs[
            "MissingValue"
        ] = np.array([np.float32(np.float32(-1e30))], dtype=np.float32)
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_amfcloudpressure.attrs[
            "Offset"
        ] = np.array([np.float64(np.float64(0.0))], dtype=np.float64)
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_amfcloudpressure.attrs[
            "ScaleFactor"
        ] = np.array([np.float64(np.float64(1.0))], dtype=np.float64)
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_amfcloudpressure.attrs[
            "Title"
        ] = "Adjusted Cloud Pressure for AMF Computation"
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_amfcloudpressure.attrs[
            "UniqueFieldDefinition"
        ] = "OMI-Specific"
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_amfcloudpressure.attrs[
            "Units"
        ] = "hPa"
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_amfcloudpressure.attrs[
            "ValidRange"
        ] = np.array(
            [np.float32(np.float32(0.0)), np.float32(np.float32(1e30))],
            dtype=np.float32,
        )
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_amfcloudpressure.attrs[
            "_FillValue"
        ] = np.array([np.float32(np.float32(-1e30))], dtype=np.float32)

        # HDFEOS/SWATHS/OMI Total Column Amount BrO/Data Fields/AdjustedSceneAlbedo
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_adjustedscenealbedo = (
            hdfeos_swaths_omi_total_column_amount_bro_data_fields.create_dataset(
                "AdjustedSceneAlbedo",
                data=np.arange(dim_0_1997 * dim_0_60, dtype=np.float32).reshape(
                    (
                        dim_0_1997,
                        dim_0_60,
                    )
                ),
            )
        )
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_adjustedscenealbedo.attrs[
            "MissingValue"
        ] = np.array([np.float32(np.float32(-1e30))], dtype=np.float32)
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_adjustedscenealbedo.attrs[
            "Offset"
        ] = np.array([np.float64(np.float64(0.0))], dtype=np.float64)
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_adjustedscenealbedo.attrs[
            "ScaleFactor"
        ] = np.array([np.float64(np.float64(1.0))], dtype=np.float64)
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_adjustedscenealbedo.attrs[
            "Title"
        ] = "Adjusted Scene Albedo"
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_adjustedscenealbedo.attrs[
            "UniqueFieldDefinition"
        ] = "OMI-Specific"
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_adjustedscenealbedo.attrs[
            "Units"
        ] = "NoUnits"
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_adjustedscenealbedo.attrs[
            "ValidRange"
        ] = np.array(
            [np.float32(np.float32(0.0)), np.float32(np.float32(1.0))], dtype=np.float32
        )
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_adjustedscenealbedo.attrs[
            "_FillValue"
        ] = np.array([np.float32(np.float32(-1e30))], dtype=np.float32)

        # HDFEOS/SWATHS/OMI Total Column Amount BrO/Data Fields/AirMassFactor
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_airmassfactor = (
            hdfeos_swaths_omi_total_column_amount_bro_data_fields.create_dataset(
                "AirMassFactor",
                data=np.arange(dim_0_1997 * dim_0_60, dtype=np.float32).reshape(
                    (
                        dim_0_1997,
                        dim_0_60,
                    )
                ),
            )
        )
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_airmassfactor.attrs[
            "MissingValue"
        ] = np.array([np.float32(np.float32(-1e30))], dtype=np.float32)
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_airmassfactor.attrs[
            "Offset"
        ] = np.array([np.float64(np.float64(0.0))], dtype=np.float64)
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_airmassfactor.attrs[
            "ScaleFactor"
        ] = np.array([np.float64(np.float64(1.0))], dtype=np.float64)
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_airmassfactor.attrs[
            "Title"
        ] = "Molecule Specific Air Mass Factor (AMF)"
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_airmassfactor.attrs[
            "UniqueFieldDefinition"
        ] = "OMI-Specific"
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_airmassfactor.attrs[
            "Units"
        ] = "NoUnits"
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_airmassfactor.attrs[
            "ValidRange"
        ] = np.array(
            [np.float32(np.float32(0.0)), np.float32(np.float32(1e30))],
            dtype=np.float32,
        )
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_airmassfactor.attrs[
            "_FillValue"
        ] = np.array([np.float32(np.float32(-1e30))], dtype=np.float32)

        # HDFEOS/SWATHS/OMI Total Column Amount BrO/Data Fields/AirMassFactorDiagnosticFlag
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_airmassfactordiagnosticflag = hdfeos_swaths_omi_total_column_amount_bro_data_fields.create_dataset(
            "AirMassFactorDiagnosticFlag",
            data=np.arange(dim_0_1997 * dim_0_60, dtype=np.int16).reshape(
                (
                    dim_0_1997,
                    dim_0_60,
                )
            ),
        )
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_airmassfactordiagnosticflag.attrs[
            "MissingValue"
        ] = np.array([np.int16(np.int16(-30000))], dtype=np.int16)
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_airmassfactordiagnosticflag.attrs[
            "Offset"
        ] = np.array([np.float64(np.float64(0.0))], dtype=np.float64)
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_airmassfactordiagnosticflag.attrs[
            "ScaleFactor"
        ] = np.array([np.float64(np.float64(1.0))], dtype=np.float64)
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_airmassfactordiagnosticflag.attrs[
            "Title"
        ] = "Diagnostic Flag for Molecule Specific AMF"
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_airmassfactordiagnosticflag.attrs[
            "UniqueFieldDefinition"
        ] = "OMI-Specific"
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_airmassfactordiagnosticflag.attrs[
            "Units"
        ] = "NoUnits"
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_airmassfactordiagnosticflag.attrs[
            "ValidRange"
        ] = np.array(
            [np.int16(np.int16(-2)), np.int16(np.int16(13127))], dtype=np.int16
        )
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_airmassfactordiagnosticflag.attrs[
            "_FillValue"
        ] = np.array([np.int16(np.int16(-30000))], dtype=np.int16)

        # HDFEOS/SWATHS/OMI Total Column Amount BrO/Data Fields/AirMassFactorGeometric
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_airmassfactorgeometric = (
            hdfeos_swaths_omi_total_column_amount_bro_data_fields.create_dataset(
                "AirMassFactorGeometric",
                data=np.arange(dim_0_1997 * dim_0_60, dtype=np.float32).reshape(
                    (
                        dim_0_1997,
                        dim_0_60,
                    )
                ),
            )
        )
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_airmassfactorgeometric.attrs[
            "MissingValue"
        ] = np.array([np.float32(np.float32(-1e30))], dtype=np.float32)
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_airmassfactorgeometric.attrs[
            "Offset"
        ] = np.array([np.float64(np.float64(0.0))], dtype=np.float64)
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_airmassfactorgeometric.attrs[
            "ScaleFactor"
        ] = np.array([np.float64(np.float64(1.0))], dtype=np.float64)
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_airmassfactorgeometric.attrs[
            "Title"
        ] = "Geometric Air Mass Factor (AMF)"
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_airmassfactorgeometric.attrs[
            "UniqueFieldDefinition"
        ] = "OMI-Specific"
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_airmassfactorgeometric.attrs[
            "Units"
        ] = "NoUnits"
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_airmassfactorgeometric.attrs[
            "ValidRange"
        ] = np.array(
            [np.float32(np.float32(0.0)), np.float32(np.float32(1e30))],
            dtype=np.float32,
        )
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_airmassfactorgeometric.attrs[
            "_FillValue"
        ] = np.array([np.float32(np.float32(-1e30))], dtype=np.float32)

        # HDFEOS/SWATHS/OMI Total Column Amount BrO/Data Fields/AverageColumnAmount
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_averagecolumnamount = (
            hdfeos_swaths_omi_total_column_amount_bro_data_fields.create_dataset(
                "AverageColumnAmount",
                data=np.arange(dim_0_1, dtype=np.float64),
            )
        )
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_averagecolumnamount.attrs[
            "MissingValue"
        ] = np.array([np.float64(np.float64(-1e30))], dtype=np.float64)
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_averagecolumnamount.attrs[
            "Offset"
        ] = np.array([np.float64(np.float64(0.0))], dtype=np.float64)
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_averagecolumnamount.attrs[
            "ScaleFactor"
        ] = np.array([np.float64(np.float64(1.0))], dtype=np.float64)
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_averagecolumnamount.attrs[
            "Title"
        ] = "Average Column Amount"
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_averagecolumnamount.attrs[
            "UniqueFieldDefinition"
        ] = "OMI-Specific"
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_averagecolumnamount.attrs[
            "Units"
        ] = "molec/cm2"
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_averagecolumnamount.attrs[
            "ValidRange"
        ] = np.array(
            [np.float64(np.float64(-1e30)), np.float64(np.float64(1e30))],
            dtype=np.float64,
        )
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_averagecolumnamount.attrs[
            "_FillValue"
        ] = np.array([np.float64(np.float64(-1e30))], dtype=np.float64)

        # HDFEOS/SWATHS/OMI Total Column Amount BrO/Data Fields/AverageColumnUncertainty
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_averagecolumnuncertainty = hdfeos_swaths_omi_total_column_amount_bro_data_fields.create_dataset(
            "AverageColumnUncertainty",
            data=np.arange(dim_0_1, dtype=np.float64),
        )
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_averagecolumnuncertainty.attrs[
            "MissingValue"
        ] = np.array([np.float64(np.float64(-1e30))], dtype=np.float64)
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_averagecolumnuncertainty.attrs[
            "Offset"
        ] = np.array([np.float64(np.float64(0.0))], dtype=np.float64)
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_averagecolumnuncertainty.attrs[
            "ScaleFactor"
        ] = np.array([np.float64(np.float64(1.0))], dtype=np.float64)
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_averagecolumnuncertainty.attrs[
            "Title"
        ] = "Average Column Uncertainty"
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_averagecolumnuncertainty.attrs[
            "UniqueFieldDefinition"
        ] = "OMI-Specific"
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_averagecolumnuncertainty.attrs[
            "Units"
        ] = "molec/cm2"
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_averagecolumnuncertainty.attrs[
            "ValidRange"
        ] = np.array(
            [np.float64(np.float64(0.0)), np.float64(np.float64(1e30))],
            dtype=np.float64,
        )
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_averagecolumnuncertainty.attrs[
            "_FillValue"
        ] = np.array([np.float64(np.float64(-1e30))], dtype=np.float64)

        # HDFEOS/SWATHS/OMI Total Column Amount BrO/Data Fields/AverageFittingRMS
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_averagefittingrms = (
            hdfeos_swaths_omi_total_column_amount_bro_data_fields.create_dataset(
                "AverageFittingRMS",
                data=np.arange(dim_0_1, dtype=np.float64),
            )
        )
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_averagefittingrms.attrs[
            "MissingValue"
        ] = np.array([np.float64(np.float64(-1e30))], dtype=np.float64)
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_averagefittingrms.attrs[
            "Offset"
        ] = np.array([np.float64(np.float64(0.0))], dtype=np.float64)
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_averagefittingrms.attrs[
            "ScaleFactor"
        ] = np.array([np.float64(np.float64(1.0))], dtype=np.float64)
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_averagefittingrms.attrs[
            "Title"
        ] = "Average Fitting RMS"
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_averagefittingrms.attrs[
            "UniqueFieldDefinition"
        ] = "OMI-Specific"
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_averagefittingrms.attrs[
            "Units"
        ] = "NoUnits"
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_averagefittingrms.attrs[
            "ValidRange"
        ] = np.array(
            [np.float64(np.float64(0.0)), np.float64(np.float64(1e30))],
            dtype=np.float64,
        )
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_averagefittingrms.attrs[
            "_FillValue"
        ] = np.array([np.float64(np.float64(-1e30))], dtype=np.float64)

        # HDFEOS/SWATHS/OMI Total Column Amount BrO/Data Fields/ColumnAmount
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_columnamount = (
            hdfeos_swaths_omi_total_column_amount_bro_data_fields.create_dataset(
                "ColumnAmount",
                data=np.arange(dim_0_1997 * dim_0_60, dtype=np.float64).reshape(
                    (
                        dim_0_1997,
                        dim_0_60,
                    )
                ),
            )
        )
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_columnamount.attrs[
            "MissingValue"
        ] = np.array([np.float64(np.float64(-1e30))], dtype=np.float64)
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_columnamount.attrs[
            "Offset"
        ] = np.array([np.float64(np.float64(0.0))], dtype=np.float64)
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_columnamount.attrs[
            "ScaleFactor"
        ] = np.array([np.float64(np.float64(1.0))], dtype=np.float64)
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_columnamount.attrs[
            "Title"
        ] = "Column Amount"
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_columnamount.attrs[
            "UniqueFieldDefinition"
        ] = "OMI-Specific"
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_columnamount.attrs[
            "Units"
        ] = "molec/cm2"
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_columnamount.attrs[
            "ValidRange"
        ] = np.array(
            [np.float64(np.float64(-1e30)), np.float64(np.float64(1e30))],
            dtype=np.float64,
        )
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_columnamount.attrs[
            "_FillValue"
        ] = np.array([np.float64(np.float64(-1e30))], dtype=np.float64)

        # HDFEOS/SWATHS/OMI Total Column Amount BrO/Data Fields/ColumnAmountDestriped
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_columnamountdestriped = (
            hdfeos_swaths_omi_total_column_amount_bro_data_fields.create_dataset(
                "ColumnAmountDestriped",
                data=np.arange(dim_0_1997 * dim_0_60, dtype=np.float64).reshape(
                    (
                        dim_0_1997,
                        dim_0_60,
                    )
                ),
            )
        )
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_columnamountdestriped.attrs[
            "MissingValue"
        ] = np.array([np.float64(np.float64(-1e30))], dtype=np.float64)
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_columnamountdestriped.attrs[
            "Offset"
        ] = np.array([np.float64(np.float64(0.0))], dtype=np.float64)
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_columnamountdestriped.attrs[
            "ScaleFactor"
        ] = np.array([np.float64(np.float64(1.0))], dtype=np.float64)
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_columnamountdestriped.attrs[
            "Title"
        ] = "Column Amount with Destriping Correction"
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_columnamountdestriped.attrs[
            "UniqueFieldDefinition"
        ] = "OMI-Specific"
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_columnamountdestriped.attrs[
            "Units"
        ] = "molec/cm2"
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_columnamountdestriped.attrs[
            "ValidRange"
        ] = np.array(
            [np.float64(np.float64(-1e30)), np.float64(np.float64(1e30))],
            dtype=np.float64,
        )
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_columnamountdestriped.attrs[
            "_FillValue"
        ] = np.array([np.float64(np.float64(-1e30))], dtype=np.float64)

        # HDFEOS/SWATHS/OMI Total Column Amount BrO/Data Fields/ColumnUncertainty
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_columnuncertainty = (
            hdfeos_swaths_omi_total_column_amount_bro_data_fields.create_dataset(
                "ColumnUncertainty",
                data=np.arange(dim_0_1997 * dim_0_60, dtype=np.float64).reshape(
                    (
                        dim_0_1997,
                        dim_0_60,
                    )
                ),
            )
        )
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_columnuncertainty.attrs[
            "MissingValue"
        ] = np.array([np.float64(np.float64(-1e30))], dtype=np.float64)
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_columnuncertainty.attrs[
            "Offset"
        ] = np.array([np.float64(np.float64(0.0))], dtype=np.float64)
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_columnuncertainty.attrs[
            "ScaleFactor"
        ] = np.array([np.float64(np.float64(1.0))], dtype=np.float64)
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_columnuncertainty.attrs[
            "Title"
        ] = "Column Uncertainty"
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_columnuncertainty.attrs[
            "UniqueFieldDefinition"
        ] = "OMI-Specific"
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_columnuncertainty.attrs[
            "Units"
        ] = "molec/cm2"
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_columnuncertainty.attrs[
            "ValidRange"
        ] = np.array(
            [np.float64(np.float64(0.0)), np.float64(np.float64(1e30))],
            dtype=np.float64,
        )
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_columnuncertainty.attrs[
            "_FillValue"
        ] = np.array([np.float64(np.float64(-1e30))], dtype=np.float64)

        # HDFEOS/SWATHS/OMI Total Column Amount BrO/Data Fields/FitConvergenceFlag
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_fitconvergenceflag = (
            hdfeos_swaths_omi_total_column_amount_bro_data_fields.create_dataset(
                "FitConvergenceFlag",
                data=np.arange(dim_0_1997 * dim_0_60, dtype=np.int16).reshape(
                    (
                        dim_0_1997,
                        dim_0_60,
                    )
                ),
            )
        )
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_fitconvergenceflag.attrs[
            "MissingValue"
        ] = np.array([np.int16(np.int16(-30000))], dtype=np.int16)
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_fitconvergenceflag.attrs[
            "Offset"
        ] = np.array([np.float64(np.float64(0.0))], dtype=np.float64)
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_fitconvergenceflag.attrs[
            "ScaleFactor"
        ] = np.array([np.float64(np.float64(1.0))], dtype=np.float64)
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_fitconvergenceflag.attrs[
            "Title"
        ] = "Fitting Convergence Flag"
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_fitconvergenceflag.attrs[
            "UniqueFieldDefinition"
        ] = "OMI-Specific"
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_fitconvergenceflag.attrs[
            "Units"
        ] = "NoUnits"
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_fitconvergenceflag.attrs[
            "ValidRange"
        ] = np.array(
            [np.int16(np.int16(-10)), np.int16(np.int16(12344))], dtype=np.int16
        )
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_fitconvergenceflag.attrs[
            "_FillValue"
        ] = np.array([np.int16(np.int16(-30000))], dtype=np.int16)

        # HDFEOS/SWATHS/OMI Total Column Amount BrO/Data Fields/FittingRMS
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_fittingrms = (
            hdfeos_swaths_omi_total_column_amount_bro_data_fields.create_dataset(
                "FittingRMS",
                data=np.arange(dim_0_1997 * dim_0_60, dtype=np.float64).reshape(
                    (
                        dim_0_1997,
                        dim_0_60,
                    )
                ),
            )
        )
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_fittingrms.attrs[
            "MissingValue"
        ] = np.array([np.float64(np.float64(-1e30))], dtype=np.float64)
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_fittingrms.attrs[
            "Offset"
        ] = np.array([np.float64(np.float64(0.0))], dtype=np.float64)
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_fittingrms.attrs[
            "ScaleFactor"
        ] = np.array([np.float64(np.float64(1.0))], dtype=np.float64)
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_fittingrms.attrs[
            "Title"
        ] = "Fitting RMS"
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_fittingrms.attrs[
            "UniqueFieldDefinition"
        ] = "OMI-Specific"
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_fittingrms.attrs[
            "Units"
        ] = "NoUnits"
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_fittingrms.attrs[
            "ValidRange"
        ] = np.array(
            [np.float64(np.float64(0.0)), np.float64(np.float64(1e30))],
            dtype=np.float64,
        )
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_fittingrms.attrs[
            "_FillValue"
        ] = np.array([np.float64(np.float64(-1e30))], dtype=np.float64)

        # HDFEOS/SWATHS/OMI Total Column Amount BrO/Data Fields/MainDataQualityFlag
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_maindataqualityflag = (
            hdfeos_swaths_omi_total_column_amount_bro_data_fields.create_dataset(
                "MainDataQualityFlag",
                data=np.arange(dim_0_1997 * dim_0_60, dtype=np.int16).reshape(
                    (
                        dim_0_1997,
                        dim_0_60,
                    )
                ),
            )
        )
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_maindataqualityflag.attrs[
            "MissingValue"
        ] = np.array([np.int16(np.int16(-30000))], dtype=np.int16)
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_maindataqualityflag.attrs[
            "Offset"
        ] = np.array([np.float64(np.float64(0.0))], dtype=np.float64)
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_maindataqualityflag.attrs[
            "ScaleFactor"
        ] = np.array([np.float64(np.float64(1.0))], dtype=np.float64)
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_maindataqualityflag.attrs[
            "Title"
        ] = "Main Data Quality Flag"
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_maindataqualityflag.attrs[
            "UniqueFieldDefinition"
        ] = "OMI-Specific"
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_maindataqualityflag.attrs[
            "Units"
        ] = "NoUnits"
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_maindataqualityflag.attrs[
            "ValidRange"
        ] = np.array([np.int16(np.int16(-1)), np.int16(np.int16(2))], dtype=np.int16)
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_maindataqualityflag.attrs[
            "_FillValue"
        ] = np.array([np.int16(np.int16(-30000))], dtype=np.int16)

        # HDFEOS/SWATHS/OMI Total Column Amount BrO/Data Fields/MaximumColumnAmount
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_maximumcolumnamount = (
            hdfeos_swaths_omi_total_column_amount_bro_data_fields.create_dataset(
                "MaximumColumnAmount",
                data=np.arange(dim_0_1, dtype=np.float64),
            )
        )
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_maximumcolumnamount.attrs[
            "MissingValue"
        ] = np.array([np.float64(np.float64(-1e30))], dtype=np.float64)
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_maximumcolumnamount.attrs[
            "Offset"
        ] = np.array([np.float64(np.float64(0.0))], dtype=np.float64)
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_maximumcolumnamount.attrs[
            "ScaleFactor"
        ] = np.array([np.float64(np.float64(1.0))], dtype=np.float64)
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_maximumcolumnamount.attrs[
            "Title"
        ] = "Maximum Column Amount for QA Flag 'good'"
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_maximumcolumnamount.attrs[
            "UniqueFieldDefinition"
        ] = "OMI-Specific"
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_maximumcolumnamount.attrs[
            "Units"
        ] = "molec/cm2"
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_maximumcolumnamount.attrs[
            "ValidRange"
        ] = np.array(
            [np.float64(np.float64(-1e30)), np.float64(np.float64(1e30))],
            dtype=np.float64,
        )
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_maximumcolumnamount.attrs[
            "_FillValue"
        ] = np.array([np.float64(np.float64(-1e30))], dtype=np.float64)

        # HDFEOS/SWATHS/OMI Total Column Amount BrO/Data Fields/PixelArea
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_pixelarea = (
            hdfeos_swaths_omi_total_column_amount_bro_data_fields.create_dataset(
                "PixelArea",
                data=np.arange(dim_0_60, dtype=np.float32),
            )
        )
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_pixelarea.attrs[
            "MissingValue"
        ] = np.array([np.float32(np.float32(-1e30))], dtype=np.float32)
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_pixelarea.attrs[
            "Offset"
        ] = np.array([np.float64(np.float64(0.0))], dtype=np.float64)
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_pixelarea.attrs[
            "ScaleFactor"
        ] = np.array([np.float64(np.float64(1.0))], dtype=np.float64)
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_pixelarea.attrs[
            "Title"
        ] = "Pixel Area"
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_pixelarea.attrs[
            "UniqueFieldDefinition"
        ] = "OMI-Specific"
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_pixelarea.attrs[
            "Units"
        ] = "km^2"
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_pixelarea.attrs[
            "ValidRange"
        ] = np.array(
            [np.float32(np.float32(0.0)), np.float32(np.float32(4.132944e08))],
            dtype=np.float32,
        )
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_pixelarea.attrs[
            "_FillValue"
        ] = np.array([np.float32(np.float32(-1e30))], dtype=np.float32)

        # HDFEOS/SWATHS/OMI Total Column Amount BrO/Data Fields/PixelCornerLatitudes
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_pixelcornerlatitudes = (
            hdfeos_swaths_omi_total_column_amount_bro_data_fields.create_dataset(
                "PixelCornerLatitudes",
                data=np.arange(dim_0_1998 * dim_1_61, dtype=np.float32).reshape(
                    (
                        dim_0_1998,
                        dim_1_61,
                    )
                ),
            )
        )
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_pixelcornerlatitudes.attrs[
            "MissingValue"
        ] = np.array([np.float32(np.float32(-1e30))], dtype=np.float32)
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_pixelcornerlatitudes.attrs[
            "Offset"
        ] = np.array([np.float64(np.float64(0.0))], dtype=np.float64)
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_pixelcornerlatitudes.attrs[
            "ScaleFactor"
        ] = np.array([np.float64(np.float64(1.0))], dtype=np.float64)
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_pixelcornerlatitudes.attrs[
            "Title"
        ] = "Pixel Corner Latitude Coordinates"
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_pixelcornerlatitudes.attrs[
            "UniqueFieldDefinition"
        ] = "OMI-Specific"
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_pixelcornerlatitudes.attrs[
            "Units"
        ] = "deg"
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_pixelcornerlatitudes.attrs[
            "ValidRange"
        ] = np.array(
            [np.float32(np.float32(-90.0)), np.float32(np.float32(90.0))],
            dtype=np.float32,
        )
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_pixelcornerlatitudes.attrs[
            "_FillValue"
        ] = np.array([np.float32(np.float32(-1e30))], dtype=np.float32)

        # HDFEOS/SWATHS/OMI Total Column Amount BrO/Data Fields/PixelCornerLongitudes
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_pixelcornerlongitudes = (
            hdfeos_swaths_omi_total_column_amount_bro_data_fields.create_dataset(
                "PixelCornerLongitudes",
                data=np.arange(dim_0_1998 * dim_1_61, dtype=np.float32).reshape(
                    (
                        dim_0_1998,
                        dim_1_61,
                    )
                ),
            )
        )
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_pixelcornerlongitudes.attrs[
            "MissingValue"
        ] = np.array([np.float32(np.float32(-1e30))], dtype=np.float32)
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_pixelcornerlongitudes.attrs[
            "Offset"
        ] = np.array([np.float64(np.float64(0.0))], dtype=np.float64)
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_pixelcornerlongitudes.attrs[
            "ScaleFactor"
        ] = np.array([np.float64(np.float64(1.0))], dtype=np.float64)
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_pixelcornerlongitudes.attrs[
            "Title"
        ] = "Pixel Corner Longitude Coordinates"
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_pixelcornerlongitudes.attrs[
            "UniqueFieldDefinition"
        ] = "OMI-Specific"
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_pixelcornerlongitudes.attrs[
            "Units"
        ] = "deg"
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_pixelcornerlongitudes.attrs[
            "ValidRange"
        ] = np.array(
            [np.float32(np.float32(-180.0)), np.float32(np.float32(180.0))],
            dtype=np.float32,
        )
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_pixelcornerlongitudes.attrs[
            "_FillValue"
        ] = np.array([np.float32(np.float32(-1e30))], dtype=np.float32)

        # HDFEOS/SWATHS/OMI Total Column Amount BrO/Data Fields/RadianceReferenceColumnAmount
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_radiancereferencecolumnamount = hdfeos_swaths_omi_total_column_amount_bro_data_fields.create_dataset(
            "RadianceReferenceColumnAmount",
            data=np.arange(dim_0_60, dtype=np.float64),
        )
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_radiancereferencecolumnamount.attrs[
            "MissingValue"
        ] = np.array([np.float64(np.float64(-1e30))], dtype=np.float64)
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_radiancereferencecolumnamount.attrs[
            "Offset"
        ] = np.array([np.float64(np.float64(0.0))], dtype=np.float64)
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_radiancereferencecolumnamount.attrs[
            "ScaleFactor"
        ] = np.array([np.float64(np.float64(1.0))], dtype=np.float64)
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_radiancereferencecolumnamount.attrs[
            "Title"
        ] = "Radiance Reference Fit Colunm Amount"
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_radiancereferencecolumnamount.attrs[
            "UniqueFieldDefinition"
        ] = "OMI-Specific"
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_radiancereferencecolumnamount.attrs[
            "Units"
        ] = "molec/cm2"
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_radiancereferencecolumnamount.attrs[
            "ValidRange"
        ] = np.array(
            [np.float64(np.float64(-1e30)), np.float64(np.float64(1e30))],
            dtype=np.float64,
        )
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_radiancereferencecolumnamount.attrs[
            "_FillValue"
        ] = np.array([np.float64(np.float64(-1e30))], dtype=np.float64)

        # HDFEOS/SWATHS/OMI Total Column Amount BrO/Data Fields/RadianceReferenceColumnUncertainty
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_radiancereferencecolumnuncertainty = hdfeos_swaths_omi_total_column_amount_bro_data_fields.create_dataset(
            "RadianceReferenceColumnUncertainty",
            data=np.arange(dim_0_60, dtype=np.float64),
        )
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_radiancereferencecolumnuncertainty.attrs[
            "MissingValue"
        ] = np.array([np.float64(np.float64(-1e30))], dtype=np.float64)
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_radiancereferencecolumnuncertainty.attrs[
            "Offset"
        ] = np.array([np.float64(np.float64(0.0))], dtype=np.float64)
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_radiancereferencecolumnuncertainty.attrs[
            "ScaleFactor"
        ] = np.array([np.float64(np.float64(1.0))], dtype=np.float64)
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_radiancereferencecolumnuncertainty.attrs[
            "Title"
        ] = "Radiance Reference Fit Colunm Uncertainty"
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_radiancereferencecolumnuncertainty.attrs[
            "UniqueFieldDefinition"
        ] = "OMI-Specific"
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_radiancereferencecolumnuncertainty.attrs[
            "Units"
        ] = "molec/cm2"
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_radiancereferencecolumnuncertainty.attrs[
            "ValidRange"
        ] = np.array(
            [np.float64(np.float64(0.0)), np.float64(np.float64(1e30))],
            dtype=np.float64,
        )
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_radiancereferencecolumnuncertainty.attrs[
            "_FillValue"
        ] = np.array([np.float64(np.float64(-1e30))], dtype=np.float64)

        # HDFEOS/SWATHS/OMI Total Column Amount BrO/Data Fields/RadianceReferenceColumnXTRFit
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_radiancereferencecolumnxtrfit = hdfeos_swaths_omi_total_column_amount_bro_data_fields.create_dataset(
            "RadianceReferenceColumnXTRFit",
            data=np.arange(dim_0_60, dtype=np.float64),
        )
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_radiancereferencecolumnxtrfit.attrs[
            "MissingValue"
        ] = np.array([np.float64(np.float64(-1e30))], dtype=np.float64)
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_radiancereferencecolumnxtrfit.attrs[
            "Offset"
        ] = np.array([np.float64(np.float64(0.0))], dtype=np.float64)
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_radiancereferencecolumnxtrfit.attrs[
            "ScaleFactor"
        ] = np.array([np.float64(np.float64(1.0))], dtype=np.float64)
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_radiancereferencecolumnxtrfit.attrs[
            "Title"
        ] = "Radiance Reference Fit Colunm XTR Fit"
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_radiancereferencecolumnxtrfit.attrs[
            "UniqueFieldDefinition"
        ] = "OMI-Specific"
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_radiancereferencecolumnxtrfit.attrs[
            "Units"
        ] = "molec/cm2"
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_radiancereferencecolumnxtrfit.attrs[
            "ValidRange"
        ] = np.array(
            [np.float64(np.float64(-1e30)), np.float64(np.float64(1e30))],
            dtype=np.float64,
        )
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_radiancereferencecolumnxtrfit.attrs[
            "_FillValue"
        ] = np.array([np.float64(np.float64(-1e30))], dtype=np.float64)

        # HDFEOS/SWATHS/OMI Total Column Amount BrO/Data Fields/RadianceReferenceConvergenceFlag
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_radiancereferenceconvergenceflag = hdfeos_swaths_omi_total_column_amount_bro_data_fields.create_dataset(
            "RadianceReferenceConvergenceFlag",
            data=np.arange(dim_0_60, dtype=np.int16),
        )
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_radiancereferenceconvergenceflag.attrs[
            "MissingValue"
        ] = np.array([np.int16(np.int16(-30000))], dtype=np.int16)
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_radiancereferenceconvergenceflag.attrs[
            "Offset"
        ] = np.array([np.float64(np.float64(0.0))], dtype=np.float64)
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_radiancereferenceconvergenceflag.attrs[
            "ScaleFactor"
        ] = np.array([np.float64(np.float64(1.0))], dtype=np.float64)
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_radiancereferenceconvergenceflag.attrs[
            "Title"
        ] = "Radiance Reference Fit Convergence Flag"
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_radiancereferenceconvergenceflag.attrs[
            "UniqueFieldDefinition"
        ] = "OMI-Specific"
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_radiancereferenceconvergenceflag.attrs[
            "Units"
        ] = "NoUnits"
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_radiancereferenceconvergenceflag.attrs[
            "ValidRange"
        ] = np.array(
            [np.int16(np.int16(-10)), np.int16(np.int16(12344))], dtype=np.int16
        )
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_radiancereferenceconvergenceflag.attrs[
            "_FillValue"
        ] = np.array([np.int16(np.int16(-30000))], dtype=np.int16)

        # HDFEOS/SWATHS/OMI Total Column Amount BrO/Data Fields/RadianceReferenceFittingRMS
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_radiancereferencefittingrms = hdfeos_swaths_omi_total_column_amount_bro_data_fields.create_dataset(
            "RadianceReferenceFittingRMS",
            data=np.arange(dim_0_60, dtype=np.float64),
        )
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_radiancereferencefittingrms.attrs[
            "MissingValue"
        ] = np.array([np.float64(np.float64(-1e30))], dtype=np.float64)
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_radiancereferencefittingrms.attrs[
            "Offset"
        ] = np.array([np.float64(np.float64(0.0))], dtype=np.float64)
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_radiancereferencefittingrms.attrs[
            "ScaleFactor"
        ] = np.array([np.float64(np.float64(1.0))], dtype=np.float64)
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_radiancereferencefittingrms.attrs[
            "Title"
        ] = "Radiance Reference Fit RMS"
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_radiancereferencefittingrms.attrs[
            "UniqueFieldDefinition"
        ] = "OMI-Specific"
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_radiancereferencefittingrms.attrs[
            "Units"
        ] = "NoUnits"
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_radiancereferencefittingrms.attrs[
            "ValidRange"
        ] = np.array(
            [np.float64(np.float64(0.0)), np.float64(np.float64(1e30))],
            dtype=np.float64,
        )
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_radiancereferencefittingrms.attrs[
            "_FillValue"
        ] = np.array([np.float64(np.float64(-1e30))], dtype=np.float64)

        # HDFEOS/SWATHS/OMI Total Column Amount BrO/Data Fields/RadianceReferenceLatitudeRange
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_radiancereferencelatituderange = hdfeos_swaths_omi_total_column_amount_bro_data_fields.create_dataset(
            "RadianceReferenceLatitudeRange",
            data=np.arange(dim_0_4, dtype=np.float32),
        )
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_radiancereferencelatituderange.attrs[
            "MissingValue"
        ] = np.array([np.float32(np.float32(-1e30))], dtype=np.float32)
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_radiancereferencelatituderange.attrs[
            "Offset"
        ] = np.array([np.float64(np.float64(0.0))], dtype=np.float64)
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_radiancereferencelatituderange.attrs[
            "ScaleFactor"
        ] = np.array([np.float64(np.float64(1.0))], dtype=np.float64)
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_radiancereferencelatituderange.attrs[
            "Title"
        ] = "Radiance Reference Fit Latitude Range"
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_radiancereferencelatituderange.attrs[
            "UniqueFieldDefinition"
        ] = "OMI-Specific"
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_radiancereferencelatituderange.attrs[
            "Units"
        ] = "NoUnits"
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_radiancereferencelatituderange.attrs[
            "ValidRange"
        ] = np.array(
            [np.float32(np.float32(-90.0)), np.float32(np.float32(90.0))],
            dtype=np.float32,
        )
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_radiancereferencelatituderange.attrs[
            "_FillValue"
        ] = np.array([np.float32(np.float32(-1e30))], dtype=np.float32)

        # HDFEOS/SWATHS/OMI Total Column Amount BrO/Data Fields/RadianceWavCalConvergenceFlag
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_radiancewavcalconvergenceflag = hdfeos_swaths_omi_total_column_amount_bro_data_fields.create_dataset(
            "RadianceWavCalConvergenceFlag",
            data=np.arange(dim_0_60, dtype=np.int16),
        )
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_radiancewavcalconvergenceflag.attrs[
            "MissingValue"
        ] = np.array([np.int16(np.int16(-30000))], dtype=np.int16)
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_radiancewavcalconvergenceflag.attrs[
            "Offset"
        ] = np.array([np.float64(np.float64(0.0))], dtype=np.float64)
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_radiancewavcalconvergenceflag.attrs[
            "ScaleFactor"
        ] = np.array([np.float64(np.float64(1.0))], dtype=np.float64)
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_radiancewavcalconvergenceflag.attrs[
            "Title"
        ] = "Radiance Wavelength Calibration Convergence Flag"
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_radiancewavcalconvergenceflag.attrs[
            "UniqueFieldDefinition"
        ] = "OMI-Specific"
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_radiancewavcalconvergenceflag.attrs[
            "Units"
        ] = "NoUnits"
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_radiancewavcalconvergenceflag.attrs[
            "ValidRange"
        ] = np.array(
            [np.int16(np.int16(-10)), np.int16(np.int16(12344))], dtype=np.int16
        )
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_radiancewavcalconvergenceflag.attrs[
            "_FillValue"
        ] = np.array([np.int16(np.int16(-30000))], dtype=np.int16)

        # HDFEOS/SWATHS/OMI Total Column Amount BrO/Data Fields/RadianceWavCalLatitudeRange
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_radiancewavcallatituderange = hdfeos_swaths_omi_total_column_amount_bro_data_fields.create_dataset(
            "RadianceWavCalLatitudeRange",
            data=np.arange(dim_0_4, dtype=np.float32),
        )
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_radiancewavcallatituderange.attrs[
            "MissingValue"
        ] = np.array([np.float32(np.float32(-1e30))], dtype=np.float32)
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_radiancewavcallatituderange.attrs[
            "Offset"
        ] = np.array([np.float64(np.float64(0.0))], dtype=np.float64)
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_radiancewavcallatituderange.attrs[
            "ScaleFactor"
        ] = np.array([np.float64(np.float64(1.0))], dtype=np.float64)
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_radiancewavcallatituderange.attrs[
            "Title"
        ] = "Radiance Wavelength Calibration Latitude Range"
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_radiancewavcallatituderange.attrs[
            "UniqueFieldDefinition"
        ] = "OMI-Specific"
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_radiancewavcallatituderange.attrs[
            "Units"
        ] = "NoUnits"
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_radiancewavcallatituderange.attrs[
            "ValidRange"
        ] = np.array(
            [np.float32(np.float32(-90.0)), np.float32(np.float32(90.0))],
            dtype=np.float32,
        )
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_radiancewavcallatituderange.attrs[
            "_FillValue"
        ] = np.array([np.float32(np.float32(-1e30))], dtype=np.float32)

        # HDFEOS/SWATHS/OMI Total Column Amount BrO/Data Fields/SlantColumnAmount
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_slantcolumnamount = (
            hdfeos_swaths_omi_total_column_amount_bro_data_fields.create_dataset(
                "SlantColumnAmount",
                data=np.arange(dim_0_1997 * dim_0_60, dtype=np.float64).reshape(
                    (
                        dim_0_1997,
                        dim_0_60,
                    )
                ),
            )
        )
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_slantcolumnamount.attrs[
            "MissingValue"
        ] = np.array([np.float64(np.float64(-1e30))], dtype=np.float64)
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_slantcolumnamount.attrs[
            "Offset"
        ] = np.array([np.float64(np.float64(0.0))], dtype=np.float64)
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_slantcolumnamount.attrs[
            "ScaleFactor"
        ] = np.array([np.float64(np.float64(1.0))], dtype=np.float64)
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_slantcolumnamount.attrs[
            "Title"
        ] = "Slant Column Amount"
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_slantcolumnamount.attrs[
            "UniqueFieldDefinition"
        ] = "OMI-Specific"
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_slantcolumnamount.attrs[
            "Units"
        ] = "molec/cm2"
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_slantcolumnamount.attrs[
            "ValidRange"
        ] = np.array(
            [np.float64(np.float64(-1e30)), np.float64(np.float64(1e30))],
            dtype=np.float64,
        )
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_slantcolumnamount.attrs[
            "_FillValue"
        ] = np.array([np.float64(np.float64(-1e30))], dtype=np.float64)

        # HDFEOS/SWATHS/OMI Total Column Amount BrO/Data Fields/SlantColumnAmountDestriped
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_slantcolumnamountdestriped = hdfeos_swaths_omi_total_column_amount_bro_data_fields.create_dataset(
            "SlantColumnAmountDestriped",
            data=np.arange(dim_0_1997 * dim_0_60, dtype=np.float64).reshape(
                (
                    dim_0_1997,
                    dim_0_60,
                )
            ),
        )
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_slantcolumnamountdestriped.attrs[
            "MissingValue"
        ] = np.array([np.float64(np.float64(-1e30))], dtype=np.float64)
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_slantcolumnamountdestriped.attrs[
            "Offset"
        ] = np.array([np.float64(np.float64(0.0))], dtype=np.float64)
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_slantcolumnamountdestriped.attrs[
            "ScaleFactor"
        ] = np.array([np.float64(np.float64(1.0))], dtype=np.float64)
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_slantcolumnamountdestriped.attrs[
            "Title"
        ] = "Slant Column Amount with Destriping Correction"
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_slantcolumnamountdestriped.attrs[
            "UniqueFieldDefinition"
        ] = "OMI-Specific"
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_slantcolumnamountdestriped.attrs[
            "Units"
        ] = "molec/cm2"
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_slantcolumnamountdestriped.attrs[
            "ValidRange"
        ] = np.array(
            [np.float64(np.float64(-1e30)), np.float64(np.float64(1e30))],
            dtype=np.float64,
        )
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_slantcolumnamountdestriped.attrs[
            "_FillValue"
        ] = np.array([np.float64(np.float64(-1e30))], dtype=np.float64)

        # HDFEOS/SWATHS/OMI Total Column Amount BrO/Data Fields/SlantColumnUncertainty
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_slantcolumnuncertainty = (
            hdfeos_swaths_omi_total_column_amount_bro_data_fields.create_dataset(
                "SlantColumnUncertainty",
                data=np.arange(dim_0_1997 * dim_0_60, dtype=np.float64).reshape(
                    (
                        dim_0_1997,
                        dim_0_60,
                    )
                ),
            )
        )
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_slantcolumnuncertainty.attrs[
            "MissingValue"
        ] = np.array([np.float64(np.float64(-1e30))], dtype=np.float64)
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_slantcolumnuncertainty.attrs[
            "Offset"
        ] = np.array([np.float64(np.float64(0.0))], dtype=np.float64)
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_slantcolumnuncertainty.attrs[
            "ScaleFactor"
        ] = np.array([np.float64(np.float64(1.0))], dtype=np.float64)
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_slantcolumnuncertainty.attrs[
            "Title"
        ] = "Slant Column Uncertainty"
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_slantcolumnuncertainty.attrs[
            "UniqueFieldDefinition"
        ] = "OMI-Specific"
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_slantcolumnuncertainty.attrs[
            "Units"
        ] = "molec/cm2"
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_slantcolumnuncertainty.attrs[
            "ValidRange"
        ] = np.array(
            [np.float64(np.float64(0.0)), np.float64(np.float64(1e30))],
            dtype=np.float64,
        )
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_slantcolumnuncertainty.attrs[
            "_FillValue"
        ] = np.array([np.float64(np.float64(-1e30))], dtype=np.float64)

        # HDFEOS/SWATHS/OMI Total Column Amount BrO/Data Fields/SlantFitConvergenceFlag
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_slantfitconvergenceflag = hdfeos_swaths_omi_total_column_amount_bro_data_fields.create_dataset(
            "SlantFitConvergenceFlag",
            data=np.arange(dim_0_1997 * dim_0_60, dtype=np.int16).reshape(
                (
                    dim_0_1997,
                    dim_0_60,
                )
            ),
        )
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_slantfitconvergenceflag.attrs[
            "MissingValue"
        ] = np.array([np.int16(np.int16(-30000))], dtype=np.int16)
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_slantfitconvergenceflag.attrs[
            "Offset"
        ] = np.array([np.float64(np.float64(0.0))], dtype=np.float64)
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_slantfitconvergenceflag.attrs[
            "ScaleFactor"
        ] = np.array([np.float64(np.float64(1.0))], dtype=np.float64)
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_slantfitconvergenceflag.attrs[
            "Title"
        ] = "Slant Fitting Convergence Flag"
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_slantfitconvergenceflag.attrs[
            "UniqueFieldDefinition"
        ] = "OMI-Specific"
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_slantfitconvergenceflag.attrs[
            "Units"
        ] = "NoUnits"
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_slantfitconvergenceflag.attrs[
            "ValidRange"
        ] = np.array(
            [np.int16(np.int16(-10)), np.int16(np.int16(12344))], dtype=np.int16
        )
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_slantfitconvergenceflag.attrs[
            "_FillValue"
        ] = np.array([np.int16(np.int16(-30000))], dtype=np.int16)

        # HDFEOS/SWATHS/OMI Total Column Amount BrO/Data Fields/SlantFittingRMS
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_slantfittingrms = (
            hdfeos_swaths_omi_total_column_amount_bro_data_fields.create_dataset(
                "SlantFittingRMS",
                data=np.arange(dim_0_1997 * dim_0_60, dtype=np.float64).reshape(
                    (
                        dim_0_1997,
                        dim_0_60,
                    )
                ),
            )
        )
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_slantfittingrms.attrs[
            "MissingValue"
        ] = np.array([np.float64(np.float64(-1e30))], dtype=np.float64)
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_slantfittingrms.attrs[
            "Offset"
        ] = np.array([np.float64(np.float64(0.0))], dtype=np.float64)
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_slantfittingrms.attrs[
            "ScaleFactor"
        ] = np.array([np.float64(np.float64(1.0))], dtype=np.float64)
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_slantfittingrms.attrs[
            "Title"
        ] = "Slant Fitting RMS"
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_slantfittingrms.attrs[
            "UniqueFieldDefinition"
        ] = "OMI-Specific"
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_slantfittingrms.attrs[
            "Units"
        ] = "NoUnits"
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_slantfittingrms.attrs[
            "ValidRange"
        ] = np.array(
            [np.float64(np.float64(0.0)), np.float64(np.float64(1e30))],
            dtype=np.float64,
        )
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_slantfittingrms.attrs[
            "_FillValue"
        ] = np.array([np.float64(np.float64(-1e30))], dtype=np.float64)

        # HDFEOS/SWATHS/OMI Total Column Amount BrO/Data Fields/SolarWavCalConvergenceFlag
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_solarwavcalconvergenceflag = hdfeos_swaths_omi_total_column_amount_bro_data_fields.create_dataset(
            "SolarWavCalConvergenceFlag",
            data=np.arange(dim_0_60, dtype=np.int16),
        )
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_solarwavcalconvergenceflag.attrs[
            "MissingValue"
        ] = np.array([np.int16(np.int16(-30000))], dtype=np.int16)
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_solarwavcalconvergenceflag.attrs[
            "Offset"
        ] = np.array([np.float64(np.float64(0.0))], dtype=np.float64)
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_solarwavcalconvergenceflag.attrs[
            "ScaleFactor"
        ] = np.array([np.float64(np.float64(1.0))], dtype=np.float64)
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_solarwavcalconvergenceflag.attrs[
            "Title"
        ] = "Solar Wavelength Calibration Convergence Flag"
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_solarwavcalconvergenceflag.attrs[
            "UniqueFieldDefinition"
        ] = "OMI-Specific"
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_solarwavcalconvergenceflag.attrs[
            "Units"
        ] = "NoUnits"
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_solarwavcalconvergenceflag.attrs[
            "ValidRange"
        ] = np.array(
            [np.int16(np.int16(-10)), np.int16(np.int16(12344))], dtype=np.int16
        )
        hdfeos_swaths_omi_total_column_amount_bro_data_fields_solarwavcalconvergenceflag.attrs[
            "_FillValue"
        ] = np.array([np.int16(np.int16(-30000))], dtype=np.int16)

        # HDFEOS/SWATHS/OMI Total Column Amount BrO/Geolocation Fields/Latitude
        hdfeos_swaths_omi_total_column_amount_bro_geolocation_fields_latitude = (
            hdfeos_swaths_omi_total_column_amount_bro_geolocation_fields.create_dataset(
                "Latitude",
                data=np.linspace(-90.0, 90.0, dim_0_1997 * dim_0_60)
                .reshape(
                    (
                        dim_0_1997,
                        dim_0_60,
                    )
                )
                .astype(np.float32),
            )
        )
        hdfeos_swaths_omi_total_column_amount_bro_geolocation_fields_latitude.attrs[
            "MissingValue"
        ] = np.array([np.float32(np.float32(-1e30))], dtype=np.float32)
        hdfeos_swaths_omi_total_column_amount_bro_geolocation_fields_latitude.attrs[
            "Offset"
        ] = np.array([np.float64(np.float64(0.0))], dtype=np.float64)
        hdfeos_swaths_omi_total_column_amount_bro_geolocation_fields_latitude.attrs[
            "ScaleFactor"
        ] = np.array([np.float64(np.float64(1.0))], dtype=np.float64)
        hdfeos_swaths_omi_total_column_amount_bro_geolocation_fields_latitude.attrs[
            "Title"
        ] = "Geodetic Latitude"
        hdfeos_swaths_omi_total_column_amount_bro_geolocation_fields_latitude.attrs[
            "UniqueFieldDefinition"
        ] = "Aura-Shared"
        hdfeos_swaths_omi_total_column_amount_bro_geolocation_fields_latitude.attrs[
            "Units"
        ] = "deg"
        hdfeos_swaths_omi_total_column_amount_bro_geolocation_fields_latitude.attrs[
            "ValidRange"
        ] = np.array(
            [np.float32(np.float32(-90.0)), np.float32(np.float32(90.0))],
            dtype=np.float32,
        )
        hdfeos_swaths_omi_total_column_amount_bro_geolocation_fields_latitude.attrs[
            "_FillValue"
        ] = np.array([np.float32(np.float32(-1e30))], dtype=np.float32)

        # HDFEOS/SWATHS/OMI Total Column Amount BrO/Geolocation Fields/Longitude
        hdfeos_swaths_omi_total_column_amount_bro_geolocation_fields_longitude = (
            hdfeos_swaths_omi_total_column_amount_bro_geolocation_fields.create_dataset(
                "Longitude",
                data=np.linspace(-180.0, 180.0, dim_0_1997 * dim_0_60)
                .reshape(
                    (
                        dim_0_1997,
                        dim_0_60,
                    )
                )
                .astype(np.float32),
            )
        )
        hdfeos_swaths_omi_total_column_amount_bro_geolocation_fields_longitude.attrs[
            "MissingValue"
        ] = np.array([np.float32(np.float32(-1e30))], dtype=np.float32)
        hdfeos_swaths_omi_total_column_amount_bro_geolocation_fields_longitude.attrs[
            "Offset"
        ] = np.array([np.float64(np.float64(0.0))], dtype=np.float64)
        hdfeos_swaths_omi_total_column_amount_bro_geolocation_fields_longitude.attrs[
            "ScaleFactor"
        ] = np.array([np.float64(np.float64(1.0))], dtype=np.float64)
        hdfeos_swaths_omi_total_column_amount_bro_geolocation_fields_longitude.attrs[
            "Title"
        ] = "Geodetic Longitude"
        hdfeos_swaths_omi_total_column_amount_bro_geolocation_fields_longitude.attrs[
            "UniqueFieldDefinition"
        ] = "Aura-Shared"
        hdfeos_swaths_omi_total_column_amount_bro_geolocation_fields_longitude.attrs[
            "Units"
        ] = "deg"
        hdfeos_swaths_omi_total_column_amount_bro_geolocation_fields_longitude.attrs[
            "ValidRange"
        ] = np.array(
            [np.float32(np.float32(-180.0)), np.float32(np.float32(180.0))],
            dtype=np.float32,
        )
        hdfeos_swaths_omi_total_column_amount_bro_geolocation_fields_longitude.attrs[
            "_FillValue"
        ] = np.array([np.float32(np.float32(-1e30))], dtype=np.float32)

        # HDFEOS/SWATHS/OMI Total Column Amount BrO/Geolocation Fields/SolarAzimuthAngle
        hdfeos_swaths_omi_total_column_amount_bro_geolocation_fields_solarazimuthangle = hdfeos_swaths_omi_total_column_amount_bro_geolocation_fields.create_dataset(
            "SolarAzimuthAngle",
            data=np.arange(dim_0_1997 * dim_0_60, dtype=np.float32).reshape(
                (
                    dim_0_1997,
                    dim_0_60,
                )
            ),
        )
        hdfeos_swaths_omi_total_column_amount_bro_geolocation_fields_solarazimuthangle.attrs[
            "MissingValue"
        ] = np.array([np.float32(np.float32(-1e30))], dtype=np.float32)
        hdfeos_swaths_omi_total_column_amount_bro_geolocation_fields_solarazimuthangle.attrs[
            "Offset"
        ] = np.array([np.float64(np.float64(0.0))], dtype=np.float64)
        hdfeos_swaths_omi_total_column_amount_bro_geolocation_fields_solarazimuthangle.attrs[
            "ScaleFactor"
        ] = np.array([np.float64(np.float64(1.0))], dtype=np.float64)
        hdfeos_swaths_omi_total_column_amount_bro_geolocation_fields_solarazimuthangle.attrs[
            "Title"
        ] = "Solar Azimuth Angle"
        hdfeos_swaths_omi_total_column_amount_bro_geolocation_fields_solarazimuthangle.attrs[
            "UniqueFieldDefinition"
        ] = "Aura-Shared"
        hdfeos_swaths_omi_total_column_amount_bro_geolocation_fields_solarazimuthangle.attrs[
            "Units"
        ] = "deg"
        hdfeos_swaths_omi_total_column_amount_bro_geolocation_fields_solarazimuthangle.attrs[
            "ValidRange"
        ] = np.array(
            [np.float32(np.float32(-180.0)), np.float32(np.float32(180.0))],
            dtype=np.float32,
        )
        hdfeos_swaths_omi_total_column_amount_bro_geolocation_fields_solarazimuthangle.attrs[
            "_FillValue"
        ] = np.array([np.float32(np.float32(-1e30))], dtype=np.float32)

        # HDFEOS/SWATHS/OMI Total Column Amount BrO/Geolocation Fields/SolarZenithAngle
        hdfeos_swaths_omi_total_column_amount_bro_geolocation_fields_solarzenithangle = hdfeos_swaths_omi_total_column_amount_bro_geolocation_fields.create_dataset(
            "SolarZenithAngle",
            data=np.arange(dim_0_1997 * dim_0_60, dtype=np.float32).reshape(
                (
                    dim_0_1997,
                    dim_0_60,
                )
            ),
        )
        hdfeos_swaths_omi_total_column_amount_bro_geolocation_fields_solarzenithangle.attrs[
            "MissingValue"
        ] = np.array([np.float32(np.float32(-1e30))], dtype=np.float32)
        hdfeos_swaths_omi_total_column_amount_bro_geolocation_fields_solarzenithangle.attrs[
            "Offset"
        ] = np.array([np.float64(np.float64(0.0))], dtype=np.float64)
        hdfeos_swaths_omi_total_column_amount_bro_geolocation_fields_solarzenithangle.attrs[
            "ScaleFactor"
        ] = np.array([np.float64(np.float64(1.0))], dtype=np.float64)
        hdfeos_swaths_omi_total_column_amount_bro_geolocation_fields_solarzenithangle.attrs[
            "Title"
        ] = "Solar Zenith Angle"
        hdfeos_swaths_omi_total_column_amount_bro_geolocation_fields_solarzenithangle.attrs[
            "UniqueFieldDefinition"
        ] = "Aura-Shared"
        hdfeos_swaths_omi_total_column_amount_bro_geolocation_fields_solarzenithangle.attrs[
            "Units"
        ] = "deg"
        hdfeos_swaths_omi_total_column_amount_bro_geolocation_fields_solarzenithangle.attrs[
            "ValidRange"
        ] = np.array(
            [np.float32(np.float32(0.0)), np.float32(np.float32(180.0))],
            dtype=np.float32,
        )
        hdfeos_swaths_omi_total_column_amount_bro_geolocation_fields_solarzenithangle.attrs[
            "_FillValue"
        ] = np.array([np.float32(np.float32(-1e30))], dtype=np.float32)

        # HDFEOS/SWATHS/OMI Total Column Amount BrO/Geolocation Fields/SpacecraftAltitude
        hdfeos_swaths_omi_total_column_amount_bro_geolocation_fields_spacecraftaltitude = hdfeos_swaths_omi_total_column_amount_bro_geolocation_fields.create_dataset(
            "SpacecraftAltitude",
            data=np.arange(dim_0_1997, dtype=np.float32),
        )
        hdfeos_swaths_omi_total_column_amount_bro_geolocation_fields_spacecraftaltitude.attrs[
            "MissingValue"
        ] = np.array([np.float32(np.float32(-1e30))], dtype=np.float32)
        hdfeos_swaths_omi_total_column_amount_bro_geolocation_fields_spacecraftaltitude.attrs[
            "Offset"
        ] = np.array([np.float64(np.float64(0.0))], dtype=np.float64)
        hdfeos_swaths_omi_total_column_amount_bro_geolocation_fields_spacecraftaltitude.attrs[
            "ScaleFactor"
        ] = np.array([np.float64(np.float64(1.0))], dtype=np.float64)
        hdfeos_swaths_omi_total_column_amount_bro_geolocation_fields_spacecraftaltitude.attrs[
            "Title"
        ] = "Altitude of Aura Spacecraft"
        hdfeos_swaths_omi_total_column_amount_bro_geolocation_fields_spacecraftaltitude.attrs[
            "UniqueFieldDefinition"
        ] = "Aura-Shared"
        hdfeos_swaths_omi_total_column_amount_bro_geolocation_fields_spacecraftaltitude.attrs[
            "Units"
        ] = "m"
        hdfeos_swaths_omi_total_column_amount_bro_geolocation_fields_spacecraftaltitude.attrs[
            "ValidRange"
        ] = np.array(
            [np.float32(np.float32(0.0)), np.float32(np.float32(1e30))],
            dtype=np.float32,
        )
        hdfeos_swaths_omi_total_column_amount_bro_geolocation_fields_spacecraftaltitude.attrs[
            "_FillValue"
        ] = np.array([np.float32(np.float32(-1e30))], dtype=np.float32)

        # HDFEOS/SWATHS/OMI Total Column Amount BrO/Geolocation Fields/TerrainHeight
        hdfeos_swaths_omi_total_column_amount_bro_geolocation_fields_terrainheight = (
            hdfeos_swaths_omi_total_column_amount_bro_geolocation_fields.create_dataset(
                "TerrainHeight",
                data=np.arange(dim_0_1997 * dim_0_60, dtype=np.int16).reshape(
                    (
                        dim_0_1997,
                        dim_0_60,
                    )
                ),
            )
        )
        hdfeos_swaths_omi_total_column_amount_bro_geolocation_fields_terrainheight.attrs[
            "MissingValue"
        ] = np.array([np.int16(np.int16(-30000))], dtype=np.int16)
        hdfeos_swaths_omi_total_column_amount_bro_geolocation_fields_terrainheight.attrs[
            "Offset"
        ] = np.array([np.float64(np.float64(0.0))], dtype=np.float64)
        hdfeos_swaths_omi_total_column_amount_bro_geolocation_fields_terrainheight.attrs[
            "ScaleFactor"
        ] = np.array([np.float64(np.float64(1.0))], dtype=np.float64)
        hdfeos_swaths_omi_total_column_amount_bro_geolocation_fields_terrainheight.attrs[
            "Title"
        ] = "Terrain Height"
        hdfeos_swaths_omi_total_column_amount_bro_geolocation_fields_terrainheight.attrs[
            "UniqueFieldDefinition"
        ] = "Aura-Shared"
        hdfeos_swaths_omi_total_column_amount_bro_geolocation_fields_terrainheight.attrs[
            "Units"
        ] = "m"
        hdfeos_swaths_omi_total_column_amount_bro_geolocation_fields_terrainheight.attrs[
            "ValidRange"
        ] = np.array(
            [np.int16(np.int16(-1000)), np.int16(np.int16(10000))], dtype=np.int16
        )
        hdfeos_swaths_omi_total_column_amount_bro_geolocation_fields_terrainheight.attrs[
            "_FillValue"
        ] = np.array([np.int16(np.int16(-30000))], dtype=np.int16)

        # HDFEOS/SWATHS/OMI Total Column Amount BrO/Geolocation Fields/Time
        hdfeos_swaths_omi_total_column_amount_bro_geolocation_fields_time = (
            hdfeos_swaths_omi_total_column_amount_bro_geolocation_fields.create_dataset(
                "Time",
                data=np.linspace(0.0, 1000000000.0, dim_0_1997).astype(np.float64),
            )
        )
        hdfeos_swaths_omi_total_column_amount_bro_geolocation_fields_time.attrs[
            "MissingValue"
        ] = np.array([np.float64(np.float64(-1e30))], dtype=np.float64)
        hdfeos_swaths_omi_total_column_amount_bro_geolocation_fields_time.attrs[
            "Offset"
        ] = np.array([np.float64(np.float64(0.0))], dtype=np.float64)
        hdfeos_swaths_omi_total_column_amount_bro_geolocation_fields_time.attrs[
            "ScaleFactor"
        ] = np.array([np.float64(np.float64(1.0))], dtype=np.float64)
        hdfeos_swaths_omi_total_column_amount_bro_geolocation_fields_time.attrs[
            "Title"
        ] = "Time in TAI units"
        hdfeos_swaths_omi_total_column_amount_bro_geolocation_fields_time.attrs[
            "UniqueFieldDefinition"
        ] = "Aura-Shared"
        hdfeos_swaths_omi_total_column_amount_bro_geolocation_fields_time.attrs[
            "Units"
        ] = "s"
        hdfeos_swaths_omi_total_column_amount_bro_geolocation_fields_time.attrs[
            "ValidRange"
        ] = np.array(
            [np.float64(np.float64(0.0)), np.float64(np.float64(10000000000.0))],
            dtype=np.float64,
        )
        hdfeos_swaths_omi_total_column_amount_bro_geolocation_fields_time.attrs[
            "_FillValue"
        ] = np.array([np.float64(np.float64(-1e30))], dtype=np.float64)

        # HDFEOS/SWATHS/OMI Total Column Amount BrO/Geolocation Fields/TimeUTC
        hdfeos_swaths_omi_total_column_amount_bro_geolocation_fields_timeutc = (
            hdfeos_swaths_omi_total_column_amount_bro_geolocation_fields.create_dataset(
                "TimeUTC",
                data=np.arange(dim_0_1997 * dim_1_6, dtype=np.int16).reshape(
                    (
                        dim_0_1997,
                        dim_1_6,
                    )
                ),
            )
        )
        hdfeos_swaths_omi_total_column_amount_bro_geolocation_fields_timeutc.attrs[
            "MissingValue"
        ] = np.array([np.int16(np.int16(-30000))], dtype=np.int16)
        hdfeos_swaths_omi_total_column_amount_bro_geolocation_fields_timeutc.attrs[
            "Offset"
        ] = np.array([np.float64(np.float64(0.0))], dtype=np.float64)
        hdfeos_swaths_omi_total_column_amount_bro_geolocation_fields_timeutc.attrs[
            "ScaleFactor"
        ] = np.array([np.float64(np.float64(1.0))], dtype=np.float64)
        hdfeos_swaths_omi_total_column_amount_bro_geolocation_fields_timeutc.attrs[
            "Title"
        ] = "Coordianted Universal Time"
        hdfeos_swaths_omi_total_column_amount_bro_geolocation_fields_timeutc.attrs[
            "UniqueFieldDefinition"
        ] = "OMI-Specific"
        hdfeos_swaths_omi_total_column_amount_bro_geolocation_fields_timeutc.attrs[
            "Units"
        ] = "NoUnits"
        hdfeos_swaths_omi_total_column_amount_bro_geolocation_fields_timeutc.attrs[
            "ValidRange"
        ] = np.array([np.int16(np.int16(0)), np.int16(np.int16(9999))], dtype=np.int16)
        hdfeos_swaths_omi_total_column_amount_bro_geolocation_fields_timeutc.attrs[
            "_FillValue"
        ] = np.array([np.int16(np.int16(-30000))], dtype=np.int16)

        # HDFEOS/SWATHS/OMI Total Column Amount BrO/Geolocation Fields/ViewingAzimuthAngle
        hdfeos_swaths_omi_total_column_amount_bro_geolocation_fields_viewingazimuthangle = hdfeos_swaths_omi_total_column_amount_bro_geolocation_fields.create_dataset(
            "ViewingAzimuthAngle",
            data=np.arange(dim_0_1997 * dim_0_60, dtype=np.float32).reshape(
                (
                    dim_0_1997,
                    dim_0_60,
                )
            ),
        )
        hdfeos_swaths_omi_total_column_amount_bro_geolocation_fields_viewingazimuthangle.attrs[
            "MissingValue"
        ] = np.array([np.float32(np.float32(-1e30))], dtype=np.float32)
        hdfeos_swaths_omi_total_column_amount_bro_geolocation_fields_viewingazimuthangle.attrs[
            "Offset"
        ] = np.array([np.float64(np.float64(0.0))], dtype=np.float64)
        hdfeos_swaths_omi_total_column_amount_bro_geolocation_fields_viewingazimuthangle.attrs[
            "ScaleFactor"
        ] = np.array([np.float64(np.float64(1.0))], dtype=np.float64)
        hdfeos_swaths_omi_total_column_amount_bro_geolocation_fields_viewingazimuthangle.attrs[
            "Title"
        ] = "Viewing Azimuth Angle"
        hdfeos_swaths_omi_total_column_amount_bro_geolocation_fields_viewingazimuthangle.attrs[
            "UniqueFieldDefinition"
        ] = "OMI-Specific"
        hdfeos_swaths_omi_total_column_amount_bro_geolocation_fields_viewingazimuthangle.attrs[
            "Units"
        ] = "deg"
        hdfeos_swaths_omi_total_column_amount_bro_geolocation_fields_viewingazimuthangle.attrs[
            "ValidRange"
        ] = np.array(
            [np.float32(np.float32(-180.0)), np.float32(np.float32(180.0))],
            dtype=np.float32,
        )
        hdfeos_swaths_omi_total_column_amount_bro_geolocation_fields_viewingazimuthangle.attrs[
            "_FillValue"
        ] = np.array([np.float32(np.float32(-1e30))], dtype=np.float32)

        # HDFEOS/SWATHS/OMI Total Column Amount BrO/Geolocation Fields/ViewingZenithAngle
        hdfeos_swaths_omi_total_column_amount_bro_geolocation_fields_viewingzenithangle = hdfeos_swaths_omi_total_column_amount_bro_geolocation_fields.create_dataset(
            "ViewingZenithAngle",
            data=np.arange(dim_0_1997 * dim_0_60, dtype=np.float32).reshape(
                (
                    dim_0_1997,
                    dim_0_60,
                )
            ),
        )
        hdfeos_swaths_omi_total_column_amount_bro_geolocation_fields_viewingzenithangle.attrs[
            "MissingValue"
        ] = np.array([np.float32(np.float32(-1e30))], dtype=np.float32)
        hdfeos_swaths_omi_total_column_amount_bro_geolocation_fields_viewingzenithangle.attrs[
            "Offset"
        ] = np.array([np.float64(np.float64(0.0))], dtype=np.float64)
        hdfeos_swaths_omi_total_column_amount_bro_geolocation_fields_viewingzenithangle.attrs[
            "ScaleFactor"
        ] = np.array([np.float64(np.float64(1.0))], dtype=np.float64)
        hdfeos_swaths_omi_total_column_amount_bro_geolocation_fields_viewingzenithangle.attrs[
            "Title"
        ] = "Viewing Zenith Angle"
        hdfeos_swaths_omi_total_column_amount_bro_geolocation_fields_viewingzenithangle.attrs[
            "UniqueFieldDefinition"
        ] = "OMI-Specific"
        hdfeos_swaths_omi_total_column_amount_bro_geolocation_fields_viewingzenithangle.attrs[
            "Units"
        ] = "deg"
        hdfeos_swaths_omi_total_column_amount_bro_geolocation_fields_viewingzenithangle.attrs[
            "ValidRange"
        ] = np.array(
            [np.float32(np.float32(0.0)), np.float32(np.float32(180.0))],
            dtype=np.float32,
        )
        hdfeos_swaths_omi_total_column_amount_bro_geolocation_fields_viewingzenithangle.attrs[
            "_FillValue"
        ] = np.array([np.float32(np.float32(-1e30))], dtype=np.float32)

        # HDFEOS/SWATHS/OMI Total Column Amount BrO/Geolocation Fields/XTrackQualityFlags
        hdfeos_swaths_omi_total_column_amount_bro_geolocation_fields_xtrackqualityflags = hdfeos_swaths_omi_total_column_amount_bro_geolocation_fields.create_dataset(
            "XTrackQualityFlags",
            data=np.arange(dim_0_1997 * dim_0_60, dtype=np.int8).reshape(
                (
                    dim_0_1997,
                    dim_0_60,
                )
            ),
        )
        hdfeos_swaths_omi_total_column_amount_bro_geolocation_fields_xtrackqualityflags.attrs[
            "MissingValue"
        ] = np.array([np.int8(np.int8(-127))], dtype=np.int8)
        hdfeos_swaths_omi_total_column_amount_bro_geolocation_fields_xtrackqualityflags.attrs[
            "Offset"
        ] = np.array([np.float64(np.float64(0.0))], dtype=np.float64)
        hdfeos_swaths_omi_total_column_amount_bro_geolocation_fields_xtrackqualityflags.attrs[
            "ScaleFactor"
        ] = np.array([np.float64(np.float64(1.0))], dtype=np.float64)
        hdfeos_swaths_omi_total_column_amount_bro_geolocation_fields_xtrackqualityflags.attrs[
            "Title"
        ] = "Cross-Track Quality Flags"
        hdfeos_swaths_omi_total_column_amount_bro_geolocation_fields_xtrackqualityflags.attrs[
            "UniqueFieldDefinition"
        ] = "OMI-Specific"
        hdfeos_swaths_omi_total_column_amount_bro_geolocation_fields_xtrackqualityflags.attrs[
            "Units"
        ] = "NoUnits"
        hdfeos_swaths_omi_total_column_amount_bro_geolocation_fields_xtrackqualityflags.attrs[
            "ValidRange"
        ] = np.array([np.int8(np.int8(0)), np.int8(np.int8(127))], dtype=np.int8)
        hdfeos_swaths_omi_total_column_amount_bro_geolocation_fields_xtrackqualityflags.attrs[
            "_FillValue"
        ] = np.array([np.int8(np.int8(-127))], dtype=np.int8)

        # HDFEOS/SWATHS/OMI Total Column Amount BrO/Geolocation Fields/XTrackQualityFlagsExpanded
        hdfeos_swaths_omi_total_column_amount_bro_geolocation_fields_xtrackqualityflagsexpanded = hdfeos_swaths_omi_total_column_amount_bro_geolocation_fields.create_dataset(
            "XTrackQualityFlagsExpanded",
            data=np.arange(dim_0_1997 * dim_0_60, dtype=np.int16).reshape(
                (
                    dim_0_1997,
                    dim_0_60,
                )
            ),
        )
        hdfeos_swaths_omi_total_column_amount_bro_geolocation_fields_xtrackqualityflagsexpanded.attrs[
            "MissingValue"
        ] = np.array([np.int16(np.int16(-30000))], dtype=np.int16)
        hdfeos_swaths_omi_total_column_amount_bro_geolocation_fields_xtrackqualityflagsexpanded.attrs[
            "Offset"
        ] = np.array([np.float64(np.float64(0.0))], dtype=np.float64)
        hdfeos_swaths_omi_total_column_amount_bro_geolocation_fields_xtrackqualityflagsexpanded.attrs[
            "ScaleFactor"
        ] = np.array([np.float64(np.float64(1.0))], dtype=np.float64)
        hdfeos_swaths_omi_total_column_amount_bro_geolocation_fields_xtrackqualityflagsexpanded.attrs[
            "Title"
        ] = "Expanded Cross-Track Quality Flags"
        hdfeos_swaths_omi_total_column_amount_bro_geolocation_fields_xtrackqualityflagsexpanded.attrs[
            "UniqueFieldDefinition"
        ] = "OMI-Specific"
        hdfeos_swaths_omi_total_column_amount_bro_geolocation_fields_xtrackqualityflagsexpanded.attrs[
            "Units"
        ] = "NoUnits"
        hdfeos_swaths_omi_total_column_amount_bro_geolocation_fields_xtrackqualityflagsexpanded.attrs[
            "ValidRange"
        ] = np.array([np.int16(np.int16(0)), np.int16(np.int16(11147))], dtype=np.int16)
        hdfeos_swaths_omi_total_column_amount_bro_geolocation_fields_xtrackqualityflagsexpanded.attrs[
            "_FillValue"
        ] = np.array([np.int16(np.int16(-30000))], dtype=np.int16)

        # HDFEOS INFORMATION/ArchiveMetadata
        hdfeos_information_archivemetadata_text = (
            "\n"
            "GROUP                  = ARCHIVEDMETADATA\n"
            "  GROUPTYPE            = MASTERGROUP\n"
            "\n"
            "  OBJECT                 = LONGNAME\n"
            "    NUM_VAL              = 1\n"
            '    VALUE                = "OMI/Aura Bromine Monoxide (BrO) Total Column 1-Orbit L2 Swath 13x24km"\n'
            "  END_OBJECT             = LONGNAME\n"
            "\n"
            "  OBJECT                 = ESDTDESCRIPTORREVISION\n"
            "    NUM_VAL              = 1\n"
            '    VALUE                = "3.0.4"\n'
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
            '      VALUE                = "OMI-Aura_L2-OMBRO_2020m0116t1207-o82471_v003-2020m0116t182003.he5"\n'
            "    END_OBJECT             = LOCALGRANULEID\n"
            "\n"
            "    OBJECT                 = PRODUCTIONDATETIME\n"
            "      NUM_VAL              = 1\n"
            '      VALUE                = "2020-01-16T18:20:03.000Z"\n'
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
            '          VALUE                = "Flag set to Passed if NrofGoodOutputSamples >= 75% of NrofGoodInputSamples, to Suspect if NrofGoodOutputSamples >= 50% but < 75% of NrofGoodInputSamples, and to Failed if NrofGoodOutputSamples < 50% of NrofGoodInputSamples."\n'
            "        END_OBJECT             = AUTOMATICQUALITYFLAGEXPLANATION\n"
            "\n"
            "        OBJECT                 = AUTOMATICQUALITYFLAG\n"
            "          NUM_VAL              = 1\n"
            '          CLASS                = "1"\n'
            '          VALUE                = "Passed"\n'
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
            "          VALUE                = 26\n"
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
            '        VALUE                = "Total Column Bromine Oxide"\n'
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
            '        VALUE                = "2020-01-16"\n'
            "      END_OBJECT             = EQUATORCROSSINGDATE\n"
            "\n"
            "      OBJECT                 = EQUATORCROSSINGTIME\n"
            '        CLASS                = "1"\n'
            "        NUM_VAL              = 1\n"
            '        VALUE                = "13:02:08.000000"\n'
            "      END_OBJECT             = EQUATORCROSSINGTIME\n"
            "\n"
            "      OBJECT                 = ORBITNUMBER\n"
            '        CLASS                = "1"\n'
            "        NUM_VAL              = 1\n"
            "        VALUE                = 82471\n"
            "      END_OBJECT             = ORBITNUMBER\n"
            "\n"
            "      OBJECT                 = EQUATORCROSSINGLONGITUDE\n"
            '        CLASS                = "1"\n'
            "        NUM_VAL              = 1\n"
            "        VALUE                = 10.49\n"
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
            '      VALUE                = "OMBRO"\n'
            "    END_OBJECT             = SHORTNAME\n"
            "\n"
            "  END_GROUP              = COLLECTIONDESCRIPTIONCLASS\n"
            "\n"
            "  GROUP                  = INPUTGRANULE\n"
            "\n"
            "    OBJECT                 = INPUTPOINTER\n"
            "      NUM_VAL              = 20\n"
            '      VALUE                = ("OMI-Aura_L1-OML1BRUG_2020m0116t1207-o82471_v003-2020m0116t174434.he4", "OMI-Aura_L2-OMCLDO2_2020m0116t1207-o82471_v003-2020m0116t175450.he5")\n'
            "    END_OBJECT             = INPUTPOINTER\n"
            "\n"
            "  END_GROUP              = INPUTGRANULE\n"
            "\n"
            "  GROUP                  = RANGEDATETIME\n"
            "\n"
            "    OBJECT                 = RANGEENDINGDATE\n"
            "      NUM_VAL              = 1\n"
            '      VALUE                = "2020-01-16"\n'
            "    END_OBJECT             = RANGEENDINGDATE\n"
            "\n"
            "    OBJECT                 = RANGEENDINGTIME\n"
            "      NUM_VAL              = 1\n"
            '      VALUE                = "13:46:09.000000"\n'
            "    END_OBJECT             = RANGEENDINGTIME\n"
            "\n"
            "    OBJECT                 = RANGEBEGINNINGDATE\n"
            "      NUM_VAL              = 1\n"
            '      VALUE                = "2020-01-16"\n'
            "    END_OBJECT             = RANGEBEGINNINGDATE\n"
            "\n"
            "    OBJECT                 = RANGEBEGINNINGTIME\n"
            "      NUM_VAL              = 1\n"
            '      VALUE                = "12:07:16.000000"\n'
            "    END_OBJECT             = RANGEBEGINNINGTIME\n"
            "\n"
            "  END_GROUP              = RANGEDATETIME\n"
            "\n"
            "  GROUP                  = PGEVERSIONCLASS\n"
            "\n"
            "    OBJECT                 = PGEVERSION\n"
            "      NUM_VAL              = 1\n"
            '      VALUE                = ("3.0.9")\n'
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
            '        VALUE                = "CCD Ultra Violet"\n'
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
            '          VALUE                = "1997"\n'
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
            "          NUM_VAL              = 3\n"
            '          CLASS                = "9"\n'
            '          VALUE                = ("0", "1", "2")\n'
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
            '          VALUE                = "226"\n'
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
            '          VALUE                = "6"\n'
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
            '		SwathName="OMI Total Column Amount BrO"\n'
            "		GROUP=Dimension\n"
            "			OBJECT=Dimension_1\n"
            '				DimensionName="nTimes"\n'
            "				Size=1997\n"
            "			END_OBJECT=Dimension_1\n"
            "			OBJECT=Dimension_2\n"
            '				DimensionName="nXtrack"\n'
            "				Size=60\n"
            "			END_OBJECT=Dimension_2\n"
            "			OBJECT=Dimension_3\n"
            '				DimensionName="nTimes+1"\n'
            "				Size=1998\n"
            "			END_OBJECT=Dimension_3\n"
            "			OBJECT=Dimension_4\n"
            '				DimensionName="nXtrack+1"\n'
            "				Size=61\n"
            "			END_OBJECT=Dimension_4\n"
            "			OBJECT=Dimension_5\n"
            '				DimensionName="nUTCdim"\n'
            "				Size=6\n"
            "			END_OBJECT=Dimension_5\n"
            "			OBJECT=Dimension_6\n"
            '				DimensionName="1"\n'
            "				Size=1\n"
            "			END_OBJECT=Dimension_6\n"
            "			OBJECT=Dimension_7\n"
            '				DimensionName="2"\n'
            "				Size=2\n"
            "			END_OBJECT=Dimension_7\n"
            "			OBJECT=Dimension_8\n"
            '				DimensionName="4"\n'
            "				Size=4\n"
            "			END_OBJECT=Dimension_8\n"
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
            '				GeoFieldName="SolarAzimuthAngle"\n'
            "				DataType=H5T_NATIVE_FLOAT\n"
            '				DimList=("nTimes","nXtrack")\n'
            '				MaxdimList=("nTimes","nXtrack")\n'
            "				CompressionType=HE5_HDFE_COMP_SHUF_DEFLATE\n"
            "				DeflateLevel=9\n"
            "			END_OBJECT=GeoField_3\n"
            "			OBJECT=GeoField_4\n"
            '				GeoFieldName="SolarZenithAngle"\n'
            "				DataType=H5T_NATIVE_FLOAT\n"
            '				DimList=("nTimes","nXtrack")\n'
            '				MaxdimList=("nTimes","nXtrack")\n'
            "				CompressionType=HE5_HDFE_COMP_SHUF_DEFLATE\n"
            "				DeflateLevel=9\n"
            "			END_OBJECT=GeoField_4\n"
            "			OBJECT=GeoField_5\n"
            '				GeoFieldName="SpacecraftAltitude"\n'
            "				DataType=H5T_NATIVE_FLOAT\n"
            '				DimList=("nTimes")\n'
            '				MaxdimList=("nTimes")\n'
            "			END_OBJECT=GeoField_5\n"
            "			OBJECT=GeoField_6\n"
            '				GeoFieldName="TerrainHeight"\n'
            "				DataType=H5T_NATIVE_SHORT\n"
            '				DimList=("nTimes","nXtrack")\n'
            '				MaxdimList=("nTimes","nXtrack")\n'
            "				CompressionType=HE5_HDFE_COMP_SHUF_DEFLATE\n"
            "				DeflateLevel=9\n"
            "			END_OBJECT=GeoField_6\n"
            "			OBJECT=GeoField_7\n"
            '				GeoFieldName="Time"\n'
            "				DataType=H5T_NATIVE_DOUBLE\n"
            '				DimList=("nTimes")\n'
            '				MaxdimList=("nTimes")\n'
            "			END_OBJECT=GeoField_7\n"
            "			OBJECT=GeoField_8\n"
            '				GeoFieldName="TimeUTC"\n'
            "				DataType=H5T_NATIVE_SHORT\n"
            '				DimList=("nTimes","nUTCdim")\n'
            '				MaxdimList=("nTimes","nUTCdim")\n'
            "				CompressionType=HE5_HDFE_COMP_SHUF_DEFLATE\n"
            "				DeflateLevel=9\n"
            "			END_OBJECT=GeoField_8\n"
            "			OBJECT=GeoField_9\n"
            '				GeoFieldName="ViewingAzimuthAngle"\n'
            "				DataType=H5T_NATIVE_FLOAT\n"
            '				DimList=("nTimes","nXtrack")\n'
            '				MaxdimList=("nTimes","nXtrack")\n'
            "				CompressionType=HE5_HDFE_COMP_SHUF_DEFLATE\n"
            "				DeflateLevel=9\n"
            "			END_OBJECT=GeoField_9\n"
            "			OBJECT=GeoField_10\n"
            '				GeoFieldName="ViewingZenithAngle"\n'
            "				DataType=H5T_NATIVE_FLOAT\n"
            '				DimList=("nTimes","nXtrack")\n'
            '				MaxdimList=("nTimes","nXtrack")\n'
            "				CompressionType=HE5_HDFE_COMP_SHUF_DEFLATE\n"
            "				DeflateLevel=9\n"
            "			END_OBJECT=GeoField_10\n"
            "			OBJECT=GeoField_11\n"
            '				GeoFieldName="XTrackQualityFlags"\n'
            "				DataType=H5T_NATIVE_SCHAR\n"
            '				DimList=("nTimes","nXtrack")\n'
            '				MaxdimList=("nTimes","nXtrack")\n'
            "				CompressionType=HE5_HDFE_COMP_SHUF_DEFLATE\n"
            "				DeflateLevel=9\n"
            "			END_OBJECT=GeoField_11\n"
            "			OBJECT=GeoField_12\n"
            '				GeoFieldName="XTrackQualityFlagsExpanded"\n'
            "				DataType=H5T_NATIVE_SHORT\n"
            '				DimList=("nTimes","nXtrack")\n'
            '				MaxdimList=("nTimes","nXtrack")\n'
            "				CompressionType=HE5_HDFE_COMP_SHUF_DEFLATE\n"
            "				DeflateLevel=9\n"
            "			END_OBJECT=GeoField_12\n"
            "		END_GROUP=GeoField\n"
            "		GROUP=DataField\n"
            "			OBJECT=DataField_1\n"
            '				DataFieldName="SolarWavCalConvergenceFlag"\n'
            "				DataType=H5T_NATIVE_SHORT\n"
            '				DimList=("nXtrack")\n'
            '				MaxdimList=("nXtrack")\n'
            "			END_OBJECT=DataField_1\n"
            "			OBJECT=DataField_2\n"
            '				DataFieldName="RadianceWavCalConvergenceFlag"\n'
            "				DataType=H5T_NATIVE_SHORT\n"
            '				DimList=("nXtrack")\n'
            '				MaxdimList=("nXtrack")\n'
            "			END_OBJECT=DataField_2\n"
            "			OBJECT=DataField_3\n"
            '				DataFieldName="RadianceWavCalLatitudeRange"\n'
            "				DataType=H5T_NATIVE_FLOAT\n"
            '				DimList=("4")\n'
            '				MaxdimList=("4")\n'
            "			END_OBJECT=DataField_3\n"
            "			OBJECT=DataField_4\n"
            '				DataFieldName="RadianceReferenceConvergenceFlag"\n'
            "				DataType=H5T_NATIVE_SHORT\n"
            '				DimList=("nXtrack")\n'
            '				MaxdimList=("nXtrack")\n'
            "			END_OBJECT=DataField_4\n"
            "			OBJECT=DataField_5\n"
            '				DataFieldName="RadianceReferenceLatitudeRange"\n'
            "				DataType=H5T_NATIVE_FLOAT\n"
            '				DimList=("4")\n'
            '				MaxdimList=("4")\n'
            "			END_OBJECT=DataField_5\n"
            "			OBJECT=DataField_6\n"
            '				DataFieldName="RadianceReferenceColumnAmount"\n'
            "				DataType=H5T_NATIVE_DOUBLE\n"
            '				DimList=("nXtrack")\n'
            '				MaxdimList=("nXtrack")\n'
            "			END_OBJECT=DataField_6\n"
            "			OBJECT=DataField_7\n"
            '				DataFieldName="RadianceReferenceColumnUncertainty"\n'
            "				DataType=H5T_NATIVE_DOUBLE\n"
            '				DimList=("nXtrack")\n'
            '				MaxdimList=("nXtrack")\n'
            "			END_OBJECT=DataField_7\n"
            "			OBJECT=DataField_8\n"
            '				DataFieldName="RadianceReferenceColumnXTRFit"\n'
            "				DataType=H5T_NATIVE_DOUBLE\n"
            '				DimList=("nXtrack")\n'
            '				MaxdimList=("nXtrack")\n'
            "			END_OBJECT=DataField_8\n"
            "			OBJECT=DataField_9\n"
            '				DataFieldName="RadianceReferenceFittingRMS"\n'
            "				DataType=H5T_NATIVE_DOUBLE\n"
            '				DimList=("nXtrack")\n'
            '				MaxdimList=("nXtrack")\n'
            "			END_OBJECT=DataField_9\n"
            "			OBJECT=DataField_10\n"
            '				DataFieldName="AirMassFactor"\n'
            "				DataType=H5T_NATIVE_FLOAT\n"
            '				DimList=("nTimes","nXtrack")\n'
            '				MaxdimList=("nTimes","nXtrack")\n'
            "				CompressionType=HE5_HDFE_COMP_SHUF_DEFLATE\n"
            "				DeflateLevel=9\n"
            "			END_OBJECT=DataField_10\n"
            "			OBJECT=DataField_11\n"
            '				DataFieldName="AirMassFactorDiagnosticFlag"\n'
            "				DataType=H5T_NATIVE_SHORT\n"
            '				DimList=("nTimes","nXtrack")\n'
            '				MaxdimList=("nTimes","nXtrack")\n'
            "				CompressionType=HE5_HDFE_COMP_SHUF_DEFLATE\n"
            "				DeflateLevel=9\n"
            "			END_OBJECT=DataField_11\n"
            "			OBJECT=DataField_12\n"
            '				DataFieldName="AirMassFactorGeometric"\n'
            "				DataType=H5T_NATIVE_FLOAT\n"
            '				DimList=("nTimes","nXtrack")\n'
            '				MaxdimList=("nTimes","nXtrack")\n'
            "				CompressionType=HE5_HDFE_COMP_SHUF_DEFLATE\n"
            "				DeflateLevel=9\n"
            "			END_OBJECT=DataField_12\n"
            "			OBJECT=DataField_13\n"
            '				DataFieldName="AverageColumnAmount"\n'
            "				DataType=H5T_NATIVE_DOUBLE\n"
            '				DimList=("1")\n'
            '				MaxdimList=("1")\n'
            "			END_OBJECT=DataField_13\n"
            "			OBJECT=DataField_14\n"
            '				DataFieldName="AverageColumnUncertainty"\n'
            "				DataType=H5T_NATIVE_DOUBLE\n"
            '				DimList=("1")\n'
            '				MaxdimList=("1")\n'
            "			END_OBJECT=DataField_14\n"
            "			OBJECT=DataField_15\n"
            '				DataFieldName="AverageFittingRMS"\n'
            "				DataType=H5T_NATIVE_DOUBLE\n"
            '				DimList=("1")\n'
            '				MaxdimList=("1")\n'
            "			END_OBJECT=DataField_15\n"
            "			OBJECT=DataField_16\n"
            '				DataFieldName="ColumnAmount"\n'
            "				DataType=H5T_NATIVE_DOUBLE\n"
            '				DimList=("nTimes","nXtrack")\n'
            '				MaxdimList=("nTimes","nXtrack")\n'
            "				CompressionType=HE5_HDFE_COMP_SHUF_DEFLATE\n"
            "				DeflateLevel=9\n"
            "			END_OBJECT=DataField_16\n"
            "			OBJECT=DataField_17\n"
            '				DataFieldName="ColumnAmountDestriped"\n'
            "				DataType=H5T_NATIVE_DOUBLE\n"
            '				DimList=("nTimes","nXtrack")\n'
            '				MaxdimList=("nTimes","nXtrack")\n'
            "				CompressionType=HE5_HDFE_COMP_SHUF_DEFLATE\n"
            "				DeflateLevel=9\n"
            "			END_OBJECT=DataField_17\n"
            "			OBJECT=DataField_18\n"
            '				DataFieldName="ColumnUncertainty"\n'
            "				DataType=H5T_NATIVE_DOUBLE\n"
            '				DimList=("nTimes","nXtrack")\n'
            '				MaxdimList=("nTimes","nXtrack")\n'
            "				CompressionType=HE5_HDFE_COMP_SHUF_DEFLATE\n"
            "				DeflateLevel=9\n"
            "			END_OBJECT=DataField_18\n"
            "			OBJECT=DataField_19\n"
            '				DataFieldName="FittingRMS"\n'
            "				DataType=H5T_NATIVE_DOUBLE\n"
            '				DimList=("nTimes","nXtrack")\n'
            '				MaxdimList=("nTimes","nXtrack")\n'
            "				CompressionType=HE5_HDFE_COMP_SHUF_DEFLATE\n"
            "				DeflateLevel=9\n"
            "			END_OBJECT=DataField_19\n"
            "			OBJECT=DataField_20\n"
            '				DataFieldName="FitConvergenceFlag"\n'
            "				DataType=H5T_NATIVE_SHORT\n"
            '				DimList=("nTimes","nXtrack")\n'
            '				MaxdimList=("nTimes","nXtrack")\n'
            "				CompressionType=HE5_HDFE_COMP_SHUF_DEFLATE\n"
            "				DeflateLevel=9\n"
            "			END_OBJECT=DataField_20\n"
            "			OBJECT=DataField_21\n"
            '				DataFieldName="MainDataQualityFlag"\n'
            "				DataType=H5T_NATIVE_SHORT\n"
            '				DimList=("nTimes","nXtrack")\n'
            '				MaxdimList=("nTimes","nXtrack")\n'
            "				CompressionType=HE5_HDFE_COMP_SHUF_DEFLATE\n"
            "				DeflateLevel=9\n"
            "			END_OBJECT=DataField_21\n"
            "			OBJECT=DataField_22\n"
            '				DataFieldName="MaximumColumnAmount"\n'
            "				DataType=H5T_NATIVE_DOUBLE\n"
            '				DimList=("1")\n'
            '				MaxdimList=("1")\n'
            "			END_OBJECT=DataField_22\n"
            "			OBJECT=DataField_23\n"
            '				DataFieldName="PixelCornerLatitudes"\n'
            "				DataType=H5T_NATIVE_FLOAT\n"
            '				DimList=("nTimes+1","nXtrack+1")\n'
            '				MaxdimList=("nTimes+1","nXtrack+1")\n'
            "				CompressionType=HE5_HDFE_COMP_SHUF_DEFLATE\n"
            "				DeflateLevel=9\n"
            "			END_OBJECT=DataField_23\n"
            "			OBJECT=DataField_24\n"
            '				DataFieldName="PixelCornerLongitudes"\n'
            "				DataType=H5T_NATIVE_FLOAT\n"
            '				DimList=("nTimes+1","nXtrack+1")\n'
            '				MaxdimList=("nTimes+1","nXtrack+1")\n'
            "				CompressionType=HE5_HDFE_COMP_SHUF_DEFLATE\n"
            "				DeflateLevel=9\n"
            "			END_OBJECT=DataField_24\n"
            "			OBJECT=DataField_25\n"
            '				DataFieldName="PixelArea"\n'
            "				DataType=H5T_NATIVE_FLOAT\n"
            '				DimList=("nXtrack")\n'
            '				MaxdimList=("nXtrack")\n'
            "			END_OBJECT=DataField_25\n"
            "			OBJECT=DataField_26\n"
            '				DataFieldName="AMFCloudFraction"\n'
            "				DataType=H5T_NATIVE_FLOAT\n"
            '				DimList=("nTimes","nXtrack")\n'
            '				MaxdimList=("nTimes","nXtrack")\n'
            "				CompressionType=HE5_HDFE_COMP_SHUF_DEFLATE\n"
            "				DeflateLevel=9\n"
            "			END_OBJECT=DataField_26\n"
            "			OBJECT=DataField_27\n"
            '				DataFieldName="AMFCloudPressure"\n'
            "				DataType=H5T_NATIVE_FLOAT\n"
            '				DimList=("nTimes","nXtrack")\n'
            '				MaxdimList=("nTimes","nXtrack")\n'
            "				CompressionType=HE5_HDFE_COMP_SHUF_DEFLATE\n"
            "				DeflateLevel=9\n"
            "			END_OBJECT=DataField_27\n"
            "			OBJECT=DataField_28\n"
            '				DataFieldName="AdjustedSceneAlbedo"\n'
            "				DataType=H5T_NATIVE_FLOAT\n"
            '				DimList=("nTimes","nXtrack")\n'
            '				MaxdimList=("nTimes","nXtrack")\n'
            "				CompressionType=HE5_HDFE_COMP_SHUF_DEFLATE\n"
            "				DeflateLevel=9\n"
            "			END_OBJECT=DataField_28\n"
            "			OBJECT=DataField_29\n"
            '				DataFieldName="SlantColumnAmount"\n'
            "				DataType=H5T_NATIVE_DOUBLE\n"
            '				DimList=("nTimes","nXtrack")\n'
            '				MaxdimList=("nTimes","nXtrack")\n'
            "				CompressionType=HE5_HDFE_COMP_SHUF_DEFLATE\n"
            "				DeflateLevel=9\n"
            "			END_OBJECT=DataField_29\n"
            "			OBJECT=DataField_30\n"
            '				DataFieldName="SlantColumnAmountDestriped"\n'
            "				DataType=H5T_NATIVE_DOUBLE\n"
            '				DimList=("nTimes","nXtrack")\n'
            '				MaxdimList=("nTimes","nXtrack")\n'
            "				CompressionType=HE5_HDFE_COMP_SHUF_DEFLATE\n"
            "				DeflateLevel=9\n"
            "			END_OBJECT=DataField_30\n"
            "			OBJECT=DataField_31\n"
            '				DataFieldName="SlantColumnUncertainty"\n'
            "				DataType=H5T_NATIVE_DOUBLE\n"
            '				DimList=("nTimes","nXtrack")\n'
            '				MaxdimList=("nTimes","nXtrack")\n'
            "				CompressionType=HE5_HDFE_COMP_SHUF_DEFLATE\n"
            "				DeflateLevel=9\n"
            "			END_OBJECT=DataField_31\n"
            "			OBJECT=DataField_32\n"
            '				DataFieldName="SlantFittingRMS"\n'
            "				DataType=H5T_NATIVE_DOUBLE\n"
            '				DimList=("nTimes","nXtrack")\n'
            '				MaxdimList=("nTimes","nXtrack")\n'
            "				CompressionType=HE5_HDFE_COMP_SHUF_DEFLATE\n"
            "				DeflateLevel=9\n"
            "			END_OBJECT=DataField_32\n"
            "			OBJECT=DataField_33\n"
            '				DataFieldName="SlantFitConvergenceFlag"\n'
            "				DataType=H5T_NATIVE_SHORT\n"
            '				DimList=("nTimes","nXtrack")\n'
            '				MaxdimList=("nTimes","nXtrack")\n'
            "				CompressionType=HE5_HDFE_COMP_SHUF_DEFLATE\n"
            "				DeflateLevel=9\n"
            "			END_OBJECT=DataField_33\n"
            "		END_GROUP=DataField\n"
            "		GROUP=ProfileField\n"
            "		END_GROUP=ProfileField\n"
            "		GROUP=MergedFields\n"
            "		END_GROUP=MergedFields\n"
            "	END_GROUP=SWATH_1\n"
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
