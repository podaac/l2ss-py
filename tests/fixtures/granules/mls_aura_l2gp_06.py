import h5py
import numpy as np
import pytest


@pytest.fixture(scope="session")
def fake_mls_aura_l2gp_oh_file(tmp_path_factory):
    """
    Generated from MLS-Aura_L2GP-OH_v06-01-c01_2005d160.he5
    """
    filepath = (
        tmp_path_factory.mktemp("data") / "MLS-Aura_L2GP-OH_v06-01-c01_2005d160.he5"
    )

    dim_0_3495 = 6
    dim_0_49 = 2
    dim_1_49 = 2

    with h5py.File(filepath, "w") as f:
        # create groups
        hdfeos = f.require_group("HDFEOS")
        hdfeos_additional = f.require_group("HDFEOS/ADDITIONAL")
        hdfeos_additional_file_attributes = f.require_group(
            "HDFEOS/ADDITIONAL/FILE_ATTRIBUTES"
        )
        hdfeos_swaths = f.require_group("HDFEOS/SWATHS")
        hdfeos_swaths_oh = f.require_group("HDFEOS/SWATHS/OH")
        hdfeos_swaths_oh_data_fields = f.require_group("HDFEOS/SWATHS/OH/Data Fields")
        hdfeos_swaths_oh_geolocation_fields = f.require_group(
            "HDFEOS/SWATHS/OH/Geolocation Fields"
        )
        hdfeos_swaths_oh_apriori = f.require_group("HDFEOS/SWATHS/OH-APriori")
        hdfeos_swaths_oh_apriori_data_fields = f.require_group(
            "HDFEOS/SWATHS/OH-APriori/Data Fields"
        )
        hdfeos_swaths_oh_apriori_geolocation_fields = f.require_group(
            "HDFEOS/SWATHS/OH-APriori/Geolocation Fields"
        )
        hdfeos_information = f.require_group("HDFEOS INFORMATION")

        # attributes for HDFEOS/ADDITIONAL/FILE_ATTRIBUTES
        hdfeos_additional_file_attributes.attrs["A Priori geos5"] = (
            "/workops/jobs/science/1720152020.0000000/GEOS.it.asm.asm_inst_3hr_glo_L576x361_v72.GEOS5294.2005-06-09T0000.V01.nc4,/workops/jobs/science/1720152020.0000000/GEOS.it.asm.asm_inst_3hr_glo_L576x361_v72.GEOS5294.2005-06-09T0000.V01.nc4,/workops/jobs/science/1720152020.0000000/GEOS.it.asm.asm_inst_3hr_glo_L576x361_v72.GEOS5294.2005-06-09T0000.V01.nc4,/workops/jobs/science/1720152020.0000000/GEOS.it.asm.asm_inst_3hr_glo_L576x361_v72.GEOS5294.2005-06-09T0300.V01.nc4,/workops/jobs/science/1720152020.0000000/GEOS.it.asm.asm_inst_3hr_glo_L576x361_v72.GEOS5294.2005-06-09T0300.V01.nc4,/workops/jobs/science/1720152020.0000000/GEOS.it.asm.asm_inst_3hr_glo_L576x361_v72.GEOS5294.2005-06-09T0300.V01.nc4,/workops/jobs/science/1720152020.0000000/GEOS.it.asm.asm_inst_3hr_glo_L576x361_v72.GEOS5294.2005-06-09T0600.V01.nc4,/workops/jobs/science/1720152020.0000000/GEOS.it.asm.asm_inst_3hr_glo_L576x361_v72.GEOS5294.2005-06-09T0600.V01.nc4,/workops/jobs/science/1720152020.0000000/GEOS.it.asm.asm_inst_3hr_glo_L576x361_v72.GEOS5294.2005-06-09T0600.V01.nc4,/workops/jobs/science/1720152020.0000000/GEOS.it.asm.asm_inst_3hr_glo_L576x361_v72.GEOS5294.2005-06-09T0900.V01.nc4,/workops/jobs/science/1720152020.0000000/GEOS.it.asm.asm_inst_3hr_glo_L576x361_v72.GEOS5294.2005-06-09T0900.V01.nc4,/workops/jobs/science/1720152020.0000000/GEOS.it.asm.asm_inst_3hr_glo_L576x361_v72.GEOS5294.2005-06-09T0900.V01.nc4,/workops/jobs/science/1720152020.0000000/GEOS.it.asm.asm_inst_3hr_glo_L576x361_v72.GEOS5294.2005-06-09T1200.V01.nc4,/workops/jobs/science/1720152020.0000000/GEOS.it.asm.asm_inst_3hr_glo_L576x361_v72.GEOS5294.2005-06-09T1200.V01.nc4,/workops/jobs/science/1720152020.0000000/GEOS.it.asm.asm_inst_3hr_glo_L576x361_v72.GEOS5294.2005-06-09T1200.V01.nc4,/workops/jobs/science/1720152020.0000000/GEOS.it.asm.asm_inst_3hr_glo_L576x361_v72.GEOS5294.2005-06-09T1500.V01.nc4,/workops/jobs/science/1720152020.0000000/GEOS.it.asm.asm_inst_3hr_glo_L576x361_v72.GEOS5294.2005-06-09T1500.V01.nc4,/workops/jobs/science/1720152020.0000000/GEOS.it.asm.asm_inst_3hr_glo_L576x361_v72.GEOS5294.2005-06-09T1500.V01.nc4,/workops/jobs/science/1720152020.0000000/GEOS.it.asm.asm_inst_3hr_glo_L576x361_v72.GEOS5294.2005-06-09T1800.V01.nc4,/workops/jobs/science/1720152020.0000000/GEOS.it.asm.asm_inst_3hr_glo_L576x361_v72.GEOS5294.2005-06-09T1800.V01.nc4,/workops/jobs/science/1720152020.0000000/GEOS.it.asm.asm_inst_3hr_glo_L576x361_v72.GEOS5294.2005-06-09T1800.V01.nc4,/workops/jobs/science/1720152020.0000000/GEOS.it.asm.asm_inst_3hr_glo_L576x361_v72.GEOS5294.2005-06-09T2100.V01.nc4,/workops/jobs/science/1720152020.0000000/GEOS.it.asm.asm_inst_3hr_glo_L576x361_v72.GEOS5294.2005-06-09T2100.V01.nc4,/workops/jobs/science/1720152020.0000000/GEOS.it.asm.asm_inst_3hr_glo_L576x361_v72.GEOS5294.2005-06-09T2100.V01.nc4,/workops/jobs/science/1720152020.0000000/GEOS.it.asm.asm_inst_3hr_glo_L576x361_v72.GEOS5294.2005-06-10T0000.V01.nc4,/workops/jobs/science/1720152020.0000000/GEOS.it.asm.asm_inst_3hr_glo_L576x361_v72.GEOS5294.2005-06-10T0000.V01.nc4,/workops/jobs/science/1720152020.0000000/GEOS.it.asm.asm_inst_3hr_glo_L576x361_v72.GEOS5294.2005-06-10T0000.V01.nc4"
        )
        hdfeos_additional_file_attributes.attrs["A Priori gmao"] = " "
        hdfeos_additional_file_attributes.attrs["A Priori l2aux"] = " "
        hdfeos_additional_file_attributes.attrs["A Priori l2gp"] = " "
        hdfeos_additional_file_attributes.attrs["A Priori ncep"] = " "
        hdfeos_additional_file_attributes.attrs["EndUTC"] = (
            "2005-06-09T23:59:59.999999Z"
        )
        hdfeos_additional_file_attributes.attrs["FirstMAF"] = np.array(
            [np.int32(np.int32(1137216))], dtype=np.int32
        )
        hdfeos_additional_file_attributes.attrs["GranuleDay"] = np.array(
            [np.int32(np.int32(9))], dtype=np.int32
        )
        hdfeos_additional_file_attributes.attrs["GranuleDayOfYear"] = np.array(
            [np.int32(np.int32(160))], dtype=np.int32
        )
        hdfeos_additional_file_attributes.attrs["GranuleMonth"] = np.array(
            [np.int32(np.int32(6))], dtype=np.int32
        )
        hdfeos_additional_file_attributes.attrs["GranuleYear"] = np.array(
            [np.int32(np.int32(2005))], dtype=np.int32
        )
        hdfeos_additional_file_attributes.attrs["HostName"] = "eagle.jpl.nasa.gov"
        hdfeos_additional_file_attributes.attrs["InstrumentName"] = "MLS Aura"
        hdfeos_additional_file_attributes.attrs["LastMAF"] = np.array(
            [np.int32(np.int32(1140723))], dtype=np.int32
        )
        hdfeos_additional_file_attributes.attrs["OrbitNumber"] = np.array(
            [
                np.int32(np.int32(4786)),
                np.int32(np.int32(4787)),
                np.int32(np.int32(4788)),
                np.int32(np.int32(4789)),
                np.int32(np.int32(4790)),
                np.int32(np.int32(4791)),
                np.int32(np.int32(4792)),
                np.int32(np.int32(4793)),
                np.int32(np.int32(4794)),
                np.int32(np.int32(4795)),
                np.int32(np.int32(4796)),
                np.int32(np.int32(4797)),
                np.int32(np.int32(4798)),
                np.int32(np.int32(4799)),
                np.int32(np.int32(4800)),
                np.int32(np.int32(4801)),
            ],
            dtype=np.int32,
        )
        hdfeos_additional_file_attributes.attrs["OrbitPeriod"] = np.array(
            [
                np.float64(np.float64(5933.007736027241)),
                np.float64(np.float64(5932.985953986645)),
                np.float64(np.float64(5932.965357005596)),
                np.float64(np.float64(5932.99115395546)),
                np.float64(np.float64(5933.066319048405)),
                np.float64(np.float64(5933.126186966896)),
                np.float64(np.float64(5933.108877003193)),
                np.float64(np.float64(5933.046346008778)),
                np.float64(np.float64(5932.970535993576)),
                np.float64(np.float64(5932.966917037964)),
                np.float64(np.float64(5932.96606194973)),
                np.float64(np.float64(5933.002448022366)),
                np.float64(np.float64(5933.108573019505)),
                np.float64(np.float64(5933.10717600584)),
                np.float64(np.float64(5933.052326977253)),
                np.float64(np.float64(5933.052326977253)),
            ],
            dtype=np.float64,
        )
        hdfeos_additional_file_attributes.attrs["PGEVersion"] = "v6.01"
        hdfeos_additional_file_attributes.attrs["ProcessLevel"] = "L2"
        hdfeos_additional_file_attributes.attrs["ProductionLocation"] = (
            "eagle.jpl.nasa.gov"
        )
        hdfeos_additional_file_attributes.attrs["StartUTC"] = (
            "2005-06-09T00:00:00.000000Z"
        )
        hdfeos_additional_file_attributes.attrs["TAI93At0zOfGranule"] = np.array(
            [np.float64(np.float64(392428805.0))], dtype=np.float64
        )
        hdfeos_additional_file_attributes.attrs["geos5 type"] = (
            "geos5_7,geos5_7,geos5_7,geos5_7,geos5_7,geos5_7,geos5_7,geos5_7,geos5_7,geos5_7,geos5_7,geos5_7,geos5_7,geos5_7,geos5_7,geos5_7,geos5_7,geos5_7,geos5_7,geos5_7,geos5_7,geos5_7,geos5_7,geos5_7,geos5_7,geos5_7,geos5_7"
        )
        hdfeos_additional_file_attributes.attrs["identifier_product_doi"] = (
            "10.5067/AURA/MLS/DATA2617"
        )

        # attributes for HDFEOS/SWATHS/OH
        hdfeos_swaths_oh.attrs["Pressure"] = np.array(
            [
                np.float32(np.float32(1000.0)),
                np.float32(np.float32(681.29205)),
                np.float32(np.float32(464.15887)),
                np.float32(np.float32(316.22775)),
                np.float32(np.float32(215.44347)),
                np.float32(np.float32(146.77992)),
                np.float32(np.float32(100.0)),
                np.float32(np.float32(68.1292)),
                np.float32(np.float32(46.41589)),
                np.float32(np.float32(31.622776)),
                np.float32(np.float32(21.544348)),
                np.float32(np.float32(14.677993)),
                np.float32(np.float32(10.0)),
                np.float32(np.float32(6.8129206)),
                np.float32(np.float32(4.6415887)),
                np.float32(np.float32(3.1622777)),
                np.float32(np.float32(2.1544347)),
                np.float32(np.float32(1.4677993)),
                np.float32(np.float32(1.0)),
                np.float32(np.float32(0.68129206)),
                np.float32(np.float32(0.4641589)),
                np.float32(np.float32(0.31622776)),
                np.float32(np.float32(0.21544346)),
                np.float32(np.float32(0.14677992)),
                np.float32(np.float32(0.1)),
                np.float32(np.float32(0.068129204)),
                np.float32(np.float32(0.046415888)),
                np.float32(np.float32(0.031622775)),
                np.float32(np.float32(0.021544347)),
                np.float32(np.float32(0.014677993)),
                np.float32(np.float32(0.01)),
                np.float32(np.float32(0.006812921)),
                np.float32(np.float32(0.004641589)),
                np.float32(np.float32(0.0031622776)),
                np.float32(np.float32(0.0021544348)),
                np.float32(np.float32(0.0014677993)),
                np.float32(np.float32(0.001)),
                np.float32(np.float32(0.0006812921)),
                np.float32(np.float32(0.0004641589)),
                np.float32(np.float32(0.00031622776)),
                np.float32(np.float32(0.00021544346)),
                np.float32(np.float32(0.00014677993)),
                np.float32(np.float32(1e-04)),
                np.float32(np.float32(6.812921e-05)),
                np.float32(np.float32(4.6415887e-05)),
                np.float32(np.float32(3.1622778e-05)),
                np.float32(np.float32(2.1544347e-05)),
                np.float32(np.float32(1.4677993e-05)),
                np.float32(np.float32(1e-05)),
            ],
            dtype=np.float32,
        )
        hdfeos_swaths_oh.attrs["VerticalCoordinate"] = "Pressure"

        # attributes for HDFEOS/SWATHS/OH-APriori
        hdfeos_swaths_oh_apriori.attrs["Pressure"] = np.array(
            [
                np.float32(np.float32(1000.0)),
                np.float32(np.float32(681.29205)),
                np.float32(np.float32(464.15887)),
                np.float32(np.float32(316.22775)),
                np.float32(np.float32(215.44347)),
                np.float32(np.float32(146.77992)),
                np.float32(np.float32(100.0)),
                np.float32(np.float32(68.1292)),
                np.float32(np.float32(46.41589)),
                np.float32(np.float32(31.622776)),
                np.float32(np.float32(21.544348)),
                np.float32(np.float32(14.677993)),
                np.float32(np.float32(10.0)),
                np.float32(np.float32(6.8129206)),
                np.float32(np.float32(4.6415887)),
                np.float32(np.float32(3.1622777)),
                np.float32(np.float32(2.1544347)),
                np.float32(np.float32(1.4677993)),
                np.float32(np.float32(1.0)),
                np.float32(np.float32(0.68129206)),
                np.float32(np.float32(0.4641589)),
                np.float32(np.float32(0.31622776)),
                np.float32(np.float32(0.21544346)),
                np.float32(np.float32(0.14677992)),
                np.float32(np.float32(0.1)),
                np.float32(np.float32(0.068129204)),
                np.float32(np.float32(0.046415888)),
                np.float32(np.float32(0.031622775)),
                np.float32(np.float32(0.021544347)),
                np.float32(np.float32(0.014677993)),
                np.float32(np.float32(0.01)),
                np.float32(np.float32(0.006812921)),
                np.float32(np.float32(0.004641589)),
                np.float32(np.float32(0.0031622776)),
                np.float32(np.float32(0.0021544348)),
                np.float32(np.float32(0.0014677993)),
                np.float32(np.float32(0.001)),
                np.float32(np.float32(0.0006812921)),
                np.float32(np.float32(0.0004641589)),
                np.float32(np.float32(0.00031622776)),
                np.float32(np.float32(0.00021544346)),
                np.float32(np.float32(0.00014677993)),
                np.float32(np.float32(1e-04)),
                np.float32(np.float32(6.812921e-05)),
                np.float32(np.float32(4.6415887e-05)),
                np.float32(np.float32(3.1622778e-05)),
                np.float32(np.float32(2.1544347e-05)),
                np.float32(np.float32(1.4677993e-05)),
                np.float32(np.float32(1e-05)),
            ],
            dtype=np.float32,
        )
        hdfeos_swaths_oh_apriori.attrs["VerticalCoordinate"] = "Pressure"

        # attributes for HDFEOS INFORMATION
        hdfeos_information.attrs["HDFEOSVersion"] = "HDFEOS_5.1.14"

        # HDFEOS/SWATHS/OH/Data Fields/Convergence
        hdfeos_swaths_oh_data_fields_convergence = (
            hdfeos_swaths_oh_data_fields.create_dataset(
                "Convergence",
                data=np.arange(dim_0_3495, dtype=np.float32),
            )
        )
        hdfeos_swaths_oh_data_fields_convergence.attrs["MissingValue"] = np.array(
            [np.float32(np.float32(-999.99))], dtype=np.float32
        )
        hdfeos_swaths_oh_data_fields_convergence.attrs["Title"] = "OHConvergence"
        hdfeos_swaths_oh_data_fields_convergence.attrs["UniqueFieldDefinition"] = (
            "MLS-Specific"
        )
        hdfeos_swaths_oh_data_fields_convergence.attrs["Units"] = "NoUnits"
        hdfeos_swaths_oh_data_fields_convergence.attrs["_FillValue"] = np.array(
            [np.float32(np.float32(-999.99))], dtype=np.float32
        )

        # HDFEOS/SWATHS/OH/Data Fields/L2gpPrecision
        hdfeos_swaths_oh_data_fields_l2gpprecision = (
            hdfeos_swaths_oh_data_fields.create_dataset(
                "L2gpPrecision",
                data=np.arange(dim_0_3495 * dim_0_49, dtype=np.float32).reshape(
                    (
                        dim_0_3495,
                        dim_0_49,
                    )
                ),
            )
        )
        hdfeos_swaths_oh_data_fields_l2gpprecision.attrs["MissingValue"] = np.array(
            [np.float32(np.float32(-999.99))], dtype=np.float32
        )
        hdfeos_swaths_oh_data_fields_l2gpprecision.attrs["Title"] = "OHPrecision"
        hdfeos_swaths_oh_data_fields_l2gpprecision.attrs["UniqueFieldDefinition"] = (
            "MLS-Specific"
        )
        hdfeos_swaths_oh_data_fields_l2gpprecision.attrs["Units"] = "vmr"
        hdfeos_swaths_oh_data_fields_l2gpprecision.attrs["_FillValue"] = np.array(
            [np.float32(np.float32(-999.99))], dtype=np.float32
        )

        # HDFEOS/SWATHS/OH/Data Fields/L2gpValue
        hdfeos_swaths_oh_data_fields_l2gpvalue = (
            hdfeos_swaths_oh_data_fields.create_dataset(
                "L2gpValue",
                data=np.arange(dim_0_3495 * dim_0_49, dtype=np.float32).reshape(
                    (
                        dim_0_3495,
                        dim_0_49,
                    )
                ),
            )
        )
        hdfeos_swaths_oh_data_fields_l2gpvalue.attrs["MissingValue"] = np.array(
            [np.float32(np.float32(-999.99))], dtype=np.float32
        )
        hdfeos_swaths_oh_data_fields_l2gpvalue.attrs["Title"] = "OH"
        hdfeos_swaths_oh_data_fields_l2gpvalue.attrs["UniqueFieldDefinition"] = (
            "MLS-Specific"
        )
        hdfeos_swaths_oh_data_fields_l2gpvalue.attrs["Units"] = "vmr"
        hdfeos_swaths_oh_data_fields_l2gpvalue.attrs["_FillValue"] = np.array(
            [np.float32(np.float32(-999.99))], dtype=np.float32
        )

        # HDFEOS/SWATHS/OH/Data Fields/Quality
        hdfeos_swaths_oh_data_fields_quality = (
            hdfeos_swaths_oh_data_fields.create_dataset(
                "Quality",
                data=np.arange(dim_0_3495, dtype=np.float32),
            )
        )
        hdfeos_swaths_oh_data_fields_quality.attrs["MissingValue"] = np.array(
            [np.float32(np.float32(-999.99))], dtype=np.float32
        )
        hdfeos_swaths_oh_data_fields_quality.attrs["Title"] = "OHQuality"
        hdfeos_swaths_oh_data_fields_quality.attrs["UniqueFieldDefinition"] = (
            "MLS-Specific"
        )
        hdfeos_swaths_oh_data_fields_quality.attrs["Units"] = "NoUnits"
        hdfeos_swaths_oh_data_fields_quality.attrs["_FillValue"] = np.array(
            [np.float32(np.float32(-999.99))], dtype=np.float32
        )

        # HDFEOS/SWATHS/OH/Data Fields/Status
        hdfeos_swaths_oh_data_fields_status = (
            hdfeos_swaths_oh_data_fields.create_dataset(
                "Status",
                data=np.arange(dim_0_3495, dtype=np.int32),
            )
        )
        hdfeos_swaths_oh_data_fields_status.attrs["MissingValue"] = np.array(
            [np.int32(np.int32(513))], dtype=np.int32
        )
        hdfeos_swaths_oh_data_fields_status.attrs["Title"] = "OHStatus"
        hdfeos_swaths_oh_data_fields_status.attrs["UniqueFieldDefinition"] = (
            "MLS-Specific"
        )
        hdfeos_swaths_oh_data_fields_status.attrs["Units"] = "NoUnits"
        hdfeos_swaths_oh_data_fields_status.attrs["_FillValue"] = np.array(
            [np.int32(np.int32(513))], dtype=np.int32
        )

        # HDFEOS/SWATHS/OH/Geolocation Fields/ChunkNumber
        hdfeos_swaths_oh_geolocation_fields_chunknumber = (
            hdfeos_swaths_oh_geolocation_fields.create_dataset(
                "ChunkNumber",
                data=np.arange(dim_0_3495, dtype=np.int32),
            )
        )
        hdfeos_swaths_oh_geolocation_fields_chunknumber.attrs["MissingValue"] = (
            np.array([np.int32(np.int32(-999))], dtype=np.int32)
        )
        hdfeos_swaths_oh_geolocation_fields_chunknumber.attrs["Title"] = "ChunkNumber"
        hdfeos_swaths_oh_geolocation_fields_chunknumber.attrs[
            "UniqueFieldDefinition"
        ] = "MLS-Specific"
        hdfeos_swaths_oh_geolocation_fields_chunknumber.attrs["Units"] = "NoUnits"
        hdfeos_swaths_oh_geolocation_fields_chunknumber.attrs["_FillValue"] = np.array(
            [np.int32(np.int32(-999))], dtype=np.int32
        )

        # HDFEOS/SWATHS/OH/Geolocation Fields/Latitude
        hdfeos_swaths_oh_geolocation_fields_latitude = (
            hdfeos_swaths_oh_geolocation_fields.create_dataset(
                "Latitude",
                data=np.linspace(-90.0, 90.0, dim_0_3495).astype(np.float32),
            )
        )
        hdfeos_swaths_oh_geolocation_fields_latitude.attrs["MissingValue"] = np.array(
            [np.float32(np.float32(-999.99))], dtype=np.float32
        )
        hdfeos_swaths_oh_geolocation_fields_latitude.attrs["Title"] = "Latitude"
        hdfeos_swaths_oh_geolocation_fields_latitude.attrs["UniqueFieldDefinition"] = (
            "HIRDLS-MLS-TES-Shared"
        )
        hdfeos_swaths_oh_geolocation_fields_latitude.attrs["Units"] = "deg"
        hdfeos_swaths_oh_geolocation_fields_latitude.attrs["_FillValue"] = np.array(
            [np.float32(np.float32(-999.99))], dtype=np.float32
        )

        # HDFEOS/SWATHS/OH/Geolocation Fields/LineOfSightAngle
        hdfeos_swaths_oh_geolocation_fields_lineofsightangle = (
            hdfeos_swaths_oh_geolocation_fields.create_dataset(
                "LineOfSightAngle",
                data=np.arange(dim_0_3495, dtype=np.float32),
            )
        )
        hdfeos_swaths_oh_geolocation_fields_lineofsightangle.attrs["MissingValue"] = (
            np.array([np.float32(np.float32(-999.99))], dtype=np.float32)
        )
        hdfeos_swaths_oh_geolocation_fields_lineofsightangle.attrs["Title"] = (
            "LineOfSightAngle"
        )
        hdfeos_swaths_oh_geolocation_fields_lineofsightangle.attrs[
            "UniqueFieldDefinition"
        ] = "MLS-Specific"
        hdfeos_swaths_oh_geolocation_fields_lineofsightangle.attrs["Units"] = "deg"
        hdfeos_swaths_oh_geolocation_fields_lineofsightangle.attrs["_FillValue"] = (
            np.array([np.float32(np.float32(-999.99))], dtype=np.float32)
        )

        # HDFEOS/SWATHS/OH/Geolocation Fields/LocalSolarTime
        hdfeos_swaths_oh_geolocation_fields_localsolartime = (
            hdfeos_swaths_oh_geolocation_fields.create_dataset(
                "LocalSolarTime",
                data=np.arange(dim_0_3495, dtype=np.float32),
            )
        )
        hdfeos_swaths_oh_geolocation_fields_localsolartime.attrs["MissingValue"] = (
            np.array([np.float32(np.float32(-999.99))], dtype=np.float32)
        )
        hdfeos_swaths_oh_geolocation_fields_localsolartime.attrs["Title"] = (
            "LocalSolarTime"
        )
        hdfeos_swaths_oh_geolocation_fields_localsolartime.attrs[
            "UniqueFieldDefinition"
        ] = "HIRDLS-MLS-TES-Shared"
        hdfeos_swaths_oh_geolocation_fields_localsolartime.attrs["Units"] = "h"
        hdfeos_swaths_oh_geolocation_fields_localsolartime.attrs["_FillValue"] = (
            np.array([np.float32(np.float32(-999.99))], dtype=np.float32)
        )

        # HDFEOS/SWATHS/OH/Geolocation Fields/Longitude
        hdfeos_swaths_oh_geolocation_fields_longitude = (
            hdfeos_swaths_oh_geolocation_fields.create_dataset(
                "Longitude",
                data=np.linspace(-180.0, 180.0, dim_0_3495).astype(np.float32),
            )
        )
        hdfeos_swaths_oh_geolocation_fields_longitude.attrs["MissingValue"] = np.array(
            [np.float32(np.float32(-999.99))], dtype=np.float32
        )
        hdfeos_swaths_oh_geolocation_fields_longitude.attrs["Title"] = "Longitude"
        hdfeos_swaths_oh_geolocation_fields_longitude.attrs["UniqueFieldDefinition"] = (
            "HIRDLS-MLS-TES-Shared"
        )
        hdfeos_swaths_oh_geolocation_fields_longitude.attrs["Units"] = "deg"
        hdfeos_swaths_oh_geolocation_fields_longitude.attrs["_FillValue"] = np.array(
            [np.float32(np.float32(-999.99))], dtype=np.float32
        )

        # HDFEOS/SWATHS/OH/Geolocation Fields/OrbitGeodeticAngle
        hdfeos_swaths_oh_geolocation_fields_orbitgeodeticangle = (
            hdfeos_swaths_oh_geolocation_fields.create_dataset(
                "OrbitGeodeticAngle",
                data=np.arange(dim_0_3495, dtype=np.float32),
            )
        )
        hdfeos_swaths_oh_geolocation_fields_orbitgeodeticangle.attrs["MissingValue"] = (
            np.array([np.float32(np.float32(-999.99))], dtype=np.float32)
        )
        hdfeos_swaths_oh_geolocation_fields_orbitgeodeticangle.attrs["Title"] = (
            "OrbitGeodeticAngle"
        )
        hdfeos_swaths_oh_geolocation_fields_orbitgeodeticangle.attrs[
            "UniqueFieldDefinition"
        ] = "MLS-Specific"
        hdfeos_swaths_oh_geolocation_fields_orbitgeodeticangle.attrs["Units"] = "deg"
        hdfeos_swaths_oh_geolocation_fields_orbitgeodeticangle.attrs["_FillValue"] = (
            np.array([np.float32(np.float32(-999.99))], dtype=np.float32)
        )

        # HDFEOS/SWATHS/OH/Geolocation Fields/Pressure
        hdfeos_swaths_oh_geolocation_fields_pressure = (
            hdfeos_swaths_oh_geolocation_fields.create_dataset(
                "Pressure",
                data=np.arange(dim_0_49, dtype=np.float32),
            )
        )
        hdfeos_swaths_oh_geolocation_fields_pressure.attrs["MissingValue"] = np.array(
            [np.float32(np.float32(-999.99))], dtype=np.float32
        )
        hdfeos_swaths_oh_geolocation_fields_pressure.attrs["Title"] = "Pressure"
        hdfeos_swaths_oh_geolocation_fields_pressure.attrs["UniqueFieldDefinition"] = (
            "Aura-Shared"
        )
        hdfeos_swaths_oh_geolocation_fields_pressure.attrs["Units"] = "hPa"
        hdfeos_swaths_oh_geolocation_fields_pressure.attrs["_FillValue"] = np.array(
            [np.float32(np.float32(-999.99))], dtype=np.float32
        )

        # HDFEOS/SWATHS/OH/Geolocation Fields/SolarZenithAngle
        hdfeos_swaths_oh_geolocation_fields_solarzenithangle = (
            hdfeos_swaths_oh_geolocation_fields.create_dataset(
                "SolarZenithAngle",
                data=np.arange(dim_0_3495, dtype=np.float32),
            )
        )
        hdfeos_swaths_oh_geolocation_fields_solarzenithangle.attrs["MissingValue"] = (
            np.array([np.float32(np.float32(-999.99))], dtype=np.float32)
        )
        hdfeos_swaths_oh_geolocation_fields_solarzenithangle.attrs["Title"] = (
            "SolarZenithAngle"
        )
        hdfeos_swaths_oh_geolocation_fields_solarzenithangle.attrs[
            "UniqueFieldDefinition"
        ] = "HIRDLS-MLS-TES-Shared"
        hdfeos_swaths_oh_geolocation_fields_solarzenithangle.attrs["Units"] = "deg"
        hdfeos_swaths_oh_geolocation_fields_solarzenithangle.attrs["_FillValue"] = (
            np.array([np.float32(np.float32(-999.99))], dtype=np.float32)
        )

        # HDFEOS/SWATHS/OH/Geolocation Fields/Time
        hdfeos_swaths_oh_geolocation_fields_time = (
            hdfeos_swaths_oh_geolocation_fields.create_dataset(
                "Time",
                data=np.linspace(0.0, 1000000000.0, dim_0_3495).astype(np.float64),
            )
        )
        hdfeos_swaths_oh_geolocation_fields_time.attrs["MissingValue"] = np.array(
            [np.float64(np.float64(-999.989990234375))], dtype=np.float64
        )
        hdfeos_swaths_oh_geolocation_fields_time.attrs["Title"] = "Time"
        hdfeos_swaths_oh_geolocation_fields_time.attrs["UniqueFieldDefinition"] = (
            "Aura-Shared"
        )
        hdfeos_swaths_oh_geolocation_fields_time.attrs["Units"] = "s"
        hdfeos_swaths_oh_geolocation_fields_time.attrs["_FillValue"] = np.array(
            [np.float64(np.float64(-999.989990234375))], dtype=np.float64
        )

        # HDFEOS/SWATHS/OH/nLevels
        hdfeos_swaths_oh_nlevels = hdfeos_swaths_oh.create_dataset(
            "nLevels",
            data=np.arange(dim_0_49, dtype=np.float32),
        )

        # HDFEOS/SWATHS/OH/nTimes
        hdfeos_swaths_oh_ntimes = hdfeos_swaths_oh.create_dataset(
            "nTimes",
            data=np.arange(dim_0_3495, dtype=np.float32),
        )

        # HDFEOS/SWATHS/OH/nTimesTotal
        hdfeos_swaths_oh_ntimestotal = hdfeos_swaths_oh.create_dataset(
            "nTimesTotal",
            data=np.arange(dim_0_3495, dtype=np.float32),
        )

        # HDFEOS/SWATHS/OH-APriori/Data Fields/Convergence
        hdfeos_swaths_oh_apriori_data_fields_convergence = (
            hdfeos_swaths_oh_apriori_data_fields.create_dataset(
                "Convergence",
                data=np.arange(dim_0_3495, dtype=np.float32),
            )
        )
        hdfeos_swaths_oh_apriori_data_fields_convergence.attrs["MissingValue"] = (
            np.array([np.float32(np.float32(-999.99))], dtype=np.float32)
        )
        hdfeos_swaths_oh_apriori_data_fields_convergence.attrs["Title"] = (
            "OH-APrioriConvergence"
        )
        hdfeos_swaths_oh_apriori_data_fields_convergence.attrs[
            "UniqueFieldDefinition"
        ] = "MLS-Specific"
        hdfeos_swaths_oh_apriori_data_fields_convergence.attrs["Units"] = "NoUnits"
        hdfeos_swaths_oh_apriori_data_fields_convergence.attrs["_FillValue"] = np.array(
            [np.float32(np.float32(-999.99))], dtype=np.float32
        )

        # HDFEOS/SWATHS/OH-APriori/Data Fields/L2gpPrecision
        hdfeos_swaths_oh_apriori_data_fields_l2gpprecision = (
            hdfeos_swaths_oh_apriori_data_fields.create_dataset(
                "L2gpPrecision",
                data=np.arange(dim_0_3495 * dim_0_49, dtype=np.float32).reshape(
                    (
                        dim_0_3495,
                        dim_0_49,
                    )
                ),
            )
        )
        hdfeos_swaths_oh_apriori_data_fields_l2gpprecision.attrs["MissingValue"] = (
            np.array([np.float32(np.float32(-999.99))], dtype=np.float32)
        )
        hdfeos_swaths_oh_apriori_data_fields_l2gpprecision.attrs["Title"] = (
            "OH-APrioriPrecision"
        )
        hdfeos_swaths_oh_apriori_data_fields_l2gpprecision.attrs[
            "UniqueFieldDefinition"
        ] = "MLS-Specific"
        hdfeos_swaths_oh_apriori_data_fields_l2gpprecision.attrs["Units"] = "vmr"
        hdfeos_swaths_oh_apriori_data_fields_l2gpprecision.attrs["_FillValue"] = (
            np.array([np.float32(np.float32(-999.99))], dtype=np.float32)
        )

        # HDFEOS/SWATHS/OH-APriori/Data Fields/L2gpValue
        hdfeos_swaths_oh_apriori_data_fields_l2gpvalue = (
            hdfeos_swaths_oh_apriori_data_fields.create_dataset(
                "L2gpValue",
                data=np.arange(dim_0_3495 * dim_0_49, dtype=np.float32).reshape(
                    (
                        dim_0_3495,
                        dim_0_49,
                    )
                ),
            )
        )
        hdfeos_swaths_oh_apriori_data_fields_l2gpvalue.attrs["MissingValue"] = np.array(
            [np.float32(np.float32(-999.99))], dtype=np.float32
        )
        hdfeos_swaths_oh_apriori_data_fields_l2gpvalue.attrs["Title"] = "OH-APriori"
        hdfeos_swaths_oh_apriori_data_fields_l2gpvalue.attrs[
            "UniqueFieldDefinition"
        ] = "MLS-Specific"
        hdfeos_swaths_oh_apriori_data_fields_l2gpvalue.attrs["Units"] = "vmr"
        hdfeos_swaths_oh_apriori_data_fields_l2gpvalue.attrs["_FillValue"] = np.array(
            [np.float32(np.float32(-999.99))], dtype=np.float32
        )

        # HDFEOS/SWATHS/OH-APriori/Data Fields/Quality
        hdfeos_swaths_oh_apriori_data_fields_quality = (
            hdfeos_swaths_oh_apriori_data_fields.create_dataset(
                "Quality",
                data=np.arange(dim_0_3495, dtype=np.float32),
            )
        )
        hdfeos_swaths_oh_apriori_data_fields_quality.attrs["MissingValue"] = np.array(
            [np.float32(np.float32(-999.99))], dtype=np.float32
        )
        hdfeos_swaths_oh_apriori_data_fields_quality.attrs["Title"] = (
            "OH-APrioriQuality"
        )
        hdfeos_swaths_oh_apriori_data_fields_quality.attrs["UniqueFieldDefinition"] = (
            "MLS-Specific"
        )
        hdfeos_swaths_oh_apriori_data_fields_quality.attrs["Units"] = "NoUnits"
        hdfeos_swaths_oh_apriori_data_fields_quality.attrs["_FillValue"] = np.array(
            [np.float32(np.float32(-999.99))], dtype=np.float32
        )

        # HDFEOS/SWATHS/OH-APriori/Data Fields/Status
        hdfeos_swaths_oh_apriori_data_fields_status = (
            hdfeos_swaths_oh_apriori_data_fields.create_dataset(
                "Status",
                data=np.arange(dim_0_3495, dtype=np.int32),
            )
        )
        hdfeos_swaths_oh_apriori_data_fields_status.attrs["MissingValue"] = np.array(
            [np.int32(np.int32(513))], dtype=np.int32
        )
        hdfeos_swaths_oh_apriori_data_fields_status.attrs["Title"] = "OH-APrioriStatus"
        hdfeos_swaths_oh_apriori_data_fields_status.attrs["UniqueFieldDefinition"] = (
            "MLS-Specific"
        )
        hdfeos_swaths_oh_apriori_data_fields_status.attrs["Units"] = "NoUnits"
        hdfeos_swaths_oh_apriori_data_fields_status.attrs["_FillValue"] = np.array(
            [np.int32(np.int32(513))], dtype=np.int32
        )

        # HDFEOS/SWATHS/OH-APriori/Geolocation Fields/ChunkNumber
        hdfeos_swaths_oh_apriori_geolocation_fields_chunknumber = (
            hdfeos_swaths_oh_apriori_geolocation_fields.create_dataset(
                "ChunkNumber",
                data=np.arange(dim_0_3495, dtype=np.int32),
            )
        )
        hdfeos_swaths_oh_apriori_geolocation_fields_chunknumber.attrs[
            "MissingValue"
        ] = np.array([np.int32(np.int32(-999))], dtype=np.int32)
        hdfeos_swaths_oh_apriori_geolocation_fields_chunknumber.attrs["Title"] = (
            "ChunkNumber"
        )
        hdfeos_swaths_oh_apriori_geolocation_fields_chunknumber.attrs[
            "UniqueFieldDefinition"
        ] = "MLS-Specific"
        hdfeos_swaths_oh_apriori_geolocation_fields_chunknumber.attrs["Units"] = (
            "NoUnits"
        )
        hdfeos_swaths_oh_apriori_geolocation_fields_chunknumber.attrs["_FillValue"] = (
            np.array([np.int32(np.int32(-999))], dtype=np.int32)
        )

        # HDFEOS/SWATHS/OH-APriori/Geolocation Fields/Latitude
        hdfeos_swaths_oh_apriori_geolocation_fields_latitude = (
            hdfeos_swaths_oh_apriori_geolocation_fields.create_dataset(
                "Latitude",
                data=np.linspace(-90.0, 90.0, dim_0_3495).astype(np.float32),
            )
        )
        hdfeos_swaths_oh_apriori_geolocation_fields_latitude.attrs["MissingValue"] = (
            np.array([np.float32(np.float32(-999.99))], dtype=np.float32)
        )
        hdfeos_swaths_oh_apriori_geolocation_fields_latitude.attrs["Title"] = "Latitude"
        hdfeos_swaths_oh_apriori_geolocation_fields_latitude.attrs[
            "UniqueFieldDefinition"
        ] = "HIRDLS-MLS-TES-Shared"
        hdfeos_swaths_oh_apriori_geolocation_fields_latitude.attrs["Units"] = "deg"
        hdfeos_swaths_oh_apriori_geolocation_fields_latitude.attrs["_FillValue"] = (
            np.array([np.float32(np.float32(-999.99))], dtype=np.float32)
        )

        # HDFEOS/SWATHS/OH-APriori/Geolocation Fields/LineOfSightAngle
        hdfeos_swaths_oh_apriori_geolocation_fields_lineofsightangle = (
            hdfeos_swaths_oh_apriori_geolocation_fields.create_dataset(
                "LineOfSightAngle",
                data=np.arange(dim_0_3495, dtype=np.float32),
            )
        )
        hdfeos_swaths_oh_apriori_geolocation_fields_lineofsightangle.attrs[
            "MissingValue"
        ] = np.array([np.float32(np.float32(-999.99))], dtype=np.float32)
        hdfeos_swaths_oh_apriori_geolocation_fields_lineofsightangle.attrs["Title"] = (
            "LineOfSightAngle"
        )
        hdfeos_swaths_oh_apriori_geolocation_fields_lineofsightangle.attrs[
            "UniqueFieldDefinition"
        ] = "MLS-Specific"
        hdfeos_swaths_oh_apriori_geolocation_fields_lineofsightangle.attrs["Units"] = (
            "deg"
        )
        hdfeos_swaths_oh_apriori_geolocation_fields_lineofsightangle.attrs[
            "_FillValue"
        ] = np.array([np.float32(np.float32(-999.99))], dtype=np.float32)

        # HDFEOS/SWATHS/OH-APriori/Geolocation Fields/LocalSolarTime
        hdfeos_swaths_oh_apriori_geolocation_fields_localsolartime = (
            hdfeos_swaths_oh_apriori_geolocation_fields.create_dataset(
                "LocalSolarTime",
                data=np.arange(dim_0_3495, dtype=np.float32),
            )
        )
        hdfeos_swaths_oh_apriori_geolocation_fields_localsolartime.attrs[
            "MissingValue"
        ] = np.array([np.float32(np.float32(-999.99))], dtype=np.float32)
        hdfeos_swaths_oh_apriori_geolocation_fields_localsolartime.attrs["Title"] = (
            "LocalSolarTime"
        )
        hdfeos_swaths_oh_apriori_geolocation_fields_localsolartime.attrs[
            "UniqueFieldDefinition"
        ] = "HIRDLS-MLS-TES-Shared"
        hdfeos_swaths_oh_apriori_geolocation_fields_localsolartime.attrs["Units"] = "h"
        hdfeos_swaths_oh_apriori_geolocation_fields_localsolartime.attrs[
            "_FillValue"
        ] = np.array([np.float32(np.float32(-999.99))], dtype=np.float32)

        # HDFEOS/SWATHS/OH-APriori/Geolocation Fields/Longitude
        hdfeos_swaths_oh_apriori_geolocation_fields_longitude = (
            hdfeos_swaths_oh_apriori_geolocation_fields.create_dataset(
                "Longitude",
                data=np.linspace(-180.0, 180.0, dim_0_3495).astype(np.float32),
            )
        )
        hdfeos_swaths_oh_apriori_geolocation_fields_longitude.attrs["MissingValue"] = (
            np.array([np.float32(np.float32(-999.99))], dtype=np.float32)
        )
        hdfeos_swaths_oh_apriori_geolocation_fields_longitude.attrs["Title"] = (
            "Longitude"
        )
        hdfeos_swaths_oh_apriori_geolocation_fields_longitude.attrs[
            "UniqueFieldDefinition"
        ] = "HIRDLS-MLS-TES-Shared"
        hdfeos_swaths_oh_apriori_geolocation_fields_longitude.attrs["Units"] = "deg"
        hdfeos_swaths_oh_apriori_geolocation_fields_longitude.attrs["_FillValue"] = (
            np.array([np.float32(np.float32(-999.99))], dtype=np.float32)
        )

        # HDFEOS/SWATHS/OH-APriori/Geolocation Fields/OrbitGeodeticAngle
        hdfeos_swaths_oh_apriori_geolocation_fields_orbitgeodeticangle = (
            hdfeos_swaths_oh_apriori_geolocation_fields.create_dataset(
                "OrbitGeodeticAngle",
                data=np.arange(dim_0_3495, dtype=np.float32),
            )
        )
        hdfeos_swaths_oh_apriori_geolocation_fields_orbitgeodeticangle.attrs[
            "MissingValue"
        ] = np.array([np.float32(np.float32(-999.99))], dtype=np.float32)
        hdfeos_swaths_oh_apriori_geolocation_fields_orbitgeodeticangle.attrs[
            "Title"
        ] = "OrbitGeodeticAngle"
        hdfeos_swaths_oh_apriori_geolocation_fields_orbitgeodeticangle.attrs[
            "UniqueFieldDefinition"
        ] = "MLS-Specific"
        hdfeos_swaths_oh_apriori_geolocation_fields_orbitgeodeticangle.attrs[
            "Units"
        ] = "deg"
        hdfeos_swaths_oh_apriori_geolocation_fields_orbitgeodeticangle.attrs[
            "_FillValue"
        ] = np.array([np.float32(np.float32(-999.99))], dtype=np.float32)

        # HDFEOS/SWATHS/OH-APriori/Geolocation Fields/Pressure
        hdfeos_swaths_oh_apriori_geolocation_fields_pressure = (
            hdfeos_swaths_oh_apriori_geolocation_fields.create_dataset(
                "Pressure",
                data=np.arange(dim_0_49, dtype=np.float32),
            )
        )
        hdfeos_swaths_oh_apriori_geolocation_fields_pressure.attrs["MissingValue"] = (
            np.array([np.float32(np.float32(-999.99))], dtype=np.float32)
        )
        hdfeos_swaths_oh_apriori_geolocation_fields_pressure.attrs["Title"] = "Pressure"
        hdfeos_swaths_oh_apriori_geolocation_fields_pressure.attrs[
            "UniqueFieldDefinition"
        ] = "Aura-Shared"
        hdfeos_swaths_oh_apriori_geolocation_fields_pressure.attrs["Units"] = "hPa"
        hdfeos_swaths_oh_apriori_geolocation_fields_pressure.attrs["_FillValue"] = (
            np.array([np.float32(np.float32(-999.99))], dtype=np.float32)
        )

        # HDFEOS/SWATHS/OH-APriori/Geolocation Fields/SolarZenithAngle
        hdfeos_swaths_oh_apriori_geolocation_fields_solarzenithangle = (
            hdfeos_swaths_oh_apriori_geolocation_fields.create_dataset(
                "SolarZenithAngle",
                data=np.arange(dim_0_3495, dtype=np.float32),
            )
        )
        hdfeos_swaths_oh_apriori_geolocation_fields_solarzenithangle.attrs[
            "MissingValue"
        ] = np.array([np.float32(np.float32(-999.99))], dtype=np.float32)
        hdfeos_swaths_oh_apriori_geolocation_fields_solarzenithangle.attrs["Title"] = (
            "SolarZenithAngle"
        )
        hdfeos_swaths_oh_apriori_geolocation_fields_solarzenithangle.attrs[
            "UniqueFieldDefinition"
        ] = "HIRDLS-MLS-TES-Shared"
        hdfeos_swaths_oh_apriori_geolocation_fields_solarzenithangle.attrs["Units"] = (
            "deg"
        )
        hdfeos_swaths_oh_apriori_geolocation_fields_solarzenithangle.attrs[
            "_FillValue"
        ] = np.array([np.float32(np.float32(-999.99))], dtype=np.float32)

        # HDFEOS/SWATHS/OH-APriori/Geolocation Fields/Time
        hdfeos_swaths_oh_apriori_geolocation_fields_time = (
            hdfeos_swaths_oh_apriori_geolocation_fields.create_dataset(
                "Time",
                data=np.linspace(0.0, 1000000000.0, dim_0_3495).astype(np.float64),
            )
        )
        hdfeos_swaths_oh_apriori_geolocation_fields_time.attrs["MissingValue"] = (
            np.array([np.float64(np.float64(-999.989990234375))], dtype=np.float64)
        )
        hdfeos_swaths_oh_apriori_geolocation_fields_time.attrs["Title"] = "Time"
        hdfeos_swaths_oh_apriori_geolocation_fields_time.attrs[
            "UniqueFieldDefinition"
        ] = "Aura-Shared"
        hdfeos_swaths_oh_apriori_geolocation_fields_time.attrs["Units"] = "s"
        hdfeos_swaths_oh_apriori_geolocation_fields_time.attrs["_FillValue"] = np.array(
            [np.float64(np.float64(-999.989990234375))], dtype=np.float64
        )

        # HDFEOS/SWATHS/OH-APriori/nLevels
        hdfeos_swaths_oh_apriori_nlevels = hdfeos_swaths_oh_apriori.create_dataset(
            "nLevels",
            data=np.arange(dim_0_49, dtype=np.float32),
        )

        # HDFEOS/SWATHS/OH-APriori/nTimes
        hdfeos_swaths_oh_apriori_ntimes = hdfeos_swaths_oh_apriori.create_dataset(
            "nTimes",
            data=np.arange(dim_0_3495, dtype=np.float32),
        )

        # HDFEOS/SWATHS/OH-APriori/nTimesTotal
        hdfeos_swaths_oh_apriori_ntimestotal = hdfeos_swaths_oh_apriori.create_dataset(
            "nTimesTotal",
            data=np.arange(dim_0_3495, dtype=np.float32),
        )

        # HDFEOS INFORMATION/coremetadata.0
        hdfeos_information_coremetadata_0 = hdfeos_information.create_dataset(
            "coremetadata.0",
            data=np.array(0, dtype=np.dtype("|S65535")),
        )

        # HDFEOS INFORMATION/xmlmetadata
        hdfeos_information_xmlmetadata = hdfeos_information.create_dataset(
            "xmlmetadata",
            data=np.array(0, dtype=np.dtype("|S65535")),
        )

        # HDFEOS INFORMATION/StructMetadata.0
        hdfeos_information_structmetadata_0_text = (
            "GROUP=SwathStructure\n"
            "	GROUP=SWATH_1\n"
            '		SwathName="OH"\n'
            "		GROUP=Dimension\n"
            "			OBJECT=Dimension_1\n"
            '				DimensionName="nTimes"\n'
            "				Size=3495\n"
            "			END_OBJECT=Dimension_1\n"
            "			OBJECT=Dimension_2\n"
            '				DimensionName="nTimesTotal"\n'
            "				Size=3495\n"
            "			END_OBJECT=Dimension_2\n"
            "			OBJECT=Dimension_3\n"
            '				DimensionName="nLevels"\n'
            "				Size=49\n"
            "			END_OBJECT=Dimension_3\n"
            "		END_GROUP=Dimension\n"
            "		GROUP=DimensionMap\n"
            "		END_GROUP=DimensionMap\n"
            "		GROUP=IndexDimensionMap\n"
            "		END_GROUP=IndexDimensionMap\n"
            "		GROUP=GeoField\n"
            "			OBJECT=GeoField_1\n"
            '				GeoFieldName="Latitude"\n'
            "				DataType=H5T_NATIVE_FLOAT\n"
            '				DimList=("nTimes")\n'
            '				MaxdimList=("nTimes")\n'
            "			END_OBJECT=GeoField_1\n"
            "			OBJECT=GeoField_2\n"
            '				GeoFieldName="Longitude"\n'
            "				DataType=H5T_NATIVE_FLOAT\n"
            '				DimList=("nTimes")\n'
            '				MaxdimList=("nTimes")\n'
            "			END_OBJECT=GeoField_2\n"
            "			OBJECT=GeoField_3\n"
            '				GeoFieldName="Time"\n'
            "				DataType=H5T_NATIVE_DOUBLE\n"
            '				DimList=("nTimes")\n'
            '				MaxdimList=("nTimes")\n'
            "			END_OBJECT=GeoField_3\n"
            "			OBJECT=GeoField_4\n"
            '				GeoFieldName="LocalSolarTime"\n'
            "				DataType=H5T_NATIVE_FLOAT\n"
            '				DimList=("nTimes")\n'
            '				MaxdimList=("nTimes")\n'
            "			END_OBJECT=GeoField_4\n"
            "			OBJECT=GeoField_5\n"
            '				GeoFieldName="SolarZenithAngle"\n'
            "				DataType=H5T_NATIVE_FLOAT\n"
            '				DimList=("nTimes")\n'
            '				MaxdimList=("nTimes")\n'
            "			END_OBJECT=GeoField_5\n"
            "			OBJECT=GeoField_6\n"
            '				GeoFieldName="LineOfSightAngle"\n'
            "				DataType=H5T_NATIVE_FLOAT\n"
            '				DimList=("nTimes")\n'
            '				MaxdimList=("nTimes")\n'
            "			END_OBJECT=GeoField_6\n"
            "			OBJECT=GeoField_7\n"
            '				GeoFieldName="OrbitGeodeticAngle"\n'
            "				DataType=H5T_NATIVE_FLOAT\n"
            '				DimList=("nTimes")\n'
            '				MaxdimList=("nTimes")\n'
            "			END_OBJECT=GeoField_7\n"
            "			OBJECT=GeoField_8\n"
            '				GeoFieldName="ChunkNumber"\n'
            "				DataType=H5T_NATIVE_INT\n"
            '				DimList=("nTimes")\n'
            '				MaxdimList=("nTimes")\n'
            "			END_OBJECT=GeoField_8\n"
            "			OBJECT=GeoField_9\n"
            '				GeoFieldName="Pressure"\n'
            "				DataType=H5T_NATIVE_FLOAT\n"
            '				DimList=("nLevels")\n'
            '				MaxdimList=("nLevels")\n'
            "			END_OBJECT=GeoField_9\n"
            "		END_GROUP=GeoField\n"
            "		GROUP=DataField\n"
            "			OBJECT=DataField_1\n"
            '				DataFieldName="L2gpValue"\n'
            "				DataType=H5T_NATIVE_FLOAT\n"
            '				DimList=("nTimes","nLevels")\n'
            '				MaxdimList=("nTimes","nLevels")\n'
            "			END_OBJECT=DataField_1\n"
            "			OBJECT=DataField_2\n"
            '				DataFieldName="L2gpPrecision"\n'
            "				DataType=H5T_NATIVE_FLOAT\n"
            '				DimList=("nTimes","nLevels")\n'
            '				MaxdimList=("nTimes","nLevels")\n'
            "			END_OBJECT=DataField_2\n"
            "			OBJECT=DataField_3\n"
            '				DataFieldName="Status"\n'
            "				DataType=H5T_NATIVE_INT\n"
            '				DimList=("nTimes")\n'
            '				MaxdimList=("nTimes")\n'
            "			END_OBJECT=DataField_3\n"
            "			OBJECT=DataField_4\n"
            '				DataFieldName="Quality"\n'
            "				DataType=H5T_NATIVE_FLOAT\n"
            '				DimList=("nTimes")\n'
            '				MaxdimList=("nTimes")\n'
            "			END_OBJECT=DataField_4\n"
            "			OBJECT=DataField_5\n"
            '				DataFieldName="Convergence"\n'
            "				DataType=H5T_NATIVE_FLOAT\n"
            '				DimList=("nTimes")\n'
            '				MaxdimList=("nTimes")\n'
            "			END_OBJECT=DataField_5\n"
            "		END_GROUP=DataField\n"
            "		GROUP=ProfileField\n"
            "		END_GROUP=ProfileField\n"
            "		GROUP=MergedFields\n"
            "		END_GROUP=MergedFields\n"
            "	END_GROUP=SWATH_1\n"
            "	GROUP=SWATH_2\n"
            '		SwathName="OH-APriori"\n'
            "		GROUP=Dimension\n"
            "			OBJECT=Dimension_1\n"
            '				DimensionName="nTimes"\n'
            "				Size=3495\n"
            "			END_OBJECT=Dimension_1\n"
            "			OBJECT=Dimension_2\n"
            '				DimensionName="nTimesTotal"\n'
            "				Size=3495\n"
            "			END_OBJECT=Dimension_2\n"
            "			OBJECT=Dimension_3\n"
            '				DimensionName="nLevels"\n'
            "				Size=49\n"
            "			END_OBJECT=Dimension_3\n"
            "		END_GROUP=Dimension\n"
            "		GROUP=DimensionMap\n"
            "		END_GROUP=DimensionMap\n"
            "		GROUP=IndexDimensionMap\n"
            "		END_GROUP=IndexDimensionMap\n"
            "		GROUP=GeoField\n"
            "			OBJECT=GeoField_1\n"
            '				GeoFieldName="Latitude"\n'
            "				DataType=H5T_NATIVE_FLOAT\n"
            '				DimList=("nTimes")\n'
            '				MaxdimList=("nTimes")\n'
            "			END_OBJECT=GeoField_1\n"
            "			OBJECT=GeoField_2\n"
            '				GeoFieldName="Longitude"\n'
            "				DataType=H5T_NATIVE_FLOAT\n"
            '				DimList=("nTimes")\n'
            '				MaxdimList=("nTimes")\n'
            "			END_OBJECT=GeoField_2\n"
            "			OBJECT=GeoField_3\n"
            '				GeoFieldName="Time"\n'
            "				DataType=H5T_NATIVE_DOUBLE\n"
            '				DimList=("nTimes")\n'
            '				MaxdimList=("nTimes")\n'
            "			END_OBJECT=GeoField_3\n"
            "			OBJECT=GeoField_4\n"
            '				GeoFieldName="LocalSolarTime"\n'
            "				DataType=H5T_NATIVE_FLOAT\n"
            '				DimList=("nTimes")\n'
            '				MaxdimList=("nTimes")\n'
            "			END_OBJECT=GeoField_4\n"
            "			OBJECT=GeoField_5\n"
            '				GeoFieldName="SolarZenithAngle"\n'
            "				DataType=H5T_NATIVE_FLOAT\n"
            '				DimList=("nTimes")\n'
            '				MaxdimList=("nTimes")\n'
            "			END_OBJECT=GeoField_5\n"
            "			OBJECT=GeoField_6\n"
            '				GeoFieldName="LineOfSightAngle"\n'
            "				DataType=H5T_NATIVE_FLOAT\n"
            '				DimList=("nTimes")\n'
            '				MaxdimList=("nTimes")\n'
            "			END_OBJECT=GeoField_6\n"
            "			OBJECT=GeoField_7\n"
            '				GeoFieldName="OrbitGeodeticAngle"\n'
            "				DataType=H5T_NATIVE_FLOAT\n"
            '				DimList=("nTimes")\n'
            '				MaxdimList=("nTimes")\n'
            "			END_OBJECT=GeoField_7\n"
            "			OBJECT=GeoField_8\n"
            '				GeoFieldName="ChunkNumber"\n'
            "				DataType=H5T_NATIVE_INT\n"
            '				DimList=("nTimes")\n'
            '				MaxdimList=("nTimes")\n'
            "			END_OBJECT=GeoField_8\n"
            "			OBJECT=GeoField_9\n"
            '				GeoFieldName="Pressure"\n'
            "				DataType=H5T_NATIVE_FLOAT\n"
            '				DimList=("nLevels")\n'
            '				MaxdimList=("nLevels")\n'
            "			END_OBJECT=GeoField_9\n"
            "		END_GROUP=GeoField\n"
            "		GROUP=DataField\n"
            "			OBJECT=DataField_1\n"
            '				DataFieldName="L2gpValue"\n'
            "				DataType=H5T_NATIVE_FLOAT\n"
            '				DimList=("nTimes","nLevels")\n'
            '				MaxdimList=("nTimes","nLevels")\n'
            "			END_OBJECT=DataField_1\n"
            "			OBJECT=DataField_2\n"
            '				DataFieldName="L2gpPrecision"\n'
            "				DataType=H5T_NATIVE_FLOAT\n"
            '				DimList=("nTimes","nLevels")\n'
            '				MaxdimList=("nTimes","nLevels")\n'
            "			END_OBJECT=DataField_2\n"
            "			OBJECT=DataField_3\n"
            '				DataFieldName="Status"\n'
            "				DataType=H5T_NATIVE_INT\n"
            '				DimList=("nTimes")\n'
            '				MaxdimList=("nTimes")\n'
            "			END_OBJECT=DataField_3\n"
            "			OBJECT=DataField_4\n"
            '				DataFieldName="Quality"\n'
            "				DataType=H5T_NATIVE_FLOAT\n"
            '				DimList=("nTimes")\n'
            '				MaxdimList=("nTimes")\n'
            "			END_OBJECT=DataField_4\n"
            "			OBJECT=DataField_5\n"
            '				DataFieldName="Convergence"\n'
            "				DataType=H5T_NATIVE_FLOAT\n"
            '				DimList=("nTimes")\n'
            '				MaxdimList=("nTimes")\n'
            "			END_OBJECT=DataField_5\n"
            "		END_GROUP=DataField\n"
            "		GROUP=ProfileField\n"
            "		END_GROUP=ProfileField\n"
            "		GROUP=MergedFields\n"
            "		END_GROUP=MergedFields\n"
            "	END_GROUP=SWATH_2\n"
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
