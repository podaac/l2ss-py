from pathlib import Path
from typing import NamedTuple

import numpy as np
import pytest
import xarray as xr
from podaac.subsetter import subset
from podaac.subsetter.utils.variables_utils import get_vars_with_paths


class VariableTestCase(NamedTuple):
    input: str
    want_var: set[str]
    want_coord: set[str]


_test_table: list[VariableTestCase] = [
    VariableTestCase(
        input="MODIS_T-JPL-L2P-v2014.0.nc",
        want_var={"/sst_dtime"},
        want_coord={"/lat", "/lon", "/time"},
    ),
    VariableTestCase(
        input="MODIS_A-JPL-L2P-v2014.0.nc",
        want_var={"/sea_surface_temperature"},
        want_coord={"/lat", "/lon", "/time"},
    ),
    VariableTestCase(
        input="cyg04.ddmi.s20210228-000000-e20210228-235959.l1.power-brcs-cdr.a10.d10.nc",
        want_var={
            "/sc_pos_z",
            "/sc_vel_y",
            "/sp_vel_z",
        },
        want_coord={"/sp_lat", "/sp_lon", "/ddm_timestamp_utc", "/ddm", "/sample"},
    ),
    VariableTestCase(
        input="SWOT_L2_LR_SSH_Expert_368_012_20121111T235910_20121112T005015_DG10_01.nc",
        want_var={
            "/mean_sea_surface_dtu",
            "/latitude_avg_ssh",
            "/geoid",
            "/x_factor",
            "/mean_sea_surface_cnescls_uncert",
            "/simulated_error_orbital",
            "/internal_tide_hret",
        },
        want_coord={"/latitude", "/longitude", "/time"},
    ),
    VariableTestCase(
        input="20200101000001-JPL-L2P_GHRSST-SSTskin-MODIS_T-N-v02.0-fv01.0.nc",
        want_var={
            "/quality_level",
            "/sst_dtime",
            "/sea_surface_temperature_4um",
            "/quality_level_4um",
            "/l2p_flags",
            "/sses_standard_deviation_4um",
        },
        want_coord={"/lat", "/lon", "/time"},
    ),
    VariableTestCase(
        input="JA1_GPN_2PeP001_002_20020115_060706_20020115_070316.nc",
        want_var={
            "/alt_state_flag_oper",
            "/qual_inst_corr_1hz_swh_c",
            "/sea_state_bias_ku",
            "/range_used_20hz_ku",
        },
        want_coord={"/lat", "/lon", "/lat_20hz", "/lon_20hz", "/time", "/meas_ind"},
    ),
    VariableTestCase(
        input="AMSR2-L2B_v08_r38622-v02.0-fv01.0.nc",
        want_var={
            "/quality_level",
            "/sses_standard_deviation",
            "/diurnal_amplitude",
            "/wind_speed",
            "/rain_rate",
            "/l2p_flags",
            "/dt_analysis",
        },
        want_coord={"/lat", "/lon", "/time"},
    ),
    VariableTestCase(
        input="Merged_TOPEX_Jason_OSTM_Jason-3_Cycle_002.V4_2.nc",
        want_var={"/Surface_Type", "/reference_orbit", "/Distance_to_coast", "/index"},
        want_coord={"/latitude", "/longitude", "/time"},
    ),
    VariableTestCase(
        input="ascat_20150702_084200_metopa_45145_eps_o_250_2300_ovw.l2.nc",
        want_var={"/wvc_index", "/wind_speed", "/ice_age", "/ice_prob", "/wind_dir"},
        want_coord={"/lat", "/lon", "/time"},
    ),
    VariableTestCase(
        input="ascat_20150702_102400_metopa_45146_eps_o_250_2300_ovw.l2.nc",
        want_var={"/wvc_index", "/wind_speed", "/ice_age", "/ice_prob", "/wind_dir"},
        want_coord={"/lat", "/lon", "/time"},
    ),
    VariableTestCase(
        input="20180101005944-REMSS-L2P_GHRSST-SSTsubskin-AMSR2-L2B_rt_r29918-v02.0-fv01.0.nc",
        want_var={
            "/quality_level",
            "/sses_standard_deviation",
            "/diurnal_amplitude",
            "/wind_speed",
            "/rain_rate",
            "/l2p_flags",
            "/dt_analysis",
        },
        want_coord={"/lat", "/lon", "/time"},
    ),
    VariableTestCase(
        input="TEMPO_HCHO_L2_V01_20240110T170237Z_S005G08.nc",
        want_var={
            "/support_data/amf_cloud_fraction",
            "/geolocation/longitude_bounds",
            "/support_data/amf_cloud_pressure",
            "/geolocation/viewing_azimuth_angle",
        },
        want_coord={"/mirror_step", "/xtrack", "/geolocation/latitude", "/geolocation/longitude", "/geolocation/time"},
    ),
    VariableTestCase(
        input="VIIRS_NPP-NAVO-L2P-v3.0.nc",
        want_var={
            "/quality_level",
            "/brightness_temperature_12um",
            "/sea_surface_temperature",
            "/sses_bias",
            "/adi_dtime_from_sst",
        },
        want_coord={"/lat", "/lon", "/time"},
    ),
    VariableTestCase(
        input="20190927000500-JPL-L2P_GHRSST-SSTskin-MODIS_A-D-v02.0-fv01.0.nc",
        want_var={"/quality_level", "/wind_speed", "/sea_surface_temperature", "/sses_bias"},
        want_coord={"/lat", "/lon", "/time"},
    ),
]


@pytest.mark.parametrize("case", _test_table, ids=lambda c: c.input)
def test_specified_variables(case, data_dir: str, tmp_path: Path):
    """
    Test that the variables which are specified when calling the subset
    operation are present in the resulting subsetted data file plus
    their required dimension scale/coordinate variables
    """

    output_path = tmp_path / case.input

    subset.subset(
        file_to_subset=Path(data_dir) / case.input,
        bbox=np.array(((-180, 180), (-90, 90))),
        output_file=output_path,
        variables=list(case.want_var),  # only specify wanted data variables
    )

    with xr.open_datatree(output_path, decode_times=False, decode_coords=False) as out_tree:
        # all vars is the super set containing data + coord vars
        all_vars = get_vars_with_paths(out_tree)

        # wanted variable should be a subset of all vars
        assert case.want_var <= all_vars

        # wanted coordinate vars should be a subset of all vars as well
        assert case.want_coord <= all_vars

        # and the symmetric difference of the variable super set and
        # the union of data and coordinate vars should be an empty set
        # indicating that nothing is present that is not expected to
        # be present. E.g. extra dimension scale vars
        assert (case.want_var | case.want_coord) ^ all_vars == set()
