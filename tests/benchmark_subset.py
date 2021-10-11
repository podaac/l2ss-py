# Copyright 2019, by the California Institute of Technology.
# ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.
# Any commercial use must be negotiated with the Office of Technology
# Transfer at the California Institute of Technology.
#
# This software may be subject to U.S. export control laws. By accepting
# this software, the user agrees to comply with all applicable U.S. export
# laws and regulations. User has the responsibility to obtain export
# licenses, or other export authority as may be required before exporting
# such information to foreign countries or providing access to foreign
# persons.

"""
===================
benchmark_subset.py
===================

Benchmark the L2 Subset functionality
"""

import shutil
import tempfile
import os
from os.path import dirname, join

import numpy as np
import pytest

from podaac.subsetter import subset

TEST_DIR = dirname(os.path.realpath(__file__))
TEST_DATA_DIR = join(TEST_DIR, 'data')
OUTPUT_DIR = tempfile.mkdtemp(dir=TEST_DATA_DIR)


def teardown_module():
    """
    Remove the temporary directory
    """
    shutil.rmtree(OUTPUT_DIR)


@pytest.mark.benchmark
def test_ascat1_bbox(benchmark):
    """
    Run the pytest-benchmark plugin to time the execution of the
    L2 subsetter on the
    ascat_20150702_084200_metopa_45145_eps_o_250_2300_ovw.l2.nc granule.
    This will subset the bottom left 1/4 of the granule.
    """
    benchmark(subset.subset,
              os.path.join(TEST_DATA_DIR,
                           "ascat_20150702_084200_metopa_45145_eps_o_250_2300_ovw.l2.nc"),
              np.array(((-180, 0), (-90, 0))),
              os.path.join(OUTPUT_DIR,
                           "ascat1_bbox.ss.nc"))


@pytest.mark.benchmark
def test_ascat1_no_bbox(benchmark):
    """
    Run the pytest-benchmark plugin to time the execution of the
    L2 subsetter on the
    ascat_20150702_084200_metopa_45145_eps_o_250_2300_ovw.l2.nc granule.
    This will subset the entire granule. (full-size bbox)
    """
    benchmark(subset.subset,
              os.path.join(TEST_DATA_DIR,
                           "ascat_20150702_084200_metopa_45145_eps_o_250_2300_ovw.l2.nc"),
              np.array(((-180, 180), (-90, 90))),
              os.path.join(OUTPUT_DIR,
                           "ascat1.ss.nc"))


@pytest.mark.benchmark
def test_ascat2_bbox(benchmark):
    """
    Run the pytest-benchmark plugin to time the execution of the
    L2 subsetter on the
    ascat_20150702_102400_metopa_45146_eps_o_250_2300_ovw.l2.nc granule.
    This will subset the bottom left 1/4 of the granule.
    """
    benchmark(subset.subset,
              os.path.join(TEST_DATA_DIR,
                           "ascat_20150702_102400_metopa_45146_eps_o_250_2300_ovw.l2.nc"),
              np.array(((-180, 0), (-90, 0))),
              os.path.join(OUTPUT_DIR,
                           "ascat2_bbox.ss.nc"))


@pytest.mark.benchmark
def test_ascat2_no_bbox(benchmark):
    """
    Run the pytest-benchmark plugin to time the execution of the
    L2 subsetter on the
    ascat_20150702_102400_metopa_45146_eps_o_250_2300_ovw.l2.nc granule.
    This will subset the entire granule. (full-size bbox)
    """
    benchmark(subset.subset,
              os.path.join(TEST_DATA_DIR,
                           "ascat_20150702_102400_metopa_45146_eps_o_250_2300_ovw.l2.nc"),
              np.array(((-180, 180), (-90, 90))),
              os.path.join(OUTPUT_DIR,
                           "ascat2.ss.nc"))


@pytest.mark.benchmark
def test_modis_a_bbox(benchmark):
    """
    Run the pytest-benchmark plugin to time the execution of the
    L2 subsetter on the MODIS_A-JPL-L2P-v2014.0.nc granule.
    This will subset the bottom left 1/4 of the granule.
    """
    benchmark(subset.subset,
              os.path.join(TEST_DATA_DIR,
                           "MODIS_A-JPL-L2P-v2014.0.nc"),
              np.array(((65.8, 86.35), (40.1, 50.15))),
              os.path.join(OUTPUT_DIR,
                           "modisA_bbox.ss.nc"))


@pytest.mark.benchmark
def test_modis_a_no_bbox(benchmark):
    """
    Run the pytest-benchmark plugin to time the execution of the
    L2 subsetter on the MODIS_A-JPL-L2P-v2014.0.nc granule.
    This will subset the entire granule. (full-size bbox)
    """
    benchmark(subset.subset,
              os.path.join(TEST_DATA_DIR,
                           "MODIS_A-JPL-L2P-v2014.0.nc"),
              np.array(((-180, 180), (-90, 90))),
              os.path.join(OUTPUT_DIR,
                           "modisA.ss.nc"))


@pytest.mark.benchmark
def test_modis_t_bbox(benchmark):
    """
    Run the pytest-benchmark plugin to time the execution of the
    L2 subsetter on the MODIS_T-JPL-L2P-v2014.0.nc granule.
    This will subset the bottom left 1/4 of the granule.
    """
    benchmark(subset.subset,
              os.path.join(TEST_DATA_DIR,
                           "MODIS_T-JPL-L2P-v2014.0.nc"),
              np.array(((-78.7, -60.7), (-54.8, -44))),
              os.path.join(OUTPUT_DIR,
                           "modisT_bbox.ss.nc"))


@pytest.mark.benchmark
def test_modis_t_no_bbox(benchmark):
    """
    Run the pytest-benchmark plugin to time the execution of the
    L2 subsetter on the MODIS_T-JPL-L2P-v2014.0.nc granule.
    This will subset the entire granule. (full-size bbox)
    """
    benchmark(subset.subset,
              os.path.join(TEST_DATA_DIR,
                           "MODIS_T-JPL-L2P-v2014.0.nc"),
              np.array(((-180, 180), (-90, 90))),
              os.path.join(OUTPUT_DIR,
                           "modisT.ss.nc"))


@pytest.mark.benchmark
def test_viirs_bbox(benchmark):
    """
    Run the pytest-benchmark plugin to time the execution of the
    L2 subsetter on the VIIRS_NPP-NAVO-L2P-v3.0.nc granule.
    This will subset the bottom left 1/4 of the granule.
    """
    benchmark(subset.subset,
              os.path.join(TEST_DATA_DIR,
                           "VIIRS_NPP-NAVO-L2P-v3.0.nc"),
              np.array(((-172.3, -126.95), (62.3, 70.65))),
              os.path.join(OUTPUT_DIR,
                           "viirs_bbox.ss.nc"))


@pytest.mark.benchmark
def test_viirs_no_bbox(benchmark):
    """
    Run the pytest-benchmark plugin to time the execution of the
    L2 subsetter on the VIIRS_NPP-NAVO-L2P-v3.0.nc granule.
    This will subset the entire granule. (full-size bbox)
    """
    benchmark(subset.subset,
              os.path.join(TEST_DATA_DIR,
                           "VIIRS_NPP-NAVO-L2P-v3.0.nc"),
              np.array(((-180, 180), (-90, 90))),
              os.path.join(OUTPUT_DIR,
                           "viirs.ss.nc"))


@pytest.mark.benchmark
def test_ghrsst_bbox(benchmark):
    """
    Run the pytest-benchmark plugin to time the execution of the
    L2 subsetter on the AMSR2-L2B_v08_r38622-v02.0-fv01.0.nc granule.
    This will subset the bottom left 1/4 of the granule.
    """
    benchmark(subset.subset,
              os.path.join(TEST_DATA_DIR,
                           "20180101005944-REMSS-L2P_GHRSST-SSTsubskin-AMSR2-L2B_rt_r29918-v02.0-fv01.0.nc"),
              np.array(((-180, 0), (-90, 0))),
              os.path.join(OUTPUT_DIR,
                           "ghrsst_bbox.ss.nc"))

@pytest.mark.benchmark
def test_ghrsst_no_bbox(benchmark):
    """
    Run the pytest-benchmark plugin to time the execution of the
    L2 subsetter on the AMSR2-L2B_v08_r38622-v02.0-fv01.0.nc granule.
    This will subset the entire granule. (full-size bbox)
    """
    benchmark(subset.subset,
              os.path.join(TEST_DATA_DIR,
                           "20180101005944-REMSS-L2P_GHRSST-SSTsubskin-AMSR2-L2B_rt_r29918-v02.0-fv01.0.nc"),
              np.array(((-180, 180), (-90, 90))),
              os.path.join(OUTPUT_DIR,
                           "ghrsst.ss.nc"))

@pytest.mark.benchmark
def test_jason1_bbox(benchmark):
    """
    Run the pytest-benchmark plugin to time the execution of the
    L2 subsetter on the
    JA1_GPN_2PeP001_002_20020115_060706_20020115_070316.nc granule. This
    will subset the bottom left 1/4 of the granule.
    """
    benchmark(subset.subset,
              os.path.join(TEST_DATA_DIR,
                           "JA1_GPN_2PeP001_002_20020115_060706_20020115_070316.nc"),
              np.array(((-180, 0), (-90, 0))),
              os.path.join(OUTPUT_DIR,
                           "jason1_bbox.ss.nc"))

@pytest.mark.benchmark
def test_jason1_no_bbox(benchmark):
    """
    Run the pytest-benchmark plugin to time the execution of the
    L2 subsetter on the
    JA1_GPN_2PeP001_002_20020115_060706_20020115_070316.nc granule.
    This will subset the entire granule. (full-size bbox)
    """
    benchmark(subset.subset,
              os.path.join(TEST_DATA_DIR,
                           "JA1_GPN_2PeP001_002_20020115_060706_20020115_070316.nc"),
              np.array(((-180, 180), (-90, 90))),
              os.path.join(OUTPUT_DIR,
                           "jason1.ss.nc"))


@pytest.mark.benchmark
def test_s6_bbox(benchmark):
    s6_file_name = 'S6A_P4_2__LR_STD__ST_002_140_20201207T011501_20201207T013023_F00.nc'
    s6_out_file_name = 'SS_S6A_P4_2__LR_STD__ST_002_140_20201207T011501_20201207T013023_F0.nc'
    shutil.copyfile(os.path.join(TEST_DATA_DIR, s6_file_name),
                    os.path.join(OUTPUT_DIR, s6_file_name))
    benchmark.pedantic(
        subset.subset,
        args=[
            os.path.join(OUTPUT_DIR, s6_file_name),
            np.array(((150, 180), (-90, -50))),
            os.path.join(OUTPUT_DIR, s6_out_file_name)
        ],
        iterations=1
    )

@pytest.mark.benchmark
def test_s6_no_bbox(benchmark):
    s6_file_name = 'S6A_P4_2__LR_STD__ST_002_140_20201207T011501_20201207T013023_F00.nc'
    s6_out_file_name = 'SS_S6A_P4_2__LR_STD__ST_002_140_20201207T011501_20201207T013023_F0.nc'
    shutil.copyfile(os.path.join(TEST_DATA_DIR, s6_file_name),
                    os.path.join(OUTPUT_DIR, s6_file_name))
    benchmark.pedantic(
        subset.subset,
        args=[
            os.path.join(OUTPUT_DIR, s6_file_name),
            np.array(((-180, 180), (-90, 90))),
            os.path.join(OUTPUT_DIR, s6_out_file_name)
        ],
        iterations=1
    )
