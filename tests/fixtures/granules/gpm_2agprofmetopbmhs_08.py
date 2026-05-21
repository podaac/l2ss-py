import numpy as np
import pytest
import xarray as xr


@pytest.fixture(scope="session")
def fake_gpm_2agprofmetopbmhs_08_file(
    tmp_path_factory,
):
    """
    Generated from 2A.METOPB.MHS.GPROFNNv1.20260228-S233156-E011316.069788.V08A.nc
    """
    filepath = (
        tmp_path_factory.mktemp("data")
        / "2A.METOPB.MHS.GPROFNNv1.20260228-S233156-E011316.069788.V08A.nc"
    )

    nlyrs = 2
    npixel = 6
    nscan = 10

    root_ds = xr.Dataset(
        data_vars={},
        attrs={
            "FileHeader": "DOI=10.5067/GPM/MHS/METOPB/GPROF/2A/08;\nDOIauthority=http://dx.doi.org/;\nDOIshortName=2AGPROFMETOPBMHS;\nAlgorithmID=2AGPROFMHS;\nAlgorithmVersion=NN0.8.14.dev;\nFileName=2A.METOPB.MHS.GPROFNNv1.20260228-S233156-E011316.069788.V08A.nc;\nSatelliteName=METOPB;\nInstrumentName=MHS;\nGenerationDateTime=2026-03-06T19:28:04.000Z;\nStartGranuleDateTime=2026-02-28T23:31:57.000Z;\nStopGranuleDateTime=2026-03-01T01:13:15.000Z;\nGranuleNumber=069788;\nNumberOfSwaths=1;\nNumberOfGrids=0;\nGranuleStart=SOUTHERNMOST_LATITUDE;\nTimeInterval=ORBIT;\nProcessingSystem=PPS;\nProductVersion=V08A;\nEmptyGranule=NOT_EMPTY;\nMissingData=0;\n",
            "InputRecord": "InputFileNames=1C.METOPB.MHS.XCAL2016-V.20260228-S233156-E011316.069788.V08A.nc,PAR_IP_2AGPROF_METOPB_MHS_20260301_69788_r.bin (binary);\nInputAlgorithmVersions=2016-V;\nInputGenerationDateTimes=2026-03-06T19:25:38.000Z;\n",
            "NavigationRecord": "LongitudeOnEquator=-44.554371;\nUTCDateTimeOnEquator=2026-02-28T23:57:17.834Z;\nMeanSolarBetaAngle=48.756489;\nEphemerisFileName=;\nAttitudeFileName=;\nGeoControlFileName=;\nEphemerisSource=;\nAttitudeSource=;\nGeoToolkitVersion=;\nSensorAlignmentFirstRotationAngle=;\nSensorAlignmentSecondRotationAngle=;\nSensorAlignmentThirdRotationAngle=;\nSensorAlignmentFirstRotationAxis=;\nSensorAlignmentSecondRotationAxis=;\nSensorAlignmentThirdRotationAxis=;\n",
            "FileInfo": "DataFormatVersion=8b;\nTKCodeBuildVersion=0;\nMetadataVersion=8b;\nFormatPackage=netCDF-4.9.2;\nBlueprintFilename=GPM.V8.2AGPROFMHS.blueprint.xml;\nBlueprintVersion=BV_68;\nTKIOVersion=3.102;\nMetadataStyle=PVL;\nEndianType=LITTLE_ENDIAN;\n",
            "GprofInfo": "Satellite=METOPB;\nSensor=MHS;\nPreProcessorVersion=2504s;\nPostProcessorVersion=gprof2024_pp_8_2_1_V08 (2025-09-09);\nProfileDatabaseFilename=GANAL_V8;\nOriginalRadiometerFilename=1C.METOPB.MHS.XCAL2016-V.20260228-S233156-E011316.069788.V08A.nc;\nProfileStructureFlag=1;\nspares=;\n",
        },
    )
    root_ds.to_netcdf(
        filepath,
        mode="w",
        engine="netcdf4",
    )

    gprofdheadr_ds = xr.Dataset(
        data_vars={
            "hgtTopLayer": xr.Variable(
                dims=["nlyrs"],
                data=np.arange(nlyrs, dtype=np.float32),
                attrs={
                    "DimensionNames": "nlyrs",
                    "Units": "km",
                    "units": "km",
                    "_FillValue": -9999.900390625,
                    "CodeMissingValue": "-9999.9",
                },
            ),
        },
        attrs={},
    )
    gprofdheadr_ds.to_netcdf(
        filepath,
        mode="a",
        group="GprofDHeadr",
        engine="netcdf4",
    )

    s1_ds = xr.Dataset(
        data_vars={
            "Latitude": xr.Variable(
                dims=["nscan", "npixel"],
                data=np.linspace(-90.0, 90.0, nscan * npixel)
                .reshape(
                    (
                        nscan,
                        npixel,
                    )
                )
                .astype(np.float32),
                attrs={
                    "DimensionNames": "nscan,npixel",
                    "Units": "degrees",
                    "units": "degrees",
                    "_FillValue": -9999.900390625,
                    "CodeMissingValue": "-9999.9",
                },
            ),
            "Longitude": xr.Variable(
                dims=["nscan", "npixel"],
                data=np.linspace(-180.0, 180.0, nscan * npixel)
                .reshape(
                    (
                        nscan,
                        npixel,
                    )
                )
                .astype(np.float32),
                attrs={
                    "DimensionNames": "nscan,npixel",
                    "Units": "degrees",
                    "units": "degrees",
                    "_FillValue": -9999.900390625,
                    "CodeMissingValue": "-9999.9",
                },
            ),
            "sunLocalTime": xr.Variable(
                dims=["nscan", "npixel"],
                data=np.arange(nscan * npixel, dtype=np.float32).reshape(
                    (
                        nscan,
                        npixel,
                    )
                ),
                attrs={
                    "DimensionNames": "nscan,npixel",
                    "Units": "hours",
                    "units": "hours",
                    "_FillValue": -9999.900390625,
                    "CodeMissingValue": "-9999.9",
                },
            ),
            "pixelStatus": xr.Variable(
                dims=["nscan", "npixel"],
                data=np.arange(nscan * npixel, dtype=np.int8).reshape(
                    (
                        nscan,
                        npixel,
                    )
                ),
                attrs={
                    "DimensionNames": "nscan,npixel",
                    "_FillValue": -99,
                    "CodeMissingValue": "-99",
                },
            ),
            "qualityFlag": xr.Variable(
                dims=["nscan", "npixel"],
                data=np.arange(nscan * npixel, dtype=np.int8).reshape(
                    (
                        nscan,
                        npixel,
                    )
                ),
                attrs={
                    "DimensionNames": "nscan,npixel",
                    "_FillValue": -99,
                    "CodeMissingValue": "-99",
                },
            ),
            "L1CqualityFlag": xr.Variable(
                dims=["nscan", "npixel"],
                data=np.arange(nscan * npixel, dtype=np.int16).reshape(
                    (
                        nscan,
                        npixel,
                    )
                ),
                attrs={
                    "DimensionNames": "nscan,npixel",
                    "_FillValue": -9999,
                    "CodeMissingValue": "-9999",
                },
            ),
            "totalColumnWaterVapor": xr.Variable(
                dims=["nscan", "npixel"],
                data=np.arange(nscan * npixel, dtype=np.float32).reshape(
                    (
                        nscan,
                        npixel,
                    )
                ),
                attrs={
                    "DimensionNames": "nscan,npixel",
                    "Units": "kg/m2",
                    "units": "kg/m2",
                    "_FillValue": -9999.900390625,
                    "CodeMissingValue": "-9999.9",
                },
            ),
            "temp2m": xr.Variable(
                dims=["nscan", "npixel"],
                data=np.arange(nscan * npixel, dtype=np.float32).reshape(
                    (
                        nscan,
                        npixel,
                    )
                ),
                attrs={
                    "DimensionNames": "nscan,npixel",
                    "Units": "K",
                    "units": "K",
                    "_FillValue": -9999.900390625,
                    "CodeMissingValue": "-9999.9",
                },
            ),
            "convectiveFraction": xr.Variable(
                dims=["nscan", "npixel"],
                data=np.arange(nscan * npixel, dtype=np.float32).reshape(
                    (
                        nscan,
                        npixel,
                    )
                ),
                attrs={
                    "DimensionNames": "nscan,npixel",
                    "_FillValue": -9999.900390625,
                    "CodeMissingValue": "-9999.9",
                },
            ),
            "moistureConvergence": xr.Variable(
                dims=["nscan", "npixel"],
                data=np.arange(nscan * npixel, dtype=np.float32).reshape(
                    (
                        nscan,
                        npixel,
                    )
                ),
                attrs={
                    "DimensionNames": "nscan,npixel",
                    "_FillValue": -9999.900390625,
                    "CodeMissingValue": "-9999.9",
                },
            ),
            "leafAreaIndex": xr.Variable(
                dims=["nscan", "npixel"],
                data=np.arange(nscan * npixel, dtype=np.float32).reshape(
                    (
                        nscan,
                        npixel,
                    )
                ),
                attrs={
                    "DimensionNames": "nscan,npixel",
                    "_FillValue": -9999.900390625,
                    "CodeMissingValue": "-9999.9",
                },
            ),
            "snowDepth": xr.Variable(
                dims=["nscan", "npixel"],
                data=np.arange(nscan * npixel, dtype=np.float32).reshape(
                    (
                        nscan,
                        npixel,
                    )
                ),
                attrs={
                    "DimensionNames": "nscan,npixel",
                    "Units": "m",
                    "units": "m",
                    "_FillValue": -9999.900390625,
                    "CodeMissingValue": "-9999.9",
                },
            ),
            "orographicWind": xr.Variable(
                dims=["nscan", "npixel"],
                data=np.arange(nscan * npixel, dtype=np.float32).reshape(
                    (
                        nscan,
                        npixel,
                    )
                ),
                attrs={
                    "DimensionNames": "nscan,npixel",
                    "_FillValue": -9999.900390625,
                    "CodeMissingValue": "-9999.9",
                },
            ),
            "windSpeed10m": xr.Variable(
                dims=["nscan", "npixel"],
                data=np.arange(nscan * npixel, dtype=np.float32).reshape(
                    (
                        nscan,
                        npixel,
                    )
                ),
                attrs={
                    "DimensionNames": "nscan,npixel",
                    "Units": "m/s",
                    "units": "m/s",
                    "_FillValue": -9999.900390625,
                    "CodeMissingValue": "-9999.9",
                },
            ),
            "mountainIndex": xr.Variable(
                dims=["nscan", "npixel"],
                data=np.arange(nscan * npixel, dtype=np.int16).reshape(
                    (
                        nscan,
                        npixel,
                    )
                ),
                attrs={
                    "DimensionNames": "nscan,npixel",
                    "_FillValue": -9999,
                    "CodeMissingValue": "-9999",
                },
            ),
            "landFraction": xr.Variable(
                dims=["nscan", "npixel"],
                data=np.arange(nscan * npixel, dtype=np.int16).reshape(
                    (
                        nscan,
                        npixel,
                    )
                ),
                attrs={
                    "DimensionNames": "nscan,npixel",
                    "_FillValue": -9999,
                    "CodeMissingValue": "-9999",
                },
            ),
            "iceFraction": xr.Variable(
                dims=["nscan", "npixel"],
                data=np.arange(nscan * npixel, dtype=np.int16).reshape(
                    (
                        nscan,
                        npixel,
                    )
                ),
                attrs={
                    "DimensionNames": "nscan,npixel",
                    "_FillValue": -9999,
                    "CodeMissingValue": "-9999",
                },
            ),
            "elevation": xr.Variable(
                dims=["nscan", "npixel"],
                data=np.arange(nscan * npixel, dtype=np.int16).reshape(
                    (
                        nscan,
                        npixel,
                    )
                ),
                attrs={
                    "DimensionNames": "nscan,npixel",
                    "Units": "m",
                    "units": "m",
                    "_FillValue": -9999,
                    "CodeMissingValue": "-9999",
                },
            ),
            "snowFlag": xr.Variable(
                dims=["nscan", "npixel"],
                data=np.arange(nscan * npixel, dtype=np.int8).reshape(
                    (
                        nscan,
                        npixel,
                    )
                ),
                attrs={
                    "DimensionNames": "nscan,npixel",
                    "_FillValue": -99,
                    "CodeMissingValue": "-99",
                },
            ),
            "sunGlintAngle": xr.Variable(
                dims=["nscan", "npixel"],
                data=np.arange(nscan * npixel, dtype=np.int8).reshape(
                    (
                        nscan,
                        npixel,
                    )
                ),
                attrs={
                    "DimensionNames": "nscan,npixel",
                    "Units": "degrees",
                    "units": "degrees",
                    "_FillValue": -99,
                    "CodeMissingValue": "-99",
                },
            ),
            "probabilityOfPrecip": xr.Variable(
                dims=["nscan", "npixel"],
                data=np.arange(nscan * npixel, dtype=np.int8).reshape(
                    (
                        nscan,
                        npixel,
                    )
                ),
                attrs={
                    "DimensionNames": "nscan,npixel",
                    "Units": "percent",
                    "units": "percent",
                    "_FillValue": -99,
                    "CodeMissingValue": "-99",
                },
            ),
            "precipitationYesNoFlag": xr.Variable(
                dims=["nscan", "npixel"],
                data=np.arange(nscan * npixel, dtype=np.int16).reshape(
                    (
                        nscan,
                        npixel,
                    )
                ),
                attrs={
                    "DimensionNames": "nscan,npixel",
                    "_FillValue": -9999,
                    "CodeMissingValue": "-9999",
                },
            ),
            "surfacePrecipitation": xr.Variable(
                dims=["nscan", "npixel"],
                data=np.arange(nscan * npixel, dtype=np.float32).reshape(
                    (
                        nscan,
                        npixel,
                    )
                ),
                attrs={
                    "DimensionNames": "nscan,npixel",
                    "Units": "mm/hr",
                    "units": "mm/hr",
                    "_FillValue": -9999.900390625,
                    "CodeMissingValue": "-9999.9",
                },
            ),
            "frozenPrecipitation": xr.Variable(
                dims=["nscan", "npixel"],
                data=np.arange(nscan * npixel, dtype=np.float32).reshape(
                    (
                        nscan,
                        npixel,
                    )
                ),
                attrs={
                    "DimensionNames": "nscan,npixel",
                    "Units": "mm/hr",
                    "units": "mm/hr",
                    "_FillValue": -9999.900390625,
                    "CodeMissingValue": "-9999.9",
                },
            ),
            "convectivePrecipitation": xr.Variable(
                dims=["nscan", "npixel"],
                data=np.arange(nscan * npixel, dtype=np.float32).reshape(
                    (
                        nscan,
                        npixel,
                    )
                ),
                attrs={
                    "DimensionNames": "nscan,npixel",
                    "Units": "mm/hr",
                    "units": "mm/hr",
                    "_FillValue": -9999.900390625,
                    "CodeMissingValue": "-9999.9",
                },
            ),
            "rainWaterPath": xr.Variable(
                dims=["nscan", "npixel"],
                data=np.arange(nscan * npixel, dtype=np.float32).reshape(
                    (
                        nscan,
                        npixel,
                    )
                ),
                attrs={
                    "DimensionNames": "nscan,npixel",
                    "Units": "kg/m^2",
                    "units": "kg/m^2",
                    "_FillValue": -9999.900390625,
                    "CodeMissingValue": "-9999.9",
                },
            ),
            "cloudWaterPath": xr.Variable(
                dims=["nscan", "npixel"],
                data=np.arange(nscan * npixel, dtype=np.float32).reshape(
                    (
                        nscan,
                        npixel,
                    )
                ),
                attrs={
                    "DimensionNames": "nscan,npixel",
                    "Units": "kg/m^2",
                    "units": "kg/m^2",
                    "_FillValue": -9999.900390625,
                    "CodeMissingValue": "-9999.9",
                },
            ),
            "iceWaterPath": xr.Variable(
                dims=["nscan", "npixel"],
                data=np.arange(nscan * npixel, dtype=np.float32).reshape(
                    (
                        nscan,
                        npixel,
                    )
                ),
                attrs={
                    "DimensionNames": "nscan,npixel",
                    "Units": "kg/m^2",
                    "units": "kg/m^2",
                    "_FillValue": -9999.900390625,
                    "CodeMissingValue": "-9999.9",
                },
            ),
            "mostLikelyPrecipitation": xr.Variable(
                dims=["nscan", "npixel"],
                data=np.arange(nscan * npixel, dtype=np.float32).reshape(
                    (
                        nscan,
                        npixel,
                    )
                ),
                attrs={
                    "DimensionNames": "nscan,npixel",
                    "Units": "mm/hr",
                    "units": "mm/hr",
                    "_FillValue": -9999.900390625,
                    "CodeMissingValue": "-9999.9",
                },
            ),
            "precip1stTertial": xr.Variable(
                dims=["nscan", "npixel"],
                data=np.arange(nscan * npixel, dtype=np.float32).reshape(
                    (
                        nscan,
                        npixel,
                    )
                ),
                attrs={
                    "DimensionNames": "nscan,npixel",
                    "Units": "mm/hr",
                    "units": "mm/hr",
                    "_FillValue": -9999.900390625,
                    "CodeMissingValue": "-9999.9",
                },
            ),
            "precip2ndTertial": xr.Variable(
                dims=["nscan", "npixel"],
                data=np.arange(nscan * npixel, dtype=np.float32).reshape(
                    (
                        nscan,
                        npixel,
                    )
                ),
                attrs={
                    "DimensionNames": "nscan,npixel",
                    "_FillValue": -9999.900390625,
                    "CodeMissingValue": "-9999.9",
                },
            ),
            "rainWaterContent": xr.Variable(
                dims=["nscan", "npixel", "nlyrs"],
                data=np.arange(nscan * npixel * nlyrs, dtype=np.float32).reshape(
                    (
                        nscan,
                        npixel,
                        nlyrs,
                    )
                ),
                attrs={
                    "DimensionNames": "nscan,npixel,nlyrs",
                    "Units": "g/m3",
                    "units": "g/m3",
                    "_FillValue": -9999.900390625,
                    "CodeMissingValue": "-9999.9",
                },
            ),
            "cloudWaterContent": xr.Variable(
                dims=["nscan", "npixel", "nlyrs"],
                data=np.arange(nscan * npixel * nlyrs, dtype=np.float32).reshape(
                    (
                        nscan,
                        npixel,
                        nlyrs,
                    )
                ),
                attrs={
                    "DimensionNames": "nscan,npixel,nlyrs",
                    "Units": "g/m3",
                    "units": "g/m3",
                    "_FillValue": -9999.900390625,
                    "CodeMissingValue": "-9999.9",
                },
            ),
            "iceWaterContent": xr.Variable(
                dims=["nscan", "npixel", "nlyrs"],
                data=np.arange(nscan * npixel * nlyrs, dtype=np.float32).reshape(
                    (
                        nscan,
                        npixel,
                        nlyrs,
                    )
                ),
                attrs={
                    "DimensionNames": "nscan,npixel,nlyrs",
                    "Units": "g/m3",
                    "units": "g/m3",
                    "_FillValue": -9999.900390625,
                    "CodeMissingValue": "-9999.9",
                },
            ),
            "latentHeating": xr.Variable(
                dims=["nscan", "npixel", "nlyrs"],
                data=np.arange(nscan * npixel * nlyrs, dtype=np.float32).reshape(
                    (
                        nscan,
                        npixel,
                        nlyrs,
                    )
                ),
                attrs={
                    "DimensionNames": "nscan,npixel,nlyrs",
                    "Units": "K/hr",
                    "units": "K/hr",
                    "_FillValue": -9999.900390625,
                    "CodeMissingValue": "-9999.9",
                },
            ),
        },
        attrs={
            "SwathHeader": "NumberScansInSet=1;\nMaximumNumberScansTotal=4300;\nNumberScansBeforeGranule=0;\nNumberScansGranule=2280;\nNumberScansAfterGranule=0;\nNumberPixels=90;\nScanType=CROSSTRACK;\n"
        },
    )
    s1_ds.to_netcdf(
        filepath,
        mode="a",
        group="S1",
        engine="netcdf4",
    )

    s1_scantime_ds = xr.Dataset(
        data_vars={
            "Year": xr.Variable(
                dims= ("nscan",),
                data= np.full(nscan, 2026, dtype=np.int16),
                attrs= {
                    "dimensionnames": "nscan",
                    "units": "years",
                    "_fillvalue": -9999,
                    "codemissingvalue": "-9999",
                },
            ),
            "Month": xr.Variable(**{
                "dims": ("nscan",),
                "data": np.full(nscan, 2, dtype=np.int8),
                "attrs": {
                    "dimensionnames": "nscan",
                    "units": "months",
                    "_fillvalue": -99,
                    "codemissingvalue": "-99",
                },
            }),
            "DayOfMonth": xr.Variable(**{
                "dims": ("nscan",),
                "data": np.full(nscan, 28, dtype=np.int8),
                "attrs": {
                    "dimensionnames": "nscan",
                    "units": "days",
                    "_fillvalue": -99,
                    "codemissingvalue": "-99",
                },
            }),
            "Hour": xr.Variable(**{
                "dims": ("nscan",),
                "data": np.full(nscan, 23, dtype=np.int8),
                "attrs": {
                    "dimensionnames": "nscan",
                    "units": "hours",
                    "_fillvalue": -99,
                    "codemissingvalue": "-99",
                },
            }),
            "Minute": xr.Variable(**{
                "dims": ("nscan",),
                "data": np.full(nscan, 31, dtype=np.int8),
                "attrs": {
                    "dimensionnames": "nscan",
                    "units": "minutes",
                    "_fillvalue": -99,
                    "codemissingvalue": "-99",
                },
            }),
            "Second": xr.Variable(**{
                "dims": ("nscan",),
                "data": np.full(nscan, 56, dtype=np.int8),
                "attrs": {
                    "dimensionnames": "nscan",
                    "units": "s",
                    "_fillvalue": -99,
                    "codemissingvalue": "-99",
                },
            }),
            "MilliSecond": xr.Variable(**{
                "dims": ("nscan",),
                "data": np.full(nscan, 0, dtype=np.int16),
                "attrs": {
                    "dimensionnames": "nscan",
                    "units": "ms",
                    "_fillvalue": -9999,
                    "codemissingvalue": "-9999",
                },
            }),
            "DayOfYear": xr.Variable(**{
                "dims": ("nscan",),
                "data": np.full(nscan, 59, dtype=np.int16),
                "attrs": {
                    "dimensionnames": "nscan",
                    "units": "days",
                    "_fillvalue": -9999,
                    "codemissingvalue": "-9999",
                },
            }),
            "SecondOfDay": xr.Variable(**{
                "dims": ("nscan",),
                "data": np.linspace(84716.0, 84726.0, nscan).astype(np.float64),
                "attrs": {
                    "dimensionnames": "nscan",
                    "units": "s",
                    "_fillvalue": -9999.9,
                    "codemissingvalue": "-9999.9",
                },
            }),
        },
        attrs={},
    )
    s1_scantime_ds.to_netcdf(
        filepath,
        mode="a",
        group="S1/ScanTime",
        engine="netcdf4",
    )

    s1_scstatus_ds = xr.Dataset(
        data_vars={
            "SCorientation": xr.Variable(
                dims=["nscan"],
                data=np.arange(nscan, dtype=np.int16),
                attrs={
                    "DimensionNames": "nscan",
                    "Units": "degrees",
                    "units": "degrees",
                    "_FillValue": -9999,
                    "CodeMissingValue": "-9999",
                },
            ),
            "SClatitude": xr.Variable(
                dims=["nscan"],
                data=np.arange(nscan, dtype=np.float32),
                attrs={
                    "DimensionNames": "nscan",
                    "Units": "degrees",
                    "units": "degrees",
                    "_FillValue": -9999.900390625,
                    "CodeMissingValue": "-9999.9",
                },
            ),
            "SClongitude": xr.Variable(
                dims=["nscan"],
                data=np.arange(nscan, dtype=np.float32),
                attrs={
                    "DimensionNames": "nscan",
                    "Units": "degrees",
                    "units": "degrees",
                    "_FillValue": -9999.900390625,
                    "CodeMissingValue": "-9999.9",
                },
            ),
            "SCaltitude": xr.Variable(
                dims=["nscan"],
                data=np.arange(nscan, dtype=np.float32),
                attrs={
                    "DimensionNames": "nscan",
                    "Units": "km",
                    "units": "km",
                    "_FillValue": -9999.900390625,
                    "CodeMissingValue": "-9999.9",
                },
            ),
            "FractionalGranuleNumber": xr.Variable(
                dims=["nscan"],
                data=np.arange(nscan, dtype=np.float64),
                attrs={
                    "DimensionNames": "nscan",
                    "_FillValue": -9999.9,
                    "CodeMissingValue": "-9999.9",
                },
            ),
        },
        attrs={},
    )
    s1_scstatus_ds.to_netcdf(
        filepath,
        mode="a",
        group="S1/SCstatus",
        engine="netcdf4",
    )

    return filepath
