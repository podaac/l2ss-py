import h5py
import numpy as np
import pytest


@pytest.fixture(scope="session")
def fake_gpm_2adprenv_07_file(
    tmp_path_factory,
):
    """
    Generated from 2A-ENV.GPM.DPR.V9-20211125.20140308-S220950-E234217.000144.V07A.HDF5
    """
    filepath = (
        tmp_path_factory.mktemp("data")
        / "2A-ENV.GPM.DPR.V9-20211125.20140308-S220950-E234217.000144.V07A.HDF5"
    )

    nbin = 18
    nbinHS = 14
    nray = 10
    nrayHS = 6
    nscan = 22
    nwater = 2
    nwind = 2

    with h5py.File(filepath, "w") as f:
        # create groups
        fs = f.require_group("FS")
        fs_scantime = f.require_group("FS/ScanTime")
        fs_verenv = f.require_group("FS/VERENV")
        hs = f.require_group("HS")
        hs_scantime = f.require_group("HS/ScanTime")
        hs_verenv = f.require_group("HS/VERENV")

        # root attributes
        f.attrs["FileHeader"] = """DOI=;
DOIshortName=2ADPRENV;
AlgorithmID=2ADPRENV;
AlgorithmVersion=9.20211125;
FileName=2A-ENV.GPM.DPR.V9-20211125.20140308-S220950-E234217.000144.V07A.HDF5;
SatelliteName=GPM;
InstrumentName=DPR;
GenerationDateTime=2021-12-17T11:21:47.000Z;
StartGranuleDateTime=2014-03-08T22:09:50.674Z;
StopGranuleDateTime=2014-03-08T23:42:18.044Z;
GranuleNumber=144;
NumberOfSwaths=2;
NumberOfGrids=0;
GranuleStart=SOUTHERNMOST_LATITUDE;
TimeInterval=ORBIT;
ProcessingSystem=PPS;
ProductVersion=V07A;
EmptyGranule=NOT_EMPTY;
MissingData=0;
"""
        f.attrs[
            "InputRecord"
        ] = """InputFileNames=2A-ENV.GPM.Ku.V9-20211125.20140308-S220950-E234217.000144.V07A.HDF5,2A-ENV.GPM.Ka.V9-20211125.20140308-S220950-E234217.000144.V07A.HDF5;
InputAlgorithmVersions=9.20211125,9.20211125;
InputGenerationDateTimes=2021-12-17T10:55:54.000Z,2021-12-17T10:52:55.000Z;
"""
        f.attrs["NavigationRecord"] = """LongitudeOnEquator=-116.149478;
UTCDateTimeOnEquator=2014-03-08T22:32:57.505Z;
MeanSolarBetaAngle=32.603267;
EphemerisFileName=;
AttitudeFileName=;
GeoControlFileName=;
EphemerisSource=7_PVT_WITH_FALLBACK_AS_FLAGGED;
AttitudeSource=1_ON_BOARD_CALCULATED_PITCH_ROLL_YAW;
GeoToolkitVersion=V7.0   09.25.2020 GeoTKstruct.h ;
SensorAlignmentFirstRotationAngle=3.991320;
SensorAlignmentSecondRotationAngle=-0.008000;
SensorAlignmentThirdRotationAngle=-0.003300;
SensorAlignmentFirstRotationAxis=2;
SensorAlignmentSecondRotationAxis=1;
SensorAlignmentThirdRotationAxis=3;
"""
        f.attrs["FileInfo"] = """DataFormatVersion=7b;
TKCodeBuildVersion=0;
MetadataVersion=7b;
FormatPackage=HDF5-1.10.5;
BlueprintFilename=GPM.V7.2ADPRENV.blueprint.xml;
BlueprintVersion=BV_68;
TKIOVersion=3.99;
MetadataStyle=PVL;
EndianType=LITTLE_ENDIAN;
"""
        f.attrs["JAXAInfo"] = """GranuleFirstScanUTCDateTime=2014-03-08T22:09:51.089Z;
GranuleLastScanUTCDateTime=2014-03-08T23:42:17.853Z;
TotalQualityCode=Good;
FirstScanLat=-65.142609;
FirstScanLon=159.803345;
LastScanLat=-65.143753;
LastScanLon=136.329346;
NumberOfRainPixelsFS=12582;
NumberOfRainPixelsHS=5746;
ProcessingSubSystem=ALGORITHM;
ProcessingMode=STD;
LightSpeed=299792458;
DielectricFactorKa=0.898900;
DielectricFactorKu=0.925500;
"""

        # attributes for FS
        fs.attrs["FS_SwathHeader"] = """NumberScansInSet=1;
MaximumNumberScansTotal=10000;
NumberScansBeforeGranule=0;
NumberScansGranule=7925;
NumberScansAfterGranule=0;
NumberPixels=49;
ScanType=CROSSTRACK;
"""

        # attributes for HS
        hs.attrs["HS_SwathHeader"] = """NumberScansInSet=1;
MaximumNumberScansTotal=10000;
NumberScansBeforeGranule=0;
NumberScansGranule=7925;
NumberScansAfterGranule=0;
NumberPixels=24;
ScanType=CROSSTRACK;
"""

        # FS/Latitude
        fs_latitude = fs.create_dataset(
            "Latitude",
            data=np.linspace(-90.0, 90.0, nscan * nray)
            .reshape(
                (
                    nscan,
                    nray,
                )
            )
            .astype(np.float32),
        )
        fs_latitude.attrs["DimensionNames"] = "nscan,nray"
        fs_latitude.attrs["Units"] = "degrees"
        fs_latitude.attrs["units"] = "degrees"
        fs_latitude.attrs["_FillValue"] = np.float32(-9999.900390625)
        fs_latitude.attrs["CodeMissingValue"] = "-9999.9"

        # FS/Longitude
        fs_longitude = fs.create_dataset(
            "Longitude",
            data=np.linspace(-180.0, 180.0, nscan * nray)
            .reshape(
                (
                    nscan,
                    nray,
                )
            )
            .astype(np.float32),
        )
        fs_longitude.attrs["DimensionNames"] = "nscan,nray"
        fs_longitude.attrs["Units"] = "degrees"
        fs_longitude.attrs["units"] = "degrees"
        fs_longitude.attrs["_FillValue"] = np.float32(-9999.900390625)
        fs_longitude.attrs["CodeMissingValue"] = "-9999.9"

        # FS/ScanTime/DayOfMonth
        fs_scantime_dayofmonth = fs_scantime.create_dataset(
            "DayOfMonth",
            data=np.full(nscan, 30, dtype=np.int8),
        )
        fs_scantime_dayofmonth.attrs["DimensionNames"] = "nscan"
        fs_scantime_dayofmonth.attrs["Units"] = "days"
        fs_scantime_dayofmonth.attrs["units"] = "days"
        fs_scantime_dayofmonth.attrs["_FillValue"] = np.int8(-99)
        fs_scantime_dayofmonth.attrs["CodeMissingValue"] = "-99"

        # FS/ScanTime/DayOfYear
        fs_scantime_dayofyear = fs_scantime.create_dataset(
            "DayOfYear",
            data=np.full(nscan, 150, dtype=np.int16),
        )
        fs_scantime_dayofyear.attrs["DimensionNames"] = "nscan"
        fs_scantime_dayofyear.attrs["Units"] = "days"
        fs_scantime_dayofyear.attrs["units"] = "days"
        fs_scantime_dayofyear.attrs["_FillValue"] = np.int16(-9999)
        fs_scantime_dayofyear.attrs["CodeMissingValue"] = "-9999"

        # FS/ScanTime/Hour
        fs_scantime_hour = fs_scantime.create_dataset(
            "Hour",
            data=np.full(nscan, 9, dtype=np.int8),
        )
        fs_scantime_hour.attrs["DimensionNames"] = "nscan"
        fs_scantime_hour.attrs["Units"] = "hours"
        fs_scantime_hour.attrs["units"] = "hours"
        fs_scantime_hour.attrs["_FillValue"] = np.int8(-99)
        fs_scantime_hour.attrs["CodeMissingValue"] = "-99"

        # FS/ScanTime/MilliSecond
        fs_scantime_millisecond = fs_scantime.create_dataset(
            "MilliSecond",
            data=np.full(nscan, 0, dtype=np.int16),
        )
        fs_scantime_millisecond.attrs["DimensionNames"] = "nscan"
        fs_scantime_millisecond.attrs["Units"] = "ms"
        fs_scantime_millisecond.attrs["units"] = "ms"
        fs_scantime_millisecond.attrs["_FillValue"] = np.int16(-9999)
        fs_scantime_millisecond.attrs["CodeMissingValue"] = "-9999"

        # FS/ScanTime/Minute
        fs_scantime_minute = fs_scantime.create_dataset(
            "Minute",
            data=np.full(nscan, 57, dtype=np.int8),
        )
        fs_scantime_minute.attrs["DimensionNames"] = "nscan"
        fs_scantime_minute.attrs["Units"] = "minutes"
        fs_scantime_minute.attrs["units"] = "minutes"
        fs_scantime_minute.attrs["_FillValue"] = np.int8(-99)
        fs_scantime_minute.attrs["CodeMissingValue"] = "-99"

        # FS/ScanTime/Month
        fs_scantime_month = fs_scantime.create_dataset(
            "Month",
            data=np.full(nscan, 5, dtype=np.int8),
        )
        fs_scantime_month.attrs["DimensionNames"] = "nscan"
        fs_scantime_month.attrs["Units"] = "months"
        fs_scantime_month.attrs["units"] = "months"
        fs_scantime_month.attrs["_FillValue"] = np.int8(-99)
        fs_scantime_month.attrs["CodeMissingValue"] = "-99"

        # FS/ScanTime/Second
        fs_scantime_second = fs_scantime.create_dataset(
            "Second",
            data=np.full(nscan, 0, dtype=np.int8),
        )
        fs_scantime_second.attrs["DimensionNames"] = "nscan"
        fs_scantime_second.attrs["Units"] = "s"
        fs_scantime_second.attrs["units"] = "s"
        fs_scantime_second.attrs["_FillValue"] = np.int8(-99)
        fs_scantime_second.attrs["CodeMissingValue"] = "-99"

        # FS/ScanTime/SecondOfDay
        fs_scantime_secondofday = fs_scantime.create_dataset(
            "SecondOfDay",
            data=np.full(nscan, 35820.0, dtype=np.float64),
        )
        fs_scantime_secondofday.attrs["DimensionNames"] = "nscan"
        fs_scantime_secondofday.attrs["Units"] = "s"
        fs_scantime_secondofday.attrs["units"] = "s"
        fs_scantime_secondofday.attrs["_FillValue"] = np.float64(-9999.9)
        fs_scantime_secondofday.attrs["CodeMissingValue"] = "-9999.9"

        # FS/ScanTime/Year
        fs_scantime_year = fs_scantime.create_dataset(
            "Year",
            data=np.full(nscan, 2021, dtype=np.int16),
        )
        fs_scantime_year.attrs["DimensionNames"] = "nscan"
        fs_scantime_year.attrs["Units"] = "years"
        fs_scantime_year.attrs["units"] = "years"
        fs_scantime_year.attrs["_FillValue"] = np.int16(-9999)
        fs_scantime_year.attrs["CodeMissingValue"] = "-9999"

        # FS/VERENV/airPressure
        fs_verenv_airpressure = fs_verenv.create_dataset(
            "airPressure",
            data=np.arange(nscan * nray * nbin, dtype=np.float32).reshape(
                (
                    nscan,
                    nray,
                    nbin,
                )
            ),
        )
        fs_verenv_airpressure.attrs["DimensionNames"] = "nscan,nray,nbin"
        fs_verenv_airpressure.attrs["Units"] = "hPa"
        fs_verenv_airpressure.attrs["units"] = "hPa"
        fs_verenv_airpressure.attrs["_FillValue"] = np.float32(-9999.900390625)
        fs_verenv_airpressure.attrs["CodeMissingValue"] = "-9999.9"

        # FS/VERENV/cloudLiquidWater
        fs_verenv_cloudliquidwater = fs_verenv.create_dataset(
            "cloudLiquidWater",
            data=np.arange(nscan * nray * nbin * nwater, dtype=np.float32).reshape(
                (
                    nscan,
                    nray,
                    nbin,
                    nwater,
                )
            ),
        )
        fs_verenv_cloudliquidwater.attrs["DimensionNames"] = "nscan,nray,nbin,nwater"
        fs_verenv_cloudliquidwater.attrs["Units"] = "kg/m^3"
        fs_verenv_cloudliquidwater.attrs["units"] = "kg/m^3"
        fs_verenv_cloudliquidwater.attrs["_FillValue"] = np.float32(-9999.900390625)
        fs_verenv_cloudliquidwater.attrs["CodeMissingValue"] = "-9999.9"

        # FS/VERENV/skinTemperature
        fs_verenv_skintemperature = fs_verenv.create_dataset(
            "skinTemperature",
            data=np.arange(nscan * nray, dtype=np.float32).reshape(
                (
                    nscan,
                    nray,
                )
            ),
        )
        fs_verenv_skintemperature.attrs["DimensionNames"] = "nscan,nray"
        fs_verenv_skintemperature.attrs["Units"] = "K"
        fs_verenv_skintemperature.attrs["units"] = "K"
        fs_verenv_skintemperature.attrs["_FillValue"] = np.float32(-9999.900390625)
        fs_verenv_skintemperature.attrs["CodeMissingValue"] = "-9999.9"

        # FS/VERENV/surfacePressure
        fs_verenv_surfacepressure = fs_verenv.create_dataset(
            "surfacePressure",
            data=np.arange(nscan * nray, dtype=np.float32).reshape(
                (
                    nscan,
                    nray,
                )
            ),
        )
        fs_verenv_surfacepressure.attrs["DimensionNames"] = "nscan,nray"
        fs_verenv_surfacepressure.attrs["Units"] = "hPa"
        fs_verenv_surfacepressure.attrs["units"] = "hPa"
        fs_verenv_surfacepressure.attrs["_FillValue"] = np.float32(-9999.900390625)
        fs_verenv_surfacepressure.attrs["CodeMissingValue"] = "-9999.9"

        # FS/VERENV/surfaceTemperature
        fs_verenv_surfacetemperature = fs_verenv.create_dataset(
            "surfaceTemperature",
            data=np.arange(nscan * nray, dtype=np.float32).reshape(
                (
                    nscan,
                    nray,
                )
            ),
        )
        fs_verenv_surfacetemperature.attrs["DimensionNames"] = "nscan,nray"
        fs_verenv_surfacetemperature.attrs["Units"] = "K"
        fs_verenv_surfacetemperature.attrs["units"] = "K"
        fs_verenv_surfacetemperature.attrs["_FillValue"] = np.float32(-9999.900390625)
        fs_verenv_surfacetemperature.attrs["CodeMissingValue"] = "-9999.9"

        # FS/VERENV/surfaceWind
        fs_verenv_surfacewind = fs_verenv.create_dataset(
            "surfaceWind",
            data=np.arange(nscan * nray * nwind, dtype=np.float32).reshape(
                (
                    nscan,
                    nray,
                    nwind,
                )
            ),
        )
        fs_verenv_surfacewind.attrs["DimensionNames"] = "nscan,nray,nwind"
        fs_verenv_surfacewind.attrs["Units"] = "m/s"
        fs_verenv_surfacewind.attrs["units"] = "m/s"
        fs_verenv_surfacewind.attrs["_FillValue"] = np.float32(-9999.900390625)
        fs_verenv_surfacewind.attrs["CodeMissingValue"] = "-9999.9"

        # FS/VERENV/waterVapor
        fs_verenv_watervapor = fs_verenv.create_dataset(
            "waterVapor",
            data=np.arange(nscan * nray * nbin * nwater, dtype=np.float32).reshape(
                (
                    nscan,
                    nray,
                    nbin,
                    nwater,
                )
            ),
        )
        fs_verenv_watervapor.attrs["DimensionNames"] = "nscan,nray,nbin,nwater"
        fs_verenv_watervapor.attrs["Units"] = "kg/m^3"
        fs_verenv_watervapor.attrs["units"] = "kg/m^3"
        fs_verenv_watervapor.attrs["_FillValue"] = np.float32(-9999.900390625)
        fs_verenv_watervapor.attrs["CodeMissingValue"] = "-9999.9"

        # HS/Latitude
        hs_latitude = hs.create_dataset(
            "Latitude",
            data=np.linspace(-90.0, 90.0, nscan * nrayHS)
            .reshape(
                (
                    nscan,
                    nrayHS,
                )
            )
            .astype(np.float32),
        )
        hs_latitude.attrs["DimensionNames"] = "nscan,nrayHS"
        hs_latitude.attrs["Units"] = "degrees"
        hs_latitude.attrs["units"] = "degrees"
        hs_latitude.attrs["_FillValue"] = np.float32(-9999.900390625)
        hs_latitude.attrs["CodeMissingValue"] = "-9999.9"

        # HS/Longitude
        hs_longitude = hs.create_dataset(
            "Longitude",
            data=np.linspace(-180.0, 180.0, nscan * nrayHS)
            .reshape(
                (
                    nscan,
                    nrayHS,
                )
            )
            .astype(np.float32),
        )
        hs_longitude.attrs["DimensionNames"] = "nscan,nrayHS"
        hs_longitude.attrs["Units"] = "degrees"
        hs_longitude.attrs["units"] = "degrees"
        hs_longitude.attrs["_FillValue"] = np.float32(-9999.900390625)
        hs_longitude.attrs["CodeMissingValue"] = "-9999.9"

        hs_scantime_dayofmonth = hs_scantime.create_dataset(
            "DayOfMonth",
            data=np.full(nscan, 30, dtype=np.int8),
        )
        hs_scantime_dayofmonth.attrs["DimensionNames"] = "nscan"
        hs_scantime_dayofmonth.attrs["Units"] = "days"
        hs_scantime_dayofmonth.attrs["units"] = "days"
        hs_scantime_dayofmonth.attrs["_FillValue"] = np.int8(-99)
        hs_scantime_dayofmonth.attrs["CodeMissingValue"] = "-99"

        # HS/ScanTime/DayOfYear
        hs_scantime_dayofyear = hs_scantime.create_dataset(
            "DayOfYear",
            data=np.full(nscan, 150, dtype=np.int16),
        )
        hs_scantime_dayofyear.attrs["DimensionNames"] = "nscan"
        hs_scantime_dayofyear.attrs["Units"] = "days"
        hs_scantime_dayofyear.attrs["units"] = "days"
        hs_scantime_dayofyear.attrs["_FillValue"] = np.int16(-9999)
        hs_scantime_dayofyear.attrs["CodeMissingValue"] = "-9999"

        # HS/ScanTime/Hour
        hs_scantime_hour = hs_scantime.create_dataset(
            "Hour",
            data=np.full(nscan, 9, dtype=np.int8),
        )
        hs_scantime_hour.attrs["DimensionNames"] = "nscan"
        hs_scantime_hour.attrs["Units"] = "hours"
        hs_scantime_hour.attrs["units"] = "hours"
        hs_scantime_hour.attrs["_FillValue"] = np.int8(-99)
        hs_scantime_hour.attrs["CodeMissingValue"] = "-99"

        # HS/ScanTime/MilliSecond
        hs_scantime_millisecond = hs_scantime.create_dataset(
            "MilliSecond",
            data=np.full(nscan, 0, dtype=np.int16),
        )
        hs_scantime_millisecond.attrs["DimensionNames"] = "nscan"
        hs_scantime_millisecond.attrs["Units"] = "ms"
        hs_scantime_millisecond.attrs["units"] = "ms"
        hs_scantime_millisecond.attrs["_FillValue"] = np.int16(-9999)
        hs_scantime_millisecond.attrs["CodeMissingValue"] = "-9999"

        # HS/ScanTime/Minute
        hs_scantime_minute = hs_scantime.create_dataset(
            "Minute",
            data=np.full(nscan, 57, dtype=np.int8),
        )
        hs_scantime_minute.attrs["DimensionNames"] = "nscan"
        hs_scantime_minute.attrs["Units"] = "minutes"
        hs_scantime_minute.attrs["units"] = "minutes"
        hs_scantime_minute.attrs["_FillValue"] = np.int8(-99)
        hs_scantime_minute.attrs["CodeMissingValue"] = "-99"

        # HS/ScanTime/Month
        hs_scantime_month = hs_scantime.create_dataset(
            "Month",
            data=np.full(nscan, 5, dtype=np.int8),
        )
        hs_scantime_month.attrs["DimensionNames"] = "nscan"
        hs_scantime_month.attrs["Units"] = "months"
        hs_scantime_month.attrs["units"] = "months"
        hs_scantime_month.attrs["_FillValue"] = np.int8(-99)
        hs_scantime_month.attrs["CodeMissingValue"] = "-99"

        # HS/ScanTime/Second
        hs_scantime_second = hs_scantime.create_dataset(
            "Second",
            data=np.full(nscan, 0, dtype=np.int8),
        )
        hs_scantime_second.attrs["DimensionNames"] = "nscan"
        hs_scantime_second.attrs["Units"] = "s"
        hs_scantime_second.attrs["units"] = "s"
        hs_scantime_second.attrs["_FillValue"] = np.int8(-99)
        hs_scantime_second.attrs["CodeMissingValue"] = "-99"

        # HS/ScanTime/SecondOfDay
        hs_scantime_secondofday = hs_scantime.create_dataset(
            "SecondOfDay",
            data=np.full(nscan, 35820.0, dtype=np.float64),
        )
        hs_scantime_secondofday.attrs["DimensionNames"] = "nscan"
        hs_scantime_secondofday.attrs["Units"] = "s"
        hs_scantime_secondofday.attrs["units"] = "s"
        hs_scantime_secondofday.attrs["_FillValue"] = np.float64(-9999.9)
        hs_scantime_secondofday.attrs["CodeMissingValue"] = "-9999.9"

        # HS/ScanTime/Year
        hs_scantime_year = hs_scantime.create_dataset(
            "Year",
            data=np.full(nscan, 2021, dtype=np.int16),
        )
        hs_scantime_year.attrs["DimensionNames"] = "nscan"
        hs_scantime_year.attrs["Units"] = "years"
        hs_scantime_year.attrs["units"] = "years"
        hs_scantime_year.attrs["_FillValue"] = np.int16(-9999)
        hs_scantime_year.attrs["CodeMissingValue"] = "-9999"

        # HS/VERENV/airPressure
        hs_verenv_airpressure = hs_verenv.create_dataset(
            "airPressure",
            data=np.arange(nscan * nrayHS * nbinHS, dtype=np.float32).reshape(
                (
                    nscan,
                    nrayHS,
                    nbinHS,
                )
            ),
        )
        hs_verenv_airpressure.attrs["DimensionNames"] = "nscan,nrayHS,nbinHS"
        hs_verenv_airpressure.attrs["Units"] = "hPa"
        hs_verenv_airpressure.attrs["units"] = "hPa"
        hs_verenv_airpressure.attrs["_FillValue"] = np.float32(-9999.900390625)
        hs_verenv_airpressure.attrs["CodeMissingValue"] = "-9999.9"

        # HS/VERENV/cloudLiquidWater
        hs_verenv_cloudliquidwater = hs_verenv.create_dataset(
            "cloudLiquidWater",
            data=np.arange(nscan * nrayHS * nbinHS * nwater, dtype=np.float32).reshape(
                (
                    nscan,
                    nrayHS,
                    nbinHS,
                    nwater,
                )
            ),
        )
        hs_verenv_cloudliquidwater.attrs["DimensionNames"] = (
            "nscan,nrayHS,nbinHS,nwater"
        )
        hs_verenv_cloudliquidwater.attrs["Units"] = "kg/m^3"
        hs_verenv_cloudliquidwater.attrs["units"] = "kg/m^3"
        hs_verenv_cloudliquidwater.attrs["_FillValue"] = np.float32(-9999.900390625)
        hs_verenv_cloudliquidwater.attrs["CodeMissingValue"] = "-9999.9"

        # HS/VERENV/skinTemperature
        hs_verenv_skintemperature = hs_verenv.create_dataset(
            "skinTemperature",
            data=np.arange(nscan * nrayHS, dtype=np.float32).reshape(
                (
                    nscan,
                    nrayHS,
                )
            ),
        )
        hs_verenv_skintemperature.attrs["DimensionNames"] = "nscan,nrayHS"
        hs_verenv_skintemperature.attrs["Units"] = "K"
        hs_verenv_skintemperature.attrs["units"] = "K"
        hs_verenv_skintemperature.attrs["_FillValue"] = np.float32(-9999.900390625)
        hs_verenv_skintemperature.attrs["CodeMissingValue"] = "-9999.9"

        # HS/VERENV/surfacePressure
        hs_verenv_surfacepressure = hs_verenv.create_dataset(
            "surfacePressure",
            data=np.arange(nscan * nrayHS, dtype=np.float32).reshape(
                (
                    nscan,
                    nrayHS,
                )
            ),
        )
        hs_verenv_surfacepressure.attrs["DimensionNames"] = "nscan,nrayHS"
        hs_verenv_surfacepressure.attrs["Units"] = "hPa"
        hs_verenv_surfacepressure.attrs["units"] = "hPa"
        hs_verenv_surfacepressure.attrs["_FillValue"] = np.float32(-9999.900390625)
        hs_verenv_surfacepressure.attrs["CodeMissingValue"] = "-9999.9"

        # HS/VERENV/surfaceTemperature
        hs_verenv_surfacetemperature = hs_verenv.create_dataset(
            "surfaceTemperature",
            data=np.arange(nscan * nrayHS, dtype=np.float32).reshape(
                (
                    nscan,
                    nrayHS,
                )
            ),
        )
        hs_verenv_surfacetemperature.attrs["DimensionNames"] = "nscan,nrayHS"
        hs_verenv_surfacetemperature.attrs["Units"] = "K"
        hs_verenv_surfacetemperature.attrs["units"] = "K"
        hs_verenv_surfacetemperature.attrs["_FillValue"] = np.float32(-9999.900390625)
        hs_verenv_surfacetemperature.attrs["CodeMissingValue"] = "-9999.9"

        # HS/VERENV/surfaceWind
        hs_verenv_surfacewind = hs_verenv.create_dataset(
            "surfaceWind",
            data=np.arange(nscan * nrayHS * nwind, dtype=np.float32).reshape(
                (
                    nscan,
                    nrayHS,
                    nwind,
                )
            ),
        )
        hs_verenv_surfacewind.attrs["DimensionNames"] = "nscan,nrayHS,nwind"
        hs_verenv_surfacewind.attrs["Units"] = "m/s"
        hs_verenv_surfacewind.attrs["units"] = "m/s"
        hs_verenv_surfacewind.attrs["_FillValue"] = np.float32(-9999.900390625)
        hs_verenv_surfacewind.attrs["CodeMissingValue"] = "-9999.9"

        # HS/VERENV/waterVapor
        hs_verenv_watervapor = hs_verenv.create_dataset(
            "waterVapor",
            data=np.arange(nscan * nrayHS * nbinHS * nwater, dtype=np.float32).reshape(
                (
                    nscan,
                    nrayHS,
                    nbinHS,
                    nwater,
                )
            ),
        )
        hs_verenv_watervapor.attrs["DimensionNames"] = "nscan,nrayHS,nbinHS,nwater"
        hs_verenv_watervapor.attrs["Units"] = "kg/m^3"
        hs_verenv_watervapor.attrs["units"] = "kg/m^3"
        hs_verenv_watervapor.attrs["_FillValue"] = np.float32(-9999.900390625)
        hs_verenv_watervapor.attrs["CodeMissingValue"] = "-9999.9"

    return filepath
