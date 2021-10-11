
# l2ss-py

[![Coverage](https://sonarcloud.io/api/project_badges/measure?project=podaac_l2ss-py&metric=coverage)](https://sonarcloud.io/dashboard?id=podaac_l2ss-py)  
develop: [![Develop Build](https://github.com/podaac/l2ss-py/actions/workflows/build-pipeline.yml/badge.svg?branch=develop&event=push)](https://github.com/podaac/l2ss-py/actions/workflows/build-pipeline.yml)  
main: [![Main Build](https://github.com/podaac/l2ss-py/actions/workflows/build-pipeline.yml/badge.svg?branch=main&event=push)](https://github.com/podaac/l2ss-py/actions/workflows/build-pipeline.yml)

Harmony service for subsetting L2 data. l2ss-py supports:

- Spatial subsetting
- Temporal subsetting
- Variable subsetting

If you would like to contribute to l2ss-py, refer to the [contribution document](CONTRIBUTING.md).


## How to test l2ss-py locally

### Unit tests

There are comprehensive unit tests for l2ss-py. The tests can be run as follows:

```
poetry run pytest -m "not aws and not integration" tests/
```

You can generate coverage reports as follows:

```
poetry run pytest --junitxml=build/reports/pytest.xml --cov=podaac/ --cov-report=html -m "not aws and not integration" tests/
```

### l2ss-py script

You can run l2ss-py on a single granule without using Harmony. In order 
to run this, the l2ss-py package must be installed in your current 
Python interpreter

```
$ l2ss-py --help                                                                                                                    
usage: run_subsetter.py [-h] [--bbox BBOX BBOX BBOX BBOX]
                        [--variables VARIABLES [VARIABLES ...]]
                        [--min-time MIN_TIME] [--max-time MAX_TIME] [--cut]
                        input_file output_file

Run l2ss-py

positional arguments:
  input_file            File to subset
  output_file           Output file

optional arguments:
  -h, --help            show this help message and exit
  --bbox BBOX BBOX BBOX BBOX
                        Bounding box in the form min_lon min_lat max_lon
                        max_lat
  --variables VARIABLES [VARIABLES ...]
                        Variables, only include if variable subset is desired.
                        Should be a space separated list of variable names
                        e.g. sst wind_dir sst_error ...
  --min-time MIN_TIME   Min time. Should be ISO-8601 format. Only include if
                        temporal subset is desired.
  --max-time MAX_TIME   Max time. Should be ISO-8601 format. Only include if
                        temporal subset is desired.
  --cut                 If provided, scanline will be cut
```

For example:

```
l2ss-py /path/to/input.nc /path/to/output.nc --bbox -50 -10 50 10 --variables wind_speed wind_dir ice_age time --min-time '2015-07-02T09:00:00' --max-time '2015-07-02T10:00:00' --cut
```

### Running Harmony locally

In order to fully test l2ss-py with Harmony, you can run Harmony locally. This requires the data exists in UAT Earthdata Cloud.

1. Set up local Harmony instance. Instructions [here](https://github.com/nasa/harmony#Quick-Start)
2. Add concept ID for your data to [services.yml](https://github.com/nasa/harmony/blob/main/config/services.yml)
3. Execute a local Harmony l2ss-py request. For example:
    ```
   localhost:3000/YOUR_COLLECTION_ID/ogc-api-coverages/1.0.0/collections/all/coverage/rangeset?format=application%2Fx-netcdf4&subset=lat(-10%3A10)&subset=lon(-10%3A10)&maxResults=2
   ```
