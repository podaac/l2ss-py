# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Added
### Changed
### Deprecated 
### Removed
### Fixed
### Security


## [2.14.0]
### Added
- [issue/314](https://github.com/podaac/l2ss-py/pull/324):
	- Update python libraries
	- Update harmony service library version
	- Add in feature pixel subsetting
- Added detailed instructions to the README.md for integrating l2ss-py with new EOSDIS collections
### Changed
### Deprecated 
### Removed
### Fixed
### Security


## [2.13.0]
### Added
### Changed
### Deprecated 
### Removed
- Removed the "time" dependency when doing spatial only subsetting.
### Fixed
- Updated xarray enhancement get_indexers_from_nd function for SMAP_RSS_L2_SSS_V6.
- Fix minor bug in checking for empty indexers and same data bounds.
### Security


## [2.12.0]
### Added
- Added custom exception to progate error messages to harmony.
- Added an automated release step for main branch in github actions.
- Handle OB.DAAC files, which do not have time variables
- Update harmony library from 1.0 to 2.0 which changes the import harmony structure
### Changed
### Deprecated 
### Removed
### Fixed
### Security


## [2.11.0]
### Added
- [issue/267](https://github.com/podaac/l2ss-py/pull/261): Add xtrack and atrack dimension options for get_nd_indexers when bounding box subsetting is performed on SNDR.
- Fix temporal subsetting ghrsst dataset by adding time delta to time variable.
- Add a function to test ghrsst dataset ability to access variables when mask_and_scale is true. 
- Update l2ss-py to allow the return of empty granules.
### Changed
### Deprecated 
### Removed
- Removed old cmr association files, Dockerfile, and cmr updater script
### Fixed
### Security


## [2.10.0]
### Added
- [issue/260](https://github.com/podaac/l2ss-py/pull/261): Add gpm cleanup function to add a timeMidScan variable if the timeMidScan variable isn't present. Function takes the years, months, days etc ScanTime variables and creates a single time variable using datetime.datetime library.
### Changed
- Update code to determine lat lon time variables
- Update xarray version
- [pull/248](https://github.com/podaac/l2ss-py/pull/248): add Harmony extra_args.cut parameter to subset_params in service adapter 
### Deprecated 
### Removed
### Fixed
- [issue/258](https://github.com/podaac/l2ss-py/issues/258): fix so that ScanTime is what determines a GPM file 
### Security


## [2.9.0]
### Added
- [issue/240](https://github.com/podaac/l2ss-py/issues/240): for time vars that need datetime conversion, allow numpy arrays of n dimension to be able to convert. Create the same shape array in start time, then add the seconds since the start date to get our datetime format for temporal subsetting.
### Changed
- Moved 'Push Tag' step to end of Github Action
### Deprecated 
### Removed
### Fixed
- [issue/234](https://github.com/podaac/l2ss-py/issues/234): allow indexing on lat/lon variables with 3 dimensions. Rows axis is 2, collumns axis is 1
### Security
- Updated dependency versions to latest possible


## [2.8.0]
### Added
### Changed
- Upgraded `cmr-umm-updater` to 0.6.0
### Deprecated 
### Removed
### Fixed
- Fix shapefile subsetting if there is more than a 2d in lon lat in shapefile that includes a third dimension
- [pull/227](https://github.com/podaac/l2ss-py/pull/227): Fix null time values in TEMPO results when spatial+temporal subsetting
- [pull/231](https://github.com/podaac/l2ss-py/pull/231): Improve fix for null time values by ensuring null valuesdon't persist on edges of longest row of True values
### Security
- Updated dependency versions to latest possible

## [2.7.0]
### Added
### Changed
- [pull/216](https://github.com/podaac/l2ss-py/pull/216): Updated all python to 3.10 and update all depencency versions
### Deprecated 
### Removed
### Fixed
- [issue/209](https://github.com/podaac/l2ss-py/issues/209): Update code so dims are created if they don't already exists. 
- Update way we modify shapefile from 180 to 360 lon lats.
- [issue/205](https://github.com/podaac/l2ss-py/issues/205): Add coding capability for when groups have same dimension names but different values. Xarray rename dims is utilized
- [issue/220](https://github.com/podaac/l2ss-py/issues/220): Check if the time variables being found haven't already been found. Add time_vars as an extra arguement in compute_time_variable_name
### Security

## [2.6.0]
### Added
### Changed
- [issue/182](https://github.com/podaac/l2ss-py/issues/182): Update code so doesn't remove '/' on attribute values.
- [issue/178](https://github.com/podaac/l2ss-py/issues/178): Add function to make sure dimension in subset is same as original file
- Update github action umm updater to 0.5.0
- [issue/172](https://github.com/podaac/l2ss-py/issues/178): Fix shapefile subsetting by passing correct variable to subset function.
### Deprecated 
### Removed
### Fixed
- [issue/119](https://github.com/podaac/l2ss-py/issues/119): GPM variable dimensions are renamed from "phony_dim" to the dimension names in the variable attribute "DimensionNames"
- [issue/184](https://github.com/podaac/l2ss-py/issues/184): boundary box argument at the command line is changed to allow decimal numbers (i.e., floats) for coordinates
- [issue/189](https://github.com/podaac/l2ss-py/issues/189): Fix temporal subsetting for SWOT collections, use mask_and_scale args for opening granule file if we have an overflow in time fill value, use original dataset encoding when writing file.
- [issue/194](https://github.com/podaac/l2ss-py/issues/194): Return coordinate variables if requested in a variable subset
### Security


## [2.5.0]
### Added
### Changed
### Deprecated 
### Removed
### Fixed
- [issue/153](https://github.com/podaac/l2ss-py/issues/153): Remove the asc_node_tai93 variable when blank in the SNDR collections for xarray.decode_times to decode.
- PODAAC-5538: Reduce memory footprint of l2ss by loading each variable individually to write to memory
- [issue/155](https://github.com/podaac/l2ss-py/issues/155): lat var name prefix now generalized to unique group names. Group variables determined in subset_bbox using the unique group name.
- [issue/162](https://github.com/podaac/l2ss-py/issues/162): allow time variable subsetting differences handled for MLS and OCO3 collections. At the moment: assuming 'he5' files to be tai93 files is a fine assumption. Xarray does not decode these times in he5 files
- [issue/168](https://github.com/podaac/l2ss-py/issues/168): A separate copy of test data is used to get expected results before subsetting at the beginning of two test functions --- test_subset_empty_bbox() and test_specified_variables() --- that rely on opening the dataset more than once.
- PODAAC-5537: Fixed AQUARIUS_L2_SSS_V5 dataset
- Update github action docker versioning to use pep440
- [issue/170](https://github.com/podaac/l2ss-py/issues/170): Prioritize getting time variables if 'time' == var_time_name over 'time' being in the var name such as 'SunLocalTime'.
- [issue/173](https://github.com/podaac/l2ss-py/issues/173): Variables without any spatial dimensions in "indexers" array should get copied over to the subsetting new_dataset after the xarray.where() function is applied.
### Security

## [2.4.0]
### Added
### Changed
- [issue/142](https://github.com/podaac/l2ss-py/issues/142): Changed handling of duplicate dimensions as part of integration with new TEMPO ozone profile data.
### Deprecated 
### Removed
### Fixed
- [issue/149](https://github.com/podaac/l2ss-py/issues/149): Fixed compression level for netCDF4 object variable creation into a string. Will need to address after netcdf4 rebuilds library. https://github.com/Unidata/netcdf4-python/issues/1236
- [issue/143](https://github.com/podaac/l2ss-py/issues/143): Fixed bug when not specifying any variable subsetting for grouped datasets.
### Security

## [2.3.0]
### Added
- [issue/126](https://github.com/podaac/l2ss-py/issues/126): Added flexibility to variable subsetting
for variables to not have leading slash in the front
- [issue/136](https://github.com/podaac/l2ss-py/issues/136): Added type annotations throughout the package code
### Changed
### Deprecated 
### Removed
### Fixed
- PODAAC-5065: integration with SMAP_RSS_L2_SSS_V5, fix way xarray open granules that have `seconds since 2000-1-1 0:0:0 0` as a time unit.
- [issue/127](https://github.com/podaac/l2ss-py/issues/127): Fixed bug when subsetting variables in grouped datasets. Variable names passed to `subset` will now have `/` replaced by `GROUP_DELIM` so they can be located in flattened datasets 
### Security

## [2.2.0]
### Added
### Changed
- [issue/115](https://github.com/podaac/l2ss-py/issues/115): Added notes to README about installing "extra" harmony dependencies to avoid test suite fails. 
- [issue/85](https://github.com/podaac/l2ss-py/issues/85): Added initial poetry setup guidance to the README
- [issue/122](https://github.com/podaac/l2ss-py/issues/122): Changed renaming of duplicate dimension from netcdf4 to xarray per issues in the netcdf.rename function. https://github.com/Unidata/netcdf-c/issues/1672 	
### Deprecated 
### Removed
### Fixed
- [issue/119](https://github.com/podaac/l2ss-py/issues/119): Add extra line for variables without any dimensions after a squeeze in compute_time_vars():	
- [issue/110](https://github.com/podaac/l2ss-py/issues/110): Get the start date in convert_times and reconvert times into original type in _recombine groups method.
### Security

## [2.1.1]
### Changed
- [issue/113](https://github.com/podaac/l2ss-py/issues/113): SNDR collections use `timedelta` as data type, added extra line of logic to handle this datatype in xarray_enhancements. SNDR file added for test cases with these variable types.

## [2.1.0]
### Added
### Changed
- [issue/106](https://github.com/podaac/l2ss-py/issues/106): he5 temporal subsetting for determining start time. Find start time in units attributes or work back from first UTC time.
- [issue/93](https://github.com/podaac/l2ss-py/issues/93): Added input argument to cmr updater to disable association removal
### Deprecated 
### Removed
### Fixed
- [issue/105](https://github.com/podaac/l2ss-py/issues/105): Added function to convert np object to python native objects.
### Security


## [2.0.0]
### Added
- [issue/98](https://github.com/podaac/l2ss-py/issues/98): Added logic to handle time decoding for he5 tai93 files. Changed the min and max inputs to tai93 format and compared to the time format in the file
### Changed 
- [pull/101](https://github.com/podaac/l2ss-py/pull/101): Updated docker image to `python3.9-slim`
- [issue/99](https://github.com/podaac/l2ss-py/issues/99): Updated python dependencies including v1.0.20 of harmony-service-lib
### Deprecated 
### Removed
- **Breaking Change** [issue/99](https://github.com/podaac/l2ss-py/issues/99): Removed support for python 3.7
### Fixed
- [issue/95](https://github.com/podaac/l2ss-py/issues/95): Fix non variable subsets for OMI since variables are not in the same group as the lat lon variables 

### Security


## [1.5.0]
### Added
- Added Shapefile option to UMM-S entry
- Added optional coordinate variable params
- [issues/78](https://github.com/podaac/l2ss-py/issues/72): Pass coordinate variables from service to l2ss-py
### Changed 
- Updated dependency versions
- [issues/88](https://github.com/podaac/l2ss-py/issues/88): Build pipeline manually pushes tag rather than use action-push-tag
### Deprecated 
### Removed
### Fixed
- [issues/72](https://github.com/podaac/l2ss-py/issues/72). Fix SMAP_RSS_L2_SSS_V4 subsetting, changed calculate chunk function.
- [issues/9](https://github.com/podaac/l2ss-py/issues/9). Determinate coordinate variables using cf_xarray.
### Security
- Changed CLI step in build action to use snyk monitor so that report is uploaded to SNYK podaac org

## [1.4.0]
### Added
- [issues/46](https://github.com/podaac/l2ss-py/issues/46). Flattening of h5py file.
- [issues/39](https://github.com/podaac/l2ss-py/issues/39): Exposed shapefile subsetting capability to Harmony
- [issues/58](https://github.com/podaac/l2ss-py/issues/58). Expand coordinates to accomodate OMI files 
	latitude variable in OMI has a capital L for Latitude that needs to be added to the list in 
	get_coordinate_variable_names. 

### Changed 
### Deprecated 
### Removed
- Remove OCO3 test. Get_coordinate_variables passed OMI and fails OCO3 because OCO3 has a multiple Latitude variable. Subset with bbox method is not
	applied properly to OCO3. Further manipulating will need to be done - OMI is a higher priority.
### Fixed
- [issues/61](https://github.com/podaac/l2ss-py/issues/61). Variables without dimensions should be included in the output subset. Previous code was
	adding dimension to variables in tropomi and SNDR as well as not have enough memory to {SCALAR} dimensions.

## [1.3.1]
### Added
- [issues/50](https://github.com/podaac/l2ss-py/issues/50): Spatial bounds are computed correctly for grouped empty subset operations
- Added `timeout` option to `cmr-umm-updater`
### Changed 
- Upgraded `cmr-umm-updater` to 0.2.1
### Deprecated 
### Removed
### Fixed
- [issues/48](https://github.com/podaac/l2ss-py/issues/48): get_epoch_time_var was not able to pick up the 'time' variable for the TROPOMI CH4 collection. Extra elif statement was added to get the full time variable returned.
- [issues/54](https://github.com/podaac/l2ss-py/issues/54): Skip encoding when xr dataset is empty
### Security

## [1.3.0]
### Added
- [issues/27](https://github.com/podaac/l2ss-py/issues/27): Xarray is unable to handle variables with duplicate dimensions. Module dimension_cleanup.py added to handle variables that may have duplicate dimensions. Method remove_duplicate_dims() creates a new dimension identical dimension to the dimensions originally duplicated so the dimension does not need to be duplicated and can have the same shape and values.
- [issues/24](https://github.com/podaac/l2ss-py/issues/24): Added support for time as lines
### Changed 
- [issues/36](https://github.com/podaac/l2ss-py/issues/36): Empty datasets will now maintain attributes, variables, and dimensions where each variable contains a single data point where the value is masked.
### Deprecated 
### Removed
### Fixed
- [issues/34](https://github.com/podaac/l2ss-py/issues/34): Fixed bug that did not allow variable subsetting in OCO3 files. Fix includes adding the variable list in subset_bbox method 
### Security

## [1.2.0]
### Added
### Changed 
- Updated dependency versions
### Deprecated 
### Removed
### Fixed
- [issues/32](https://github.com/podaac/l2ss-py/issues/32): Fixed bug when given variables to subset that have a '/' character in the variable name, they would not appear in the output.
- [issues/20](https://github.com/podaac/l2ss-py/issues/20): Fixed bug where spatial indexing was including extra dimensions causing output file to drastically increase in size
- [issues/10](https://github.com/podaac/l2ss-py/issues/10): Fixed bug where variable dimensions are assumed to be the same across all variables in a group in recombine_group_dataset method. When variables are written out, shape must match.
- [issues/28](https://github.com/podaac/l2ss-py/issues/28): Fixed bug where variable dtype would be type object, the code would raise exception. Fix adds logic to handle type object 
### Security

## [1.1.0]
### Added
- [issues/11](https://github.com/podaac/l2ss-py/issues/11): Added .squeeze on lat in get_time_var method. Added GROUP_DELIM to the root group variables.
- [issues/17](https://github.com/podaac/l2ss-py/issues/17): Integrated with cmr-umm-updater
### Changed 
- [issues/15](https://github.com/podaac/l2ss-py/issues/15): Changed the way groups are handled so that variables at the root of a file are not ignored. Groups now include the '/' level group
### Deprecated 
### Removed
### Fixed
- Fixed bug where temporal and variable subsetting resulted in failure
### Security

## [1.0.0]
### Added
- PODAAC-3620: Added a script for running l2ss-py locally without Harmony
- PODAAC-3620: Updated README with details about how to test l2ss-py
### Changed 
- Moved to GitHub.com!
### Deprecated 
### Removed 
### Fixed 
- PODAAC-3657: Appropriate Harmony base URL is used in UMM-S entry based on venue
### Security

## [0.16.0]
### Added
### Changed 
- PODAAC-3530: Improved logic that determines coordinate variables
- Updated UMM-S record to indicate temporal subsetting is available 
### Deprecated 
### Removed 
### Fixed 
- PODAAC-3627: Fix subsetting MERGED_TP_J1_OSTM_OST_CYCLES_V42 collection
### Security

## [0.15.0]
### Added
- Added VIIRS and S6 collection associations
- PODAAC-3441: Added temporal subsetting capability
### Changed
- Updated dependency versions. (harmony-serivce-lib to 1.0.9)
### Deprecated 
### Removed 
### Fixed 
- PODAAC-3494
  - Fix filename derived_from in the json_history metadata.
- PODAAC-3493
  - Fix subsetted granule is larger than original file.
### Security

## [0.13.0]
### Added
- PODAAC-3353
  - Sync associations with hitide umm-t
- PODAAC-3361
  - Add history_json attribute after subsetting
### Changed 
### Deprecated 
### Removed 
### Fixed 
- Removed ending slash from UMM-S entry so it works with EDSC
### Security

## [0.11.0]

### Added
- PODAAC-3209
  - Added the ability to subset granules with groups
### Changed 
- PODAAC-3158
  - Upgraded `harmony-service-lib` to 1.0.4
  - Use `harmony-service-lib` from PyPI instead of nexus/maven
- PODAAC-2660
  - Coord variables are retained in a variable subset, even if not requested
- PODAAC-3353
  - Sync associations with hitide umm-t
### Deprecated 
### Removed 
### Fixed 
### Security
- PODAAC-3158
  - Updated dependencies to address Snyk warning

## [0.10.0]

### Added
### Changed 
### Deprecated 
### Removed 
### Fixed 
### Security
- PODAAC-3011 
    - Added pillow 8.1.0 and pyyaml 5.4 to fix vulnerabilities from snyk

## [0.9.0]

### Added
### Changed 
- HARMONY-616 - Updated to 0.0.30 of the harmony service library
- Updated UMM-S record to indicate spatial and variable subsetting are supported
### Deprecated 
### Removed 
### Fixed 
### Security

## [0.8.0]

### Added
- PCESA-2282 - Added harmony-service to deploy podaac/subsetter directly into Harmony ecosystem
- PCESA-2307 - Added variable subsetting to harmony-service
- PCESA-2280 - Subset returns new spatial bounds after subset
- PCESA-2324 - Added shapefile subset capabilities to the subset.py module
### Changed 
- PCESA-2308 
    - Updated Jenkins pipeline to push to ECC nexus
    - Moved harmony service into the built poetry project
### Deprecated 
### Removed 
### Fixed 
### Security

## [0.7.0]

### Added
- PCESA-1750 - Added UMM-S updater service and cmr/ directory that stores the json profile and associations.txt (which contains concept-ids) locally
### Changed 
- PCESA-2231 - Updated to use the new SNS baseworker, Job Service, and Staging Service 
- PCESA-2195 - Subset will not fail when bounding box contains no data. Instead, an empty file is returned.
- PCESA-2296 - Updated L2SS to use both CAE artifactory and the PODAAC artifactory settings, added tool.poetry.source in pypyoject.toml 
### Deprecated 
### Removed 
### Fixed 
### Security

## [0.6.1]

### Fixed
- Added missing ops.tfvars

## [0.6.0]

### Added
- PCESA-2177 - Added CodeBuild to build pytest Integration Tests
- PCESA-2176 - Added pytest integration tests (IT) to run at SIT
- PCESA-2192 - Added automatic End-to-End deployment (Artifactory and ECR) to Jenkins pipeline
### Changed 
- PCESA-2174 - Simultaneously deploy sit and sit-#### stacks via terraform+workspaces to the SIT environment.
- PCESA-2175 - L2SS jenkins job, upon creation of a PR, deploys the l2ss to the sit environment using the developer/PR workspace and stack naming conventions
- PCESA-1789 - Increased memory of ECS tasks to 750
- PCESA-2178 - Upon completion of the automated testing destroy the SIT DEV stack.
### Fixed 
- PCESA-2203 - Fixed the JobException to use the parent exception error message.
- PCESA-2202 - Update L2SS to destroy.sh to verify that terraform workspace matches ticket, then after complete delete workspace ticket 

## [0.5.0]
### Changed
- PCESA-1639 - Use t3 instances to enable 'unlimited' credits by default for cpu bursting
- PCESA-1815 - Parse incoming `variables` json field and pass to subsetter

## [0.4.0]
### Added
- PCESA-1779 - Added ESB subscription to SNS topic instead of placing message on SQS
### Changed 
### Deprecated 
### Removed 
### Fixed 
### Security 

## [0.3.0]
### Added
- PCESA-1530 - Throw error when bbox cannot be parsed
- PCESA-1530 - Throw error when 'lat' and 'lon' not in variables
- PCESA-1530 - Throw error when data dimensions is < 2
- PCESA-1824 - Added new JSON format logging using python-json-logger library
### Changed 
- PCESA-1550 - Updated to use amazon2 linux ECS ami 
- PCESA-1413 - Added pre-baked terraform usage to Jenkins
### Deprecated 
### Removed 
### Fixed 
### Security 
