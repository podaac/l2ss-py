# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Added
- [issues/50](https://github.com/podaac/l2ss-py/issues/50): Spatial bounds are computed correctly for grouped empty subset operations
### Changed 
### Deprecated 
### Removed
### Fixed
- [issues/48](https://github.com/podaac/l2ss-py/issues/48): get_epoch_time_var was not able to pick up the 'time' variable for the TROPOMI CH4 collection. Extra elif statement was added to get the full time variable returned.
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
