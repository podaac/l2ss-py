
# NASA EOSDIS Integrations

## Overview

This document provides detailed instructions for DAACs to integrate their collections with the PODAAC l2ss-py Harmony service.

### Adding new collections to L2ss-py Harmony Service

1. #### Add UMM-V for new collections to both OPS and UAT.  Must define latitude and longitude.  Time is optional but preferred if used in the collection.

   a. The latitude and longitude variables require a Type of "COORDINATE" and a SubType of "LATITUDE" or "LONGITUDE" respectively.

2. #### Add the new association to the l2ss-py service in UAT and OPS

   a. Go to the UAT Earthdata [Metadata Management Tool UAT](https://mmt.uat.earthdata.nasa.gov)

   b. Login with your Earthdata credentials

   c. Go to Services (On the left-hand side), and click the "All Services" link

   d. Search for "podaac l2" in the Search Bar

   e. Click the `PODAAC L2 Cloud Subsetter` service from provider `POCLOUD`

   f. Click the 3 dots on the upper right side and then click `Collection Associations`

   g. Click `Add Collection Associations`

   h. Use the Search Field to find the collection you want to add.

   i. Check the box next to the collections you want to add and click `Associate Selected Collections`

   j. Repeat steps above for OPS, go to the OPS Earthdata [Metadata Management Tool OPS](https://mmt.earthdata.nasa.gov)


3. #### Manually test the new association (OPTIONAL)

   a. (Test with UAT Harmony) Test the new association with Harmony

      i. Test subsetting in EDSC UAT

      ii. Test the collection in Harmony UAT with curl

   b. (Test l2ss-py locally) Test a granule from the collection with l2ss-py locally to ensure the collection is working as expected

4. #### (Autotest) Within 3 days, check how the collection is doing in the l2ss-py-autotest

   a. Check for errors in the PRs in [l2ss-py-autotest](https://github.com/podaac/l2ss-py-autotest/pulls)

      i. Go to the Pull requests tab and Filter using your `short name` or `concept id`.  Remove `is:open` to check closed PRs also.

      ii. If you see it in the list as `Open` then it either hasn't been tested yet or it failed.

      iii. If you find it `Closed` and `Merged` then it passed and you can skip step 5.

   b. If you find the PR as `Open` with a red X (failed), click the PR and then click the `Checks` tab

   c. Look on the left side to see the status of the PR and see which part failed, which will have a red X next to it.  Usually it is the `Tested with Harmony` that fails.

   d. Click the failed item and it will show which tests failed.  There are 2 types of tests, the `test_spatial_subset` and the `test_temporal_subset`

   e. Click the `Raw output` button to see the details of the tests.

   f. Use this information to determine if the error is in the l2ss-py code or in the l2ss-py-autotest.

   g. If your collection doesn't use `time` at all, then you can create a PR to skip the `test_temporal_subset` from this collection.  Update the appropriate file in this [directory](https://github.com/podaac/l2ss-py-autotest/tree/main/tests/skip).  Add the collection concept id to the appropriate file.

5. #### If there are errors in your l2ss-py-autotest collection PRs then:

   a. Try to determine if the error is in the l2ss-py code or in the l2ss-py-autotest.
   
   b. Fork either the l2ss-py repo or the l2ss-py-autotest repo based on previous step.

   c. Create a new branch from `develop` named `feature/fix-<collection-name>`

   d. Make the necessary changes in the branch and test locally until the error is fixed.

   e. Make a PR to the l2ss-py or l2ss-py-autotest repo into the `develop` branch

   f. Ask for a review from the PODAAC team.  Add `jamesfwood` and `sliu` as Reviewers.

   g. [PODAAC] The PODAAC team will review the PR and merge it into the `develop` branch

   h. [PODAAC] The PODAAC team will make a new release of l2ss-py and deploy it to UAT

   i. Within 3 days, the l2ss-py-autotest will run and retest the collection in UAT.  If the collection passes the autotest, then everything is good in UAT.  If the collection fails the autotest, then the DAAC will need to repeat step 5.

   j. [PODAAC] Once everything is in the l2ss-py release, the PODAAC team will merge the release to main and deploy it to OPS.  This step may take some time since it may be waiting for other changes to be completed.

   k. Within 3 days of deploying the release to OPS, the l2ss-py-autotest will run and retest the collection in OPS.  If the collection passes the autotest, then proceed to next step.  If the collection fails the autotest, then the DAAC will need to repeat step 5.

6. #### Test the collection with the l2ss-py Harmony service in OPS

   a. Test subsetting the collection in EDSC OPS

   b. curl the collection from the l2ss-py Harmony service in OPS
