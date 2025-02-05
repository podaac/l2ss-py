
# NASA EOSDIS Integrations

## Overview

This document provides detailed instructions for DAACs to integrate their collections with the PODAAC l2ss-py Harmony service.

### Adding new collections to L2ss-py Harmony Service

1. Add UMM-V for new collections to both OPS and UAT.  Must define latitude and longitude.  Time is optional but preferred if used in the collection.

   a. The latitude and longitude variables require a Type of "COORDINATE" and a subtype of "LATITUDE" or "LONGITUDE"

2. Add the new association to the l2ss-py service in UAT and OPS

   a. Go to the UAT Earthdata [Metadata Management Tool](https://mmt.uat.earthdata.nasa.gov)

   b. Login with your Earthdata credentials

   c. Go to Services (On the left-hand side), and click the "All Services" link

   d. Repeat steps above for OPS

3. [UAT] Manually test the new association (Optional)

   a. (Test with UAT Harmony) Test the new association with Harmony

      i. Test subsetting in EDSC UAT

      ii. Test the collection in Harmony UAT with curl

   b. (Test l2ss-py locally) Test a granule from the collection with l2ss-py locally to ensure the collection is working as expected

4. (Autotest) Within 3 days, check how the collection is doing in the l2ss-py-autotest

   a. Check the errors in the PRs

   b. add more...

5. If there are errors with l2ss-py-autotest PRs then:

   a. Try to determine if the error is in the l2ss-py code or in the l2ss-py-autotest.
   
   b. Fork either the l2ss-py repo or the l2ss-py-autotest repo based on previous step.

   b. Create a new branch from `develop` named `feature/<collection-name>`

   c. Make the necessary changes in the branch and test locally until the error is fixed.

   d. Make a PR to the l2ss-py or l2ss-py-autotest repo into the `develop` branch

   e. Ask for a review from the PODAAC team.  Add `jamesfwood` and `sliu` as Reviewers.

   f. [PODAAC] The PODAAC team will review the PR and merge it into the `develop` branch

   g. [PODAAC] The PODAAC team will make a new release of l2ss-py and deploy it to UAT

   h. **[UAT]** Within 3 days, the l2ss-py-autotest will run and retest the collection in UAT.  If the collection passes the autotest, then everything is good in UAT.  If the collection fails the autotest, then the DAAC will need to repeat step 5.

   i. Once everything is in the l2ss-py release, the PODAAC team will merge the release to main and deploy it to OPS.  This step may take some time before it is completed.

   j. **[OPS]** Within 3 days of deploying the release to OPS, the l2ss-py-autotest will run and retest the collection in OPS.  If the collection passes the autotest, then proceed to next step.  If the collection fails the autotest, then the DAAC will need to repeat step 5.

6. Manually test the collection with the l2ss-py Harmony service in OPS

   a. curl the collection from the l2ss-py Harmony service in OPS

   b. Test subsetting the collection in EDSC OPS
