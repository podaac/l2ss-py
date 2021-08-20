### UMM-S Update
This directory holds files for updating the service's CMR UMM-S profile.

Core files for CMR UMM-S update and associations are:
* cmr.Dockerfile (for running the script via Jenkins)
* run_umms_updater.sh (for executing the command line request)
* cmr_umms_s.json (UMM-S profile to keep updated locally)
* associations.txt (list of concept_ids, one per line, to be associated with UMM-S)