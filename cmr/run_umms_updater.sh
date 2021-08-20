#!/usr/bin/env bash
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

set -Exo pipefail

tf_venue=$1
l2ss_version=$2
file=cmr/l2ss_cmr_umm_s.json

set +x

cmr_user=$(aws ssm get-parameter --profile "ngap-service-${tf_venue}" --with-decryption --name "urs_user" --output text --query Parameter.Value)
cmr_pass=$(aws ssm get-parameter --profile "ngap-service-${tf_venue}" --with-decryption --name "urs_password" --output text --query Parameter.Value)

jq --arg a $l2ss_version '.Version = $a' $file > cmr/cmr.json
umms_updater -d -f cmr/cmr.json -a cmr/${tf_venue}_associations.txt -p POCLOUD -e ${tf_venue} -cu "$cmr_user" -cp "$cmr_pass"

set -x