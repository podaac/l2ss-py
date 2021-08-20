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

# This script is intended to be run by the CI/CD pipeline to push a docker tag previously built by build-docker.sh

set -Eeo pipefail

POSITIONAL=()
while [[ $# -gt 0 ]]
do
key="$1"

case $key in
    -t|--docker-tag)
    docker_tag="$2"
    shift # past argument
    shift # past value
    ;;
    -r|--registry)
    ARTIFACTORY_DOCKER_REGISTRY="$2"
    shift # past argument
    shift # past value
    ;;
    -u|--artifactory-username)
    ARTIFACTORY_USER="$2"
    shift # past argument
    shift # past value
    ;;
    -p|--artifactory-password)
    ARTIFACTORY_PASSWORD="$2"
    shift # past argument
    shift # past value
    ;;
    *)    # unknown option
    POSITIONAL+=("$1") # save it in an array for later
    shift # past argument
    ;;
esac
done
set -- "${POSITIONAL[@]}" # restore positional parameters

USAGE="push-docker-artifactory.sh -t|--docker-tag docker_tag -u|--artifactory-username ARTIFACTORY_USER -p|--artifactory-password ARTIFACTORY_PASSWORD"

# shellcheck disable=SC2154
if [[ -z "${docker_tag}" ]]; then
  echo "docker_tag required." >&2
  echo "$USAGE" >&2
  exit 1
fi

# shellcheck disable=SC2154
if [[ -z "${ARTIFACTORY_USER}" ]]; then
  echo "ARTIFACTORY_USER required." >&2
  echo "$USAGE" >&2
  exit 1
fi

# shellcheck disable=SC2154
if [[ -z "${ARTIFACTORY_PASSWORD}" ]]; then
  echo "ARTIFACTORY_PASSWORD required." >&2
  echo "$USAGE" >&2
  exit 1
fi

echo "${ARTIFACTORY_PASSWORD}" | docker login --username "${ARTIFACTORY_USER}" --password-stdin "${ARTIFACTORY_DOCKER_REGISTRY}"
docker tag "${docker_tag}" "${ARTIFACTORY_DOCKER_REGISTRY}/${docker_tag}"
docker push "${ARTIFACTORY_DOCKER_REGISTRY}/${docker_tag}"