# Level 2 Subsetter Service Docker Image

This directory contains the `Dockerfile` used to build the Docker image capable of running the L2 Subsetter service.

It includes a number of helper scripts to be run by the CI/CD pipeline but can also be run locally to build the image.

## Building

There are two ways to build the L2 Subsetter Service, from the PO.DAAC 
Artifactory or from a local poetry build. 

### Building from Artifactory

Use the `build-docker.sh` script to build the docker image. There are two required arguments that must
be set:

1. service-name: The name of the service being built (from pyproject.toml)
2. service-version: The version of the service being built

This script will then call Docker build which will in turn retrieve the given version of the service from Artifactory
and install it into the Docker image. The docker tag of the built image will be returned from the script.

Example:

```shell script
./docker/build-docker.sh -n podaac-subsetter -v 0.2.0
```

### Building from local code

First build the project with Poetry.

```
poetry build
```

That will create a folder `dist/` and a wheel file that is named with the version of the software that was built. 
Similar to building from Artifactory, the `buld-docker.sh` script can be used to build the docker image from the
local wheel file. In this case there are still two required arguments that must be set:

1. service-name: The name of the service being built (from pyproject.toml)
2. service-version: The version of the service being built (also from pyproject.toml)

In order to use the local wheel file, call the `build-docker.sh` script with the optional argument `--local`. This
will cause the docker image to use the local wheel file instead of downloading the software from Artifactory. 
The docker tag of the built image will be returned from the script.

Example:

```shell script
./docker/build-docker.sh -n podaac-subsetter -v 0.3.0a3 --local
```

## Running

The Docker image can be run directly using the `docker run` command.

## Pushing to ECR

The `push-docker-ecr.sh` script can be used to push a docker image to AWS ECR. There are two required arguments:

1. tf-venue: The target venue for uploading (sit, uat, or ops).
2. docker-tag: The docker tage of the image being pushed

The easiest way to use the `push-docker-ecr.sh` script is to first call `build-docker.sh` and save the output to the
`docker_tag` environment variable. Then call `push-docker-ecr.sh`.

Example:

```shell script
export docker_tag=$(./docker/build-docker.sh -n podaac-subsetter -v 0.2.0)
./docker/push-docker-ecr.sh -v sit -t $docker_tag
```