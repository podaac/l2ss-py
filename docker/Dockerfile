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

FROM python:3.7-slim

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get upgrade -y \
    gcc \
    libnetcdf-dev \
    libhdf5-dev \
    hdf5-helpers \
    && pip3 install --upgrade pip \
    && pip3 install cython \
    && apt-get clean

# Create a new user
RUN adduser --quiet --disabled-password --shell /bin/sh --home /home/dockeruser --gecos "" --uid 1000 dockeruser
USER dockeruser
ENV HOME /home/dockeruser
ENV PYTHONPATH "${PYTHONPATH}:/home/dockeruser/.local/bin"
ENV PATH="/home/dockeruser/.local/bin:${PATH}"

# Add artifactory as a trusted pip index
RUN mkdir $HOME/.pip
RUN echo "[global]" >> $HOME/.pip/pip.conf && \
    echo "index-url = https://pypi.org/simple"" >> $HOME/.pip/pip.conf" && \
    echo "trusted-host = pypi.org" >> $HOME/.pip/pip.conf

# The 'SOURCE' argument is what will be used in 'pip install'.
ARG SOURCE

# Set this argument if running the pip install on a local directory, so
# the local dist files are copied into the container.
ARG DIST_PATH

USER root
RUN mkdir -p /worker && chown dockeruser /worker
USER dockeruser
WORKDIR /worker

COPY --chown=dockeruser $DIST_PATH $DIST_PATH
USER dockeruser
RUN pip3 install --use-feature=2020-resolver --force $SOURCE --user \
    && rm -rf $DIST_PATH

# Run the subsetter
ENTRYPOINT ["l2ss_harmony"]