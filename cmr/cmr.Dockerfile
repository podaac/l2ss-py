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

FROM python:3.13.0b3-slim

## Create a new user
RUN adduser --quiet --disabled-password --shell /bin/sh --home /home/dockeruser --gecos "" --uid 300 dockeruser

# Get jq
RUN apt update -y
RUN apt install jq -y

USER dockeruser
ENV HOME /home/dockeruser
ENV PYTHONPATH "${PYTHONPATH}:/home/dockeruser/.local/bin"
ENV PATH="/home/dockeruser/.local/bin:${PATH}"

# Add artifactory as a trusted pip index
RUN mkdir $HOME/.pip
RUN echo "[global]" >> $HOME/.pip/pip.conf
RUN echo "index-url = https://pypi.org/simple"" >> $HOME/.pip/pip.conf
RUN echo "trusted-host = pypi.org" >> $HOME/.pip/pip.conf

WORKDIR "/home/dockeruser"

RUN pip install --upgrade pip \
    && pip install awscli --upgrade \
    && pip install podaac-dev-tools>=0.6.0

RUN pip list
CMD ["sh"]