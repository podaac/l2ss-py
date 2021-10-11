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

import os

import pytest


@pytest.fixture(scope='function')
def mock_environ(tmp_path):
    old_vals = {key: os.environ.get(key, None) for key in [
        'AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY',
        'AWS_SECURITY_TOKEN', 'AWS_SESSION_TOKEN',
        'AWS_REGION', 'AWS_DEFAULT_REGION',
        'SHARED_SECRET_KEY', 'ENV',
        'DATA_DIRECTORY'
    ]}

    os.environ['AWS_ACCESS_KEY_ID'] = 'foo'
    os.environ['AWS_SECRET_ACCESS_KEY'] = 'foo'
    os.environ['AWS_SECURITY_TOKEN'] = 'foo'
    os.environ['AWS_SESSION_TOKEN'] = 'foo'
    os.environ['AWS_REGION'] = 'us-west-2'
    os.environ['AWS_DEFAULT_REGION'] = 'us-west-2'
    os.environ['SHARED_SECRET_KEY'] = "shhhhhhhhhhhhhhhhhhhhhhhhhhhhhhh"
    os.environ['ENV'] = "test"
    os.environ['DATA_DIRECTORY'] = str(tmp_path)

    yield

    for key, value in old_vals.items():
        if value:
            os.environ[key] = value
