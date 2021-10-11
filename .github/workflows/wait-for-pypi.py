#!/usr/bin/env python
import logging
import subprocess
import sys
import tempfile

import tenacity

'''
Sometimes the package published to PyPi is not immediately available for download from the index. This script
simply repeatedly tries to download a specific version of a package from PyPI (or test.pypi) until it succeeds or
a limit is exceeded.
'''


@tenacity.retry(
    wait=tenacity.wait_exponential(multiplier=1, min=4, max=10),
    retry=tenacity.retry_if_exception_type(subprocess.CalledProcessError),
    stop=tenacity.stop_after_delay(120),
    before_sleep=tenacity.before_sleep_log(logging.getLogger(__name__), logging.DEBUG)
)
def download_package(package):
    subprocess.check_call([sys.executable, '-m',
                           'pip', '--isolated', '--no-cache-dir',
                           'download', '--no-deps', '-d', tempfile.gettempdir(), '--index-url',
                           'https://pypi.org/simple/',
                           '--extra-index-url', 'https://test.pypi.org/simple/', package
                           ])


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
    package_spec = sys.argv[1]
    download_package(package_spec)
