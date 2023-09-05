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

"""
=================
subset_harmony.py
=================

Implementation of harmony-service-lib that invokes the Level 2 subsetter.
"""
import argparse
import os
import subprocess
import shutil
from tempfile import mkdtemp
from typing import List, Union

import pystac
from pystac import Asset

import harmony
import numpy as np
from harmony import BaseHarmonyAdapter
from harmony.util import download, stage, generate_output_filename, bbox_to_geometry

from podaac.subsetter import subset
from podaac.subsetter.subset import SERVICE_NAME

DATA_DIRECTORY_ENV = "DATA_DIRECTORY"


def podaac_to_harmony_bbox(bbox: np.ndarray) -> Union[np.ndarray, float]:
    """
    Convert PO.DAAC bbox ((west, east), (south, north))
    to Harmony bbox (west, south, east, north)

    Parameters
    ----------
    bbox : np.array
        Podaac bbox

    Returns
    -------
    array, int or float
        Harmony bbox
    TODO - fix this docstring type, type hint, and code to match (code currently returns a list)
    """

    return_box = [bbox.item(0), bbox.item(2), bbox.item(1), bbox.item(3)]
    return return_box


def harmony_to_podaac_bbox(bbox: list) -> np.ndarray:
    """
    Convert Harmony bbox (west, south, east, north)
    to PO.DAAC bbox ((west, east), (south, north))

    Parameters
    ----------
    bbox : list
        Harmony bbox

    Returns
    -------
    np.array
        PO.DAAC bbox
    """
    return np.array(((bbox[0], bbox[2]),
                     (bbox[1], bbox[3])))


class L2SubsetterService(BaseHarmonyAdapter):
    """
    See https://github.com/nasa/harmony-service-lib-py
    for documentation and examples.
    """

    def __init__(self, message, catalog=None, config=None):
        super().__init__(message, catalog, config)

        self.data_dir = os.getenv(DATA_DIRECTORY_ENV, '/home/dockeruser/data')

    def process_item(self, item: pystac.Item, source: harmony.message.Source) -> pystac.Item:
        """
        Performs variable and bounding box subsetting on the input STAC Item's data, returning
        an output STAC item

        Parameters
        ----------
        item : pystac.Item
            the item that should be subset
        source : harmony.message.Source
            the input source defining the variables, if any, to subset from the item

        Returns
        -------
        pystac.Item
            a STAC item describing the output of the subsetter
        """
        result = item.clone()
        result.assets = {}

        # Create a temporary dir for processing we may do
        temp_dir = mkdtemp()
        output_dir = self.data_dir
        self.prepare_output_dir(output_dir)

        try:
            # Get the data file
            asset = next(v for k, v in item.assets.items() if 'data' in (v.roles or []))
            input_filename = download(asset.href,
                                      temp_dir,
                                      logger=self.logger,
                                      access_token=self.message.accessToken,
                                      cfg=self.config)

            message = self.message

            # Dictionary of keywords params that will be passed into
            # the l2ss-py subset function
            subset_params = {}

            # Transform params to PO.DAAC subsetter arguments and invoke the subsetter
            harmony_bbox = [-180, -90, 180, 90]

            if message.subset and message.subset.bbox:
                harmony_bbox = message.subset.bbox

            if message.subset and message.subset.shape:
                subset_params['shapefile'] = download(
                    message.subset.shape.href,
                    temp_dir,
                    logger=self.logger,
                    access_token=self.message.accessToken,
                    cfg=self.config
                )

            if message.temporal:
                subset_params['min_time'] = message.temporal.start
                subset_params['max_time'] = message.temporal.end

            subset_params['bbox'] = harmony_to_podaac_bbox(harmony_bbox)

            if source.variables:
                subset_params['variables'] = [variable.name for variable in source.process('variables')]

            if source.coordinateVariables:
                coordinate_variables = list(
                    filter(lambda var: var.type and var.subtype, source.coordinateVariables)
                )

                def filter_by_subtype(variables, subtype):
                    return list(map(lambda var: var.name, filter(
                        lambda var: var.subtype == subtype, variables
                    )))
                subset_params['lat_var_names'] = filter_by_subtype(coordinate_variables, 'LATITUDE')
                subset_params['lon_var_names'] = filter_by_subtype(coordinate_variables, 'LONGITUDE')
                subset_params['time_var_names'] = filter_by_subtype(coordinate_variables, 'TIME')

            subset_params['output_file'] = f'{output_dir}/{os.path.basename(input_filename)}'
            subset_params['file_to_subset'] = input_filename
            subset_params['origin_source'] = asset.href

            self.logger.info('Calling l2ss-py subset with params %s', subset_params)
            result_bbox = subset.subset(**subset_params)

            # Stage the output file with a conventional filename
            mime = 'application/x-netcdf4'
            operations = {
                "variable_subset": subset_params.get('variables'),
                "is_subsetted": bool(result_bbox is not None)
            }
            staged_filename = generate_output_filename(asset.href, '.nc4', **operations)

            url = stage(subset_params['output_file'],
                        staged_filename,
                        mime,
                        location=message.stagingLocation,
                        logger=self.logger,
                        cfg=self.config)

            # Update the STAC record
            asset = Asset(url, title=staged_filename, media_type=mime, roles=['data'])
            result.assets['data'] = asset
            if result_bbox is not None:
                if message.subset:
                    message.subset.process('bbox')
                result.bbox = podaac_to_harmony_bbox(result_bbox)
                result.geometry = bbox_to_geometry(result.bbox)

            # Return the STAC record
            return result
        finally:
            # Clean up any intermediate resources
            shutil.rmtree(temp_dir)

    def prepare_output_dir(self, output_dir: str) -> None:
        """
        Deletes (if present) and recreates the given output_dir, ensuring it exists
        and is empty

        Parameters
        ----------
        output_dir : string
            the directory to delete and recreate
        """
        self.cmd('rm', '-rf', output_dir)
        self.cmd('mkdir', '-p', output_dir)

    def cmd(self, *args) -> List[str]:
        """
        Logs and then runs command.

        Parameters
        ----------
        args Command and args to run

        Returns
        -------
        Command output
        """
        self.logger.info("%s %s", args[0], " ".join(["'{}'".format(arg) for arg in args[1:]]))  # pylint: disable=C0209
        result_str = subprocess.check_output(args).decode("utf-8")
        return result_str.split("\n")


def main(config: harmony.util.Config = None) -> None:
    """Parse command line arguments and invoke the service to respond to
    them.

    Parameters
    ----------
    config : harmony.util.Config

    Returns
    -------
    None
    """
    parser = argparse.ArgumentParser(prog=SERVICE_NAME,
                                     description='Run the l2_subsetter service')
    harmony.setup_cli(parser)
    args = parser.parse_args()
    if harmony.is_harmony_cli(args):
        harmony.run_cli(parser, args, L2SubsetterService, cfg=config)
    else:
        parser.error("Only --harmony CLIs are supported")


if __name__ == "__main__":
    main()
