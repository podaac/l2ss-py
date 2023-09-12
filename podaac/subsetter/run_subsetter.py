"""
run_subsetter.py

This script runs L2SS-Py on the given granule.
"""
import argparse
import logging
import os
import sys

import numpy as np

from podaac.subsetter import subset


def parse_args(args: list) -> tuple:
    """
    Parse args for this script.

    Returns
    -------
    tuple
        input_file, output_file, bbox, variables, min_time, max_time
    """
    parser = argparse.ArgumentParser(description='Run l2ss-py')
    parser.add_argument(
        'input_file',
        type=str,
        help='File to subset'
    )
    parser.add_argument(
        'output_file',
        type=str,
        help='Output file'
    )
    parser.add_argument(
        '--bbox',
        type=float,
        default=[-180, -90, 180, 90],
        nargs=4,
        action='store',
        help='Bounding box in the form min_lon min_lat max_lon max_lat'
    )
    parser.add_argument(
        '--variables',
        type=str,
        default=None,
        nargs='+',
        help='Variables, only include if variable subset is desired. '
             'Should be a space separated list of variable names e.g. '
             'sst wind_dir sst_error ...'
    )
    parser.add_argument(
        '--min-time',
        type=str,
        default=None,
        help='Min time. Should be ISO-8601 format. Only include if '
             'temporal subset is desired.'
    )
    parser.add_argument(
        '--max-time',
        type=str,
        default=None,
        help='Max time. Should be ISO-8601 format. Only include if '
             'temporal subset is desired.'
    )
    parser.add_argument(
        '--cut',
        default=False,
        action='store_true',
        help='If provided, scanline will be cut'
    )
    parser.add_argument(
        '--shapefile',
        type=str,
        default=None,
        help='Path to either shapefile or geojson file used to subset '
             'the provided input granule'
    )

    args = parser.parse_args(args=args)
    bbox = np.array([[args.bbox[0], args.bbox[2]], [args.bbox[1], args.bbox[3]]])

    return args.input_file, args.output_file, bbox, args.variables, \
        args.min_time, args.max_time, args.cut, args.shapefile


def run_subsetter(args: list) -> None:
    """
    Parse arguments and run subsetter on the specified input file
    """
    input_file, output_file, bbox, variables, min_time, max_time, cut, shapefile = parse_args(args)

    logging.info('Executing subset on %s...', input_file)
    subset.subset(
        file_to_subset=input_file,
        bbox=bbox,
        output_file=output_file,
        variables=variables,
        cut=cut,
        min_time=min_time,
        max_time=max_time,
        origin_source=os.path.basename(input_file),
        shapefile=shapefile
    )
    logging.info('Subset complete. Result in %s', output_file)


def main() -> None:
    """Entry point to the script"""
    logging.basicConfig(
        stream=sys.stdout,
        format='[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s',
        level=logging.DEBUG
    )
    run_subsetter(sys.argv[1:])


if __name__ == '__main__':
    main()
