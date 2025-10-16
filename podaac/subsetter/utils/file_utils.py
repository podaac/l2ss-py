"""
===============
file_utils.py
===============

Utility functions for file and dataset handling.
"""

from typing import Optional
import dateutil
from dateutil import parser
import cftime
import xarray as xr
import xarray.coding.times
from xarray import DataTree


def calculate_chunks(dataset: xr.Dataset) -> dict:
    """
    For the given dataset, calculate if the size on any dimension is
    worth chunking. Any dimension larger than 4000 will be chunked. This
    is done to ensure that the variable can fit in memory.
    """
    if len(dataset.dims) <= 3:
        chunk = {dim: 4000 for dim in dataset.dims
                 if dataset.sizes[dim] > 4000
                 and len(dataset.dims) > 1}
    else:
        chunk = {dim: 500 for dim in dataset.dims
                 if dataset.sizes[dim] > 500}
    return chunk


def override_decode_cf_datetime() -> None:
    """
    WARNING !!! REMOVE AT EARLIEST XARRAY FIX, this is a override to xarray override_decode_cf_datetime function.
    xarray has problems decoding time units with format `seconds since 2000-1-1 0:0:0 0`, this solves by testing
    the unit to see if its parsable, if it is use original function, if not format unit into a parsable format.
    """
    orig_decode_cf_datetime = xarray.coding.times.decode_cf_datetime

    def decode_cf_datetime(num_dates, units, calendar=None, use_cftime=None):
        try:
            parser.parse(units.split('since')[-1])
            return orig_decode_cf_datetime(num_dates, units, calendar, use_cftime)
        except dateutil.parser.ParserError:
            reference_time = cftime.num2date(0, units, calendar)
            units = f"{units.split('since')[0]} since {reference_time}"
            return orig_decode_cf_datetime(num_dates, units, calendar, use_cftime)

    xarray.coding.times.decode_cf_datetime = decode_cf_datetime


def test_access_sst_dtime_values(nc_dataset):
    """
    Test accessing values of 'sst_dtime' variable in a NetCDF file.
    """
    args = {
        'decode_coords': False,
        'mask_and_scale': True,
        'decode_times': True
    }
    try:
        with xr.open_dataset(
                xr.backends.NetCDF4DataStore(nc_dataset),
                **args
        ) as dataset:
            for var_name in dataset.variables:
                dataset[var_name].values  # pylint: disable=pointless-statement
    except (TypeError, ValueError, KeyError):
        return False
    return True


def get_hdf_type(tree: DataTree) -> Optional[str]:
    """
    Determine the HDF type (OMI or MLS) from a DataTree object.
    """
    try:
        additional_attrs = tree['/HDFEOS/ADDITIONAL/FILE_ATTRIBUTES']
        if additional_attrs is not None and 'InstrumentName' in additional_attrs.attrs:
            instrument = additional_attrs.attrs['InstrumentName']
            if isinstance(instrument, bytes):
                instrument = instrument.decode("utf-8")
        else:
            return None
        if 'OMI' in instrument:
            return 'OMI'
        if 'MLS' in instrument:
            return 'MLS'
    except (KeyError, AttributeError):
        pass
    return None


def get_path(s):
    """Extracts the path by removing the last part after the final '/'."""
    path = s.rsplit('/', 1)[0] if '/' in s else s
    return path if path.startswith('/') else f'/{path}'
