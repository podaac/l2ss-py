"""
==============
test_run_subsetter.py
==============

Test the run_subsetter cli
"""
from podaac.subsetter import run_subsetter
from unittest.mock import patch, MagicMock
import numpy as np


def test_run_subsetter():
    with patch('podaac.subsetter.subset.subset') as mock_subsetter:
        args = [
            'input.nc',
            'output.nc'
        ]
        run_subsetter.run_subsetter(args)

        call_args = mock_subsetter.call_args_list[0][1]

        assert call_args['file_to_subset'] == args[0]
        assert call_args['output_file'] == args[1]
        np.testing.assert_equal(call_args['bbox'], np.array([[-180, 180], [-90, 90]]))
        assert not call_args['variables']
        assert not call_args['min_time']
        assert not call_args['max_time']
        assert not call_args['cut']

    with patch('podaac.subsetter.subset.subset') as mock_subsetter:
        args = [
            'input.nc',
            'output.nc',
            '--bbox', '-100', '-10', '100', '10',
            '--variables', 'sst', 'wind',
            '--min-time', '2015-07-02T09:00:00',
            '--max-time', '2015-07-02T10:00:00',
            '--cut'
        ]
        run_subsetter.run_subsetter(args)

        call_args = mock_subsetter.call_args_list[0][1]

        np.testing.assert_equal(call_args['bbox'], np.array([[-100, 100], [-10, 10]]))
        assert call_args['variables'] == ['sst', 'wind']
        assert call_args['min_time'] == '2015-07-02T09:00:00'
        assert call_args['max_time'] == '2015-07-02T10:00:00'
        assert call_args['cut']
