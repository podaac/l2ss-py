"""Tests for earthdata-varinfo helpers in variables_utils."""
import logging
from unittest import mock

from podaac.subsetter.utils import variables_utils


def test_varinfo_failure_returns_empty_set(caplog):
    """A varinfo failure is logged and results in an empty set."""
    with mock.patch("varinfo.VarInfoFromNetCDF4", side_effect=RuntimeError("boom")):
        with caplog.at_level(logging.WARNING):
            result = variables_utils.get_varinfo_required_variables(
                "missing.nc", ["/var"], {"/var"}
            )

    assert result == set()
    assert "resolution failed" in caplog.text


def test_varinfo_missing_file_returns_empty_set():
    """A nonexistent granule fails soft."""
    result = variables_utils.get_varinfo_required_variables(
        "does_not_exist.nc", {"/var"}, {"/var"}
    )
    assert result == set()


def test_varinfo_results_filtered_to_existing_variables():
    """Only variables present in the tree are returned."""
    fake = mock.Mock()
    fake.get_required_variables.return_value = {"/var", "/var_bounds", "/not_in_tree"}
    with mock.patch("varinfo.VarInfoFromNetCDF4", return_value=fake):
        result = variables_utils.get_varinfo_required_variables(
            "f.nc", ["/var"], {"/var", "/var_bounds"}
        )
    assert result == {"/var", "/var_bounds"}
