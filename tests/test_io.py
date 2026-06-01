# -*- coding: utf-8 -*-
"""
Unit tests covering functions in quakemigrate.io.core.

:copyright:
    2020 - 2021, QuakeMigrate developers.
:license:
    GNU General Public License, Version 3
    (https://www.gnu.org/licenses/gpl-3.0.html)

"""

import os
import tempfile
import unittest

import quakemigrate.util as util
from quakemigrate.io.core import read_stations


class TestReadStations(unittest.TestCase):
    @staticmethod
    def _write(content):
        handle, path = tempfile.mkstemp(suffix=".csv")
        os.close(handle)
        with open(path, "w") as fid:
            fid.write(content)
        return path

    def test_complete_header(self):
        """A station file with every required column is read successfully."""
        path = self._write(
            "Latitude,Longitude,Elevation,Name\n1.0,2.0,3.0,STA1\n"
        )
        try:
            stn_data = read_stations(path)
        finally:
            os.remove(path)
        for column in ("Latitude", "Longitude", "Elevation", "Name"):
            self.assertIn(column, stn_data.columns)
        # Elevation is negated (positive-up -> depth) and Name kept as str
        self.assertEqual(stn_data["Elevation"].iloc[0], -3.0)
        self.assertEqual(stn_data["Name"].iloc[0], "STA1")

    def test_missing_columns_raise(self):
        """A station file missing any required column must raise.

        Regression test: the header check was
        ``("Latitude" or "Longitude" or "Elevation" or "Name") not in
        stn_data.columns``, which short-circuits to ``"Latitude"`` and so only
        ever checked for that one column. Files missing Longitude, Elevation or
        Name slipped through and failed later with a confusing ``KeyError``
        instead of the intended ``StationFileHeaderException``.
        """
        cases = [
            "Longitude,Elevation,Name\n2.0,3.0,STA1\n",       # no Latitude
            "Latitude,Elevation,Name\n1.0,3.0,STA1\n",        # no Longitude
            "Latitude,Longitude,Name\n1.0,2.0,STA1\n",        # no Elevation
            "Latitude,Longitude,Elevation\n1.0,2.0,3.0\n",    # no Name
        ]
        for content in cases:
            path = self._write(content)
            try:
                with self.assertRaises(util.StationFileHeaderException):
                    read_stations(path)
            finally:
                os.remove(path)


if __name__ == "__main__":
    unittest.main()
