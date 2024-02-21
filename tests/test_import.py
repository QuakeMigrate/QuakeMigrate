# -*- coding: utf-8 -*-
"""
Short test script that will ensure QuakeMigrate and all required dependencies
have been correctly installed.

:copyright:
    2020 - 2021, QuakeMigrate developers.
:license:
    GNU General Public License, Version 3
    (https://www.gnu.org/licenses/gpl-3.0.html)

"""

import unittest


class TestImport(unittest.TestCase):
    def test_import(self):
        i = 0
        import sys

        if sys.version_info.major != 3:
            print("QuakeMigrate does not support Python 2.x")
            i += 1
        if sys.version_info.minor < 8:
            print("QuakeMigrate only supports Python 3.8 and up.")
            i += 1
        try:
            import matplotlib  # NOQA
        except ImportError:
            print("You have not properly installed: matplotlib")
            i += 1
        try:
            import numpy  # NOQA
        except ImportError:
            print("You have not properly installed: numpy")
            i += 1
        try:
            import obspy  # NOQA
        except ImportError:
            print("You have not properly installed: obspy")
            i += 1
        try:
            import pyproj  # NOQA
        except ImportError:
            print("You have not properly installed: pyproj")
            i += 1
        try:
            import scipy  # NOQA
        except ImportError:
            print("You have not properly installed: scipy")
            i += 1
        try:
            import quakemigrate  # NOQA
        except ImportError as e:
            print(f"QuakeMigrate does not import correctly. - {e}")
            i += 1
        self.assertEqual(i, 0)

        if matplotlib.get_backend() == "agg":
            print("Only AGG backend available - interactive plots won't work!")
            print("Consider installing Tk or Qt bindings.")


if __name__ == "__main__":
    unittest.main()
