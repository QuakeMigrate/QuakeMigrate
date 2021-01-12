import unittest


class TestImport(unittest.TestCase):
    def test_import(self):
        i = 0
        import sys
        if sys.version_info.major != 3:
            print("QuakeMigrate does not support Python 2.x")
            i += 1
        if sys.version_info.minor < 6:
            print("QuakeMigrate only supports Python 3.5 and up.")
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
             print("QuakeMigrate does not import correctly.")
             i += 1
        self.assertEqual(i, 0)


if __name__ == "__main__":
    unittest.main()
