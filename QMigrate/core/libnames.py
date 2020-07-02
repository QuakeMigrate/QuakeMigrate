# -*- coding: utf-8 -*-
"""
Helper to load compiled C library.

"""

import ctypes
from distutils import sysconfig
import pathlib


def _load_cdll(name):
    """
    Helper function to load an extension built using setuptools/distutils
    Extension.

    Parameters
    ----------
    name : str
        Name of library to load.

    Returns
    -------
    `ctypes.CDLL`
        Shared library object.

    """

    # our custom defined part of the extension file name
    libpath = pathlib.Path(__file__).parent / "src" / name
    lib = libpath.with_suffix(sysconfig.get_config_var("EXT_SUFFIX"))
    try:
        cdll = ctypes.CDLL(lib)
    except Exception as e:
        msg = (f"Could not load extension library '{libpath.name}'.\n\n{e}\n"
               "If you have chosen to install from a clone of the github "
               "repository, please ensure you have run setup.py, which will "
               "compile and install the C library.")
        raise ImportError(msg)

    return cdll
