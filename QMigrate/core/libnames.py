# -*- coding: utf-8 -*-
"""
Helpers to get library names.

Edited for our use from obspy.core.util.libnames
"""

import ctypes
from distutils import sysconfig
import pathlib


def _get_lib_path(libdir, libname):
    """
    Grab architecture/Python version specific library filename.

    Parameters
    ----------
    libdir : `pathlib.Path` object
        Path to library to load.
    libname : str
        Name of library to be loaded.

    Returns
    -------
    libpath : `pathlib.Path` object
        Path to library with architecture/Python version specific extension.

    """

    # Get extension suffix defined by Python for current platform
    ext_suffix = sysconfig.get_config_var("EXT_SUFFIX")
    # In principle "EXT_SUFFIX" is what we want.
    # "SO" seems to be deprecated on newer python
    # but: older Python seems to have empty "EXT_SUFFIX", so we fall back
    if not ext_suffix:
        try:
            ext_suffix = sysconfig.get_config_var("SO")
        except Exception as e:
            print(e.msg)
            pass
    if ext_suffix:
        libpath = (libdir / libname).with_suffix(ext_suffix)

    return libpath


def _load_cdll(name):
    """
    Helper function to load a shared library built during installation
    with ctypes.

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
    libdir = pathlib.Path(__file__).parent / "src"
    libpath = _get_lib_path(libdir, name)
    try:
        cdll = ctypes.CDLL(libpath)
    except Exception as e:
        msg = f"Could not load shared library '{libpath.name}'.\n\n {e}"
        raise ImportError(msg)

    return cdll
