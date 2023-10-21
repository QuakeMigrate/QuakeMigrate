# -*- coding: utf-8 -*-
"""
The :mod:`quakemigrate.lut` module handles the definition and generation of the
traveltime lookup tables used in QuakeMigrate.

:copyright:
    2020–2023, QuakeMigrate developers.
:license:
    GNU General Public License, Version 3
    (https://www.gnu.org/licenses/gpl-3.0.html)

"""

import pyproj

from .create_lut import compute_traveltimes, read_nlloc  # NOQA
from .lut import LUT  # NOQA


# Handle bugged version of PROJ
proj_major, proj_minor, proj_patch = pyproj.proj_version_str.split(".")
if proj_major == "6" and proj_minor == "2":
    raise ImportError(
        f"PROJ version {proj_major}.{proj_minor}.{proj_patch} is being used as the "
        "backend for pyproj. This version has a\nbug with the conversion of Z units "
        "when using units that are not metres. Please consult the \nQuakeMigrate "
        "installation instructions for how to update the PROJ backend."
    )


def update_lut(old_lut_file, save_file):
    """
    Utility function to convert old-style LUTs to new-style LUTs.

    Parameters
    ----------
    old_lut_file : str
        Path to lookup table file to update.
    save_file : str, optional
        Output path for updated lookup table.

    """

    from quakemigrate.io import read_lut

    lut = read_lut(old_lut_file)

    try:
        traveltimes = {}
        for station, phases in lut.maps.items():
            for phase, ttimes in phases.items():
                phase_code = phase.split("_")[1]
                traveltimes.setdefault(station, {}).update({phase_code: ttimes})
        lut.traveltimes = traveltimes
        del lut.maps
    except AttributeError:
        pass
    lut.phases = ["P", "S"]
    lut.fraction_tt = 0.1
    try:
        lut.node_spacing = lut.cell_size
        lut.node_count = lut.cell_count
        del lut._cell_size, lut._cell_count
    except AttributeError:
        pass

    lut.save(save_file)
