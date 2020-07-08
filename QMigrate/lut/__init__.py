# -*- coding: utf-8 -*-
"""
The :mod:`QMigrate.lut` module handles the definition and generation of the
traveltime lookup tables used in QuakeMigrate.

"""

from .create_lut import compute_traveltimes, read_nlloc  # NOQA
from .lut import LUT  # NOQA


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

    from QMigrate.io import read_lut

    lut = read_lut(old_lut_file)

    lut.traveltimes = {}
    for station, phases in lut.maps.items():
        for phase, ttimes in phases.items():
            phase_code = phase.split("_")[1]
            lut.traveltimes.setdefault(station, {}).update(
                {phase_code: ttimes})

    del lut.maps

    lut.save(save_file)
