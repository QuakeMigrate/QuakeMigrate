# -*- coding: utf-8 -*-
"""
The :mod:`quakemigrate.core` module provides Python bindings for the libraries
of compiled C routines that form the core of QuakeMigrate and do the heavy
lifting:

    * migratelib - contains routines for computing the 4-D coalescence \
    function by continuously migrating onset functions through time and space \
    and determining the continuous maximum coalescence amplitude in the 4-D \
    coalescence volume.
    * onsetlib - contains routines for computing various onset functions, \
    for now limited to the centred/overlapping STA/LTA, as well as a \
    recursive implementation of the centred STA/LTA.

:copyright:
    2020, QuakeMigrate developers.
:license:
    GNU General Public License, Version 3
    (https://www.gnu.org/licenses/gpl-3.0.html)

"""

from .lib import (migrate, find_max_coa, overlapping_sta_lta,  # NOQA
                  centred_sta_lta, recursive_sta_lta)
