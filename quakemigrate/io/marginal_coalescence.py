# -*- coding: utf-8 -*-
"""
Module to handle input/output of marginalised coalescence map.

:copyright:
    2020 - 2024, QuakeMigrate developers.
:license:
    GNU General Public License, Version 3
    (https://www.gnu.org/licenses/gpl-3.0.html)

"""

import numpy as np

import quakemigrate.util as util


def read_marginal_coalescence(fname):
    """
    Read marginalised coalescence map from .npy file. This function is primarily
    intended as a convenience tool for post-hoc plotting or analysis.

    Parameters
    ----------
    fname : str
        File containing marginalised coalescence map.

    Returns
    -------
    marginalised_coa_map : array-like
        Marginalised 3-D coalescence map.

    """

    marginalised_coa_map = np.load(fname)

    return marginalised_coa_map


@util.timeit("info")
def write_marginal_coalescence(run, marginalised_coa_map, event):
    """
    Write marginalised 3-D coalescence map to file.

    Parameters
    ----------
    run : :class:`~quakemigrate.io.core.Run` object
        Light class encapsulating i/o path information for a given run.
    marginalised_coa_map : array-like
        Marginalised 3-D coalescence map.
    event : :class:`~quakemigrate.io.event.Event` object
        Light class encapsulating waveforms, coalescence information, picks and
        location information for a given event.

    """

    fpath = run.path / "locate" / run.subname / "marginalised_coalescence_maps"
    fpath.mkdir(exist_ok=True, parents=True)

    file = (fpath / f"{event.uid}").with_suffix(".npy")
    np.save(file, marginalised_coa_map)
