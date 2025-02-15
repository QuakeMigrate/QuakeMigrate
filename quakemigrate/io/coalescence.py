"""
Module to handle input/output of coalescence maps.

:copyright:
    2020–2025, QuakeMigrate developers.
:license:
    GNU General Public License, Version 3
    (https://www.gnu.org/licenses/gpl-3.0.html)

"""

from __future__ import annotations

import numpy as np

import quakemigrate
import quakemigrate.util as util


def read_coalescence(fname: str) -> np.ndarray[np.double]:
    """
    Read a coalescence map from .npy file. This function is primarily intended as a
    convenience tool for post-hoc plotting or analysis.

    Parameters
    ----------
    fname:
        File containing coalescence map.

    Returns
    -------
     :
        3- or 4-D coalescence map.

    """

    coa_map = np.load(fname)

    return coa_map


@util.timeit("info")
def write_coalescence(
    run: quakemigrate.io.core.Run,
    coalescence_map: np.ndarray,
    event: quakemigrate.io.event.Event,
    marginalised: bool = False,
) -> None:
    """
    Write coalescence map to file. Can be 3-D (marginalised) or 4-D.

    Parameters
    ----------
    run:
        Light class encapsulating i/o path information for a given run.
    coalescence_map:
        Coalescence map.
    event:
        Light class encapsulating waveforms, coalescence information, picks and
        location information for a given event.
    marginalised:
        Toggle for whether the coalescence map has been marginalised (3-D) or not (4-D).

    """

    if marginalised:
        fpath = run.path / "locate" / run.subname / "marginalised_coalescence_maps"
    else:
        fpath = run.path / "locate" / run.subname / "coalescence_maps"
    fpath.mkdir(exist_ok=True, parents=True)

    file = (fpath / f"{event.uid}").with_suffix(".npy")
    np.save(file, coalescence_map)
