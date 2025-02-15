"""
Module to handle input/output of .amps files.

:copyright:
    2020–2025, QuakeMigrate developers.
:license:
    GNU General Public License, Version 3
    (https://www.gnu.org/licenses/gpl-3.0.html)

"""

from __future__ import annotations

import pandas as pd

import quakemigrate


def write_amplitudes(
    run: quakemigrate.io.core.Run,
    amplitudes: pd.DataFrame,
    event: quakemigrate.io.event.Event,
) -> None:
    """
    Write amplitude results to a new .amps file. This includes amplitude measurements,
    and the magnitude estimates derived from them (with station correction terms
    applied, if provided).

    Parameters
    ----------
    run:
        Light class encapsulating i/o path information for a given run.
    amplitudes:
        P- and S-wave amplitude measurements for each component of each station in the
        station file, and individual local magnitude estimates derived from them.
        Columns = ["epi_dist", "z_dist", "P_amp", "P_freq", "P_time",
                   "P_avg_amp", "P_filter_gain", "S_amp", "S_freq", "S_time",
                   "S_avg_amp", "S_filter_gain", "Noise_amp", "is_picked",
                   "ML", "ML_Err"]
        Index = Trace ID (see `obspy.Trace` object property 'id')
    event:
        Light class encapsulating waveforms, coalescence information, picks and
        location information for a given event.

    """

    fpath = run.path / "locate" / run.subname / "amplitudes"
    fpath.mkdir(exist_ok=True, parents=True)

    # Work on a copy
    amplitudes = amplitudes.copy()

    # Set floating point precision for output file
    for col in [
        "epi_dist",
        "z_dist",
        "P_amp",
        "P_avg_amp",
        "S_amp",
        "S_avg_amp",
        "Noise_amp",
    ]:
        amplitudes[col] = amplitudes[col].map(lambda x: f"{x:.5g}", na_action="ignore")
    for col in ["P_freq", "S_freq"]:
        amplitudes[col] = amplitudes[col].map(lambda x: f"{x:.2g}", na_action="ignore")
    for col in ["P_filter_gain", "S_filter_gain"]:
        amplitudes[col] = amplitudes[col].map(lambda x: f"{x:.3g}", na_action="ignore")
    # Handle case where no amplitude measurement made
    if "ML" in amplitudes.columns:
        for col in ["ML", "ML_Err"]:
            amplitudes[col] = amplitudes[col].map(
                lambda x: f"{x:.3g}", na_action="ignore"
            )

    fstem = f"{event.uid}"
    file = (fpath / fstem).with_suffix(".amps")

    amplitudes.to_csv(file, index=True)
