# -*- coding: utf-8 -*-
"""
Module to handle input/output of .amps files.

"""


def write_amplitudes(run, amplitudes, event):
    """
    Write amplitude results to a new .amps file. This includes amplitude
    measurements, and the magnitude estimates derived from them (with station
    correction terms appied, if provided).

    Parameters
    ----------
    run : :class:`~quakemigrate.io.Run` object
        Light class encapsulating i/o path information for a given run.
    amplitudes : `pandas.DataFrame` object
        P- and S-wave amplitude measurements for each component of each
        station in the station file, and individual local magnitude estimates
        derived from them.
        Columns = ["epi_dist", "z_dist", "P_amp", "P_freq", "P_time",
                   "S_amp", "S_freq", "S_time", "Noise_amp", "is_picked", "ML",
                   "ML_Err"]
        Index = Trace ID (see `obspy.Trace` object property 'id')
    event : :class:`~quakemigrate.io.Event` object
        Light class encapsulating signal, onset, and location information for a
        given event.

    """

    fpath = run.path / "locate" / run.subname / "amplitudes"
    fpath.mkdir(exist_ok=True, parents=True)

    # Work on a copy
    amplitudes = amplitudes.copy()

    # Set floating point precision for output file
    for col in ["epi_dist", "z_dist", "P_amp", "S_amp", "Noise_amp"]:
        amplitudes[col] = amplitudes[col].map(lambda x: f"{x:.5g}",
                                              na_action="ignore")
    for col in ["P_freq", "S_freq"]:
        amplitudes[col] = amplitudes[col].map(lambda x: f"{x:.2g}",
                                              na_action="ignore")
    for col in ["ML", "ML_Err"]:
        amplitudes[col] = amplitudes[col].map(lambda x: f"{x:.3g}",
                                              na_action="ignore")

    fstem = f"{event.uid}"
    file = (fpath / fstem).with_suffix(".amps")
    amplitudes.to_csv(file, index=True)
