# -*- coding: utf-8 -*-
"""
Module to handle input/output of .amps files.

"""


def write_amplitudes(run, amplitudes, event):
    """
    Write amplitude values to a new .amps file.

    Parameters
    ----------
    run : `QMigrate.io.Run` object
        Light class encapsulating i/o path information for a given run.
    amplitudes :

    event : `QMigrate.io.Event` object
        Light class encapsulating signal, onset, and location information for a
        given event.

    """

    fpath = run.path / "locate" / run.subname / "amplitudes"
    fpath.mkdir(exist_ok=True, parents=True)

    fstem = f"{event.uid}"
    file = (fpath / fstem).with_suffix(".amps")
    amplitudes.to_csv(file, index=True)
