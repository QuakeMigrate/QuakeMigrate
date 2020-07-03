# -*- coding: utf-8 -*-
"""
Module to handle input/output of cut waveforms.

"""

import logging
import warnings

import QMigrate.util as util


warnings.filterwarnings("ignore", message=("File will be written with more tha"
                                           "n one different record lengths.\nT"
                                           "his might have a negative influenc"
                                           "e on the compatibility with other "
                                           "programs."))
warnings.filterwarnings("ignore", message=("File will be written with more tha"
                                           "n one different encodings.\nThis m"
                                           "ight have a negative influence on "
                                           "the compatibility with other progr"
                                           "ams."))


@util.timeit
def write_cut_waveforms(run, event, file_format, pre_cut=0., post_cut=0.):
    """
    Output raw cut waveform data as a waveform file -- defaults to mSEED.

    Parameters
    ----------
    run : `QMigrate.io.Run` object
        Light class encapsulating i/o path information for a given run.
    event : `QMigrate.io.Event` object
        Light class encapsulating signal, onset, and location information for a
        given event.
    file_format : str, optional
        File format to write waveform data to. Options are all file formats
        supported by obspy, including: "MSEED" (default), "SAC", "SEGY",
        "GSE2"
    pre_cut : float, optional
        Specify how long before the event origin time to cut the waveform
        data from
    post_cut : float, optional
        Specify how long after the event origin time to cut the waveform
        data to

    """

    logging.info("\tSaving raw cut waveforms...")

    fpath = run.path / "locate" / run.subname / "cut_waveforms"
    fpath.mkdir(exist_ok=True, parents=True)

    st = event.data.raw_waveforms

    if pre_cut != 0.:
        for tr in st.traces:
            tr.trim(starttime=event.otime - pre_cut)
    if post_cut != 0.:
        for tr in st.traces:
            tr.trim(endtime=event.otime + post_cut)

    if file_format == "MSEED":
        suffix = ".m"
    elif file_format == "SAC":
        suffix = ".sac"
    elif file_format == "SEGY":
        suffix = ".segy"
    elif file_format == "GSE2":
        suffix = ".gse2"
    else:
        suffix = ".waveforms"

    fstem = f"{event.uid}"
    file = (fpath / fstem).with_suffix(suffix)
    st.write(str(file), format=file_format)
