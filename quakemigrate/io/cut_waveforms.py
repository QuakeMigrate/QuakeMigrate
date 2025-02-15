"""
Module to handle input/output of cut waveforms.

:copyright:
    2020–2025, QuakeMigrate developers.
:license:
    GNU General Public License, Version 3
    (https://www.gnu.org/licenses/gpl-3.0.html)

"""

from __future__ import annotations

import logging
import pathlib
import warnings

from obspy import Stream

import quakemigrate
import quakemigrate.util as util


warnings.filterwarnings(
    "ignore",
    message=(
        "File will be written with more than one different record lengths.\nThis might "
        "have a negative influence on the compatibility with other programs."
    ),
)
warnings.filterwarnings(
    "ignore",
    message=(
        "File will be written with more than one different encodings.\nThis might have "
        "a negative influence on the compatibility with other programs."
    ),
)
warnings.filterwarnings(
    "ignore",
    message=(
        "The encoding specified in trace.stats.mseed.encoding does not match the dtype "
        "of the data.\nA suitable encoding will be chosen."
    ),
)


@util.timeit("info")
def write_cut_waveforms(
    run: quakemigrate.io.core.Run,
    event: quakemigrate.io.event.Event,
    file_format: str = "MSEED",
    pre_cut: float = 0.0,
    post_cut: float = 0.0,
    waveform_type: str = "raw",
    units: str = "displacement",
) -> None:
    """
    Output cut waveform data as a waveform file -- defaults to miniSEED format.

    Parameters
    ----------
    run:
        Light class encapsulating i/o path information for a given run.
    event:
        Light class encapsulating waveforms, coalescence information, picks and location
        information for a given event.
    file_format:
        File format to write waveform data to. Options are all file formats supported by
        ObsPy, including: "MSEED" (default), "SAC", "SEGY", "GSE2"
    pre_cut:
        Specify how long before the event origin time to cut the waveform data from.
    post_cut:
        Specify how long after the event origin time to cut the waveform data to.
    waveform_type:
        Whether to output raw, real or Wood-Anderson simulated waveforms.
        Default: "raw"
        Options: {"raw", "real", "wa"}
    units:
        Whether to output displacement waveforms or velocity waveforms for real/
        Wood-Anderson corrected traces.
        Default: "displacement"
        Options: {"displacement", "velocity"}

    Raises
    ------
    AttributeError
        If real or wa waveforms are requested and no response inventory has been
        provided.

    """

    logging.info(f"\tSaving {waveform_type} cut waveforms...")

    fpath = run.path / "locate" / run.subname / f"{waveform_type}_cut_waveforms"
    fpath.mkdir(exist_ok=True, parents=True)
    fstem = f"{event.uid}"

    st = event.data.raw_waveforms

    # "if pre_cut" catches both 0. and None
    if pre_cut:
        for tr in st.traces:
            tr.trim(starttime=event.otime - pre_cut)
    if post_cut:
        for tr in st.traces:
            tr.trim(endtime=event.otime + post_cut)

    # Remove empty traces
    for tr in st:
        if not bool(tr):
            st.remove(tr)

    if waveform_type == "real" or waveform_type == "wa":
        if (
            waveform_type == "real"
            and isinstance(event.data.real_waveforms, Stream)
            and not pre_cut
            and not post_cut
        ):
            st = event.data.real_waveforms
        elif (
            waveform_type == "wa"
            and isinstance(event.data.wa_waveforms, Stream)
            and not pre_cut
            and not post_cut
        ):
            st = event.data.wa_waveforms
        else:
            try:
                st = get_waveforms(st, event, waveform_type, units)
            except AttributeError as e:
                raise AttributeError(
                    "To output real or Wood-Anderson cut waveforms you must supply an "
                    "instrument response inventory."
                ) from e

    if bool(st):
        write_waveforms(st, fpath, fstem, file_format)
    else:
        logging.info(f"\t\tNo {waveform_type} cut waveform data for event{event.uid}!")


@util.timeit("debug")
def get_waveforms(
    st: Stream, event: quakemigrate.io.event.Event, waveform_type: str, units: str
) -> Stream:
    """
    Get real or simulated waveforms for a Stream.

    Parameters
    ----------
    st:
        Stream for which to get real or simulated waveforms.
    event:
        Light class encapsulating waveforms, coalescence information, picks and location
        information for a given event.
    waveform_type:
        Whether to get real or Wood-Anderson simulated waveforms.
        Options: {"real", "wa"}
    units:
        Units to return waveforms in.
        Options: {"displacement", "velocity"}

    Returns
    -------
     :
        Stream of real or Wood-Anderson simulated waveforms in the requested units.

    """

    # Work on a copy
    st = st.copy()
    st_out = Stream()

    velocity = True if units == "velocity" else False

    for tr in st:
        # Check there is data present
        if bool(tr) and tr.data.max() != tr.data.min():
            try:
                if waveform_type == "real":
                    tr = event.data.get_real_waveform(tr, velocity)
                else:
                    tr = event.data.get_wa_waveform(tr, velocity)
                st_out.append(tr)
            except (util.ResponseNotFoundError, util.ResponseRemovalError) as e:
                logging.warning(e)

    return st_out


@util.timeit("debug")
def write_waveforms(
    st: Stream, fpath: pathlib.Path, fstem: str, file_format: str
) -> None:
    """
    Output waveform data as a waveform file -- defaults to miniSEED format.

    Parameters
    ----------
    st:
        Waveforms to be written to file.
    fpath : `pathlib.Path` object
        Path to output directory.
    fstem:
        File name (without suffix).
    file_format:
        File format to write waveform data to. Options are all file formats supported by
        ObsPy, including: "MSEED" (default), "SAC", "SEGY", "GSE2"

    """

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

    file = (fpath / fstem).with_suffix(suffix)
    st.write(str(file), format=file_format)
