# -*- coding: utf-8 -*-
"""
Module to handle input/output of .scanmseed files.

:copyright:
    2020–2023, QuakeMigrate developers.
:license:
    GNU General Public License, Version 3
    (https://www.gnu.org/licenses/gpl-3.0.html)

"""

import logging

import numpy as np
from obspy import read, Stream, Trace, UTCDateTime
from obspy.io.mseed import InternalMSEEDError
import pandas as pd

import quakemigrate.util as util


class ScanmSEED:
    """
    Light class to encapsulate the data output by the detect stage of QuakeMigrate. This
    data is stored in an `obspy.Stream` object with the channels:
    ["COA", "COA_N", "X", "Y", "Z"].

    Parameters
    ----------
    run : :class:`~quakemigrate.io.core.Run` object
        Light class encapsulating i/o path information for a given run.
    continuous_write : bool
        Option to continuously write the .scanmseed file output by
        :func:`~quakemigrate.signal.scan.QuakeScan.detect()` at the end of every time
        step. Default behaviour is to write in day chunks where possible.
    sampling_rate : int
        Desired sampling rate of input data; sampling rate at which to compute the
        coalescence function. Default: 50 Hz.

    Attributes
    ----------
    stream : `obspy.Stream` object
        Output of :func:`~quakemigrate.signal.scan.QuakeScan.detect()` stored in
        `obspy.Stream` object. The values have been multiplied by a factor to make use
        of more efficient compression.
        Channels: ["COA", "COA_N", "X", "Y", "Z"]
    written : bool
        Tracker for whether the data appended has been written recently.

    Methods
    -------
    append(times, max_coa, max_coa_n, coord, map4d=None)
        Append the output of :func:`~quakemigrate.signal.scan.QuakeScan._compute()` to
        the coalescence stream.
    empty(starttime, timestep, i, msg)
        Create an set of empty arrays for a given timestep and append to the coalescence
        stream.
    write(write_start=None, write_end=None)
        Write the coalescence stream to a .scanmseed file.

    """

    def __init__(self, run, continuous_write, sampling_rate):
        """Instantiate the ScanmSEED object."""

        self.run = run
        self.continuous_write = continuous_write
        self.sampling_rate = sampling_rate

        self.written = False
        self.stream = Stream()

    def append(self, starttime, max_coa, max_coa_n, coord, ucf):
        """
        Append latest timestep of :func:`~quakemigrate.signal.scan.QuakeScan.detect()`
        output to `obspy.Stream` object.

        Multiply channels ["COA", "COA_N", "X", "Y", "Z"] by factors of
        ["1e5", "1e5", "1e6", "1e6", "1e3"] respectively, round and convert to int32 as
        this dramatically reduces memory usage, and allows the coastream data to be
        saved in mSEED format with STEIM2 compression. The multiplication factor is
        removed when the data is read back in.

        Parameters
        ----------
        starttime : `obspy.UTCDateTime` object
            Timestamp of first sample of coalescence data.
        max_coa : `numpy.ndarray` of floats, shape(nsamples)
            Coalescence value through time.
        max_coa_n : `numpy.ndarray` of floats, shape(nsamples)
            Normalised coalescence value through time.
        coord : `numpy.ndarray` of floats, shape(nsamples)
            Location of maximum coalescence through time in input projection space.
        ucf : float
            A conversion factor based on the lookup table grid projection. Used to
            ensure the same level of precision (millimetre) is retained during
            compression, irrespective of the units of the grid projection.

        """

        # Clip max value of COA to prevent int overflow
        max_coa[max_coa > 21474.0] = 21474.0
        max_coa_n[max_coa_n > 21474.0] = 21474.0

        meta = {
            "network": "NW",
            "npts": len(max_coa) - 1,
            "sampling_rate": self.sampling_rate,
            "starttime": starttime,
        }

        self.stream += Trace(
            data=self._data2int(max_coa[:-1], 1e5),
            header={**{"station": "COA"}, **meta},
        )
        self.stream += Trace(
            data=self._data2int(max_coa_n[:-1], 1e5),
            header={**{"station": "COA_N"}, **meta},
        )
        self.stream += Trace(
            data=self._data2int(coord[:-1, 0], 1e6), header={**{"station": "X"}, **meta}
        )
        self.stream += Trace(
            data=self._data2int(coord[:-1, 1], 1e6), header={**{"station": "Y"}, **meta}
        )
        self.stream += Trace(
            data=self._data2int(coord[:-1, 2], 1e3 * ucf),
            header={**{"station": "Z"}, **meta},
        )
        self.stream.merge(method=-1)

        # Write to file if passed day line
        self.written = False
        stats = self.stream[0].stats
        if stats.starttime.julday != stats.endtime.julday:
            write_start = stats.starttime
            write_end = UTCDateTime(stats.endtime.date) - stats.delta
            self.write(write_start, write_end)
            self.stream.trim(starttime=write_end + stats.delta)

        if self.continuous_write and not self.written:
            self.write()

    def empty(self, starttime, timestep, i, msg, ucf):
        """
        Create an empty set of arrays to write to .scanmseed; used where there is no
        data available to run :func:`~quakemigrate.signal.scan.QuakeScan._compute()`.

        Parameters
        ----------
        starttime : `obspy.UTCDateTime` object
            Timestamp of first sample in the given timestep.
        timestep : float
            Length (in seconds) of timestep used in detect().
        i : int
            The ith timestep of the continuous compute.
        msg : str
            Message to output to log giving details as to why this timestep is empty.
        ucf : float
            A conversion factor based on the lookup table grid projection. Used to
            ensure the same level of precision (millimetre) is retained during
            compression, irrespective of the units of the grid projection.

        """

        logging.info(msg)

        starttime = starttime + timestep * i
        n = util.time2sample(timestep, self.sampling_rate) + 1
        max_coa = max_coa_n = np.full(n, 0)
        coord = np.full((n, 3), 0)

        self.append(starttime, max_coa, max_coa_n, coord, ucf)

    def write(self, write_start=None, write_end=None):
        """
        Write a new .scanmseed file from an `obspy.Stream` object containing the data
        output from detect(). Note: values have been multiplied by a power of ten,
        rounded and converted to an int32 array so the data can be saved as mSEED with
        STEIM2 compression. This multiplication factor is removed when the data is read
        back in with read_scanmseed().

        Parameters
        ----------
        write_start : `obspy.UTCDateTime` object, optional
            Timestamp from which to write the coalescence stream to file.
        write_end : `obspy.UTCDateTime` object, optional
            Timestamp up to which to write the coalescence stream to file.

        """

        fpath = self.run.path / "detect" / "scanmseed"
        fpath.mkdir(exist_ok=True, parents=True)

        if write_start is not None and write_end is not None:
            st = self.stream.slice(starttime=write_start, endtime=write_end)
        else:
            st = self.stream

        starttime = st[0].stats.starttime
        fstem = f"{starttime.year}_{starttime.julday:03d}"
        file = (fpath / fstem).with_suffix(".scanmseed")

        try:
            st.write(str(file), format="MSEED", encoding="STEIM2")
        except InternalMSEEDError as e:
            logging.debug(
                f"Cannot compress data: {e}\nUnable to compress data using STEIM2 - "
                "falling back on STEIM1."
            )
            st.write(str(file), format="MSEED", encoding="STEIM1")
        self.written = True

    def _data2int(self, data, factor):
        """
        Utility function to convert data to ints before writing.

        Parameters
        ----------
        data : `numpy.Array`
            Data stream to convert to integer values.
        factor : float
            Scaling factor used to control the number of decimal places saved.

        Returns
        -------
        out : `numpy.Array`, int
            Original data stream multiplied by factor and converted to int.

        """

        return np.round(data * factor).astype(np.int32)


@util.timeit()
def read_scanmseed(run, starttime, endtime, pad, ucf):
    """
    Read .scanmseed files between two time stamps. Files are labelled by year and Julian
    day.

    Parameters
    ----------
    run : :class:`~quakemigrate.io.core.Run` object
        Light class encapsulating i/o path information for a given run.
    starttime : `obspy.UTCDateTime` object
        Timestamp from which to read the coalescence stream.
    endtime : `obspy.UTCDateTime` object
        Timestamp up to which to read the coalescence stream.
    pad : float
        Read in "pad" seconds of additional data on either end.
    ucf : float
        A conversion factor based on the lookup table grid projection. Used to ensure
        the same level of precision (millimetre) is retained during compression,
        irrespective of the units of the grid projection.

    Returns
    -------
    data : `pandas.DataFrame` object
        Data output by detect() -- decimated scan.
        Columns: ["DT", "COA", "COA_N", "X", "Y", "Z"] - X/Y/Z as lon/lat/units
        where units is the user-selected units of the lookup table grid projection
        (either metres or kilometres).
    stats : `obspy.trace.Stats` object
        Container for additional header information for coalescence trace.
        Contains keys: network, station, channel, starttime, endtime,
                       sampling_rate, delta, npts, calib, _format, mseed

    """

    fpath = run.path / "detect" / "scanmseed"

    readstart, readend = starttime - pad, endtime + pad
    startday = UTCDateTime(readstart.date)

    dy = 0
    scanmseed = Stream()
    # Loop through days trying to read .scanmseed files
    while startday + (dy * 86400) <= readend:
        now = readstart + (dy * 86400)
        fstem = f"{now.year}_{now.julday:03d}"
        file = (fpath / fstem).with_suffix(".scanmseed")
        try:
            scanmseed += read(
                str(file), starttime=readstart, endtime=readend, format="MSEED"
            )
        except FileNotFoundError:
            logging.info(f"\n\t    No .scanmseed file found for day {fstem}!")
        dy += 1

    if not bool(scanmseed):
        raise util.NoScanMseedDataException

    scanmseed.merge(method=-1)
    stats = scanmseed.select(station="COA")[0].stats

    data = pd.DataFrame()
    data["DT"] = scanmseed[0].times(type="utcdatetime")
    data["COA"] = scanmseed.select(station="COA")[0].data / 1e5
    data["COA_N"] = scanmseed.select(station="COA_N")[0].data / 1e5
    data["X"] = scanmseed.select(station="X")[0].data / 1e6
    data["Y"] = scanmseed.select(station="Y")[0].data / 1e6
    data["Z"] = scanmseed.select(station="Z")[0].data / (1e3 * ucf)

    # Check if the data covers the entirety of the requested period
    if stats.starttime > starttime:
        logging.info(
            "\n\t    Warning! .scanmseed starttime is later than trigger() starttime!"
        )
    elif stats.starttime > readstart:
        logging.info("\t    Warning! No .scanmseed data found for pre-pad!")
    if stats.endtime < endtime - stats.delta:
        logging.info("\n\t    Warning! .scanmseed endtime is before trigger() endtime!")
    elif stats.endtime < readend:
        logging.info("\t    Warning! No .scanmseed data found for post-pad!")
    logging.info(f"\t    ...from {stats.starttime} - {stats.endtime}.")

    return data, stats
