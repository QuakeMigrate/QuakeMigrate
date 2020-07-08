# -*- coding: utf-8 -*-
"""
Module to perform the trigger stage of QuakeMigrate.

"""

import logging

import numpy as np
from obspy import UTCDateTime
import pandas as pd

from QMigrate.io import Run, read_scanmseed, write_triggered_events
from QMigrate.plot import trigger_summary
import QMigrate.util as util


def calculate_mad(x, scale=1.4826):
    """
    Calculates the Median Absolute Deviation (MAD) of the input array x.

    Parameters
    ----------
    x : array-like
        Coalescence array in.
    scale : float, optional
        A scaling factor for the MAD output to make the calculated MAD factor
        a consistent estimation of the standard deviation of the distribution.

    Returns
    -------
    scaled_mad : array-like
        Array of scaled mean absolute deviation values for the input array, x,
        scaled to provide an estimation of the standard deviation of the
        distribution.

    """

    x = np.asarray(x)

    if not x.size:
        return np.nan

    if np.isnan(np.sum(x)):
        return np.nan

    # Calculate median and mad values:
    med = np.apply_over_axes(np.median, x, 0)
    mad = np.median(np.abs(x - med), axis=0)

    return scale * mad


def chunks2trace(a, new_shape):
    """
    Create a trace filled with chunks of the same value.

    Parameters:
    -----------
    a : array-like
        Array of chunks.
    new_shape : tuple of ints
        (number of chunks, chunk_length)

    Returns:
    --------
    b : array-like
        Single array of values contained in `a`.

    """

    b = np.broadcast_to(a[:, None], new_shape)
    b = np.reshape(b, np.product(new_shape))

    return b


TRIGGER_FILE_COLS = ["EventNum", "CoaTime", "COA_V", "COA_X", "COA_Y", "COA_Z",
                     "MinTime", "MaxTime", "COA", "COA_NORM"]


class Trigger:
    """
    QuakeMigrate triggering class.

    Triggers candidate earthquakes from the maximum coalescence through time
    data output by the decimated detect scan, ready to be run through locate().

    Parameters
    ----------
    lut : `QMigrate.lut.LUT` object
        Contains the traveltime lookup tables for P- and S-phases, computed for
        some pre-defined velocity model.
    run_path : str
        Points to the top level directory containing all input files, under
        which the specific run directory will be created.
    run_name : str
        Name of the current QuakeMigrate run.
    kwargs : **dict
        See Trigger Attributes for details. In addition to these:
        log : bool, optional
            Toggle for logging. If True, will output to stdout and generate a
            log file. Default is to only output to stdout.
        trigger_name : str
            Optional name of a sub-run - useful when testing different trigger
            parameters, for example.

    Attributes
    ----------
    mad_window_length : float, optional
        Length of window within which to calculate the Median Average
        Deviation. Default: 3600 seconds (1 hour).
    mad_multiplier : float, optional
        A scaling factor for the MAD output to make the calculated MAD factor
        a consistent estimation of the standard deviation of the distribution.
        Default: 1.4826, which is the appropriate scaling factor for a normal
        distribution.
    marginal_window : float, optional
        Time window over which to marginalise the coalescence, making it solely
        a function of the spatial dimensions. This should be an estimate of the
        time error, as derived from an estimate of the spatial error and error
        in the velocity model. Default: 2 seconds.
    minimum_repeat : float, optional
        Minimum time interval between triggered events. Must be at least twice
        the marginal window. Default: 4 seconds.
    normalise_coalescence : bool, optional
        If True, triggering is performed on the maximum coalescence normalised
        by the mean coalescence value in the 3-D grid. Default: False.
    pad : float, optional
        Additional time padding to ensure events close to the starttime/endtime
        are not cut off and missed. Default: 120 seconds.
    run : `QMigrate.io.Run` object
        Light class encapsulating i/o path information for a given run.
    static_threshold : float, optional
        Static threshold value above which to trigger candidate events.
    threshold_method : str, optional
        Toggle between a "static" threshold and a "dynamic" threshold, based on
        the Median Average Deviation. Default: "static".

    Methods
    -------
    trigger(starttime, endtime, region=None, savefig=True)
        Trigger candidate earthquakes from decimated detect scan results.

    Raises
    ------
    Exception
        If `minimum_repeat` < 2 * `marginal_window`.
    InvalidThresholdMethodException
        If an invalid threshold method is passed in by the user.
    TimeSpanException
        If the user supplies a starttime that is after the endtime.

    """

    def __init__(self, lut, run_path, run_name, **kwargs):
        """Instantiate the Trigger object."""

        self.lut = lut

        # --- Organise i/o and logging ---
        self.run = Run(run_path, run_name, kwargs.get("trigger_name", ""),
                       "trigger")
        self.run.logger(kwargs.get("log", False))

        # --- Grab Trigger parameters or set defaults ---
        self.threshold_method = kwargs.get("threshold_method", "static")
        if self.threshold_method == "static":
            self.static_threshold = kwargs.get("static_threshold", 1.5)
        elif self.threshold_method == "dynamic":
            self.mad_window_length = kwargs.get("mad_window_length", 3600.)
            self.mad_multiplier = kwargs.get("mad_multiplier", 1.)
        else:
            raise util.InvalidThresholdMethodException
        self.marginal_window = kwargs.get("marginal_window", 2.)
        self.minimum_repeat = kwargs.get("minimum_repeat", 4.)
        self.normalise_coalescence = kwargs.get("normalise_coalescence", False)
        self.pad = kwargs.get("pad", 120.)

    def __str__(self):
        """Return short summary string of the Trigger object."""

        out = ("\tTrigger parameters:\n"
               f"\t\tPre/post pad = {self.pad} s\n"
               f"\t\tMarginal window = {self.marginal_window} s\n"
               f"\t\tMinimum repeat  = {self.minimum_repeat} s\n\n"
               f"\t\tTriggering from the ")
        out += "normalised " if self.normalise_coalescence else ""
        out += "maximum coalescence trace.\n\n"
        out += f"\t\tTrigger threshold method: {self.threshold_method}\n"
        if self.threshold_method == "static":
            out += f"\t\tStatic threshold = {self.static_threshold}\n"
        elif self.threshold_method == "dynamic":
            out += (f"\t\tMAD Window     = {self.mad_window_length}\n"
                    f"\t\tMAD Multiplier = {self.mad_multiplier}\n")

        return out

    def trigger(self, starttime, endtime, region=None, savefig=True):
        """
        Trigger candidate earthquakes from decimated scan data.

        Parameters
        ----------
        starttime : str
            Timestamp from which to trigger.
        endtime : str
            Timestamp up to which to trigger.
        region : list of floats, optional
            Only write triggered events within this region to the triggered
            events csv file (for use in locate.) Format is:
                [Xmin, Ymin, Zmin, Xmax, Ymax, Zmax]
            Units are longitude / latitude / metres (in positive-down frame).
        savefig : bool, optional
            Save triggered events figure (default) or open interactive view.

        Raises
        ------
        TimeSpanException
            If `starttime` is after `endtime`.

        """

        starttime, endtime = UTCDateTime(starttime), UTCDateTime(endtime)
        if starttime > endtime:
            raise util.TimeSpanException

        logging.info(util.log_spacer)
        logging.info("\tTRIGGER - Triggering events from .scanmseed")
        logging.info(util.log_spacer)
        logging.info(f"\n\tTriggering events from {starttime} to {endtime}\n")
        logging.info(self)
        logging.info(util.log_spacer)

        batchstart = starttime
        while batchstart < endtime:
            next_day = UTCDateTime(batchstart.date) + 86400
            batchend = next_day if next_day <= endtime else endtime
            self._trigger_batch(batchstart, batchend, region, savefig)
            batchstart = next_day

        logging.info(util.log_spacer)

    def _trigger_batch(self, batchstart, batchend, region, savefig):
        """
        Wraps all of the methods used in sequence to determine triggers.

        Parameters
        ----------
        batchstart : `obspy.UTCDateTime` object
            Timestamp from which to trigger.
        batchend : `obspy.UTCDateTime` object
            Timestamp up to which to trigger.
        region : list of floats
            Only write triggered events within this region to the triggered
            events csv file (for use in locate.) Format is:
                [Xmin, Ymin, Zmin, Xmax, Ymax, Zmax]
            Units are longitude / latitude / metres (in positive-down frame).
        savefig : bool
            Save triggered events figure (default) or open interactive view.

        """

        logging.info("\tReading in .scanmseed...")
        data, stats = read_scanmseed(self.run, batchstart, batchend, self.pad)

        logging.info("\tTriggering events...\n")
        trigger_on = "COA_N" if self.normalise_coalescence else "COA"
        threshold = self._get_threshold(data[trigger_on], stats.sampling_rate)
        candidate_events = self._identify_candidates(data, trigger_on,
                                                     threshold)

        if candidate_events.empty:
            logging.info("\tNo events triggered at this threshold - try a "
                         "lower detection threshold.")
            events = candidate_events
        else:
            refined_events = self._refine_candidates(candidate_events)
            events = self._filter_events(refined_events, batchstart, batchend,
                                         region)
            logging.info((f"\n\t\t{len(events)} triggered within the specified "
                          f"region between {batchstart} and {batchend}"))
            logging.info("\n\tWriting triggered events to file...")
            write_triggered_events(self.run, events, batchstart)

        logging.info("\n\tPlotting trigger summary...")
        trigger_summary(events, batchstart, batchend, self.run,
                        self.marginal_window, self.minimum_repeat,
                        threshold, self.normalise_coalescence, self.lut,
                        data, region=region, savefig=savefig)

    def _get_threshold(self, scandata, sampling_rate):
        """
        Determine the threshold to use when triggering candidate events.

        Parameters
        ----------
        scandata : `pandas.Series` object
            (Normalised) coalescence values for which to calculate the
            threshold.
        sampling_rate : int
            Number of samples per second of the coalescence scan data.

        Returns
        -------
        threshold : `numpy.ndarray` object
            Array of threshold values.

        """

        if self.threshold_method == "dynamic":
            # Split the data in window_length chunks
            breaks = np.arange(len(scandata))
            breaks = breaks[breaks % int(self.mad_window_length
                                         * sampling_rate) == 0][1:]
            chunks = np.split(scandata.values, breaks)

            # Calculate the mad and median values
            mad_values = np.asarray([calculate_mad(chunk) for chunk in chunks])
            median_values = np.asarray([np.median(chunk) for chunk in chunks])
            mad_trace = chunks2trace(mad_values, (len(chunks), len(chunks[0])))
            median_trace = chunks2trace(median_values, (len(chunks),
                                                        len(chunks[0])))
            mad_trace = mad_trace[:len(scandata)]
            median_trace = median_trace[:len(scandata)]

            # Set the dynamic threshold
            threshold = median_trace + (mad_trace * self.mad_multiplier)
        else:
            # Set static threshold
            threshold = np.zeros_like(scandata) + self.static_threshold

        return threshold

    def _identify_candidates(self, scandata, trigger_on, threshold):
        """
        Identify distinct periods of time for which the maximum (normalised)
        coalescence trace exceeds the chosen threshold.

        Parameters
        ----------
        scandata : `pandas.DataFrame` object
            Data output by detect() -- decimated scan.
            Columns: ["DT", "COA", "COA_N", "X", "Y", "Z"] - X/Y/Z as lon/lat/m
        trigger_on : str
            Specifies the maximum coalescence data on which to trigger events.
        threshold : `numpy.ndarray` object
            Array of threshold values.

        Returns
        -------
        triggers : `pandas.DataFrame` object
            Candidate events exceeding some threshold.

        """

        # Switch between user-facing minimum repeat definition (minimum repeat
        # interval between event triggers) and internal definition (extra
        # buffer on top of marginal window within which events cannot overlap)
        minimum_repeat = self.minimum_repeat - self.marginal_window

        thresholded = scandata[scandata[trigger_on] >= threshold]
        r = np.arange(len(thresholded))
        candidates = [d for _, d in thresholded.groupby(thresholded.index - r)]

        triggers = pd.DataFrame(columns=TRIGGER_FILE_COLS)
        for i, candidate in enumerate(candidates):
            peak = candidate.loc[candidate["COA"].idxmax()]

            # If first sample above threshold is within the marginal window
            if (peak["DT"] - candidate["DT"].iloc[0]) < self.marginal_window:
                min_dt = peak["DT"] - self.minimum_repeat
            # Otherwise just subtract the minimum repeat
            else:
                min_dt = candidate["DT"].iloc[0] - minimum_repeat

            # If last sample above threshold is within the marginal window
            if (candidate["DT"].iloc[-1] - peak["DT"]) < self.marginal_window:
                max_dt = peak["DT"] + self.minimum_repeat
            # Otherwise just add the minimum repeat
            else:
                max_dt = candidate["DT"].iloc[-1] + minimum_repeat

            trigger = pd.Series([i, peak["DT"], peak[trigger_on],
                                 peak["X"], peak["Y"], peak["Z"],
                                 min_dt, max_dt, peak["COA"], peak["COA_N"]],
                                index=TRIGGER_FILE_COLS)

            triggers = triggers.append(trigger, ignore_index=True)

        return triggers

    def _refine_candidates(self, candidate_events):
        """
        Merge candidate events for which the marginal windows overlap with the
        minimum inter-event time.

        Parameters
        ----------
        candidate_events : `pandas.DataFrame` objecy
            Candidate events corresponding to periods of time in which the
            coalescence signal exceeds some threshold.

        Returns
        -------
        events : `pandas.DataFrame` object
            Merged events with some minimum inter-event spacing in time.

        """

        # Iterate pairwise (event1, event2) over the candidate events to
        # identify overlaps between:
        #   - event1 marginal window and event2 minimum window position
        #   - event2 marginal window and event1 maximum window position
        event_count = 1
        for i, event1 in candidate_events.iterrows():
            candidate_events.loc[i, "EventNum"] = event_count
            if i + 1 == len(candidate_events):
                continue
            event2 = candidate_events.iloc[i+1]
            if all([event1["MaxTime"] < event2["CoaTime"] - self.marginal_window,
                    event2["MinTime"] > event1["CoaTime"] + self.marginal_window]):
                event_count += 1

        # Split into DataFrames by event number
        merged_candidates = [d for _, d in candidate_events.groupby(
            candidate_events["EventNum"])]

        # Update the min/max window times and build final event DataFrame
        refined_events = pd.DataFrame(columns=TRIGGER_FILE_COLS)
        for i, candidate in enumerate(merged_candidates):
            logging.info(f"\t    Triggered event {i+1} of "
                         f"{len(merged_candidates)}")
            event = candidate.loc[candidate["COA_V"].idxmax()].copy()
            event["MinTime"] = candidate["MinTime"].min()
            event["MaxTime"] = candidate["MaxTime"].max()
            refined_events = refined_events.append(event, ignore_index=True)

        return refined_events

    def _filter_events(self, events, starttime, endtime, region):
        """
        Remove events within the padding time and/or within a specific
        geographical region. Also adds a unique event identifier based on the
        coalescence time.

        Parameters
        ----------
        events : `pandas.DataFrame` object
            Refined set of events to be filtered.
        starttime : `obspy.UTCDateTime` object
            Timestamp from which to trigger.
        endtime : `obspy.UTCDateTime` object
            Timestamp up to which to trigger.
        region : list
            Only write triggered events within this region to the triggered
            events csv file (for use in locate.) Format is:
                [Xmin, Ymin, Zmin, Xmax, Ymax, Zmax]
            Units are longitude / latitude / metres (elevation; up is positive)

        Returns
        -------
        events : `pandas.DataFrame` object
            Final set of triggered events.

        """

        # Remove events which occur in the pre-pad and post-pad:
        events = events.loc[(events["CoaTime"] >= starttime) &
                            (events["CoaTime"] < endtime), :].copy()

        if region is not None:
            events = events.loc[(events["COA_X"] >= region[0]) &
                                (events["COA_Y"] >= region[1]) &
                                (events["COA_Z"] >= region[2]) &
                                (events["COA_X"] <= region[3]) &
                                (events["COA_Y"] <= region[4]) &
                                (events["COA_Z"] <= region[5]), :].copy()

        # Reset EventNum column and add a unique identifier
        events.loc[:, "EventNum"] = np.arange(len(events)) + 1
        event_uid = events["CoaTime"].astype(str)
        for char_ in ["-", ":", ".", " ", "Z", "T"]:
            event_uid = event_uid.str.replace(char_, "")
        event_uid = event_uid.apply(lambda x: x[:17].ljust(17, "0"))
        events["EventID"] = event_uid

        return events

    @property
    def minimum_repeat(self):
        """Get and set the minimum repeat time."""

        return self._minimum_repeat

    @minimum_repeat.setter
    def minimum_repeat(self, value):
        if value < 2 * self.marginal_window:
            msg = "\tMinimum repeat must be >= 2 * marginal window."
            raise Exception(msg)
        else:
            self._minimum_repeat = value
