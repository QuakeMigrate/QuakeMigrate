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


def mad(x, scale=1.4826):
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
    b = np.broadcast_to(a[:, None], new_shape)
    b = np.reshape(b, np.product(new_shape))
    return b


EVENT_FILE_COLS = ["EventNum", "CoaTime", "COA_V", "COA_X", "COA_Y", "COA_Z",
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
        self.run.logger(kwargs.get("log", True))

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
        save_fig : bool, optional
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

        logging.info("\tReading in .scanmseed...")
        data, stats = read_scanmseed(self.run, starttime, endtime, self.pad)

        logging.info("\tTriggering events...\n")
        events = self._trigger_events(starttime, endtime, data, stats, region)

        if events is None:
            logging.info("\tNo events triggered at this threshold - try a "
                         "lower detection threshold.")
        else:
            logging.info("\n\tWriting triggered events to file...")
            write_triggered_events(self.run, events, starttime, endtime)

        logging.info("\n\tPlotting trigger summary...")
        trigger_summary(events, starttime, endtime, self.run,
                        self.marginal_window, self.minimum_repeat,
                        self.threshold, self.normalise_coalescence,
                        self.lut, data, region=region, savefig=savefig)

        logging.info(util.log_spacer)

    def _trigger_events(self, starttime, endtime, scandata, stats, region):
        """
        Function to perform the triggering on the maximum coalescence through
        time data.

        Parameters
        ----------
        starttime : `obspy.UTCDateTime` object
            Timestamp from which to trigger.
        endtime : `obspy.UTCDateTime` object
            Timestamp up to which to trigger.
        scandata : `pandas.DataFrame` object
            Data output by detect() -- decimated scan.
            Columns: ["DT", "COA", "COA_N", "X", "Y", "Z"] - X/Y/Z as lon/lat/m
        coa_stats : `obspy.trace.Stats` object
            Container for additional header information for coalescence trace.
            Contains keys: network, station, channel, starttime, endtime,
                           sampling_rate, delta, npts, calib, _format, mseed
        region : `pandas.DataFrame`
            Only write triggered events within this region to the triggered
            events csv file (for use in locate.) Format is:
                [Xmin, Ymin, Zmin, Xmax, Ymax, Zmax]
            Units are longitude / latitude / metres (elevation; up is positive)

        Returns
        -------
        events : `pandas.DataFrame`
        Triggered events information.
        Columns: ["EventNum", "CoaTime", "COA_V", "COA_X", "COA_Y", "COA_Z",
                  "MinTime", "MaxTime", "COA", "COA_NORM", "EventID"].

        """

        # switch between user-facing minimum repeat definition (minimum repeat
        # interval between event triggers) and internal definition (extra
        # buffer on top of marginal window within which events can't overlap)
        minimum_repeat = self.minimum_repeat - self.marginal_window

        starttime_pad, endtime_pad = starttime - self.pad, endtime + self.pad

        if self.normalise_coalescence:
            tr = scandata["COA_N"]
        else:
            tr = scandata["COA"]
        sampling_rate = stats.sampling_rate
        if self.threshold_method == "dynamic":
            # Split the data in window_length chunks
            breaks = np.array(range(len(tr)))
            breaks = breaks[breaks % int(self.mad_window_length
                                         * sampling_rate) == 0][1:]
            chunks = np.split(tr.values, breaks)
            # Calculate the mad and median values
            mad_values = np.asarray([mad(chunk) for chunk in chunks])
            median_values = np.asarray([np.median(chunk) for chunk in chunks])
            mad_trace = chunks2trace(mad_values, (len(chunks), len(chunks[0])))
            median_trace = chunks2trace(median_values, (len(chunks),
                                        len(chunks[0])))
            mad_trace = mad_trace[:len(tr)]
            median_trace = median_trace[:len(tr)]

            # Set the dynamic threshold
            self.threshold = median_trace + (mad_trace * self.mad_multiplier)
        else:
            # Set static threshold
            self.threshold = np.zeros_like(tr) + self.static_threshold

        # Mask based on first and final time stamps and detection threshold
        coa_data = scandata[(tr >= self.threshold) &
                            (scandata["DT"] >= starttime_pad) &
                            (scandata["DT"] <= endtime_pad)].reset_index(drop=True)

        if coa_data.empty:
            return None

        ss = 1. / sampling_rate

        # Determine the triggers, defined as those points exceeding threshold
        triggers = pd.DataFrame(columns=EVENT_FILE_COLS)
        c = 0
        e = 1
        while c <= len(coa_data) - 1:
            # Determining the index when above the level and maximum value
            d = c
            try:
                # Check the next sample in the list has the correct time stamp
                while coa_data["DT"].iloc[d] + ss == coa_data["DT"].iloc[d + 1]:
                    d += 1
                    if d + 1 >= len(coa_data):
                        break
            except IndexError:
                pass
            min_idx = c
            max_idx = d
            val_idx = coa_data["COA"].iloc[np.arange(min_idx, max_idx+1)].idxmax()
            coa_max_df = coa_data.iloc[val_idx]

            # Determining the times for min, max and max coalescence value
            t_min = coa_data["DT"].iloc[min_idx]
            t_max = coa_data["DT"].iloc[max_idx]
            t_val = coa_max_df["DT"]

            if self.normalise_coalescence:
                COA_V = coa_max_df.COA_N
            else:
                COA_V = coa_max_df.COA
            COA_X, COA_Y, COA_Z = coa_max_df.X, coa_max_df.Y, coa_max_df.Z
            COA, COA_NORM = coa_max_df.COA, coa_max_df.COA_N

            # If the first sample above the threshold is within the marginal
            # window, set min time stamp to:
            # maximum value time - marginal window - minimum repeat
            if (t_val - t_min) < self.marginal_window:
                t_min = t_val - self.marginal_window - minimum_repeat
            # If the first sample is outwith, just subtract the minimum repeat
            else:
                t_min = t_min - minimum_repeat

            # If the final sample above the threshold is within the marginal
            # window,  set the max time stamp to the maximum value time +
            # marginal window + minimum repeat
            if (t_max - t_val) < self.marginal_window:
                t_max = t_val + self.marginal_window + minimum_repeat
            # If the final sample is outwith, just add the minimum repeat
            else:
                t_max = t_max + minimum_repeat

            tmp = pd.DataFrame([[e, t_val, COA_V, COA_X, COA_Y, COA_Z,
                                t_min, t_max, COA, COA_NORM]],
                               columns=EVENT_FILE_COLS)

            triggers = triggers.append(tmp, ignore_index=True)

            c = d + 1
            e += 1

        n_evts = len(triggers)
        evt_num = np.ones((n_evts), dtype=int)

        # Iterate over initially triggered events and see if there is overlap
        # between the final sample in the event window and the edge of the
        # marginal window of the next. If so, treat them as the same event
        count = 1
        for i, event in triggers.iterrows():
            evt_num[i] = count
            if (i + 1 < n_evts) and ((event["MaxTime"]
                                      - (triggers["CoaTime"].iloc[i + 1]
                                      - self.marginal_window)) < 0):
                count += 1
        triggers["EventNum"] = evt_num

        events = pd.DataFrame(columns=EVENT_FILE_COLS)
        for i in range(1, count + 1):
            logging.info(f"\t    Triggered event {i} of {count}")
            tmp = triggers[triggers["EventNum"] == i].reset_index(drop=True)
            _event = tmp.iloc[tmp.COA_V.idxmax()]
            event = pd.DataFrame([[i, _event.CoaTime, _event.COA_V,
                                   _event.COA_X, _event.COA_Y, _event.COA_Z,
                                   tmp.MinTime.min(), tmp.MaxTime.max(),
                                   _event.COA, _event.COA_NORM]],
                                 columns=EVENT_FILE_COLS)
            events = events.append(event, ignore_index=True)

        # Remove events which occur in the pre-pad and post-pad:
        events = events[(events["CoaTime"] >= starttime) &
                        (events["CoaTime"] < endtime)]

        if region is not None:
            events = events[(events["COA_X"] >= region[0]) &
                            (events["COA_Y"] >= region[1]) &
                            (events["COA_Z"] >= region[2]) &
                            (events["COA_X"] <= region[3]) &
                            (events["COA_Y"] <= region[4]) &
                            (events["COA_Z"] <= region[5])]

        # Reset EventNum column
        events.loc[:, "EventNum"] = np.arange(1, len(events) + 1)

        event_uid = events["CoaTime"].astype(str)
        for char_ in ["-", ":", ".", " ", "Z", "T"]:
            event_uid = event_uid.str.replace(char_, "")
        event_uid = event_uid.apply(lambda x: x[:17].ljust(17, "0"))
        events["EventID"] = event_uid

        if events.empty:
            return None
        else:
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
