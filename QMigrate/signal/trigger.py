# -*- coding: utf-8 -*-
"""
Module to perform the trigger stage of QuakeMigrate.

"""

import logging

import numpy as np
from obspy import UTCDateTime
import pandas as pd

from QMigrate.io import Run
from QMigrate.plot import triggered_events
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

        self.stations = stations

        self.start_time = None
        self.end_time = None

        # Detection threshold above which to trigger events
        self.detection_threshold = 1.5

        # Set some defaults for the dynamic trigger threshold
        self.mad_window_length = 3600.
        self.mad_multiplier = 10.

        # Trigger from normalised max coalescence through time
        self.normalise_coalescence = True

        # Marginalise 4D coalescence map over +/- 2 seconds around max
        # time of max coalescence
        self.marginal_window = 2.

        # Minumum time interval between triggers. Must be >= 2*marginal_window
        self.minimum_repeat = 4.

        # Pad at start and end of trigger run
        self.pad = 120

        self.sampling_rate = None
        self.coa_data = None

        self.events = None
        self.region = None

    def __str__(self):
        """Return short summary string of the Trigger object."""

        out = ("\tTrigger parameters:\n"
               f"\t\tPre/post pad = {self.pad} s\n"
               f"\t\tMarginal window = {self.marginal_window} s\n"
               f"\t\tMinimum repeat  = {self.minimum_repeat} s\n\n"
               f"\t\tTriggering from ")
        out += "normalised " if self.normalise_coalescence else ""
        out += "coalescence stream.\n\n"
        out += f"\t\tDetection threshold method: {self.threshold_method}\n"
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
        logging.info("\tTRIGGER - Triggering events from coalescence")
        logging.info(util.log_spacer)
        logging.info(f"\n\tTriggering events from {starttime} to {endtime}\n")
        logging.info(self)
        logging.info(util.log_spacer)

        if self.minimum_repeat < 2 * self.marginal_window:
            msg = "\tMinimum repeat must be >= 2 * marginal window."
            raise Exception(msg)

        logging.info("\tReading in .scanmseed...")
        read_start = self.start_time - self.pad
        read_end = self.end_time + self.pad
        self.coa_data, coa_stats = self.output.read_coastream(read_start,
                                                              read_end)

        # Check if coa_data found by read_coastream covers whole trigger period
        msg = ""
        if coa_stats.starttime > self.start_time:
            msg += ("\tWarning! scanmseed data start is after trigger()"
                    "start_time!\n")
        elif coa_stats.starttime > read_start:
            msg += "\tWarning! No scanmseed data found for pre-pad!\n"
        if coa_stats.endtime < self.end_time - 1 / coa_stats.sampling_rate:
            msg += ("\tWarning! scanmseed data end is before trigger()"
                    "end_time!\n")
        elif coa_stats.endtime < read_end:
            msg += "\tWarning! No scanmseed data found for post-pad!\n"
        self.output.log(msg, self.log)
        self.output.log("\tscanmseed read complete.", self.log)

        self.sampling_rate = coa_stats.sampling_rate

        events = self._trigger_events()

        if events is None:
            logging.info("\tNo events triggered at this threshold - try a "
                         "lower detection threshold.")
        else:
            logging.info("\tWriting triggered events to file...")
            self.output.write_triggered_events(self.events, self.start_time,
                                               self.end_time)

        triggered_events(events=self.events, start_time=self.start_time,
                         end_time=self.end_time, output=self.output,
                         marginal_window=self.marginal_window,
                         detection_threshold=self.threshold,
                         normalise_coalescence=self.normalise_coalescence,
                         log=self.log, data=self.coa_data,
                         region=self.region, stations=self.stations,
                         savefig=savefig)

        logging.info(util.log_spacer)

    def _trigger_events(self):
        """
        Function to perform the triggering on the maximum coalescence through
        time data.

        Returns
        -------
        events : pandas DataFrame
            Triggered events information. Columns: ["EventNum", "CoaTime",
                                                    "COA_V", "COA_X", "COA_Y",
                                                    "COA_Z", "MinTime",
                                                    "MaxTime", "COA",
                                                    "COA_NORM", "evt_id"]

        """

        # switch between user-facing minimum repeat definition (minimum repeat
        # interval between event triggers) and internal definition (extra
        # buffer on top of marginal window within which events can't overlap)
        minimum_repeat = self.minimum_repeat - self.marginal_window

        start_time = self.start_time - self.pad
        end_time = self.end_time + self.pad

        if self.normalise_coalescence:
            coa_data = self.coa_data["COA_N"]
        else:
            coa_data = self.coa_data["COA"]

        if self.detection_threshold == "dynamic":
            # Split the data in window_length chunks
            breaks = np.array(range(len(coa_data)))
            breaks = breaks[breaks % int(self.mad_window_length
                                         * self.sampling_rate) == 0][1:]
            chunks = np.split(coa_data.values, breaks)
            # Calculate the mad and median values
            mad_values = np.asarray([mad(chunk) for chunk in chunks])
            median_values = np.asarray([np.median(chunk) for chunk in chunks])
            mad_trace = chunks2trace(mad_values, (len(chunks), len(chunks[0])))
            median_trace = chunks2trace(median_values, (len(chunks),
                                        len(chunks[0])))
            mad_trace = mad_trace[:len(coa_data)]
            median_trace = median_trace[:len(coa_data)]

            # Set the dynamic threshold
            self.threshold = median_trace + (mad_trace * self.mad_multiplier)
        else:
            # Set static threshold
            self.threshold = np.zeros_like(coa_data) + self.detection_threshold

        # Mask based on first and final time stamps and detection threshold
        coa_data = self.coa_data[coa_data >= self.threshold]
        coa_data = coa_data[(coa_data["DT"] >= start_time) &
                            (coa_data["DT"] <= end_time)]

        coa_data = coa_data.reset_index(drop=True)

        if len(coa_data) == 0:
            return None

        event_cols = ["EventNum", "CoaTime", "COA_V", "COA_X", "COA_Y",
                      "COA_Z", "MinTime", "MaxTime", "COA", "COA_NORM"]

        ss = 1. / self.sampling_rate

        # Determine the initial triggered events
        init_events = pd.DataFrame(columns=event_cols)
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

            # Determining the times for min, max and max coalescence value
            t_min = coa_data["DT"].iloc[min_idx]
            t_max = coa_data["DT"].iloc[max_idx]
            t_val = coa_data["DT"].iloc[val_idx]

            if self.normalise_coalescence is True:
                COA_V = coa_data["COA_N"].iloc[val_idx]
            else:
                COA_V = coa_data["COA"].iloc[val_idx]
            COA_X = coa_data["X"].iloc[val_idx]
            COA_Y = coa_data["Y"].iloc[val_idx]
            COA_Z = coa_data["Z"].iloc[val_idx]
            COA = coa_data["COA"].iloc[val_idx]
            COA_NORM = coa_data["COA_N"].iloc[val_idx]

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
                               columns=event_cols)

            init_events = init_events.append(tmp, ignore_index=True)

            c = d + 1
            e += 1

        n_evts = len(init_events)
        evt_num = np.ones((n_evts), dtype=int)

        # Iterate over initially triggered events and see if there is overlap
        # between the final sample in the event window and the edge of the
        # marginal window of the next. If so, treat them as the same event
        count = 1
        for i, event in init_events.iterrows():
            evt_num[i] = count
            if (i + 1 < n_evts) and ((event["MaxTime"]
                                      - (init_events["CoaTime"].iloc[i + 1]
                                      - self.marginal_window)) < 0):
                count += 1
        init_events["EventNum"] = evt_num

        events = pd.DataFrame(columns=event_cols)
        self.output.log("\n\tTriggering...", self.log)
        for i in range(1, count + 1):
            logging.info(f"\t    Triggered event {i} of {count}")
            tmp = init_events[init_events["EventNum"] == i]
            tmp = tmp.reset_index(drop=True)
            j = np.argmax(tmp["COA_V"].values)
            min_mt = np.min(tmp["MinTime"])
            max_mt = np.max(tmp["MaxTime"])
            event = pd.DataFrame([[i, tmp["CoaTime"].iloc[j],
                                   tmp["COA_V"].iloc[j],
                                   tmp["COA_X"].iloc[j],
                                   tmp["COA_Y"].iloc[j],
                                   tmp["COA_Z"].iloc[j],
                                   min_mt,
                                   max_mt,
                                   tmp["COA"].iloc[j],
                                   tmp["COA_NORM"].iloc[j]]],
                                 columns=event_cols)
            events = events.append(event, ignore_index=True)

        # Remove events which occur in the pre-pad and post-pad:
        events = events[(events["CoaTime"] >= self.start_time) &
                        (events["CoaTime"] < self.end_time)]

        if self.region is not None:
            events = events[(events["COA_X"] >= self.region[0]) &
                            (events["COA_Y"] >= self.region[1]) &
                            (events["COA_Z"] >= self.region[2]) &
                            (events["COA_X"] <= self.region[3]) &
                            (events["COA_Y"] <= self.region[4]) &
                            (events["COA_Z"] <= self.region[5])]

        # Reset EventNum column
        events.loc[:, "EventNum"] = np.arange(1, len(events) + 1)

        event_uid = events["CoaTime"].astype(str)
        for char_ in ["-", ":", ".", " ", "Z", "T"]:
            event_uid = event_uid.str.replace(char_, "")
        event_uid = event_uid.apply(lambda x: x[:17].ljust(17, "0"))
        events["EventID"] = event_uid

        if len(events) == 0:
            events = None

        return events
