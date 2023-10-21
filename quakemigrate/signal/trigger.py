# -*- coding: utf-8 -*-
"""
Module to perform the trigger stage of QuakeMigrate.

:copyright:
    2020–2023, QuakeMigrate developers.
:license:
    GNU General Public License, Version 3
    (https://www.gnu.org/licenses/gpl-3.0.html)

"""

import logging

import numpy as np
from obspy import UTCDateTime
import pandas as pd

from quakemigrate.io import Run, read_scanmseed, write_triggered_events
from quakemigrate.plot import trigger_summary
import quakemigrate.util as util


def chunks2trace(a, new_shape):
    """
    Create a trace filled with chunks of the same value.

    Parameters:
    -----------
    a : array-like
        Array of chunks.
    new_shape : tuple of ints
        (number of chunks, chunk_length).

    Returns:
    --------
    b : array-like
        Single array of values contained in `a`.

    """

    b = np.broadcast_to(a[:, None], new_shape)
    b = np.reshape(b, np.product(new_shape))

    return b


CANDIDATES_COLS = [
    "EventNum",
    "CoaTime",
    "TRIG_COA",
    "COA_X",
    "COA_Y",
    "COA_Z",
    "MinTime",
    "MaxTime",
    "COA",
    "COA_NORM",
]

REFINED_EVENTS_COLS = [
    "EventID",
    "CoaTime",
    "TRIG_COA",
    "COA_X",
    "COA_Y",
    "COA_Z",
    "MinTime",
    "MaxTime",
    "COA",
    "COA_NORM",
]


class Trigger:
    """
    QuakeMigrate triggering class.

    Triggers candidate earthquakes from the continuous maximum coalescence through time
    data output by the decimated detect scan, ready to be run through locate().

    Parameters
    ----------
    lut : :class:`~quakemigrate.lut.lut.LUT` object
        Contains the traveltime lookup tables for the selected seismic phases, computed
        for some pre-defined velocity model.
    run_path : str
        Points to the top level directory containing all input files, under which the
        specific run directory will be created.
    run_name : str
        Name of the current QuakeMigrate run.
    kwargs : **dict
        See Trigger Attributes for details. In addition to these:
        log : bool, optional
            Toggle for logging. If True, will output to stdout and generate a log file.
            Default is to only output to stdout.
        loglevel : {"info", "debug"}, optional
            Toggle to set the logging level: "debug" will print out additional
            diagnostic information to the log and stdout. (Default "info")
        trigger_name : str
            Optional name of a sub-run - useful when testing different trigger
            parameters, for example.

    Attributes
    ----------
    mad_window_length : float, optional
        Length of window within which to calculate the Median Average
        Deviation. Default: 3600 seconds (1 hour).
    mad_multiplier : float, optional
        A scaling factor for the MAD output to make the calculated MAD factor a
        consistent estimation of the standard deviation of the distribution.
        Default: 1.4826, which is the appropriate scaling factor for a normal
        distribution.
    marginal_window : float, optional
        Half-width of window centred on the maximum coalescence time. The 4-D
        coalescence functioned is marginalised over time across this window such that
        the earthquake location and associated uncertainty can be appropriately
        calculated. It should be an estimate of the time uncertainty in the earthquake
        origin time, which itself is some combination of the expected spatial
        uncertainty and uncertainty in the seismic velocity model used.
        Default: 2 seconds.
    min_event_interval : float, optional
        Minimum time interval between triggered events. Must be at least twice the
        marginal window. Default: 4 seconds.
    normalise_coalescence : bool, optional
        If True, triggering is performed on the maximum coalescence normalised by the
        mean coalescence value in the 3-D grid. Default: False.
    pad : float, optional
        Additional time padding to ensure events close to the starttime/endtime are not
        cut off and missed. Default: 120 seconds.
    plot_trigger_summary : bool, optional
        Plot triggering through time for each batched segment. Default: True.
    run : :class:`~quakemigrate.io.core.Run` object
        Light class encapsulating i/o path information for a given run.
    static_threshold : float, optional
        Static threshold value above which to trigger candidate events.
    threshold_method : str, optional
        Toggle between a "static" threshold and a "dynamic" threshold, based on the
        Median Average Deviation. Default: "static".
    xy_files : str, optional
        Path to comma-separated value file (.csv) containing a series of coordinate
        files to plot. Columns: ["File", "Color", "Linewidth", "Linestyle"], where
        "File" is the absolute path to the file containing the coordinates to be
        plotted. E.g: "/home/user/volcano_outlines.csv,black,0.5,-". Each .csv
        coordinate file should contain coordinates only, with columns: ["Longitude",
        "Latitude"]. E.g.: "-17.5,64.8". Lines pre-pended with ``#`` will be treated as
        a comment - this can be used to include references. See the
        Volcanotectonic_Iceland example XY_files for a template.\n
        .. note:: Do not include a header line in either file.
    plot_all_stns : bool, optional
        If true, plot all stations used for detect. Otherwise, only plot stations which
        for which some data was available during the trigger time window. NOTE: if no
        station availability data is found, all stations in the LUT will be plotted.
        (Default: True)

    Methods
    -------
    trigger(starttime, endtime, region=None, interactive_plot=False)
        Trigger candidate earthquakes from decimated detect scan results.

    Raises
    ------
    ValueError
        If `min_event_interval` < 2 * `marginal_window`.
    InvalidTriggerThresholdMethodException
        If an invalid threshold method is passed in by the user.
    TimeSpanException
        If the user supplies a starttime that is after the endtime.

    """

    def __init__(self, lut, run_path, run_name, **kwargs):
        """Instantiate the Trigger object."""

        self.lut = lut

        # --- Organise i/o and logging ---
        self.run = Run(
            run_path,
            run_name,
            kwargs.get("trigger_name", ""),
            "trigger",
            loglevel=kwargs.get("loglevel", "info"),
        )
        self.run.logger(kwargs.get("log", False))

        # --- Grab Trigger parameters or set defaults ---
        self.threshold_method = kwargs.get("threshold_method", "static")
        if self.threshold_method == "static":
            self.static_threshold = kwargs.get("static_threshold", 1.5)
        elif self.threshold_method == "dynamic":
            self.mad_window_length = kwargs.get("mad_window_length", 3600.0)
            self.mad_multiplier = kwargs.get("mad_multiplier", 1.0)
        else:
            raise util.InvalidTriggerThresholdMethodException
        self.marginal_window = kwargs.get("marginal_window", 2.0)
        self.min_event_interval = kwargs.get("min_event_interval", 4.0)
        if kwargs.get("minimum_repeat"):
            self.minimum_repeat = kwargs.get("minimum_repeat")
        self.normalise_coalescence = kwargs.get("normalise_coalescence", False)
        self.pad = kwargs.get("pad", 120.0)

        # --- Plotting toggles and parameters ---
        self.plot_trigger_summary = kwargs.get("plot_trigger_summary", True)
        self.xy_files = kwargs.get("xy_files")
        self.plot_all_stns = kwargs.get("plot_all_stns", True)

    def __str__(self):
        """Return short summary string of the Trigger object."""

        out = (
            "\tTrigger parameters:\n"
            f"\t\tPre/post pad = {self.pad} s\n"
            f"\t\tMarginal window = {self.marginal_window} s\n"
            f"\t\tMinimum event interval  = {self.min_event_interval} s\n\n"
            f"\t\tTriggering from the "
        )
        out += "normalised " if self.normalise_coalescence else ""
        out += "maximum coalescence trace.\n\n"
        out += f"\t\tTrigger threshold method: {self.threshold_method}\n"
        if self.threshold_method == "static":
            out += f"\t\tStatic threshold = {self.static_threshold}\n"
        elif self.threshold_method == "dynamic":
            out += (
                f"\t\tMAD Window     = {self.mad_window_length}\n"
                f"\t\tMAD Multiplier = {self.mad_multiplier}\n"
            )

        return out

    def trigger(self, starttime, endtime, region=None, interactive_plot=False):
        """
        Trigger candidate earthquakes from decimated scan data.

        Parameters
        ----------
        starttime : str
            Timestamp from which to trigger events.
        endtime : str
            Timestamp up to which to trigger events.
        region : list of floats, optional
            Only retain triggered events located within this region. Format is:
                [Xmin, Ymin, Zmin, Xmax, Ymax, Zmax]
            As longitude / latitude / depth (units corresponding to the lookup table
            grid projection; in positive-down frame).
        interactive_plot : bool, optional
            Toggles whether to produce an interactive plot. Default: False.

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
            self._trigger_batch(batchstart, batchend, region, interactive_plot)
            batchstart = next_day

        logging.info(util.log_spacer)

    def _trigger_batch(self, batchstart, batchend, region, interactive_plot):
        """
        Wraps all of the methods used in sequence to determine triggers.

        Parameters
        ----------
        batchstart : `obspy.UTCDateTime` object
            Timestamp from which to trigger events.
        batchend : `obspy.UTCDateTime` object
            Timestamp up to which to trigger events.
        region : list of floats
            Only retain triggered events located within this region. Format is:
                [Xmin, Ymin, Zmin, Xmax, Ymax, Zmax]
            As longitude / latitude / depth (units corresponding to the lookup table
            grid projection; in positive-down frame).
        interactive_plot : bool
            Toggles whether to produce an interactive plot. Default: False.

        """

        logging.info("\tReading in .scanmseed...")
        data, stats = read_scanmseed(
            self.run, batchstart, batchend, self.pad, self.lut.unit_conversion_factor
        )

        logging.info("\n\tTriggering events...")
        trigger_on = "COA_N" if self.normalise_coalescence else "COA"
        threshold = self._get_threshold(data[trigger_on], stats.sampling_rate)
        candidate_events = self._identify_candidates(data, trigger_on, threshold)

        if candidate_events.empty:
            logging.info(
                "\tNo events triggered at this threshold - try a lower detection "
                "threshold."
            )
            events = candidate_events
            discarded = candidate_events
        else:
            refined_events = self._refine_candidates(candidate_events)
            logging.debug(refined_events)
            events = self._filter_events(refined_events, batchstart, batchend, region)
            logging.debug(events)
            discarded = refined_events[
                ~refined_events.index.isin(events.index)
            ].dropna()
            logging.debug(discarded)
            logging.info(
                f"\n\t\t{len(events)} event(s) triggered within the specified region "
                f"between {batchstart} \n\t\tand {batchend}"
            )
            logging.info("\n\tWriting triggered events to file...")
            write_triggered_events(self.run, events, batchstart)

        if self.plot_trigger_summary:
            logging.info("\n\tPlotting trigger summary...")
            trigger_summary(
                events,
                batchstart,
                batchend,
                self.run,
                self.marginal_window,
                self.min_event_interval,
                threshold,
                self.normalise_coalescence,
                self.lut,
                data,
                region,
                discarded,
                interactive=interactive_plot,
                xy_files=self.xy_files,
                plot_all_stns=self.plot_all_stns,
            )

    @util.timeit()
    def _get_threshold(self, scandata, sampling_rate):
        """
        Determine the threshold to use when triggering candidate events.

        Parameters
        ----------
        scandata : `pandas.Series` object
            (Normalised) coalescence values for which to calculate the threshold.
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
            breaks = breaks[breaks % int(self.mad_window_length * sampling_rate) == 0][
                1:
            ]
            chunks = np.split(scandata.values, breaks)

            # Calculate the mad and median values
            mad_values = np.asarray([util.calculate_mad(chunk) for chunk in chunks])
            median_values = np.asarray([np.median(chunk) for chunk in chunks])
            mad_trace = chunks2trace(mad_values, (len(chunks), len(chunks[0])))
            median_trace = chunks2trace(median_values, (len(chunks), len(chunks[0])))
            mad_trace = mad_trace[: len(scandata)]
            median_trace = median_trace[: len(scandata)]

            # Set the dynamic threshold
            threshold = median_trace + (mad_trace * self.mad_multiplier)
        else:
            # Set static threshold
            threshold = np.zeros_like(scandata) + self.static_threshold

        return threshold

    @util.timeit()
    def _identify_candidates(self, scandata, trigger_on, threshold):
        """
        Identify distinct periods of time for which the maximum (normalised) coalescence
        trace exceeds the chosen threshold.

        Parameters
        ----------
        scandata : `pandas.DataFrame` object
            Data output by detect() -- decimated scan.
            Columns: ["DT", "COA", "COA_N", "X", "Y", "Z"] - X/Y/Z as lon/lat/
            z-units corresponding to the units of the lookup table grid projection.
        trigger_on : str
            Specifies the maximum coalescence data on which to trigger events.
        threshold : `numpy.ndarray` object
            Array of threshold values.

        Returns
        -------
        triggers : `pandas.DataFrame` object
            Candidate events exceeding some threshold. Columns: ["EventNum", "CoaTime",
            "TRIG_COA", "COA_X", "COA_Y", "COA_Z", "MinTime",  "MaxTime", "COA",
            "COA_NORM"]

        """

        # Switch between user-facing minimum event interval definition (minimum interval
        # between event triggers) and internal definition (extra buffer on top of
        # marginal window within which events cannot overlap)
        min_event_interval = self.min_event_interval - self.marginal_window

        thresholded = scandata[scandata[trigger_on] >= threshold]
        r = np.arange(len(thresholded))
        candidates = [d for _, d in thresholded.groupby(thresholded.index - r)]

        triggers = pd.DataFrame(columns=CANDIDATES_COLS)
        for i, candidate in enumerate(candidates):
            peak = candidate.loc[candidate[trigger_on].idxmax()]

            # If first sample above threshold is within the marginal window
            if (peak["DT"] - candidate["DT"].iloc[0]) < self.marginal_window:
                min_dt = peak["DT"] - self.min_event_interval
            # Otherwise just subtract the minimum event interval
            else:
                min_dt = candidate["DT"].iloc[0] - min_event_interval

            # If last sample above threshold is within the marginal window
            if (candidate["DT"].iloc[-1] - peak["DT"]) < self.marginal_window:
                max_dt = peak["DT"] + self.min_event_interval
            # Otherwise just add the minimum event interval
            else:
                max_dt = candidate["DT"].iloc[-1] + min_event_interval

            trigger = pd.Series(
                [
                    i,
                    peak["DT"],
                    peak[trigger_on],
                    peak["X"],
                    peak["Y"],
                    peak["Z"],
                    min_dt,
                    max_dt,
                    peak["COA"],
                    peak["COA_N"],
                ],
                index=CANDIDATES_COLS,
            )

            triggers = pd.concat(
                [triggers, trigger.to_frame().T.convert_dtypes()], ignore_index=True
            )

        return triggers

    @util.timeit()
    def _refine_candidates(self, candidate_events):
        """
        Merge candidate events for which the marginal windows overlap with the minimum
        inter-event time.

        Parameters
        ----------
        candidate_events : `pandas.DataFrame` object
            Candidate events corresponding to periods of time in which the
            coalescence signal exceeds some threshold. Columns: ["EventNum", "CoaTime",
            "TRIG_COA", "COA_X", "COA_Y", "COA_Z", "MinTime", "MaxTime", "COA",
            "COA_NORM"]

        Returns
        -------
        events : `pandas.DataFrame` object
            Merged events with some minimum inter-event spacing in time.
            Columns: ["EventID", "CoaTime", "TRIG_COA", "COA_X", "COA_Y", "COA_Z",
            "MinTime", "MaxTime", "COA", "COA_NORM"].

        """

        # Iterate pairwise (event1, event2) over the candidate events to identify
        # overlaps between:
        #   - event1 marginal window and event2 minimum window position
        #   - event2 marginal window and event1 maximum window position
        event_count = 1
        for i, event1 in candidate_events.iterrows():
            candidate_events.loc[i, "EventNum"] = event_count
            if i + 1 == len(candidate_events):
                continue
            event2 = candidate_events.iloc[i + 1]
            if all(
                [
                    event1["MaxTime"] < event2["CoaTime"] - self.marginal_window,
                    event2["MinTime"] > event1["CoaTime"] + self.marginal_window,
                ]
            ):
                event_count += 1

        # Split into DataFrames by event number
        merged_candidates = [
            d for _, d in candidate_events.groupby(candidate_events["EventNum"])
        ]

        # Update the min/max window times and build final event DataFrame
        refined_events = pd.DataFrame(columns=REFINED_EVENTS_COLS)
        for i, candidate in enumerate(merged_candidates):
            logging.debug(f"\t    Triggered event {i+1} of {len(merged_candidates)}")
            event = candidate.loc[candidate["TRIG_COA"].idxmax()].copy()
            event["MinTime"] = candidate["MinTime"].min()
            event["MaxTime"] = candidate["MaxTime"].max()

            # Add unique identifier
            event_uid = str(event["CoaTime"])
            for char_ in ["-", ":", ".", " ", "Z", "T"]:
                event_uid = event_uid.replace(char_, "")
            event_uid = event_uid[:17].ljust(17, "0")
            event["EventID"] = event_uid

            refined_events = pd.concat(
                [refined_events, event.to_frame().T.convert_dtypes()], ignore_index=True
            )

        return refined_events

    @util.timeit()
    def _filter_events(self, events, starttime, endtime, region):
        """
        Remove events within the padding time and/or within a specific geographical
        region. Also adds a unique event identifier based on the coalescence time.

        Parameters
        ----------
        events : `pandas.DataFrame` object
            Refined set of events to be filtered. Columns: ["EventID", "CoaTime",
            "TRIG_COA", "COA_X", "COA_Y", "COA_Z", "MinTime", "MaxTime", "COA",
            "COA_NORM"].
        starttime : `obspy.UTCDateTime` object
            Timestamp from which to trigger events.
        endtime : `obspy.UTCDateTime` object
            Timestamp up to which to trigger events.
        region : list of floats
            Only retain triggered events located within this region. Format is:
                [Xmin, Ymin, Zmin, Xmax, Ymax, Zmax]
            As longitude / latitude / depth (units corresponding to the lookup table
            grid projection; in positive-down frame).

        Returns
        -------
        events : `pandas.DataFrame` object
            Final set of triggered events. Columns: ["EventID", "CoaTime", "TRIG_COA",
            "COA_X", "COA_Y", "COA_Z", "MinTime", "MaxTime", "COA", "COA_NORM"].

        """

        # Remove events which occur in the pre-pad and post-pad:
        events = events.loc[
            (events["CoaTime"] >= starttime) & (events["CoaTime"] < endtime), :
        ].copy()

        if region is not None:
            events = events.loc[
                (events["COA_X"] >= region[0])
                & (events["COA_Y"] >= region[1])
                & (events["COA_Z"] >= region[2])
                & (events["COA_X"] <= region[3])
                & (events["COA_Y"] <= region[4])
                & (events["COA_Z"] <= region[5]),
                :,
            ].copy()

        return events

    @property
    def min_event_interval(self):
        """Get and set the minimum event interval."""

        return self._min_event_interval

    @min_event_interval.setter
    def min_event_interval(self, value):
        if value < 2 * self.marginal_window:
            raise ValueError("\tMinimum event interval must be >= 2 * marginal window.")
        else:
            self._min_event_interval = value

    # --- Deprecation/Future handling ---
    @property
    def minimum_repeat(self):
        """Handler for deprecated attribute name 'minimum_repeat'."""

        return self._min_event_interval

    @minimum_repeat.setter
    def minimum_repeat(self, value):
        if value < 2 * self.marginal_window:
            raise ValueError("\tMinimum repeat must be >= 2 * marginal window.")
        print(
            "FutureWarning: Parameter name has changed - continuing.\n"
            "To remove this message, change:\n"
            "\t'minimum_repeat' -> 'min_event_interval'"
        )
        self._min_event_interval = value
