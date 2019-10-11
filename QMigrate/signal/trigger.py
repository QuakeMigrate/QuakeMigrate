# -*- coding: utf-8 -*-
"""
Module to perform the trigger stage of QuakeMigrate.

"""

import numpy as np
from obspy import UTCDateTime
import pandas as pd

import QMigrate.io.quakeio as qio
import QMigrate.plot.triggered_events as tplot

def _contains_nan(a, nan_policy='propagate'):
    policies = ['propagate', 'raise', 'omit']
    if nan_policy not in policies:
        raise ValueError("nan_policy must be one of {%s}" %
                         ', '.join("'%s'" % s for s in policies))
    try:
        # Calling np.sum to avoid creating a huge array into memory
        # e.g. np.isnan(a).any()
        with np.errstate(invalid='ignore'):
            contains_nan = np.isnan(np.sum(a))
    except TypeError:
        # This can happen when attempting to sum things which are not
        # numbers (e.g. as in the function `mode`). Try an alternative method:
        try:
            contains_nan = np.nan in set(a.ravel())
        except TypeError:
            # Don't know what to do. Fall back to omitting nan values and
            # issue a warning.
            contains_nan = False
            nan_policy = 'omit'
            warnings.warn("The input array could not be properly checked for nan "
                          "values. nan values will be ignored.", RuntimeWarning)

    if contains_nan and nan_policy == 'raise':
        raise ValueError("The input contains nan values")

    return (contains_nan, nan_policy)

def mad(x, axis=0, center=np.median, scale=1.4826,
                              nan_policy='propagate'):

    # from scipy v1.3 source
    x = np.asarray(x)

    # Consistent with `np.var` and `np.std`.
    if not x.size:
        return np.nan

    contains_nan, nan_policy = _contains_nan(x, nan_policy)

    if contains_nan and nan_policy == 'propagate':
        return np.nan

    if contains_nan and nan_policy == 'omit':
        # Way faster than carrying the masks around
        arr = np.ma.masked_invalid(x).compressed()
    else:
        arr = x

    if axis is None:
        med = center(arr)
        mad = np.median(np.abs(arr - med))
    else:
        med = np.apply_over_axes(center, arr, axis)
        mad = np.median(np.abs(arr - med), axis=axis)

    return scale * mad

def mad_chunk(chunks, axis=None):
    mads = np.asarray([mad(chunk, scale=1.) for chunk in chunks])
    means = np.asarray([np.mean(chunk) for chunk in chunks])
    return mads, means
def chunks_to_trace(a, new_shape):
    b = np.broadcast_to(a[:, None], new_shape)
    b = np.reshape(b, np.product(new_shape))
    return b
class Trigger:
    """
    QuakeMigrate triggering class

    Triggers candidate earthquakes from the maximum coalescence through time
    data output by the decimated detect scan, ready to be run through locate().

    Methods
    -------
    trigger_scn()
        Trigger candidate earthquakes from decimated detect scan results.

    """

    def __init__(self, output_path, output_name, stations, log=False, dynamic_thresh=False):
        """
        Class initialisation method.

        Parameters
        ----------
        output_path : str
            Path to output location

        output_name : str
            Name of run

        stations : pandas DataFrame
            Station information.
            Columns (in any order): ["Latitude", "Longitude", "Elevation",
                                     "Name"]

        start_time : str
            Time stamp of first sample

        end_time : str
            Time stamp of final sample

        detection_threshold : float, optional
            Coalescence value above which to trigger events

        normalise_coalescence : bool, optional
            If True, use the max coalescence normalised by the average
            coalescence value in the 3-D grid at each time step

        marginal_window : float, optional
            Estimate of time error (derived from estimate of spatial error and
            seismic velocity) over which to marginalise the coalescence

        minimum_repeat : float, optional
            Minimum time interval between triggers

        pad : float, optional
            Trigger will attempt to read in coastream data from start_time - pad
            to end_time + pad. Events will only be triggered if the origin time
            occurs within the trigger window. Default = 120 seconds

        sampling_rate : int
            Sampling rate in hertz

        coa_data : pandas DataFrame
            Data output by detect() -- decimated scan
            Columns: ["COA", "COA_N", "X", "Y", "Z"]

        """

        if output_path is not None:
            self.output = qio.QuakeIO(output_path, output_name, log)
        else:
            self.output = None

        self.log = log

        self.stations = stations

        self.start_time = None
        self.end_time = None

        # Detection threshold above which to trigger events
        if not dynamic_thresh:
            self.detection_threshold = 1.5
        else:
            self.detection_threshold = 'dynamic'
            self.mad_window_length = 3600. # 1 hour
            self.mad_multiplier = 6.

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

    def trigger(self, start_time, end_time, region=None, savefig=True):
        """
        Trigger candidate earthquakes from decimated scan data.

        Parameters
        ----------
        start_time : str
            Time stamp of first sample

        end_time : str
            Time stamp of final sample

        save_fig : bool, optional
            Save triggered events figure (default) or open for interactive view

        """

        # Convert times to UTCDateTime objects
        self.start_time = UTCDateTime(start_time)
        self.end_time = UTCDateTime(end_time)
        self.region   = region

        if self.minimum_repeat < 2 * self.marginal_window:
            msg = "\tMinimum repeat must be >= 2 * marginal window."
            raise Exception(msg)

        msg = "=" * 120 + "\n"
        msg += "   TRIGGER - Triggering events from coalescence\n"
        msg += "=" * 120 + "\n\n"
        msg += "   Parameters specified:\n"
        msg += "         Start time                = {}\n"
        msg += "         End   time                = {}\n"
        msg += "         Pre/post pad              = {} s\n\n"
        msg += "         Detection threshold       = {}\n"
        if self.detection_threshold == 'dynamic':
            msg += "         MAD Window                = {}\n"
            msg += "         MAD Multiplier            = {}\n"
        msg += "         Marginal window           = {} s\n"
        msg += "         Minimum repeat            = {} s\n\n"
        msg += "         Trigger from normalised coalescence - {}\n\n"
        msg += "=" * 120
        if self.detection_threshold == 'dynamic':
            msg = msg.format(str(self.start_time), str(self.end_time),
                            str(self.pad), self.detection_threshold,
                            self.mad_window_length, self.mad_multiplier,
                            self.marginal_window, self.minimum_repeat,
                            self.normalise_coalescence)  
        else:
            msg = msg.format(str(self.start_time), str(self.end_time),
                            str(self.pad), self.detection_threshold,
                            self.marginal_window, self.minimum_repeat,
                            self.normalise_coalescence)
        self.output.log(msg, self.log)

        self.output.log("    Reading in scanmseed...\n", self.log)
        read_start = self.start_time - self.pad
        read_end = self.end_time + self.pad
        self.coa_data, coa_stats = self.output.read_coastream(read_start,
                                                              read_end)

        # Check if coa_data found by read_coastream covers whole trigger period
        msg = ""
        if coa_stats.starttime > self.start_time:
            msg += "\tWarning! scanmseed data start is after trigger() start_time!\n"
        elif coa_stats.starttime > read_start:
            msg += "\tWarning! No scanmseed data found for pre-pad!\n"
        if coa_stats.endtime < self.end_time - 1 / coa_stats.sampling_rate:
            msg += "\tWarning! scanmseed data end is before trigger() end_time!\n"
        elif coa_stats.endtime < read_end:
            msg += "\tWarning! No scanmseed data found for post-pad!\n"
        self.output.log(msg, self.log)
        self.output.log("    scanmseed read complete.", self.log)

        self.sampling_rate = coa_stats.sampling_rate

        self.events = self._trigger_scn()

        if self.events is None:
            msg = "\tNo events triggered at this threshold - "
            msg += "try reducing the detection threshold."
            self.output.log(msg, self.log)
        else:
            self.output.write_triggered_events(self.events, self.start_time,
                                               self.end_time)

        tplot.triggered_events(events=self.events, start_time=self.start_time,
                               end_time=self.end_time, output=self.output,
                               marginal_window=self.marginal_window,
                               detection_threshold=self.threshold,
                               normalise_coalescence=self.normalise_coalescence,
                               log=self.log, data=self.coa_data,
                               region=self.region, stations=self.stations, savefig=savefig)

        self.output.log("=" * 120, self.log)

    def _trigger_scn(self):
        """
        Function to perform the triggering on the maximum coalescence through
        time data.

        Returns
        -------
        events : pandas DataFrame
            Triggered events information. Columns: ["EventNum", "CoaTime",
                                                    "COA_V", "COA_X", "COA_Y",
                                                    "COA_Z", "MinTime",
                                                    "MaxTime"]

        """

        # switch between user-facing minimum repeat definition (minimum repeat
        # interval between event triggers) and internal definition (extra
        # buffer on top of marginal window within which events can't overlap)
        minimum_repeat = self.minimum_repeat - self.marginal_window

        start_time = self.start_time - self.pad
        end_time = self.end_time + self.pad

        # grab the correct coalesence trace
        if self.normalise_coalescence:
            coa_in = self.coa_data["COA_N"]
        else:
            coa_in = self.coa_data["COA"]

        # calculate the dynamic threshold - if required
        if self.detection_threshold == 'dynamic':
            breaks = np.array(range(len(coa_in)))
            breaks = breaks[breaks % int(self.mad_window_length * self.sampling_rate) == 0][1:]
            # split the data in window_length chunks
            chunks = np.split(coa_in.values, breaks)
            mad_values, mean_values = mad_chunk(chunks)
            mad_trace = chunks_to_trace(mad_values, (len(chunks), len(chunks[0])))
            mean_trace = chunks_to_trace(mean_values, (len(chunks), len(chunks[0])))
            print(mad_trace.shape, coa_in.values.shape)
            mad_trace = mad_trace[:len(coa_in)]
            mean_trace = mean_trace[:len(coa_in)]
            print(mad_trace.shape, mean_trace.shape)
            self.threshold = mean_trace + (mad_trace * self.mad_multiplier)
        else:
            self.threshold = np.zeros_like(coa_in) + self.detection_threshold

        # Mask based on first and final time stamps and detection threshold
        coa_data = self.coa_data[coa_in >= self.threshold]
        coa_data = coa_data[(coa_data["DT"] >= start_time) &
                            (coa_data["DT"] <= end_time)]

        coa_data = coa_data.reset_index(drop=True)

        if len(coa_data) == 0:
            return None

        event_cols = ["EventNum", "CoaTime", "COA_V", "COA_X", "COA_Y",
                      "COA_Z", "MinTime", "MaxTime"]

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
            val_idx = np.argmax(coa_data["COA"].iloc[np.arange(min_idx, max_idx+1)])

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
                                t_min, t_max]],
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
        self.output.log("\n    Triggering...", self.log)
        for i in range(1, count + 1):
            self.output.log("\tTriggered event {} of {}".format(i, count),
                            self.log)
            tmp = init_events[init_events["EventNum"] == i]
            tmp = tmp.reset_index(drop=True)
            j = np.argmax(tmp["COA_V"])
            min_mt = np.min(tmp["MinTime"])
            max_mt = np.max(tmp["MaxTime"])
            event = pd.DataFrame([[i, tmp["CoaTime"].iloc[j],
                                   tmp["COA_V"].iloc[j],
                                   tmp["COA_X"].iloc[j],
                                   tmp["COA_Y"].iloc[j],
                                   tmp["COA_Z"].iloc[j],
                                   min_mt,
                                   max_mt]],
                                 columns=event_cols)
            events = events.append(event, ignore_index=True)

        # Remove events which occur in the pre-pad and post-pad:
        events = events[(events["CoaTime"] >= self.start_time) &
                        (events["CoaTime"] < self.end_time)]
        
        if self.region != None:
            events = events[(events['COA_X'] >= self.region[0]) &
                            (events['COA_Y'] >= self.region[1]) &
                            (events['COA_Z'] >= self.region[2]) &
                            (events['COA_X'] <= self.region[3]) &
                            (events['COA_Y'] <= self.region[4]) &
                            (events['COA_Z'] <= self.region[5])]

        # Reset EventNum column
        events.loc[:, "EventNum"] = np.arange(1, len(events) + 1)

        evt_id = events["CoaTime"].astype(str)
        for char_ in ["-", ":", ".", " ", "Z", "T"]:
            evt_id = evt_id.str.replace(char_, "")
        events["EventID"] = evt_id

        if len(events) == 0:
            events = None

        return events
