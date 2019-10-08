# -*- coding: utf-8 -*-
"""
Module to perform the trigger stage of QuakeMigrate.

"""

import numpy as np
from obspy import UTCDateTime
import pandas as pd

import QMigrate.io.quakeio as qio
import QMigrate.plot.triggered_events as tplot


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

    def __init__(self, output_path, output_name, stations, log=False):
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

        static_threshold : boolean
            whether to use a static threshold, or dynamic based on current noise

        dynamic_offset : [float, float]
            float1 : number of minutes on which to calculate current noise, 
            expressed as 99th percentile of coalescence stream
            float2: number on which to vertically offset the dynamic threshold from the 99th percentile

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
        self.detection_threshold = 1.5

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

        self.static_threshold = True

    def trigger(self, start_time, end_time, savefig=True):
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
        msg += "         Marginal window           = {} s\n"
        msg += "         Minimum repeat            = {} s\n\n"
        msg += "         Trigger from normalised coalescence - {}\n\n"
        msg += "=" * 120
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
                               detection_threshold=self.detection_threshold,
                               normalise_coalescence=self.normalise_coalescence,
                               log=self.log, data=self.coa_data,
                               stations=self.stations, savefig=savefig)

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

        if self.normalise_coalescence is True: ##
            coaString = "COA_N" ##
        else:
            coaString = "COA"

        # Mask based on first and final time stamps and detection threshold
        if self.static_threshold: #BQL


            coa_data = self.coa_data[self.coa_data[coaString] >=
                                     self.detection_threshold]
            coa_data = coa_data[(coa_data["DT"] >= start_time) &
                                (coa_data["DT"] <= end_time)]
            self.coa_data["Th"]= np.ones(self.coa_data[coaString].shape)*self.detection_threshold

        ### BQL ###
        else: #dynamic threshold by splitting
            coa_data = self.coa_data
            tChunk = self.dynamic_offset[0] * 60 #n min chunks
            nChunks = (len(coa_data[coaString]) / self.sampling_rate) / tChunk
            #import pdb; pdb.set_trace()
            inds = np.floor(np.linspace(0,len(coa_data[coaString]) - (tChunk*self.sampling_rate),nChunks)).astype(int)
            inde = inds.copy()
            inde[:-1] = inds[1:]
            inde[-1] = len(coa_data[coaString]) 
            correctedThreshold = np.zeros(coa_data[coaString].shape)

            for ii in range(0,len(inds)):
                correctedThreshold[inds[ii]:inde[ii]] = np.ones(correctedThreshold[inds[ii]:inde[ii]].shape)*np.power(np.percentile(coa_data[coaString].iloc[inds[ii]:inde[ii]],95),self.dynamic_offset[1])
            self.coa_data["Th"] = correctedThreshold
            coa_data = coa_data[np.greater(coa_data[coaString],correctedThreshold)]
            coa_data = coa_data[(coa_data["DT"] >= start_time) &
                (coa_data["DT"] <= end_time)]
            #import pdb; pdb.set_trace()
        ### BQL ###

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
                        d = len(coa_data) - 1 #BQL change from "d = len(coa_data)" to "d = len(coa_data) - 1" 
                        break
                min_idx = c
                max_idx = d
                #val_idx = np.argmax(coa_data["COA"].iloc[np.arange(c, d + 1)])
                if self.normalise_coalescence is True:
                    val_idx = np.argmax(coa_data["COA_N"].iloc[np.arange(c, d+1)]) 
                else:
                    val_idx = np.argmax(coa_data["COA"].iloc[np.arange(c, d+1)]) 
            except IndexError:
                # Handling for last sample if it is a single sample above
                # threshold
                min_idx = max_idx = val_idx = c

            # Determining the times for min, max and max coalescence value
            t_min = coa_data["DT"].iloc[min_idx]
            #import pdb; pdb.set_trace()
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
        # Reset EventNum column
        events.loc[:, "EventNum"] = np.arange(1, len(events) + 1)

        evt_id = events["CoaTime"].astype(str)
        for char_ in ["-", ":", ".", " ", "Z", "T"]:
            evt_id = evt_id.str.replace(char_, "")
        events["EventID"] = evt_id

        if len(events) == 0:
            events = None

        return events
