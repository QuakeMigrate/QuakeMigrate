# -*- coding: utf-8 -*-
"""
Module to perform the trigger stage of QuakeMigrate

"""

import numpy as np
from obspy import UTCDateTime
import pandas as pd

import QMigrate.io.quakeio as qio
import QMigrate.plot.triggered_events as tplot


def trigger(start_time, end_time, output_path, output_name, marginal_window,
            detection_threshold, normalise_coalescence, minimum_repeat,
            sampling_rate, stations, savefig=True, log=False):
    """
    Scans through the output from detect for peaks that exceed some detection
    threshold

    Parameters
    ----------
    start_time : str
        Time stamp of first sample

    end_time : str
        Time stamp of final sample

    output_path : str
        Path to output location

    output_name : str
        Name of run

    marginal_window : float
        Estimate of time error over which to marginalise the coalescence

    detection_threshold : float
        Coalescence value above which to trigger events

    normalise_coalescence : bool
        If True, use the coalescence normalised by the average background noise

    minimum_repeat : float
        Prior knowledge of event spacing in time

    sampling_rate : int
        Sampling rate in hertz

    stations : dict
        Station location information

    savefig : bool, optional
        Saves plots if True

    log : bool, optional
        Output processing to a log file

    """

    # Convert times to UTCDateTime objects
    start_time = UTCDateTime(start_time)
    end_time = UTCDateTime(end_time)

    if output_path is not None:
        output = qio.QuakeIO(output_path, output_name)
    else:
        output = None

    msg = "=" * 120 + "\n"
    msg += "   TRIGGER - Triggering events from coalescence\n"
    msg += "=" * 120 + "\n\n"
    msg += "   Parameters specified:\n"
    msg += "         Start time                = {}\n"
    msg += "         End   time                = {}\n\n"
    msg += "         Detection threshold       = {}\n"
    msg += "         Marginal window           = {} s\n"
    msg += "         Minimum repeat            = {} s\n\n"
    msg += "         Trigger normalised coalescence - {}\n\n"
    msg += "=" * 120
    msg = msg.format(str(start_time), str(end_time), detection_threshold,
                     marginal_window, minimum_repeat, normalise_coalescence)

    print(msg)

    if minimum_repeat < marginal_window:
        msg = "    Minimum repeat must be <= to marginal window."
        raise Exception(msg)

    # Intial detection of the events from .scn file
    print("    Reading in scan mSEED...")
    coa_data, coa_stats = output.read_decscan()
    print("    Scan mSEED read complete.")
    events = _trigger_scn(coa_data, start_time, end_time, marginal_window,
                          detection_threshold, normalise_coalescence,
                          sampling_rate, minimum_repeat)

    if events is None:
        msg = "    No events triggered at this threshold - "
        msg += "try reducing the threshold value."
        print(msg)
    else:
        output.write_triggered_events(events)

    tplot.triggered_events(events=events, start_time=start_time,
                           end_time=end_time, output=output,
                           marginal_window=marginal_window,
                           detection_threshold=detection_threshold,
                           normalise_coalescence=normalise_coalescence,
                           data=coa_data, stations=stations, savefig=savefig)

    print("=" * 120)


def _trigger_scn(coa_data, start_time, end_time, marginal_window,
                 detection_threshold, normalise_coalescence,
                 sampling_rate, minimum_repeat):
    """Function to perform the actual triggering"""

    if normalise_coalescence is True:
        coa_data["COA"] = coa_data["COA_N"]

    # Mask based on first and final time stamps and detection threshold
    coa_data = coa_data[coa_data["COA"] >= detection_threshold]
    coa_data = coa_data[(coa_data["DT"] >= start_time) &
                        (coa_data["DT"] <= end_time)]

    coa_data = coa_data.reset_index(drop=True)

    if len(coa_data) == 0:
        msg = "    No events triggered at this threshold"
        print(msg)
        return None

    event_cols = ["EventNum", "CoaTime", "COA_V", "COA_X", "COA_Y",
                  "COA_Z", "MinTime", "MaxTime"]

    ss = 1 / sampling_rate

    # Determine the initial triggered events
    init_events = pd.DataFrame(columns=event_cols)
    c = 0
    e = 1
    while c <= len(coa_data) - 1:
        # Determining the index when above the level and maximum value
        d = c

        try:
            # Check if the next sample in the list has the correct time stamp
            while coa_data["DT"].iloc[d] + ss == coa_data["DT"].iloc[d + 1]:
                d += 1
                if d + 1 >= len(coa_data):
                    d = len(coa_data)
                    break
            min_idx = c
            max_idx = d
            val_idx = np.argmax(coa_data["COA"].iloc[np.arange(c, d + 1)])
        except IndexError:
            # Handling for last sample if it is a single sample above threshold
            min_idx = max_idx = val_idx = c

        # Determining the times for min, max and max coalescence value
        t_min = coa_data["DT"].iloc[min_idx]
        t_max = coa_data["DT"].iloc[max_idx]
        t_val = coa_data["DT"].iloc[val_idx]

        COA_V = coa_data["COA"].iloc[val_idx]
        COA_X = coa_data["X"].iloc[val_idx]
        COA_Y = coa_data["Y"].iloc[val_idx]
        COA_Z = coa_data["Z"].iloc[val_idx]

        # If the first sample above the threshold is within the marginal
        # window, set min time stamp the maximum value time - marginal window
        # - minimum repeat
        if (t_val - t_min) < marginal_window:
            t_min = t_val - marginal_window - minimum_repeat
        # If the first sample is outwith, just subtract the minimum repeat
        else:
            t_min = t_min - minimum_repeat

        # If the final sample above the threshold is within the marginal
        # window,  set the max time stamp to the maximum value time + marginal
        # window + minimum repeat
        if (t_max - t_val) < marginal_window:
            t_max = t_val + marginal_window + minimum_repeat
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
    # between the final sample in the event window and the edge of the marginal
    # window of the next. If so, treat them as the same event
    count = 1
    for i, event in init_events.iterrows():
        evt_num[i] = count
        if (i + 1 < n_evts) and ((event["MaxTime"]
                                  - (init_events["CoaTime"].iloc[i + 1]
                                  - marginal_window)) < 0):
            count += 1
    init_events["EventNum"] = evt_num

    events = pd.DataFrame(columns=event_cols)
    print("\n    Triggering...")
    for i in range(1, count + 1):
        print("\tTriggered event {} of {}".format(i, count))
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

    evt_id = events["CoaTime"].astype(str)
    for char_ in ["-", ":", ".", " ", "Z", "T"]:
        evt_id = evt_id.str.replace(char_, "")
    events["EventID"] = evt_id

    if len(events) == 0:
        events = None

    return events


def stations(path, units, delimiter=","):
    """
    Reads station information from file

    Parameters
    ----------
    path : str
        Location of file containing station information
    delimiter : char, optional
        Station file delimiter, defaults to ","
    units : str

    """

    stats = pd.read_csv(path, delimiter=delimiter).values

    stn_data = {}
    if units == "lon_lat_elev":
        stn_lon = stats[:, 0]
        stn_lat = stats[:, 1]
    elif units == "lat_lon_elev":
        stn_lon = stats[:, 1]
        stn_lat = stats[:, 0]

    stn_data["Longitude"] = stn_lon
    stn_data["Latitude"] = stn_lat
    stn_data["Elevation"] = stats[:, 2]
    stn_data["Name"] = stats[:, 3]

    return stn_data
