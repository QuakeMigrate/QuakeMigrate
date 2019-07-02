
# -*- coding: utf-8 -*-
"""
Module to plot the triggered events on a decimated grid.

"""

import os

import matplotlib
try:
    os.environ["DISPLAY"]
    matplotlib.use("Qt5Agg")
except KeyError:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from obspy import UTCDateTime
import pandas as pd

import QMigrate.util as util


def triggered_events(events, start_time, end_time, output, marginal_window,
                     detection_threshold, normalise_coalescence, log,
                     data=None, stations=None, savefig=False):
    """
    Plots the data from a .scanmseed file with annotations illustrating the
    trigger results: event triggers and marginal windows on the coalescence
    traces, and map and cross section view of the gridded triggered earthquake
    locations.

    Parameters
    ----------
    events : pandas DataFrame
        Triggered events output from _trigger_scn().
        Columns: ["EventNum", "CoaTime", "COA_V", "COA_X", "COA_Y", "COA_Z",
                 "MinTime", "MaxTime"]

    start_time : UTCDateTime
        Start time of trigger run.

    end_time : UTCDateTime
        End time of trigger run

    output : QuakeIO class
        QuakeIO class initialised with output path and output name

    marginal_window : float
        Estimate of time error over which to marginalise the coalescence

    detection_threshold : float
        Coalescence value above which to trigger events

    normalise_coalescence : bool
        If True, use the coalescence normalised by the average background noise

    data : pandas DataFrame
        Data output by detect() -- decimated scan
        Columns: ["COA", "COA_N", "X", "Y", "Z"]

    stations : pandas DataFrame
        Station information.
        Columns (in any order): ["Latitude", "Longitude", "Elevation", "Name"]

    savefig : bool, optional
        Output the plot as a file. The plot is shown by default, and not saved.

    """

    if data is None:
        data, coa_stats = output.read_coastream(start_time, end_time)
        del coa_stats

    output.log("\n\tPlotting triggered events on decimated grid...", log)
    data["DT"] = pd.to_datetime(data["DT"].astype(str))

    fig = plt.figure(figsize=(30, 15))
    fig.patch.set_facecolor("white")
    coa = plt.subplot2grid((6, 16), (0, 0), colspan=9, rowspan=2)
    coa_norm = plt.subplot2grid((6, 16), (2, 0), colspan=9, rowspan=2,
                                sharex=coa)
    numstations = plt.subplot2grid((6, 16), (4, 0), colspan=9, rowspan=2,
                                   sharex=coa)
    xy = plt.subplot2grid((6, 16), (0, 10), colspan=4, rowspan=4)
    xz = plt.subplot2grid((6, 16), (4, 10), colspan=4, rowspan=2,
                          sharex=xy)
    yz = plt.subplot2grid((6, 16), (0, 14), colspan=2, rowspan=4,
                          sharey=xy)

    coa.plot(data["DT"], data["COA"], color="blue", zorder=10,
             label="Maximum coalescence", linewidth=0.2)
    coa.get_xaxis().set_ticks([])
    coa_norm.plot(data["DT"], data["COA_N"], color="blue", zorder=10,
                  label="Maximum coalescence", linewidth=0.2)

    try:
        stn_ava_data = output.read_stn_availability(start_time,
                                                    end_time).sum(axis=1).astype(int)
        a_times = [UTCDateTime(x).datetime for x in stn_ava_data.index.values]

        # Handle last step
        stn_ava_data = stn_ava_data.values
        stn_ava_data = np.append(stn_ava_data, [stn_ava_data[-1]])
        a_times.append(end_time.datetime)
        numstations.step(a_times, stn_ava_data, color="green", where="post")
        numstations.set_ylim(top=len(stations["Name"]) + 1)
    except util.NoStationAvailabilityDataException as e:
        output.log(e.msg, log)

    if events is not None:
        for i, event in events.iterrows():
            if i == 0:
                label1 = "Minimum repeat window"
                label2 = "Marginal window"
                label3 = "Detected events"
            else:
                label1 = ""
                label2 = ""
                label3 = ""

            for plot in [coa, coa_norm]:
                plot.axvspan((event["MinTime"]).datetime,
                             (event["MaxTime"]).datetime,
                             label=label1, alpha=0.5, color="red")
                plot.axvline((event["CoaTime"] - marginal_window).datetime,
                             label=label2, c="m", linestyle="--",
                             linewidth=1.75)
                plot.axvline((event["CoaTime"] + marginal_window).datetime,
                             c="m", linestyle="--", linewidth=1.75)
                plot.axvline(event["CoaTime"].datetime, label=label3,
                             c="m", linewidth=1.75)

    props = {"boxstyle": "round",
             "facecolor": "white",
             "alpha": 0.5}
    coa.set_xlim(start_time.datetime, end_time.datetime)
    coa.text(.5, .9, "Maximum coalescence",
             horizontalalignment="center",
             transform=coa.transAxes, bbox=props)
    coa.legend(loc=2)
    coa.set_ylabel("Maximum coalescence value")
    coa_norm.set_xlim(start_time.datetime, end_time.datetime)
    coa_norm.text(.5, .9, "Normalised maximum coalescence",
                  horizontalalignment="center",
                  transform=coa_norm.transAxes, bbox=props)
    coa_norm.legend(loc=2)
    coa_norm.set_ylabel("Normalised maximum coalescence value")

    numstations.set_ylabel("Number of available stations")
    numstations.set_xlabel("DateTime")

    if events is not None:
        if normalise_coalescence:
            coa_norm.axhline(detection_threshold, c="g",
                             label="Detection threshold")
        else:
            coa_norm.axhline(detection_threshold, c="g",
                             label="Detection threshold")

        # Plotting the scatter of the earthquake locations
        xy.scatter(events["COA_X"], events["COA_Y"], 50, events["COA_V"])
        yz.scatter(events["COA_Z"], events["COA_Y"], 50, events["COA_V"])
        xz.scatter(events["COA_X"], events["COA_Z"], 50, events["COA_V"])

    xy.set_title("Decimated coalescence earthquake locations")

    yz.yaxis.tick_right()
    yz.invert_xaxis()
    yz.yaxis.set_label_position("right")
    yz.set_ylabel("Latitude (deg)")
    yz.set_xlabel("Depth (m)")

    xz.yaxis.set_label_position("right")
    xz.set_xlabel("Longitude (deg)")
    xz.set_ylabel("Depth (m)")

    if stations is not None:
        xy.scatter(stations["Longitude"], stations["Latitude"], 15,
                   marker="^", color="black")
        xz.scatter(stations["Longitude"], stations["Elevation"], 15,
                   marker="^", color="black")
        yz.scatter(stations["Elevation"], stations["Latitude"], 15,
                   marker="<", color="black")
        for i, txt in enumerate(stations["Name"]):
            xy.annotate(txt, [stations["Longitude"][i],
                        stations["Latitude"][i]], color="black")

    # Save figure or open interactive matplotlib window
    if savefig:
        out = output.run / "{}_Trigger".format(output.name)
        out = str(out.with_suffix(".pdf"))
        plt.savefig(out)
    else:
        plt.show()
