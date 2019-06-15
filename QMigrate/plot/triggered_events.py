
# -*- coding: utf-8 -*-
"""
Module to plot the triggered events on a decimated grid

"""

import os

import matplotlib
try:
    os.environ["DISPLAY"]
    matplotlib.use("Qt5Agg")
except KeyError:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def triggered_events(events, start_time, end_time, output, marginal_window,
                     detection_threshold, normalise_coalescence, data=None,
                     stations=None, savefig=False):
    """
    Plots the data from a .scnmseed file

    Parameters
    ----------
    events :

    start_time : UTCDateTime

    end_time : UTCDateTime

    stations :

    savefig : bool, optional
        Output the plot as a file. The plot is just shown by default.

    """

    if data is None:
        fname = (output.run / output.name).with_suffix(".scnmseed")
        if fname.exists():
            data = output.read_decscan()
        else:
            msg = "    Please run detect to generate a .scnmseed file."
            print(msg)

    print("\n    Plotting triggered events on decimated grid...")
    data["DT"] = pd.to_datetime(data["DT"].astype(str))

    fig = plt.figure(figsize=(30, 15))
    fig.patch.set_facecolor("white")
    coa = plt.subplot2grid((6, 16), (0, 0), colspan=9, rowspan=3)
    coa_norm = plt.subplot2grid((6, 16), (3, 0), colspan=9, rowspan=3,
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
                             label=label2, c="m", linestyle="--", linewidth=1.75)
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
    coa_norm.set_xlabel("DateTime")

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
        xy.scatter(stations["Longitude"], stations["Latitude"], 20,)
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
