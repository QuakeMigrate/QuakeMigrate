# -*- coding: utf-8 -*-
"""
Module containing methods to generate event summaries and videos.

"""

import logging
import os

import matplotlib
try:
    os.environ["DISPLAY"]
    matplotlib.use("Qt5Agg")
except KeyError:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd

import QMigrate.util as util


@util.timeit
def event_summary(run, event, marginal_coalescence, lut):
    """
    Plots an event summary illustrating the locate results: slices through the
    marginalised coalescence with the location estimates (best-fitting spline
    to interpolated coalescence; Gaussian fit; covariance fit) and associated
    uncertainties; a gather of the filtered station data, sorted by distance
    from the event; and the maximum coalescence through time.

    Parameters
    ----------
    run : :class:`~QMigrate.io.Run` object
        Light class encapsulating i/o path information for a given run.
    event : :class:`~QMigrate.io.Event` object
        Light class encapsulating signal, onset, and location information
        for a given event.
    marginal_coalescence : `~numpy.ndarray` of `~numpy.double`
        Marginalised 3-D coalescence map, shape(nx, ny, nz).
    lut : :class:`~QMigrate.lut.LUT` object
        Contains the traveltime lookup tables for seismic phases, computed for
        some pre-defined velocity model.

    """

    logging.info("\tPlotting event summary figure...")

    # Extract indices and grid coordinates of maximum coalescence
    coa_map = np.ma.masked_invalid(marginal_coalescence)
    idx_max = np.column_stack(np.where(coa_map == np.nanmax(coa_map)))[0]
    slices = [coa_map[:, :, idx_max[2]],
              coa_map[:, idx_max[1], :],
              coa_map[idx_max[0], :, :].T]
    otime = event.otime

    fig = plt.figure(figsize=(25, 15))

    # Create plot axes, ordering: [SIGNAL, COA, XY, XZ, YZ]
    sig_spec = GridSpec(9, 15).new_subplotspec((0, 8), colspan=7, rowspan=7)
    fig.add_subplot(sig_spec)
    fig.canvas.draw()
    coa_spec = GridSpec(9, 15).new_subplotspec((7, 8), colspan=7, rowspan=2)
    fig.add_subplot(coa_spec)

    # --- Plot LUT, waveform gather, and max coalescence trace ---
    lut.plot(fig, (9, 15), slices, event.hypocentre, "white")
    _plot_waveform_gather(fig.axes[0], lut, event, idx_max)
    _plot_coalescence_trace(fig.axes[1], event)

    # --- Add event origin time to signal and coalescence plots ---
    for ax in fig.axes[:2]:
        ax.axvline(event.otime.datetime, ls="--", lw=2, c="#F03B20")

    # --- Create and plot covariance and Gaussian uncertainty ellipses ---
    gues = _make_ellipses(lut, event, "gaussian", "k")
    for ax, gue in zip(fig.axes[2:], gues):
        ax.add_patch(gue)

    # --- Write summary information ---
    text = plt.subplot2grid((9, 15), (0, 0), colspan=8, rowspan=2, fig=fig)
    _plot_text_summary(text, lut, event)

    fig.axes[0].legend(fontsize=14, loc=1)
    fig.axes[2].legend(fontsize=14)
    fig.tight_layout(pad=1, h_pad=0)
    plt.subplots_adjust(wspace=0.3, hspace=0.3)

    fpath = run.path / "locate" / run.subname / "summaries"
    fpath.mkdir(exist_ok=True, parents=True)
    fstem = f"{run.name}_{event.uid}_EventSummary"
    file = (fpath / fstem).with_suffix(".pdf")
    plt.savefig(file, dpi=400)
    plt.close("all")


WAVEFORM_COLOURS1 = ["#1b9e77", "#7570b3", "#FB9A99"]
WAVEFORM_COLOURS2 = ["#1f78b4", "#b2df8a", "#33a02c"]
PICK_COLOURS = ["#F03B20", "#3182BD"]


def _plot_waveform_gather(ax, lut, event, idx):
    """
    Utility function to bring all aspects of plotting the waveform gather into
    one place.

    Parameters
    ----------
    ax : `~matplotlib.Axes` object
        Axes on which to plot the waveform gather.
    lut : :class:`~QMigrate.lut.LUT` object
        Contains the traveltime lookup tables for seismic phases, computed for
        some pre-defined velocity model.
    event : :class:`~QMigrate.io.Event` object
        Light class encapsulating signal, onset, and location information
        for a given event.
    idx : `~numpy.ndarray` of `numpy.double`
        Marginalised 3-D coalescence map, shape(nx, ny, nz).

    """

    # --- Predicted traveltimes ---
    ttpf, ttsf = [lut.traveltime_to(phase, idx) for phase in ["P", "S"]]
    ttp = [(event.otime + tt).datetime for tt in ttpf]
    tts = [(event.otime + tt).datetime for tt in ttsf]
    range_order = abs(np.argsort(np.argsort(ttp)) - len(ttp)) * 2
    s = (ax.get_window_extent().height / (max(range_order)+1) * 1.2) ** 2
    max_tts = max(ttsf)
    for tt, c in zip([ttp, tts], PICK_COLOURS):
        ax.scatter(tt, range_order, s=s, c=c, marker="|", zorder=5, lw=1.5)

    # --- Waveforms ---
    times_utc = event.data.times(type="UTCDateTime")
    mint, maxt = event.otime - 0.1, event.otime + max_tts*1.5
    mint_i, maxt_i = [np.argmin(abs(times_utc - t)) for t in (mint, maxt)]
    times_plot = event.data.times(type="matplotlib")[mint_i:maxt_i]
    for i, signal in enumerate(np.rollaxis(event.data.filtered_signal, 1)):
        for data, c, comp in zip(signal, WAVEFORM_COLOURS1, "ENZ"):
            if not data.any():
                continue
            data[mint_i:]

            # Get station specific range for norm factor
            stat_maxt = event.otime + ttsf[i]*1.5
            norm = max(abs(data[mint_i:np.argmin(abs(times_utc - stat_maxt))]))

            y = data[mint_i:maxt_i] / norm + range_order[i]
            label = f"{comp} component" if i == 0 else None
            ax.plot(times_plot, y, c=c, lw=0.3, label=label, alpha=0.85)

    # --- Limits, annotations, and axis formatting ---
    ax.set_xlim([mint.datetime, maxt.datetime])
    ax.set_ylim([0, max(range_order)+2])
    ax.xaxis.set_major_formatter(util.DateFormatter("%H:%M:%S.{ms}", 2))
    ax.yaxis.set_ticks(range_order)
    ax.yaxis.set_ticklabels(event.data.stations, fontsize=14)
    ax.text(0.01, 0.975, "Range-ordered waveform gather", ha="left",
            va="center", transform=ax.transAxes, fontsize=14,
            bbox=dict(boxstyle='round', fc='w', alpha=0.8))


def _plot_coalescence_trace(ax, event):
    """
    Utility function to plot the maximum coalescence trace around the event
    origin time.

    Parameters
    ----------
    ax : `~matplotlib.Axes` object
        Axes on which to plot the coalescence trace.
    event : :class:`~QMigrate.io.Event` object
        Light class encapsulating signal, onset, and location information
        for a given event.

    """

    times = [x.datetime for x in event.coa_data["DT"]]
    ax.plot(times, event.coa_data["COA"], c="k", lw=0.5, zorder=10)
    ax.set_ylabel("Coalescence value", fontsize=14)
    ax.set_xlabel("DateTime", fontsize=14)
    ax.set_xlim([times[0], times[-1]])
    ax.xaxis.set_major_formatter(util.DateFormatter("%H:%M:%S.{ms}", 2))
    ax.text(0.01, 0.925, "Maximum coalescence", ha="left", va="center",
            transform=ax.transAxes, fontsize=14,
            bbox=dict(boxstyle='round', fc='w', alpha=0.8))


def _plot_text_summary(ax, lut, event):
    """
    Utility function to plot the event summary information.

    Parameters
    ----------
    ax : `~matplotlib.Axes` object
        Axes on which to plot the text summary.
    lut : :class:`~QMigrate.lut.LUT` object
        Contains the traveltime lookup tables for seismic phases, computed for
        some pre-defined velocity model.
    event : :class:`~QMigrate.io.Event` object
        Light class encapsulating signal, onset, and location information
        for a given event.

    """

    # Grab a conversion factor based on the grid projection to convert the
    # hypocentre depth + uncertainties to the correct units
    km_cf = 1000 / lut.unit_conversion_factor
    gau_unc = event.loc_uncertainty / km_cf
    hypo = (f"{event.hypocentre[1]:6.3f}\u00b0N \u00B1 {gau_unc[1]:5.3f} km\n"
            f"{event.hypocentre[0]:6.3f}\u00b0E \u00B1 {gau_unc[0]:5.3f} km\n"
            f"{event.hypocentre[2]/km_cf:6.3f} \u00B1 {gau_unc[2]:5.3f} km")

    # Grab the magnitude information
    mag_info = event.local_magnitude

    ax.text(0.25, 0.8, f"Event: {event.uid}", fontsize=20, fontweight="bold")
    with plt.rc_context({"font.size": 16}):
        ax.text(0.35, 0.65, "Origin time:", ha="right", va="center")
        ax.text(0.37, 0.65, f"{event.otime}", ha="left", va="center")
        ax.text(0.35, 0.55, "Hypocentre:", ha="right", va="top")
        ax.text(0.37, 0.55, hypo, ha="left", va="top")
        if mag_info is not None:
            mag, mag_err, mag_r2 = mag_info
            ax.text(0.35, 0.22, "Magnitude:", ha="right")
            ax.text(0.37, 0.22, f"{mag} \u00B1 {mag_err} Ml", ha="left")
            ax.text(0.35, 0.12, "Magnitude r^2:", ha="right")
            ax.text(0.37, 0.12, f"{mag_r2}", ha="left")
    ax.set_axis_off()


def _make_ellipses(lut, event, uncertainty, clr):
    """
    Utility function to create uncertainty ellipses for plotting.

    Parameters
    ----------
    lut : :class:`~QMigrate.lut.LUT` object
        Contains the traveltime lookup tables for seismic phases, computed for
        some pre-defined velocity model.
    event : :class:`~QMigrate.io.Event` object
        Light class encapsulating signal, onset, and location information
        for a given event.
    uncertainty : {"covariance", "gaussian"}
        Choice of uncertainty for which to generate ellipses.
    clr : str
        Colour for the ellipses - see matplotlib documentation for more
        details.

    Returns
    -------
    xy, yz, xz : `~matplotlib.Ellipse` (Patch) objects
        Ellipses for the requested uncertainty measure.

    """

    coord = event.get_hypocentre(method=uncertainty)
    error = event.get_loc_uncertainty(method=uncertainty)
    xyz = lut.coord2grid(coord)[0]
    d = abs(coord - lut.coord2grid(xyz + error, inverse=True))[0]

    xy = Ellipse((coord[0], coord[1]), 2*d[0], 2*d[1], lw=2, edgecolor=clr,
                 fill=False, label=f"{uncertainty.capitalize()} uncertainty")
    yz = Ellipse((coord[2], coord[1]), 2*d[2], 2*d[1], lw=2, edgecolor=clr,
                 fill=False)
    xz = Ellipse((coord[0], coord[2]), 2*d[0], 2*d[2], lw=2, edgecolor=clr,
                 fill=False)

    return xy, xz, yz


def _plot_xy_files(xy_files, ax):
    """
    Plot xy files supplied by user.

    The user can specify a list of xy files which are assigned to the
    xy_files variable. They are stored in a pandas DataFrame with
    columns:
        ["File", "Color", "Linewidth", "Linestyle"]
    File is the path to the xy file. Each file should have the format:
        ["Longitude", "Latitude"]

    Parameters
    ----------
    ax : `~matplotlib.Axes` object
        Axes on which to plot the xy files.

    """

    if xy_files is not None:
        xy_files = pd.read_csv(xy_files,
                               names=["File", "Color",
                                      "Linewidth", "Linestyle"],
                               header=None)
        for _, f in xy_files.iterrows():
            xy_file = pd.read_csv(f["File"], names=["Longitude",
                                                    "Latitude"],
                                  header=None)
            ax.plot(xy_file["Longitude"], xy_file["Latitude"],
                    ls=xy_file["Linestyle"], lw=xy_file["Linewidth"],
                    c=xy_file["Color"])
