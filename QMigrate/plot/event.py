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
    run : `QMigrate.io.Run` object
        Light class encapsulating i/o path information for a given run.
    event : `QMigrate.io.Event` object
        Light class encapsulating signal, onset, and location information
        for a given event.
    marginal_coalescence : array-like
        Marginalised 3-D coalescence map.
    lut : `QMigrate.lut.LUT` object
        Contains the traveltime lookup tables for seismic phases, computed for
        some pre-defined velocity model.

    """

    logging.info("\tPlotting event summary figure...")

    # Extract indices and grid coordinates of maximum coalescence
    coa_map = np.ma.masked_invalid(marginal_coalescence)
    idx_max = np.vstack(np.where(coa_map == np.nanmax(coa_map))).T[0]
    slices = [coa_map[:, :, idx_max[2]],
              coa_map[:, idx_max[1], :],
              coa_map[idx_max[0], :, :].T]
    otime = event.otime

    fig = plt.figure(figsize=(25, 15))

    # Create plot axes, ordering: [SIGNAL, COA, XY, XZ, YZ]
    sig_spec = GridSpec(9, 15).new_subplotspec((0, 9), colspan=6, rowspan=7)
    fig.add_subplot(sig_spec)
    coa_spec = GridSpec(9, 15).new_subplotspec((7, 9), colspan=6, rowspan=2)
    fig.add_subplot(coa_spec)
    lut.plot(fig, (9, 15), slices, event.hypocentre, "white")
    axes = fig.axes

    # --- Plot waveform information on the station gather ---
    ttp = lut.traveltime_to("P", idx_max)
    sidx = abs(np.argsort(np.argsort(ttp)) - max(np.argsort(np.argsort(ttp))))
    times = event.times()
    for i, signal in enumerate(np.rollaxis(event.data.filtered_signal, 1)):
        zipped = zip(signal, ["r", "b", "g"], ["E", "N", "Z"])
        for component, clr, comp in zipped:
            if component.any():
                y = component / max(abs(component)) + (sidx[i] + 1)
                label = f"{comp} component" if i == 0 else None
                axes[0].plot(times, y, c=clr, lw=0.5, label=label)

    # --- Plot predicted travel times on the station gather ---
    ttp = lut.traveltime_to("P", idx_max)
    ttp = [(otime + tt).datetime for tt in ttp]
    tts = lut.traveltime_to("S", idx_max)
    tts = [(otime + tt).datetime for tt in tts]
    axes[0].scatter(ttp, (sidx + 1), s=50, c="pink", marker="v", zorder=4,
                    lw=0.1, edgecolors="black")
    axes[0].scatter(tts, (sidx + 1), s=50, c="purple", marker="v", zorder=5,
                    lw=0.1, edgecolors="black")

    # --- Set signal trace limits ---
    axes[0].set_xlim([(otime-0.1).datetime, (event.data.endtime-0.8).datetime])
    axes[0].yaxis.set_ticks(sidx + 1)
    axes[0].yaxis.set_ticklabels(event.data.stations, fontsize=14)
    axes[0].text(0.01, 0.975, "Range-ordered waveform gather", ha="left",
                 va="center", transform=axes[0].transAxes, fontsize=16)

    # --- Plot the maximum coalescence value around the origin time ---
    times = [x.datetime for x in event.coa_data["DT"]]
    axes[1].plot(times, event.coa_data["COA"], c="k", zorder=10)
    axes[1].set_ylabel("Coalescence value", fontsize=14)
    axes[1].set_xlabel("DateTime", fontsize=14)
    axes[1].set_xlim([times[0], times[-1]])
    axes[1].text(0.01, 0.925, "Maximum coalescence", ha="left",
                 va="center", transform=axes[1].transAxes, fontsize=16)

    # --- Add event origin time to signal and coalescence plots ---
    for ax in axes[:2]:
        ax.axvline(otime.datetime, ls="--", lw=2, c="#F03B20")

    # --- Create and plot covariance and Gaussian uncertainty ellipses ---
    cues = _make_ellipses(lut, event, "Covariance", "k")
    gues = _make_ellipses(lut, event, "Gaussian", "b")
    for ax, gue, cue in zip(axes[2:], gues, cues):
        ax.add_patch(gue)
        ax.add_patch(cue)

    # ax.scatter(coord[idx1], coord[idx2], 150, c="green", marker="*",
    #            label="Maximum Coalescence Location")

    # if eq is not None and ee is not None and gee is not None:
    #     if dim == "YZ":
    #         dim = dim[::-1]
    #     ax.scatter(eq[f"LocalGaussian_{dim[0]}"],
    #                eq[f"LocalGaussian_{dim[1]}"],
    #                150, c="pink", marker="*",
    #                label="Local Gaussian Location")
    #     ax.scatter(eq[f"GlobalCovariance_{dim[0]}"],
    #                eq[f"GlobalCovariance_{dim[1]}"],
    #                150, c="blue", marker="*",
    #                label="Global Covariance Location")

    # --- Write summary information ---
    text = plt.subplot2grid((9, 15), (0, 0), colspan=8, rowspan=2, fig=fig)
    text.text(0.5, 0.8, f"Event: {event.uid}",
              ha="center", va="center", fontsize=20, fontweight="bold")
    text.text(0.4, 0.65, f"Origin time:", ha="right", va="center", fontsize=20)
    text.text(0.42, 0.65, f"{otime}", ha="left", va="center", fontsize=20)
    text.text(0.4, 0.55, f"Hypocentre:", ha="right", va="top", fontsize=20)
    gau_unc = event.df.filter(regex="LocalGaussian_Err[XYZ]").values[0] / 1000
    hypo = (f"{event.hypocentre[1]:5.3f}\u00b0 N +/- {gau_unc[1]:5.3f} km\n"
            f"{event.hypocentre[0]:5.3f}\u00b0 E +/- {gau_unc[0]:5.3f} km\n"
            f"{event.hypocentre[2]/1000:5.3f} +/- {gau_unc[2]:5.3f} km")
    text.text(0.42, 0.55, hypo, ha="left", va="top", fontsize=20)

    text.set_axis_off()

    axes[0].legend(fontsize=14)
    axes[2].legend(fontsize=14)
    fig.tight_layout(pad=1, h_pad=0)
    plt.subplots_adjust(wspace=0.3, hspace=0.3)

    fpath = run.path / "locate" / "summaries"
    fpath.mkdir(exist_ok=True, parents=True)
    fstem = f"{run.name}_{event.uid}_EventSummary"
    file = (fpath / fstem).with_suffix(".pdf")
    plt.savefig(file, dpi=400)
    plt.close("all")


def _make_ellipses(lut, event, uncertainty, clr):
    """
    Utility function to create uncertainty ellipses for plotting.

    Parameters
    ----------
    eq : `pandas.DataFrame` object
        Final location information for the event to be plotted.
        Columns = ["DT", "COA", "X", "Y", "Z",
                   "LocalGaussian_X", "LocalGaussian_Y", "LocalGaussian_Z",
                   "LocalGaussian_ErrX", "LocalGaussian_ErrY",
                   "LocalGaussian_ErrZ", "GlobalCovariance_X",
                   "GlobalCovariance_Y", "GlobalCovariance_Z",
                   "GlobalCovariance_ErrX", "GlobalCovariance_ErrY",
                   "GlobalCovariance_ErrZ", "ML", "ML_Err"]
        All X / Y as lon / lat; Z and X / Y / Z uncertainties in metres.
    uncertainty : str
        Choice of uncertainty for which to generate ellipses.
        Options are: "Covariance" or "Gaussian".
    clr : str
        Colour for the ellipses - see matplotlib documentation for more
        details.

    Returns
    -------
    xy, yz, xz : `matplotlib.Ellipse` (Patch) objects
        Ellipses for the requested uncertainty measure.

    """

    coord = event.df.filter(regex=f"{uncertainty}_[XYZ]").values[0]
    error = event.df.filter(regex=f"{uncertainty}_Err[XYZ]").values[0]
    xyz = lut.coord2grid(coord)[0]
    d = abs(coord - lut.coord2grid(xyz + error, inverse=True))[0]

    if uncertainty == "Covariance":
        label = "Global covariance uncertainty ellipse"
    elif uncertainty == "Gaussian":
        label = "Local Gaussian uncertainty ellipse"

    xy = Ellipse((coord[0], coord[1]), 2*d[0], 2*d[1], lw=2, edgecolor=clr,
                 fill=False, label=label)
    yz = Ellipse((coord[2], coord[1]), 2*d[2], 2*d[1], lw=2, edgecolor=clr,
                 fill=False)
    xz = Ellipse((coord[0], coord[2]), 2*d[0], 2*d[2], lw=2, edgecolor=clr,
                 fill=False)

    return xy, xz, yz


def _plot_map_slice(lut, ax, slice_, coord, dim, eq=None, ee=None, gee=None):
    """
    Plot slice through map in a given plane.

    Parameters
    ----------
    ax : matplotlib Axes object
        Axes on which to plot the grid slice.
    slice_ : array-like
        2-D array of coalescence values for the slice through the 3-D grid.
    coord : array-like
        Earthquake location in the input projection coordinate space.
    dim : str
        Denotes which 2-D slice is to be plotted ("XY", "XZ", "YZ").
    eq : pandas DataFrame object
        Final location information for the event to be plotted.
        Columns = ["DT", "COA", "X", "Y", "Z",
                   "LocalGaussian_X", "LocalGaussian_Y", "LocalGaussian_Z",
                   "LocalGaussian_ErrX", "LocalGaussian_ErrY",
                   "LocalGaussian_ErrZ", "GlobalCovariance_X",
                   "GlobalCovariance_Y", "GlobalCovariance_Z",
                   "GlobalCovariance_ErrX", "GlobalCovariance_ErrY",
                   "GlobalCovariance_ErrZ", "ML", "ML_Err"]
        All X / Y as lon / lat; Z and X / Y / Z uncertainties in metres.

    ee : matplotlib Ellipse (Patch) object.
        Uncertainty ellipse for the global covariance.

    gee : matplotlib Ellipse (Patch) object.
        Uncertainty ellipse for the local Gaussian.

    """

    corners = lut.coord2grid(lut.grid_corners, inverse=True)

    # Series of tests to select the correct components for the given slice
    mins = [np.min(dim) for dim in corners.T]
    maxs = [np.max(dim) for dim in corners.T]
    sizes = (np.array(maxs) - np.array(mins)) / lut.cell_count
    stack = np.c_[mins, maxs, sizes]

    if dim == "XY":
        idx1, idx2 = 0, 1
    elif dim == "XZ":
        idx1, idx2 = 0, 2
    elif dim == "YZ":
        idx1, idx2 = 2, 1

    min1, max1, size1 = stack[idx1]
    min2, max2, size2 = stack[idx2]

    # Create meshgrid with shape (X + 1, Y + 1) - pcolormesh uses the grid
    # values as fenceposts
    grid1, grid2 = np.mgrid[min1:max1 + size1:size1,
                            min2:max2 + size2:size2]

    # Ensure that the shape of grid1 and grid2 comply with the shape of the
    # slice (sometimes floating point errors can carry over and return a
    # grid with incorrect shape)
    grid1 = grid1[:slice_.shape[0]+1, :slice_.shape[1]+1]
    grid2 = grid2[:slice_.shape[0]+1, :slice_.shape[1]+1]
    ax.pcolormesh(grid1, grid2, slice_, cmap="viridis", edgecolors="face")

    # ax.set_xlim([min1, max1])
    # ax.set_ylim([min2, max2])

    ax.axvline(x=coord[idx1], ls="--", lw=2, c="white")
    ax.axhline(y=coord[idx2], ls="--", lw=2, c="white")
    ax.scatter(coord[idx1], coord[idx2], 150, c="green", marker="*",
               label="Maximum Coalescence Location")

    if eq is not None and ee is not None and gee is not None:
        if dim == "YZ":
            dim = dim[::-1]
        ax.scatter(eq[f"LocalGaussian_{dim[0]}"],
                   eq[f"LocalGaussian_{dim[1]}"],
                   150, c="pink", marker="*",
                   label="Local Gaussian Location")
        ax.scatter(eq[f"GlobalCovariance_{dim[0]}"],
                   eq[f"GlobalCovariance_{dim[1]}"],
                   150, c="blue", marker="*",
                   label="Global Covariance Location")
        ax.add_patch(ee)
        ax.add_patch(gee)


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
    ax : matplotlib Axes object
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
