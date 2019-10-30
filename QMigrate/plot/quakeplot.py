# -*- coding: utf-8 -*-
"""
Module to produce plots for QuakeMigrate.

"""

import os
import pathlib
from datetime import datetime

from obspy import UTCDateTime
import pandas as pd
import numpy as np
import matplotlib
try:
    os.environ["DISPLAY"]
    matplotlib.use("Qt5Agg")
except KeyError:
    matplotlib.use("Agg")
import matplotlib.pylab as plt
from matplotlib.patches import Ellipse
import matplotlib.image as mpimg
import matplotlib.animation as animation
from pandas.plotting import register_matplotlib_converters

import QMigrate.util as util

register_matplotlib_converters()


class QuakePlot:
    """
    QuakeMigrate plotting class

    Includes methods for various plotting options within QuakeMigrate.

    Methods
    -------
    station_traces()
        Generate plot of data & characteristic function traces and phase picks

    coalescence_video()
        Generate a video of coalescence over marginal time window about
        earthquake origin time

    event_summary()
        Generate summary plot of earthquake location and uncertainty

    """

    logo = (pathlib.Path(__file__) / "QuakeMigrate").with_suffix(".png")

    def __init__(self, lut, data, mw_coa, marginal_window, run_path, event,
                 map_4d=None, coa_map=None, options=None):
        """
        Initialisation of QuakePlot object.

        Parameters
        ----------
        lut : array-like
            QMigrate look-up table

        map_4d : array-like, optional
            4-D coalescence map output by _compute()

        coa_map : array-like
            3-D marginalised coalescence map output by _calculate_location()

        data : Archive object
            Contains read_waveforms() method and stores read data in raw and
            processed state

        mw_coa : pandas DataFrame
            Gridded maximum coa location through time across the marginal
            window. Columns = ["DT", "COA", "X", "Y", "Z"]

        event : pandas DataFrame
            Final location information for the event to be plotted
            Columns = ["DT", "COA", "X", "Y", "Z",
                       "LocalGaussian_X", "LocalGaussian_Y", "LocalGaussian_Z",
                       "LocalGaussian_ErrX", "LocalGaussian_ErrY",
                       "LocalGaussian_ErrZ", "GlobalCovariance_X",
                       "GlobalCovariance_Y", "GlobalCovariance_Z",
                       "GlobalCovariance_ErrX", "GlobalCovariance_ErrY",
                       "GlobalCovariance_ErrZ"]
            All X / Y as lon / lat; Z and X / Y / Z uncertainties in metres

        phase_picks : dict, optional
            Phase pick info, with keys:
                "Pick" : pandas DataFrame
                    Phase pick times with columns: ["Name", "Phase",
                                                    "ModelledTime",
                                                    "PickTime", "PickError",
                                                    "SNR"]
                    Each row contains the phase pick from one station/phase.
                "GAU_P" : array-like, dict
                    Numpy array stack of Gaussian pick info (each as a dict)
                    for P phase:
                        {"popt": popt,
                        "xdata": x_data,
                        "xdata_dt": x_data_dt,
                        "PickValue": max_onset,
                        "PickThreshold": threshold}
                "GAU_S" : array-like
                    Numpy array stack of Gaussian pick info (each as a dict)
                    for S phase: see "GAU_P"

        marginal_window : float
            Length of marginal window; time window about event maximum
            coalescence time (origin time) to marginalise the 4-D coalescence
            function

        run_path : path
            Path of run directory

        ### UNTESTED ###
        options : dict of additional kwargs
            Usage e.g. options={'TraceScaling': True, 'MAPColor': 'r'}

        """

        self.lut = lut
        self.map_4d = map_4d
        if self.map_4d is not None:
            self.map_max = np.nanmax(map_4d)
        self.coa_map = coa_map

        self.data = data

        self.mw_coa = mw_coa
        self.event = event

        self.marginal_window = marginal_window

        self.run_path = run_path

        self.range_order = True

        self.logo = "{}"

        if options is None:
            self.trace_scale = 1
            self.cmap = "viridis"
            self.line_station_color = "white"
            self.plot_stats = True
            self.filtered_signal = True
            self.xy_files = None
        else:
            try:
                self.trace_scale = options.TraceScaling
                self.cmap = options.MAPColor
                self.line_station_color = options.line_station_color
                self.plot_stats = options.Plot_Stations
                self.filtered_signal = options.FilteredSignal
                self.xy_files = options.xy_files

            except AttributeError:
                msg = "Error - define all plot options."
                print(msg)

        # start_time and end_time are start of pre-pad and end of post-pad,
        # respectively.
        tmp = np.arange(self.data.start_time,
                        self.data.end_time + self.data.sample_size,
                        self.data.sample_size)
        self.times = pd.to_datetime([x.datetime for x in tmp])

        # Convert event["DT"] to python datetime object
        if not isinstance(self.mw_coa["DT"].iloc[0], datetime):
            self.mw_coa["DT"] = [x.datetime for x in self.mw_coa["DT"]]

        # I think this should do nothing....
        self.mw_coa = self.mw_coa[(self.mw_coa["DT"] > self.times[0]) &
                                  (self.mw_coa["DT"] < self.times[-1])]

        self.slices = {}
        self.crosshairs = {}

    def coalescence_video(self, file_str):
        """
        Generate a video over the marginal window showing the coalescence map
        and expected arrival times overlain on the station traces.

        Parameters
        ----------
        file_str : str
            String {run_name}_{event_name}

        """

        # Find index of start and end of marginal window
        i0 = np.where(self.times == self.mw_coa["DT"].iloc[0])[0][0]
        i1 = np.where(self.times == self.mw_coa["DT"].iloc[-1])[0][0]

        Writer = animation.writers["ffmpeg"]
        writer = Writer(fps=4, metadata=dict(artist="QM"), bitrate=1800)

        fig = self._coalescence_frame(i0)
        ani = animation.FuncAnimation(fig, self._video_update,
                                      frames=np.linspace(i0+1, i1, i1-i0),
                                      blit=False, repeat=False)

        plt.show()

        util.make_directories(self.run_path, subdir="videos")
        out_str = self.run_path / "videos" / file_str
        print(out_str, str(out_str))
        ani.save("{}_CoalescenceVideo.mp4".format(str(out_str)), writer=writer)

    def event_summary(self, file_str=None):
        """
        Create summary plot for an event.

        Shows the coalescence map sliced through the maximum coalescence
        showing calculated locations and uncdrtainties, the coalescence value
        over the course of the marginal time window and a gather of the station
        traces.

        Parameters
        ----------
        file_str : str, optional
            String {run_name}_{event_name} (figure displayed by default)

        """

        # Event is only in first line of earthquake, reduces chars later on
        if self.event is not None:
            eq = self.event
        else:
            msg = "\t\tError: no event specified!"
            print(msg)
            return

        dt_max = self.mw_coa["DT"].iloc[np.argmax(self.mw_coa["COA"].values)]

        # Extract indices and grid coordinates of maximum coalescence
        coa_map = np.ma.masked_invalid(self.coa_map)
        idx_max = np.r_[np.where(coa_map == np.nanmax(coa_map))]
        coord_max = self.lut.index2coord(idx_max)[0]

        # Defining the plots to be represented
        fig = plt.figure(figsize=(25, 15))
        fig.patch.set_facecolor("white")
        xy_ax = plt.subplot2grid((3, 5), (0, 0), colspan=2, rowspan=2)
        xz_ax = plt.subplot2grid((3, 5), (2, 0), colspan=2)
        yz_ax = plt.subplot2grid((3, 5), (0, 2), rowspan=2)
        sig_ax = plt.subplot2grid((3, 5), (0, 3), colspan=2, rowspan=2)
        logo_ax = plt.subplot2grid((3, 5), (2, 2))
        coa_ax = plt.subplot2grid((3, 5), (2, 3), colspan=2)

        # Station trace ordering
        if self.range_order:
            ttp = self.lut.traveltime_to("P", idx_max)
            sidx = abs(np.argsort(np.argsort(ttp))
                       - np.max(np.argsort(np.argsort(ttp))))
        else:
            # Order alphabetically by name
            sidx = np.argsort(self.lut.station_data["Name"])[::-1]

        for i in range(self.data.signal.shape[1]):
            if not self.filtered_signal:
                signal = self.data.signal
            else:
                signal = self.data.filtered_signal

            self._plot_signal_trace(sig_ax, self.times, signal[0, i, :],
                                    sidx[i], color="r")
            self._plot_signal_trace(sig_ax, self.times, signal[1, i, :],
                                    sidx[i], color="b")
            self._plot_signal_trace(sig_ax, self.times, signal[2, i, :],
                                    sidx[i], color="g")

        # --- Plot predicted travel times on station traces ---
        dt_max = UTCDateTime(dt_max)
        ttp = self.lut.traveltime_to("P", idx_max)
        ttp = [(dt_max + tt).datetime for tt in ttp]
        tts = self.lut.traveltime_to("S", idx_max)
        tts = [(dt_max + tt).datetime for tt in tts]

        self.ptt = sig_ax.scatter(ttp, (sidx + 1), 50, "pink", marker="v",
                                  zorder=4, linewidth=0.1, edgecolors="black")
        self.stt = sig_ax.scatter(tts, (sidx + 1), 50, "purple", marker="v",
                                  zorder=5, linewidth=0.1, edgecolors="black")

        # --- Set signal trace limits ---
        sig_ax.set_xlim([(dt_max-0.1).datetime,
                        (self.data.end_time-0.8).datetime])
        sig_ax.yaxis.tick_right()
        sig_ax.yaxis.set_ticks(sidx + 1)
        sig_ax.yaxis.set_ticklabels(self.data.stations)
        sig_ax.axvline(dt_max.datetime, 0, 1000, linestyle="--", linewidth=2,
                       color="r")

        # --- Plot the slices through the 3-D coalescence volume ---
        self._plot_coalescence_value(coa_ax, dt_max.datetime)

        # --- Create covariance and Gaussian uncertainty ellipses ---
        cov_exy, cov_eyz, cov_exz = self._make_ellipses(eq, "Covariance", "k")
        gau_exy, gau_eyz, gau_exz = self._make_ellipses(eq, "Gaussian", "b")

        # --- Plot slices through coalescence map ---
        self._plot_map_slice(xy_ax, coa_map[:, :, idx_max[2]], coord_max, "XY",
                             eq, cov_exy, gau_exy)
        xy_ax.legend()

        self._plot_map_slice(xz_ax, coa_map[:, idx_max[1], :], coord_max, "XZ",
                             eq, cov_exz, gau_exz)
        xz_ax.invert_yaxis()

        self._plot_map_slice(yz_ax, coa_map[idx_max[0], :, :].T, coord_max, "YZ",
                             eq, cov_eyz, gau_eyz)

        # --- Plotting the station locations ---
        xy_ax.scatter(self.lut.station_data["Longitude"],
                      self.lut.station_data["Latitude"],
                      15, marker="^", color=self.line_station_color)
        xz_ax.scatter(self.lut.station_data["Longitude"],
                      self.lut.station_data["Elevation"],
                      15, marker="^", color=self.line_station_color)
        yz_ax.scatter(self.lut.station_data["Elevation"],
                      self.lut.station_data["Latitude"],
                      15, marker="<", color=self.line_station_color)
        for i, txt in enumerate(self.lut.station_data["Name"]):
            xy_ax.annotate(txt, [self.lut.station_data["Longitude"][i],
                                 self.lut.station_data["Latitude"][i]],
                           color=self.line_station_color)

        # --- Plotting the xy_files ---
        self._plot_xy_files(xy_ax)

        # --- Plotting the logo ---
        self._plot_logo(logo_ax, r"Earthquake Location Error")

        if file_str is None:
            plt.show()
        else:
            fig.suptitle("Event Origin Time = {}".format(dt_max.datetime))
            subdir = "summaries"
            util.make_directories(self.run_path, subdir=subdir)
            out_str = self.run_path / subdir / file_str
            plt.savefig("{}_EventSummary.pdf".format(out_str), dpi=400)
            plt.close("all")

    def _make_ellipses(self, eq, uncertainty, color):
        """
        Utility function to create uncertainty ellipses for plotting.

        Parameters
        ----------
        eq : pandas DataFrame object
            Final location information for the event to be plotted.
            Columns = ["DT", "COA", "X", "Y", "Z",
                       "LocalGaussian_X", "LocalGaussian_Y", "LocalGaussian_Z",
                       "LocalGaussian_ErrX", "LocalGaussian_ErrY",
                       "LocalGaussian_ErrZ", "GlobalCovariance_X",
                       "GlobalCovariance_Y", "GlobalCovariance_Z",
                       "GlobalCovariance_ErrX", "GlobalCovariance_ErrY",
                       "GlobalCovariance_ErrZ"]
            All X / Y as lon / lat; Z and X / Y / Z uncertainties in metres.

        uncertainty : str
            Choice of uncertainty for which to generate ellipses.
            Options are: "Covariance" or "Gaussian".

        color : str
            Colour for the ellipses - see matplotlib documentation for more
            details.

        Returns
        -------
        xy, yz, xz : matplotlib Ellipse (Patch) objects
            Ellipses for the requested uncertainty measure.

        """

        coord = eq.filter(regex="{}_[XYZ]".format(uncertainty)).values[0]
        error = eq.filter(regex="{}_Err[XYZ]".format(uncertainty)).values[0]
        xyz = self.lut.coord2grid(coord)[0]
        d = abs(coord - self.lut.coord2grid(xyz + error, inverse=True))[0]

        if uncertainty == "Covariance":
            label = "Global covariance uncertainty ellipse"
        elif uncertainty == "Gaussian":
            label = "Local Gaussian uncertainty ellipse"

        xy = Ellipse((coord[0], coord[1]), 2*d[0], 2*d[1], linewidth=2,
                     edgecolor=color, fill=False, label=label)
        yz = Ellipse((coord[2], coord[1]), 2*d[2], 2*d[1], linewidth=2,
                     edgecolor=color, fill=False)
        xz = Ellipse((coord[0], coord[2]), 2*d[0], 2*d[2], linewidth=2,
                     edgecolor=color, fill=False)

        return xy, yz, xz

    def _plot_map_slice(self, ax, slice_, coord, dim, eq=None, ee=None, gee=None):
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

        eq : pandas DataFrame object.
            Final location information for the event to be plotted.
            Columns = ["DT", "COA", "X", "Y", "Z",
                       "LocalGaussian_X", "LocalGaussian_Y", "LocalGaussian_Z",
                       "LocalGaussian_ErrX", "LocalGaussian_ErrY",
                       "LocalGaussian_ErrZ", "GlobalCovariance_X",
                       "GlobalCovariance_Y", "GlobalCovariance_Z",
                       "GlobalCovariance_ErrX", "GlobalCovariance_ErrY",
                       "GlobalCovariance_ErrZ"]
            All X / Y as lon / lat; Z and X / Y / Z uncertainties in metres.

        ee : matplotlib Ellipse (Patch) object.
            Uncertainty ellipse for the global covariance.

        gee : matplotlib Ellipse (Patch) object.
            Uncertainty ellipse for the local Gaussian.

        """

        corners = self.lut.coord2grid(self.lut.grid_corners, inverse=True)

        # Series of tests to select the correct components for the given slice
        mins = [np.min(dim) for dim in corners.T]
        maxs = [np.max(dim) for dim in corners.T]
        sizes = (np.array(maxs) - np.array(mins)) / self.lut.cell_count
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
        self.slices[dim] = ax.pcolormesh(grid1, grid2, slice_, cmap=self.cmap,
                                         edgecolors="face")

        ax.set_xlim([min1, max1])
        ax.set_ylim([min2, max2])

        vstring = "{}_v".format(dim)
        hstring = "{}_h".format(dim)
        self.crosshairs[vstring] = ax.axvline(x=coord[idx1], linestyle="--",
                                              linewidth=2,
                                              color=self.line_station_color)
        self.crosshairs[hstring] = ax.axhline(y=coord[idx2], linestyle="--",
                                              linewidth=2,
                                              color=self.line_station_color)
        ax.scatter(coord[idx1], coord[idx2], 150, c="green", marker="*",
                   label="Maximum Coalescence Location")

        if eq is not None and ee is not None and gee is not None:
            if dim == "YZ":
                dim = dim[::-1]
            ax.scatter(eq["LocalGaussian_{}".format(dim[0])],
                       eq["LocalGaussian_{}".format(dim[1])],
                       150, c="pink", marker="*",
                       label="Local Gaussian Location")
            ax.scatter(eq["GlobalCovariance_{}".format(dim[0])],
                       eq["GlobalCovariance_{}".format(dim[1])],
                       150, c="blue", marker="*",
                       label="Global Covariance Location")
            ax.add_patch(ee)
            ax.add_patch(gee)

    def _plot_signal_trace(self, ax, x, y, st_idx, color):
        """
        Plot signal trace.

        Performs a simple check to see if there is any signal data available to
        plot.

        Parameters
        ----------
        ax : matplotlib Axes object
            Axes on which to plot the signal trace.

        x : array-like
            Timestamps for the signal trace.

        y : array-like
            The amplitudes of the signal trace.

        st_idx : int
            Amount to vertically shift the signal trace. Either range ordered
            or ordered alphabetically by station name.

        color : str
            Line colour for the trace - see matplotlib documentation for more
            details.

        """

        if y.any():
            ax.plot(x, y / np.max(abs(y)) * self.trace_scale + (st_idx + 1),
                    color=color, linewidth=0.5, zorder=1)

    def _coalescence_frame(self, tslice_idx):
        """
        Plots a frame of a coalescence video at a particular time.

        Parameters
        ----------
        tslice_idx : int
            Index for the current time slice.

        """

        coord = self.mw_coa.iloc[0]
        coord = [coord["X"], coord["Y"], coord["Z"]]
        idx_frame = self.lut.index2coord(coord, inverse=True)

        # Define the axes on which to plot data
        fig = plt.figure(figsize=(25, 15))
        fig.patch.set_facecolor("white")
        xy_ax = plt.subplot2grid((3, 5), (0, 0), colspan=2, rowspan=2)
        xz_ax = plt.subplot2grid((3, 5), (2, 0), colspan=2)
        yz_ax = plt.subplot2grid((3, 5), (0, 2), rowspan=2)
        sig_ax = plt.subplot2grid((3, 5), (0, 3), colspan=2, rowspan=2)
        logo_ax = plt.subplot2grid((3, 5), (2, 2))
        coa_ax = plt.subplot2grid((3, 5), (2, 3), colspan=2)

        # Station trace ordering
        if self.range_order:
            ttp = self.lut.traveltime_to("P", idx_frame)
            sidx = abs(np.argsort(np.argsort(ttp))
                       - np.max(np.argsort(np.argsort(ttp))))
        else:
            # Order alphabetically by name
            sidx = np.argsort(self.lut.station_data["Name"])[::-1]

        for i in range(self.data.signal.shape[1]):
            if not self.filtered_signal:
                signal = self.data.signal
            else:
                signal = self.data.filtered_signal

            self._plot_signal_trace(sig_ax, self.times, signal[0, i, :],
                                    sidx[i], color="r")
            self._plot_signal_trace(sig_ax, self.times, signal[1, i, :],
                                    sidx[i], color="b")
            self._plot_signal_trace(sig_ax, self.times, signal[2, i, :],
                                    sidx[i], color="g")

        # --- Plot predicted travel times on station traces ---
        dt_max = self.mw_coa["DT"].iloc[np.argmax(self.mw_coa["COA"])]
        dt_max = UTCDateTime(dt_max)
        ttp = self.lut.traveltime_to("P", idx_frame)
        ttp = [(dt_max + tt).datetime for tt in ttp]
        tts = self.lut.traveltime_to("S", idx_frame)
        tts = [(dt_max + tt).datetime for tt in tts]

        self.ptt = sig_ax.scatter(ttp, (sidx + 1), 50, "pink", marker="v",
                                  zorder=4, linewidth=0.1, edgecolors="black")
        self.stt = sig_ax.scatter(tts, (sidx + 1), 50, "purple", marker="v",
                                  zorder=5, linewidth=0.1, edgecolors="black")

        # Set coalescence trace limits
        sig_ax.set_xlim([(dt_max-0.1).datetime,
                        (self.data.end_time-0.8).datetime])
        sig_ax.yaxis.tick_right()
        sig_ax.yaxis.set_ticks(sidx + 1)
        sig_ax.yaxis.set_ticklabels(self.data.stations)
        self.station_trace_vline = sig_ax.axvline(dt_max.datetime, 0, 1000,
                                                  linestyle="--", linewidth=2,
                                                  color="r")

        # --- Plot the slices through the 3-D coalescence volume ---
        self._plot_coalescence_value(coa_ax, dt_max.datetime)

        # --- Plot slices through coalescence map ---
        self._plot_map_slice(xy_ax, self.map_4d[:, :, idx_frame[0][2], 0],
                             coord, "XY")
        xy_ax.legend()

        self._plot_map_slice(xz_ax, self.map_4d[:, idx_frame[0][1], :, 0],
                             coord, "XZ")
        xz_ax.invert_yaxis()

        self._plot_map_slice(yz_ax, self.map_4d[idx_frame[0][0], :, :, 0].T,
                             coord, "YZ")

        # --- Plotting the station locations ---
        xy_ax.scatter(self.lut.station_data["Longitude"],
                      self.lut.station_data["Latitude"],
                      15, marker="^", color=self.line_station_color)
        xz_ax.scatter(self.lut.station_data["Longitude"],
                      self.lut.station_data["Elevation"],
                      15, marker="^", color=self.line_station_color)
        yz_ax.scatter(self.lut.station_data["Elevation"],
                      self.lut.station_data["Latitude"],
                      15, marker="<", color=self.line_station_color)
        for i, txt in enumerate(self.lut.station_data["Name"]):
            xy_ax.annotate(txt, [self.lut.station_data["Longitude"][i],
                                 self.lut.station_data["Latitude"][i]],
                           color=self.line_station_color)

        # --- Plotting the xy_files ---
        self._plot_xy_files(xy_ax)

        # --- Plotting the logo ---
        self._plot_logo(logo_ax, r"Coalescence Video")

        return fig

    def _video_update(self, frame):
        """
        Plot latest video frame.

        Parameters
        ----------
        frame : int
            Current frame number.

        """

        frame = int(frame)
        idx0 = np.where(self.times == self.mw_coa["DT"].iloc[0])[0][0]
        tslice = self.times[frame]

        idx = np.where(self.mw_coa["DT"] == tslice)[0][0]
        coord = self.mw_coa[["X", "Y", "Z"]].values[idx]
        idx_frame = self.lut.index2coord(coord, inverse=True)[0]

        # Updating the coalescence value and trace lines
        self.station_trace_vline.set_xdata(tslice)
        self.coal_val_vline.set_xdata(tslice)

        # Updating the Coalescence Maps
        self.slices["XY"].set_array(self.map_4d[:, :, idx_frame[2],
                                                int(idx0-frame)].ravel())
        self.slices["XZ"].set_array(self.map_4d[:, idx_frame[1], :,
                                                int(idx0-frame)].ravel())
        self.slices["YZ"].set_array(self.map_4d[idx_frame[0], :, :,
                                                int(idx0 - frame)].T.ravel())

        # Updating the coalescence lines
        self.crosshairs["XY_v"].set_xdata(coord[0])
        self.crosshairs["XY_h"].set_ydata(coord[1])
        self.crosshairs["YZ_v"].set_xdata(coord[2])
        self.crosshairs["YZ_h"].set_ydata(coord[1])
        self.crosshairs["XZ_v"].set_xdata(coord[0])
        self.crosshairs["XZ_h"].set_ydata(coord[2])

        # --- Update predicted travel times on station traces ---
        tslice = UTCDateTime(tslice)
        ttp = self.lut.traveltime_to("P", idx_frame)
        ttp = [np.argmin(abs(self.times - (tslice + tt).datetime)) for tt in ttp]
        tts = self.lut.traveltime_to("S", idx_frame)
        tts = [np.argmin(abs(self.times - (tslice + tt).datetime)) for tt in tts]

        self.ptt.set_offsets(np.c_[ttp, (np.arange(len(ttp)) + 1)])
        self.stt.set_offsets(np.c_[tts, (np.arange(len(tts)) + 1)])

    def _plot_xy_files(self, ax):
        """
        Plot xy files supplied by user.

        The user can specify a list of xy files which are assigned to the
        self.xy_files variable. They are stored in a pandas DataFrame with
        columns:
            ["File", "Color", "Linewidth", "Linestyle"]
        File is the path to the xy file. Each file should have the format:
            ["Longitude", "Latitude"]

        Parameters
        ----------
        ax : matplotlib Axes object
            Axes on which to plot the xy files.

        """

        if self.xy_files is not None:
            xy_files = pd.read_csv(self.xy_files,
                                   names=["File", "Color",
                                          "Linewidth", "Linestyle"],
                                   header=None)
            for i, f in xy_files.iterrows():
                xy_file = pd.read_csv(f["File"], names=["Longitude",
                                                        "Latitude"],
                                      header=None)
                ax.plot(xy_file["Longitude"], xy_file["Latitude"],
                        linestyle=xy_file["Linestyle"],
                        linewidth=xy_file["Linewidth"],
                        color=xy_file["Color"])

    def _plot_logo(self, ax, txt):
        """
        Plot QuakeMigrate logo.

        Parameters
        ----------
        ax : matplotlib Axes object
            Axes on which to plot the QuakeMigrate logo.

        txt : str
            Text specifying the type of plot.

        """

        try:
            ax.axis("off")
            im = mpimg.imread(str(self.logo))
            ax.imshow(im)
            ax.text(150, 200, txt, fontsize=14, style="italic")
        except:
            print("\t\tLogo not plotting")

    def _plot_coalescence_value(self, ax, tslice):
        """
        Plot max coalescence value in the grid through time.

        Parameters
        ----------
        ax : matplotlib Axes object
            Axes on which to plot the trace of the maximum amplitude of
            coalescence in the 3-D volume through time.

        tslice : Python Datetime object
            Timestamp at which to plot a vertical line. For the event summary,
            this corresponds to the maximum coalescence peak.

        """

        ax.plot(self.mw_coa["DT"], self.mw_coa["COA"], zorder=10)
        ax.set_ylabel("Coalescence value")
        ax.set_xlabel("Date-Time")
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")
        ax.set_xlim([self.mw_coa["DT"].iloc[0], self.mw_coa["DT"].iloc[-1]])
        for tick in ax.get_xticklabels():
            tick.set_rotation(45)

        self.coal_val_vline = ax.axvline(tslice, 0, 1000, linestyle="--",
                                         linewidth=2, color="r")
