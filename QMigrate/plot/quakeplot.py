# -*- coding: utf-8 -*-
"""
Module to produce plots for QuakeMigrate.

"""

import os
import pathlib
from datetime import datetime, timedelta

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

import QMigrate.util as util


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

    def __init__(self, lut, map_4d=None, coa_map=None, data, event_mw_data, 
                 event, phase_picks=None, marginal_window, run_path,
                 options=None):
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

        event_mw_data : pandas DataFrame
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
        if self.map_4d:
            self.map_max = np.nanmax(map_4d)
        self.coa_map = coa_map

        self.data = data

        self.event_mw_data = event_mw_data
        self.event = event
        self.phase_picks = phase_picks

        self.marginal_window = marginal_window

        self.run_path = run_path

        self.range_order = True

        self.logo = "{}"

        if options is None:
            self.trace_scale = 1
            self.cmap = "hot_r"
            self.line_station_color = "black"
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
        if not isinstance(self.event_mw_data["DT"].iloc[0], datetime):
            self.event_mw_data["DT"] = [x.datetime for x in \
                                        self.event_mw_data["DT"]]

        # I think this should do nothing....
        self.event_mw_data = self.event_mw_data[(self.event_mw_data["DT"] > \
                                                 self.times[0])
                                 & (self.event_mw_data["DT"] < self.times[-1])]
        
        self.station_trace_vline = None
        self.coal_val_vline = None
        self.xy_plot = None
        self.yz_plot = None
        self.xz_plot = None
        self.xy_vline = None
        self.xy_hline = None
        self.yz_vline = None
        self.yz_hline = None
        self.xz_vline = None
        self.xz_hline = None
        self.tp_arrival = None
        self.ts_arrival = None

    def station_traces(self, file_str=None, event_name=None):
        """
        Plot figures showing the filtered traces for each data component
        and the characteristic functions calculated from them (P and S) for
        each station. The search window to make a phase pick is displayed,
        along with the dynamic pick threshold (defined as a percentile of the
        background noise level), the phase pick time and its uncertainty (if
        made) and the gaussian fit to the characteristic function.

        Parameters
        ----------
        file_str : str, optional
            String {run_name}_{evt_id} (figure displayed by default)

        event_name : str, optional
            Earthquake UID string; for subdirectory naming within directory
            {run_path}/traces/

        """

        ## This function currently doesn't work due to a float/int issue
        # point = np.round(self.lut.coord2loc(np.array([[event["X"], 
        #                                                event["Y"],
        #                                                event["Z"]]]))).astype(int)

        loc = np.where(self.coa_map == np.nanmax(self.coa_map))
        point = np.array([[loc[0][0],
                           loc[1][0],
                           loc[2][0]]])

        # Get P- and S-traveltimes at this location
        ptt = self.lut.get_value_at("TIME_P", point)[0]
        stt = self.lut.get_value_at("TIME_S", point)[0]

        # Make output dir for this event outside of loop
        if file_str:
            subdir = "traces"
            util._make_directories(self.run_path, subdir=subdir)
            out_dir = self.run_path / subdir / event_name
            util._make_directories(out_dir)

        # Looping through all stations
        for i in range(self.data.signal.shape[1]):
            station = self.lut.station_data["Name"][i]
            gau_p = self.phase_picks["GAU_P"][i]
            gau_s = self.phase_picks["GAU_S"][i]
            fig = plt.figure(figsize=(30, 15))

            # Defining the plot
            fig.patch.set_facecolor("white")
            x_trace = plt.subplot(322)
            y_trace = plt.subplot(324)
            z_trace = plt.subplot(321)
            p_onset = plt.subplot(323)
            s_onset = plt.subplot(326)

            # Plotting the traces
            self._plot_signal_trace(x_trace, self.times,
                                 self.data.filtered_signal[0, i, :], -1, "r")
            self._plot_signal_trace(y_trace, self.times,
                                 self.data.filtered_signal[1, i, :], -1, "b")
            self._plot_signal_trace(z_trace, self.times,
                                 self.data.filtered_signal[2, i, :], -1, "g")
            p_onset.plot(self.times, self.data.p_onset[i, :], "r", 
                         linewidth=0.5)
            s_onset.plot(self.times, self.data.s_onset[i, :], "b", 
                         linewidth=0.5)

            # Defining Pick and Error
            picks = self.phase_picks["Pick"]
            phase_picks = picks[picks["Name"] == station].replace(-1, np.nan)
            phase_picks = phase_picks.reset_index(drop=True)

            for j, pick in phase_picks.iterrows():
                if np.isnan(pick["PickError"]):
                    continue

                pick_time = pick["PickTime"]
                pick_err = pick["PickError"]

                if pick["Phase"] == "P":
                    self._pick_vlines(z_trace, pick_time, pick_err)

                    yy = util.gaussian_1d(gau_p["xdata"],
                                          gau_p["popt"][0],
                                          gau_p["popt"][1],
                                          gau_p["popt"][2])
                    gau_dts = [x.datetime for x in gau_p["xdata_dt"]]
                    p_onset.plot(gau_dts, yy)
                    self._pick_vlines(p_onset, pick_time, pick_err)
                else:
                    self._pick_vlines(y_trace, pick_time, pick_err)
                    self._pick_vlines(x_trace, pick_time, pick_err)

                    yy = util.gaussian_1d(gau_s["xdata"],
                                          gau_s["popt"][0],
                                          gau_s["popt"][1],
                                          gau_s["popt"][2])
                    gau_dts = [x.datetime for x in gau_s["xdata_dt"]]
                    s_onset.plot(gau_dts, yy)
                    self._pick_vlines(s_onset, pick_time, pick_err)

            dt_max = self.event_mw_data["DT"].iloc[np.argmax(self.event_mw_data["COA"])]
            dt_max = UTCDateTime(dt_max)
            self._ttime_vlines(z_trace, dt_max, ptt[i])
            self._ttime_vlines(p_onset, dt_max, ptt[i])
            self._ttime_vlines(y_trace, dt_max, stt[i])
            self._ttime_vlines(x_trace, dt_max, stt[i])
            self._ttime_vlines(s_onset, dt_max, stt[i])

            p_onset.axhline(gau_p["PickThreshold"])
            s_onset.axhline(gau_s["PickThreshold"])

            # Refining the window as around the pick time
            min_t = (dt_max + 0.5 * ptt[i]).datetime
            max_t = (dt_max + 1.5 * stt[i]).datetime

            x_trace.set_xlim([min_t, max_t])
            y_trace.set_xlim([min_t, max_t])
            z_trace.set_xlim([min_t, max_t])
            p_onset.set_xlim([min_t, max_t])
            s_onset.set_xlim([min_t, max_t])

            suptitle = "Trace for Station {} - PPick = {}, SPick = {}"
            suptitle = suptitle.format(station,
                                       gau_p["PickValue"], gau_s["PickValue"])

            fig.suptitle(suptitle)

            if file_str is None:
                plt.show()
            else:
                out_str = out_dir / file_str
                fname = "{}_{}.pdf"
                fname = fname.format(out_str, station)
                plt.savefig(fname)
                plt.close("all")

    def coalescence_video(self, file_str=None):
        """
        Generate a video over the marginal window showing the coalescence map
        and expected arrival times overlain on the station traces.

        Parameters
        ----------
        file_str : str, optional
            String {run_name}_{event_name} (figure displayed by default)

        """

        # Find index of start and end of marginal window
        idx0 = np.where(self.times == self.event_mw_data["DT"].iloc[0])[0][0]
        idx1 = np.where(self.times == self.event_mw_data["DT"].iloc[-1])[0][0]

        Writer = animation.writers["ffmpeg"]
        writer = Writer(fps=4, metadata=dict(artist="Ulvetanna"), bitrate=1800)

        fig = self._coalescence_frame(idx0)
        ani = animation.FuncAnimation(fig, self._video_update,
                                      frames=np.linspace(idx0+1, idx1, 200),
                                      blit=False, repeat=False)

        if file_str is None:
            plt.show()
        else:
            subdir = "videos"
            util._make_directories(self.run_path, subdir=subdir)
            out_str = self.run_path / subdir / file_str
            ani.save("{}_CoalescenceVideo.mp4".format(out_str),
                     writer=writer)

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
            eq = self.event.iloc[0]
        else:
            msg = "\t\tError: no event specified!"
            print(msg)
            return

        dt_max = (self.event_mw_data["DT"].iloc[np.argmax(self.event_mw_data["COA"])]).to_pydatetime()

        # Determining the marginal window value from the coalescence function
        coa_map = np.ma.masked_invalid(self.coa_map)
        loc = np.where(coa_map == np.nanmax(coa_map))
        point = np.array([[loc[0][0],
                           loc[1][0],
                           loc[2][0]]])
        crd = self.lut.coord2loc(point, inverse=True)

        # Defining the plots to be represented
        fig = plt.figure(figsize=(30, 15))
        fig.patch.set_facecolor("white")
        xy_slice = plt.subplot2grid((3, 5), (0, 0), colspan=2, rowspan=2)
        xz_slice = plt.subplot2grid((3, 5), (2, 0), colspan=2)
        yz_slice = plt.subplot2grid((3, 5), (0, 2), rowspan=2)
        trace = plt.subplot2grid((3, 5), (0, 3), colspan=2, rowspan=2)
        logo = plt.subplot2grid((3, 5), (2, 2))
        coal_val = plt.subplot2grid((3, 5), (2, 3), colspan=2)

        # --- Ordering by distance to event ---
        if self.range_order:
            ttp = self.lut.get_value_at("TIME_P", point[0])[0]
            sidx = abs(np.argsort(np.argsort(ttp))
                       - np.max(np.argsort(np.argsort(ttp))))
        else:
            sidx = np.argsort(self.data.stations)[::-1]

        for i in range(self.data.signal.shape[1]):
            if not self.filtered_signal:
                self._plot_signal_trace(trace, self.times,
                                     self.data.signal[0, i, :],
                                     sidx[i], color="r")
                self._plot_signal_trace(trace, self.times,
                                     self.data.signal[1, i, :],
                                     sidx[i], color="b")
                self._plot_signal_trace(trace, self.times,
                                     self.data.signal[2, i, :],
                                     sidx[i], color="g")
            else:
                self._plot_signal_trace(trace, self.times,
                                     self.data.filtered_signal[0, i, :],
                                     sidx[i], color="r")
                self._plot_signal_trace(trace, self.times,
                                     self.data.filtered_signal[1, i, :],
                                     sidx[i], color="b")
                self._plot_signal_trace(trace, self.times,
                                     self.data.filtered_signal[2, i, :],
                                     sidx[i], color="g")

        # --- Plotting the Station Travel Times ---
        ttime_range = self.lut.get_value_at("TIME_P", point[0])[0].shape[0]
        tps = []
        tss = []
        dt_max = UTCDateTime(dt_max)
        tmp_p = self.lut.get_value_at("TIME_P", point[0])
        tmp_s = self.lut.get_value_at("TIME_S", point[0])
        for i in range(ttime_range):
            tps.append((dt_max + tmp_p[0][i]).datetime)
            tss.append((dt_max + tmp_s[0][i]).datetime)

        del tmp_p, tmp_s

        self.tp_arrival = trace.scatter(tps, (sidx + 1), 50, "pink",
                                        marker="v", zorder=4, linewidth=0.1,
                                        edgecolors="black")
        self.ts_arrival = trace.scatter(tss, (sidx + 1), 50, "purple",
                                        marker="v", zorder=5, linewidth=0.1,
                                        edgecolors="black")

        # Set signal trace limits
        trace.set_xlim([(dt_max-0.1).datetime, 
                        (self.data.end_time-0.8).datetime])
        trace.yaxis.tick_right()
        trace.yaxis.set_ticks(sidx + 1)
        trace.yaxis.set_ticklabels(self.data.stations)
        self.station_trace_vline = trace.axvline(dt_max.datetime, 0, 1000,
                                                 linestyle="--", linewidth=2,
                                                 color="r")

        # --- Plotting the Coalescence Function ---
        self._plot_coalescence_value(coal_val, dt_max.datetime)

        # --- Determining Error ellipse for Covariance ---
        cov_x = eq["GlobalCovariance_ErrX"] / self.lut.cell_size[0]
        cov_y = eq["GlobalCovariance_ErrY"] / self.lut.cell_size[1]
        cov_z = eq["GlobalCovariance_ErrZ"] / self.lut.cell_size[2]

        cov_crd = np.array([[eq["GlobalCovariance_X"],
                             eq["GlobalCovariance_Y"],
                             eq["GlobalCovariance_Z"]]])
        cov_loc = self.lut.coord2loc(cov_crd)
        dCo = abs(cov_crd - self.lut.coord2loc(np.array([[cov_loc[0][0] + cov_x,
                                                          cov_loc[0][1] + cov_y,
                                                          cov_loc[0][2] + cov_z]]),
                                               inverse=True))

        ellipse_XY = Ellipse((eq["GlobalCovariance_X"],
                              eq["GlobalCovariance_Y"]),
                             2 * dCo[0][0], 2 * dCo[0][1], angle=0,
                             linewidth=2, edgecolor="k", fill=False,
                             label="Global Covariance Error Ellipse")
        ellipse_YZ = Ellipse((eq["GlobalCovariance_Z"],
                              eq["GlobalCovariance_Y"]),
                             2 * dCo[0][2], 2 * dCo[0][1], angle=0,
                             linewidth=2, edgecolor="k", fill=False)
        ellipse_XZ = Ellipse((eq["GlobalCovariance_X"],
                              eq["GlobalCovariance_Z"]),
                             2 * dCo[0][0], 2 * dCo[0][2], angle=0,
                             linewidth=2, edgecolor="k", fill=False)

        # --- Determining Error ellipse for Gaussian ---
        gau_x = eq["LocalGaussian_ErrX"] / self.lut.cell_size[0]
        gau_y = eq["LocalGaussian_ErrY"] / self.lut.cell_size[1]
        gau_z = eq["LocalGaussian_ErrZ"] / self.lut.cell_size[2]

        gau_crd = np.array([[eq["LocalGaussian_X"],
                             eq["LocalGaussian_Y"],
                             eq["LocalGaussian_Z"]]])
        gau_loc = self.lut.coord2loc(gau_crd)
        dGa = abs(gau_crd - self.lut.coord2loc(np.array([[gau_loc[0][0] + gau_x,
                                                          gau_loc[0][1] + gau_y,
                                                          gau_loc[0][2] + gau_z]]),
                                               inverse=True))

        gellipse_XY = Ellipse((eq["LocalGaussian_X"],
                               eq["LocalGaussian_Y"]),
                              2 * dGa[0][0], 2 * dGa[0][1], angle=0,
                              linewidth=2, edgecolor="b", fill=False,
                              label="Local Gaussian Error Ellipse")
        gellipse_YZ = Ellipse((eq["LocalGaussian_Z"],
                               eq["LocalGaussian_Y"]),
                              2 * dGa[0][2], 2 * dGa[0][1], angle=0,
                              linewidth=2, edgecolor="b", fill=False)
        gellipse_XZ = Ellipse((eq["LocalGaussian_X"],
                               eq["LocalGaussian_Z"]),
                              2 * dGa[0][0], 2 * dGa[0][2], angle=0,
                              linewidth=2, edgecolor="b", fill=False)

        # --- Plot slices through coalescence map ---
        self._plot_map_slice(xy_slice, eq, coa_map[:, :, int(loc[2][0])], crd,
                             "X", "Y", ellipse_XY, gellipse_XY)
        xy_slice.legend()

        self._plot_map_slice(xz_slice, eq, coa_map[:, int(loc[1][0]), :], crd,
                             "X", "Z", ellipse_XZ, gellipse_XZ)
        xz_slice.invert_yaxis()

        self._plot_map_slice(yz_slice, eq, 
                             np.transpose(coa_map[int(loc[0][0]), :, :]),
                             crd, "Y", "Z", ellipse_YZ, gellipse_YZ)

        # --- Plotting the station locations ---
        xy_slice.scatter(self.lut.station_data["Longitude"],
                         self.lut.station_data["Latitude"],
                         15, marker="^", color=self.line_station_color)
        xz_slice.scatter(self.lut.station_data["Longitude"],
                         self.lut.station_data["Elevation"],
                         15, marker="^", color=self.line_station_color)
        yz_slice.scatter(self.lut.station_data["Elevation"],
                         self.lut.station_data["Latitude"],
                         15, marker="<", color=self.line_station_color)
        for i, txt in enumerate(self.lut.station_data["Name"]):
            xy_slice.annotate(txt, [self.lut.station_data["Longitude"][i],
                                    self.lut.station_data["Latitude"][i]],
                              color=self.line_station_color)

        # --- Plotting the xy_files ---
        self._plot_xy_files(xy_slice)

        # --- Plotting the logo ---
        self._plot_logo(logo, r"Earthquake Location Error", 10)

        if file_str is None:
            plt.show()
        else:
            fig.suptitle("Event Origin Time = {}".format(dt_max.datetime))
            subdir = "summaries"
            util._make_directories(self.run_path, subdir=subdir)
            out_str = self.run_path / subdir / file_str
            plt.savefig("{}_EventSummary.pdf".format(out_str), dpi=400)
            plt.close("all")

    def _plot_map_slice(self, ax, eq, slice_, crd, c1, c2, ee, gee):
        """
        Plot slice through map in a given plane.

        """

        crd_crnrs = self.lut.xyz2coord(self.lut.grid_corners)
        cells = self.lut.cell_count

        # Series of tests to select the correct components for the given slice
        if c1 == "X":
            min1, max1 = min(crd_crnrs[:, 0]), max(crd_crnrs[:, 0])
            size1 = (max1 - min1) / cells[0]
            idx1 = 0
        elif c1 == "Y":
            min2, max2 = min(crd_crnrs[:, 1]), max(crd_crnrs[:, 1])
            size2 = (max2 - min2) / cells[1]
            idx2 = 1
            min1, max1 = min(crd_crnrs[:, 2]), max(crd_crnrs[:, 2])
            size1 = (max1 - min1) / cells[2]
            idx1 = 2

        if c2 == "Y":
            min2, max2 = min(crd_crnrs[:, 1]), max(crd_crnrs[:, 1])
            size2 = (max2 - min2) / cells[1]
            idx2 = 1
        elif c2 == "Z" and c1 == "Y":
            pass
        elif c2 == "Z" and c1 == "X":
            min2, max2 = min(crd_crnrs[:, 2]), max(crd_crnrs[:, 2])
            size2 = (max2 - min2) / cells[2]
            idx2 = 2

        # Create meshgrid with shape (X + 1, Y + 1) - pcolormesh uses the grid
        # values as fenceposts
        grid1, grid2 = np.mgrid[min1:max1 + size1:size1,
                                min2:max2 + size2:size2]

        # Ensure that the shape of grid1 and grid2 comply with the shape of the
        # slice (sometimes floating point errors can carry over and return a
        # grid with incorrect shape)
        grid1 = grid1[:slice_.shape[0] + 1, :slice_.shape[1] + 1]
        grid2 = grid2[:slice_.shape[0] + 1, :slice_.shape[1] + 1]
        ax.pcolormesh(grid1, grid2, slice_, cmap=self.cmap, edgecolors="face")
        ax.set_xlim([min1, max1])
        ax.set_ylim([min2, max2])
        if c1 == "Y" and c2 == "Z":
            ax.set_xlim([max1, min1])
        elif c1 == "X" and c2 == "Z":
            ax.set_ylim([max2, min2])

        if c1 == "Y":
            c1, c2 = c2, c1

        ax.axvline(x=crd[0][idx1], linestyle="--", linewidth=2,
                   color=self.line_station_color)
        ax.axhline(y=crd[0][idx2], linestyle="--", linewidth=2,
                   color=self.line_station_color)
        ax.scatter(eq[c1], eq[c2], 150, c="green", marker="*",
                   label="Maximum Coalescence Location")
        ax.scatter(eq["LocalGaussian_{}".format(c1)],
                   eq["LocalGaussian_{}".format(c2)],
                   150, c="pink", marker="*",
                   label="Local Gaussian Location")
        ax.scatter(eq["GlobalCovariance_{}".format(c1)],
                   eq["GlobalCovariance_{}".format(c2)],
                   150, c="blue", marker="*",
                   label="Global Covariance Location")
        ax.add_patch(ee)
        ax.add_patch(gee)

    def _plot_signal_trace(self, trace, x, y, st_idx, color):
        """
        Plot signal trace.

        """

        if y.any():
            trace.plot(x, y / np.max(abs(y)) * self.trace_scale + (st_idx + 1),
                       color=color, linewidth=0.5, zorder=1)

    def _coalescence_frame(self, tslice_idx):
        """
        Plots a frame of a coalescence video at a particular time.

        Parameters
        ----------
        tslice_idx : int
            Index for the current time slice

        """

        tslice = self.times[tslice_idx]
        idx = np.where(self.event_mw_data["DT"] == tslice)[0][0]
        loc = self.lut.coord2loc(np.array([[self.event_mw_data["X"].iloc[idx],
                                            self.event_mw_data["Y"].iloc[idx],
                                            self.event_mw_data["Z"].iloc[idx]]])
                                 ).astype(int)[0]
        point = np.array([loc[0],
                          loc[1],
                          loc[2]])
        crd = np.array([[self.event_mw_data["X"].iloc[idx],
                         self.event_mw_data["Y"].iloc[idx],
                         self.event_mw_data["Z"].iloc[idx]]])[0, :]

        # --- Defining the plot area ---
        fig = plt.figure(figsize=(30, 15))
        fig.patch.set_facecolor("white")
        xy_slice = plt.subplot2grid((3, 5), (0, 0), colspan=2, rowspan=2)
        xz_slice = plt.subplot2grid((3, 5), (2, 0), colspan=2)
        yz_slice = plt.subplot2grid((3, 5), (0, 2), rowspan=2)
        trace = plt.subplot2grid((3, 5), (0, 3), colspan=2, rowspan=2)
        logo = plt.subplot2grid((3, 5), (2, 2))
        coal_val = plt.subplot2grid((3, 5), (2, 3), colspan=2)

        # --- Plotting the Traces ---
        idx0 = np.where(self.times == self.event_mw_data["DT"].iloc[0])[0][0]

        # --- Defining the stations in alphabetical order ---
        if self.range_order:
            ttp = self.lut.get_value_at("TIME_P", point)[0]
            sidx = np.argsort(ttp)[::-1]
        else:
            sidx = np.argsort(self.data.stations)[::-1]

        for i in range(self.data.signal.shape[1]):
            if not self.filtered_signal:
                self._plot_signal_trace(trace, self.times,
                                     self.data.signal[0, i, :],
                                     sidx[i], color="r")
                self._plot_signal_trace(trace, self.times,
                                     self.data.signal[1, i, :],
                                     sidx[i], color="b")
                self._plot_signal_trace(trace, self.times,
                                     self.data.signal[2, i, :],
                                     sidx[i], color="g")
            else:
                self._plot_signal_trace(trace, self.times,
                                     self.data.filtered_signal[0, i, :],
                                     sidx[i], color="r")
                self._plot_signal_trace(trace, self.times,
                                     self.data.filtered_signal[1, i, :],
                                     sidx[i], color="b")
                self._plot_signal_trace(trace, self.times,
                                     self.data.filtered_signal[2, i, :],
                                     sidx[i], color="g")

        # --- Plotting the Station Travel Times ---
        ttime_range = self.lut.get_value_at("TIME_P", point)[0].shape[0]
        dt_max = self.event_mw_data["DT"].iloc[np.argmax(self.event_mw_data["COA"])]
        tps = []
        tss = []
        dt_max = UTCDateTime(dt_max)

        tmp_p = self.lut.get_value_at("TIME_P", point)
        tmp_s = self.lut.get_value_at("TIME_S", point)
        for i in range(ttime_range):

            tps.append((dt_max + tmp_p[0][i]).datetime)
            tss.append((dt_max + tmp_s[0][i]).datetime)

        del tmp_p, tmp_s

        self.tp_arrival = trace.scatter(tps, (sidx + 1), 50, "pink",
                                        marker="v", zorder=4, linewidth=0.1,
                                        edgecolors="black")
        self.ts_arrival = trace.scatter(tss, (sidx + 1), 50, "purple",
                                        marker="v", zorder=5, linewidth=0.1,
                                        edgecolors="black")

        # Set coalescence trace limits
        # trace.set_ylim([0, i + 2])
        trace.set_xlim([(self.data.start_time).datetime, np.max(tss)])
        # trace.get_xaxis().set_ticks([])
        trace.yaxis.tick_right()
        trace.yaxis.set_ticks(sidx + 1)
        trace.yaxis.set_ticklabels(self.data.stations)
        self.station_trace_vline = trace.axvline(dt_max.datetime, 0, 1000,
                                                 linestyle="--", linewidth=2,
                                                 color="r")

        # --- Plotting the Coalescence Function ---
        self._plot_coalescence_value(coal_val, tslice)

        # --- Plotting the Coalescence Value Slices ---
        crd_crnrs = self.lut.xyz2coord(self.lut.grid_corners)
        cells = self.lut.cell_count
        xmin = min(crd_crnrs[:, 0])
        xmax = max(crd_crnrs[:, 0])
        xcells = cells[0]
        xsize = (xmax - xmin) / xcells
        ymin = min(crd_crnrs[:, 1])
        ymax = max(crd_crnrs[:, 1])
        ycells = cells[1]
        ysize = (ymax - ymin) / ycells
        zmin = min(crd_crnrs[:, 2])
        zmax = max(crd_crnrs[:, 2])
        zcells = cells[2]
        zsize = (zmax - zmin) / zcells

        # xy_slice
        grid1, grid2 = np.mgrid[xmin:xmax + xsize:xsize,
                                ymin:ymax + ysize:ysize]
        self.xy_plot = xy_slice.pcolormesh(grid1, grid2,
                                           (self.map_4d[:, :, int(loc[2]),
                                            int(tslice_idx - idx0)]
                                            / self.map_max), cmap=self.cmap)
        xy_slice.set_xlim([xmin, xmax])
        xy_slice.set_ylim([ymin, ymax])
        self.xy_vline = xy_slice.axvline(x=crd[0], linestyle="--", linewidth=2,
                                         color="k")
        self.xy_hline = xy_slice.axhline(y=crd[1], linestyle="--", linewidth=2,
                                         color="k")

        # xz_slice
        grid1, grid2 = np.mgrid[xmin:xmax + xsize:xsize,
                                zmin:zmax + zsize:zsize]
        self.xz_plot = xz_slice.pcolormesh(grid1, grid2,
                                           (self.map_4d[:, int(loc[1]), :,
                                            int(tslice_idx - idx0)]
                                            / self.map_max), cmap=self.cmap)
        xz_slice.set_xlim([xmin, xmax])
        xz_slice.set_ylim([zmax, zmin])
        self.xz_vline = xz_slice.axvline(x=crd[0], linestyle="--", linewidth=2,
                                         color="k")
        self.xz_hline = xz_slice.axhline(y=crd[2], linestyle="--", linewidth=2,
                                         color="k")
        xz_slice.invert_yaxis()

        # yz_slice
        grid1, grid2 = np.mgrid[zmin:zmax + zsize:zsize,
                                ymin:ymax + ysize:ysize]

        self.yz_plot = yz_slice.pcolormesh(grid1, grid2,
                                           (np.transpose(
                                            self.map_4d[int(loc[0]), :, :,
                                                     int(tslice_idx - idx0)])
                                            / self.map_max), cmap=self.cmap)
        yz_slice.set_xlim([zmax, zmin])
        yz_slice.set_ylim([ymin, ymax])
        self.yz_vline = yz_slice.axvline(x=crd[2], linestyle="--", linewidth=2,
                                         color="k")
        self.yz_hline = yz_slice.axhline(y=crd[1], linestyle="--", linewidth=2,
                                         color="k")

        # --- Plotting the station locations ---
        xy_slice.scatter(self.lut.station_data["Longitude"],
                         self.lut.station_data["Latitude"], 15, "k", marker="^")
        xz_slice.scatter(self.lut.station_data["Longitude"],
                         self.lut.station_data["Elevation"], 15, "k", marker="^")
        yz_slice.scatter(self.lut.station_data["Elevation"],
                         self.lut.station_data["Latitude"], 15, "k", marker="<")
        for i, txt in enumerate(self.lut.station_data["Name"]):
            xy_slice.annotate(txt, [self.lut.station_data["Longitude"][i],
                                    self.lut.station_data["Latitude"][i]])

        # --- Plotting the xy_files ---
        self._plot_xy_files(xy_slice)

        # --- Plotting the logo ---
        self._plot_logo(logo, r"Coalescence Video", 14)

        return fig

    def _video_update(self, frame):
        """
        Plot latest video frame.

        Parameters
        ----------
        frame : int
            Current frame number

        """

        frame = int(frame)
        idx0 = np.where(self.times == self.event_mw_data["DT"].iloc[0])[0][0]
        tslice = self.times[int(frame)]
        idx = np.where(self.event_mw_data["DT"] == tslice)[0][0]
        crd = np.array([[self.event_mw_data["X"].iloc[idx],
                         self.event_mw_data["Y"].iloc[idx],
                         self.event_mw_data["Z"].iloc[idx]]])
        loc = self.lut.coord2loc(crd).astype(int)[0]
        crd = crd[0, :]

        # Updating the Coalescence Value and Trace Lines
        self.station_trace_vline.set_xdata(tslice)
        self.coal_val_vline.set_xdata(tslice)

        # Updating the Coalescence Maps
        self.xy_plot.set_array((self.map_4d[:, :, loc[2], int(idx0 - frame)]
                                / self.map_max)[:-1, :-1].ravel())
        self.xz_plot.set_array((self.map_4d[:, loc[1], :, int(idx0 - frame)]
                                / self.map_max)[:-1, :-1].ravel())
        self.yz_plot.set_array((np.transpose(self.map_4d[loc[0], :, :, 
                                                      int(idx0 - frame)])
                                / self.map_max)[:-1, :-1].ravel())

        # Updating the coalescence lines
        self.xy_vline.set_xdata(crd[0])
        self.xy_hline.set_ydata(crd[1])
        self.yz_vline.set_xdata(crd[2])
        self.yz_hline.set_ydata(crd[1])
        self.xz_vline.set_xdata(crd[0])
        self.xz_hline.set_ydata(crd[2])

        # Get P- and S-traveltimes at this location
        ptt = self.lut.get_value_at("TIME_P", np.array([loc]))[0]
        stt = self.lut.get_value_at("TIME_S", np.array([loc]))[0]
        tps = []
        tss = []
        for i in range(ptt.shape[0]):
            tps.append(np.argmin(abs((self.times -
                                      (tslice + timedelta(seconds=ptt[i]))))))
            tss.append(np.argmin(abs((self.times -
                                      (tslice + timedelta(seconds=stt[i]))))))

        self.tp_arrival.set_offsets(np.c_[tps,
                                    (np.arange(len(tps)) + 1)])
        self.ts_arrival.set_offsets(np.c_[tss,
                                    (np.arange(len(tss)) + 1)])

    def _pick_vlines(self, trace, pick_time, pick_err):
        """
        Plot vlines showing phase pick time and uncertainty.

        """

        trace.axvline((pick_time - pick_err/2).datetime,
                      linestyle="--")
        trace.axvline((pick_time + pick_err/2).datetime,
                      linestyle="--")
        trace.axvline((pick_time).datetime)

    def _ttime_vlines(self, trace, dt_max, ttime):
        """
        Plot vlines showing expected arrival times based on max
        coalescence location.

        """

        trace.axvline((dt_max + ttime).datetime, color="red")
        trace.axvline((dt_max + 0.9 * ttime - self.marginal_window).datetime,
                      color="red", linestyle="--")
        trace.axvline((dt_max + 1.1 * ttime + self.marginal_window).datetime,
                      color="red", linestyle="--")

    def _plot_xy_files(self, slice_):
        """
        Plot xy files supplied by user.

        Reads file list from self.xy_files (with columns ["File", "Color",
                                                          "Linewidth",
                                                          "Linestyle"] )
        where File is the file path to the xy file to be plotted on the
        map. File should contain two columns ["Longitude", "Latitude"].

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
                slice_.plot(xy_file["Longitude"], xy_file["Latitude"],
                            linestyle=xy_file["Linestyle"],
                            linewidth=xy_file["Linewidth"],
                            color=xy_file["Color"])

    def _plot_logo(self, plot, txt, fontsize):
        """
        Plot QuakeMigrate logo.

        """

        try:
            plot.axis("off")
            im = mpimg.imread(str(self.logo))
            plot.imshow(im)
            plot.text(150, 200, txt,
                      fontsize=fontsize, style="italic")
        except:
            print("\t\tLogo not plotting")

    def _plot_coalescence_value(self, plot, tslice):
        """
        Plot max coalescence value in the grid through time.

        """

        plot.plot(self.event_mw_data["DT"], self.event_mw_data["COA"],
                  zorder=10)
        plot.set_ylabel("Coalescence value")
        plot.set_xlabel("Date-Time")
        plot.yaxis.tick_right()
        plot.yaxis.set_label_position("right")
        plot.set_xlim([self.event_mw_data["DT"].iloc[0],
                       self.event_mw_data["DT"].iloc[-1]])
        for tick in plot.get_xticklabels():
            tick.set_rotation(45)

        self.coal_val_vline = plot.axvline(tslice, 0, 1000, linestyle="--",
                                           linewidth=2, color="r")
