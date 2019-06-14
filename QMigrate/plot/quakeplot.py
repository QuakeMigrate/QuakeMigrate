# -*- coding: utf-8 -*-
"""
Module to produce gridded traveltime velocity models

"""

import os
import pathlib
from datetime import datetime, timedelta

from obspy import UTCDateTime
import pandas as pd
import numpy as np
from matplotlib.collections import PatchCollection
import matplotlib
try:
    os.environ["DISPLAY"]
    matplotlib.use("Qt5Agg")
except KeyError:
    matplotlib.use("Agg")
import matplotlib.pylab as plt
from matplotlib.patches import Rectangle, Ellipse
import matplotlib.image as mpimg
import matplotlib.animation as animation

import QMigrate.util as util


class QuakePlot:
    """
    QuakeMigrate plotting class

    Describes methods for plotting various outputs of QuakeMigrate

    Methods
    -------
    coalescence_video()
        Generate a video of coalescence over the period of earthquake location
    coalescence_location()
        Location plot

    """

    logo = (pathlib.Path(__file__) / "QuakeMigrate").with_suffix(".png")

    def __init__(self, lut, map_, coa_map, data, event, station_pick,
                 marginal_window, options=None):
        """
        Initialisation of SeisPlot object

        Parameters
        ----------
        lut :

        map_ :

        coa_map :

        data :

        event :

        station_pick :

        marginal_window :

        options :


        """

        self.lut = lut
        self.map = map_
        self.data = data
        self.event = event
        self.coa_map = coa_map
        self.stat_pick = station_pick
        self.marginal_window = marginal_window
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

        tmp = np.arange(self.data.start_time,
                        self.data.end_time + self.data.sample_size,
                        self.data.sample_size)
        self.times = pd.to_datetime([x.datetime for x in tmp])
        # Convert event["DT"] to python datetime object
        if not isinstance(self.event["DT"].iloc[0], datetime):
            self.event["DT"] = [x.datetime for x in self.event["DT"]]

        self.event = self.event[(self.event["DT"] > self.times[0])
                                & (self.event["DT"] < self.times[-1])]

        self.map_max = np.nanmax(map_)

        self.coal_trace_vline = None
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

    def coalescence_trace(self, output_file=None):
        """
        Plots a figure showing the behaviour of the coalescence value through
        time as a trace

        Parameters
        ----------
        output_file : str, optional
            Directory to output PDF of figure (figure displayed by default)

        """

        # Determining the marginal window value from the coalescence function
        map_ = self.coa_map
        loc = np.where(map_ == np.max(map_))
        point = np.array([loc[0][0],
                          loc[1][0],
                          loc[2][0]])

        # Get P- and S-traveltimes at this location
        ptt = self.lut.get_value_at("TIME_P", point)[0]
        stt = self.lut.get_value_at("TIME_S", point)[0]

        # Looping through all stations
        for i in range(self.data.signal.shape[1]):
            station = self.lut.station_data["Name"][i]
            gau_p = self.stat_pick["GAU_P"][i]
            gau_s = self.stat_pick["GAU_S"][i]
            fig = plt.figure(figsize=(30, 15))

            # Defining the plot
            fig.patch.set_facecolor("white")
            x_trace = plt.subplot(322)
            y_trace = plt.subplot(324)
            z_trace = plt.subplot(321)
            p_onset = plt.subplot(323)
            s_onset = plt.subplot(326)

            # Plotting the traces
            self._plot_coa_trace(x_trace, self.times,
                                 self.data.filtered_signal[0, i, :], -1, "r")
            self._plot_coa_trace(y_trace, self.times,
                                 self.data.filtered_signal[1, i, :], -1, "b")
            self._plot_coa_trace(z_trace, self.times,
                                 self.data.filtered_signal[2, i, :], -1, "g")
            p_onset.plot(self.times, self.data.p_onset[i, :], "r", linewidth=0.5)
            s_onset.plot(self.times, self.data.s_onset[i, :], "b", linewidth=0.5)

            # Defining Pick and Error
            picks = self.stat_pick["Pick"]
            stat_pick = picks[picks["Name"] == station].replace(-1, np.nan)
            stat_pick = stat_pick.reset_index(drop=True)

            for j, pick in stat_pick.iterrows():
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

            dt_max = self.event["DT"].iloc[np.argmax(self.event["COA"])]
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

            if output_file is None:
                plt.show()
            else:
                fname = "{}_{}.pdf"
                fname = fname.format(output_file, station)
                plt.savefig(fname)
                plt.close("all")

    def coalescence_video(self, output_file=None):
        idx0 = np.where(self.times == self.event["DT"].iloc[0])[0][0]
        idx1 = np.where(self.times == self.event["DT"].iloc[-1])[0][0]

        Writer = animation.writers["ffmpeg"]
        writer = Writer(fps=4, metadata=dict(artist="Ulvetanna"), bitrate=1800)

        fig = self._coalescence_image(idx0)
        ani = animation.FuncAnimation(fig, self._video_update,
                                      frames=np.linspace(idx0+1, idx1, 200),
                                      blit=False, repeat=False)

        if output_file is None:
            plt.show()
        else:
            ani.save("{}_CoalescenceVideo.mp4".format(output_file),
                     writer=writer)

    def coalescence_marginal(self, output_file=None, earthquake=None):
        """
        Generate a marginal window about the event to determine the error

        Parameters
        ----------
        output_file : str, optional

        earthquake : str, optional

        TO-DO
        -----
        Redefine the marginal as instead of the whole coalescence period,
        Gaussian fit to the coalescence value then take the 1st std to
        define the time window and use this

        """

        # Event is only in first line of earthquake, reduces chars later on
        if earthquake is not None:
            eq = earthquake.iloc[0]
        else:
            msg = "No event specified."
            print(msg)
            return

        dt_max = (self.event["DT"].iloc[np.argmax(self.event["COA"])]).to_pydatetime()

        # Determining the marginal window value from the coalescence function
        map_ = np.ma.masked_invalid(self.coa_map)
        loc = np.where(map_ == np.nanmax(map_))
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
                self._plot_coa_trace(trace, self.times,
                                     self.data.signal[0, i, :],
                                     sidx[i], color="r")
                self._plot_coa_trace(trace, self.times,
                                     self.data.signal[1, i, :],
                                     sidx[i], color="b")
                self._plot_coa_trace(trace, self.times,
                                     self.data.signal[2, i, :],
                                     sidx[i], color="g")
            else:
                self._plot_coa_trace(trace, self.times,
                                     self.data.filtered_signal[0, i, :],
                                     sidx[i], color="r")
                self._plot_coa_trace(trace, self.times,
                                     self.data.filtered_signal[1, i, :],
                                     sidx[i], color="b")
                self._plot_coa_trace(trace, self.times,
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

        # Set coalescence trace limits
        # trace.set_ylim([0, i + 2])
        trace.set_xlim([(dt_max-0.1).datetime, (self.data.end_time-0.8).datetime])
        # trace.get_xaxis().set_ticks([])
        trace.yaxis.tick_right()
        trace.yaxis.set_ticks(sidx + 1)
        trace.yaxis.set_ticklabels(self.data.stations)
        self.coal_trace_vline = trace.axvline(dt_max.datetime, 0, 1000,
                                              linestyle="--", linewidth=2,
                                              color="r")

        # --- Plotting the Coalescence Function ---
        self._plot_coal(coal_val, dt_max.datetime)

        # --- Determining Error ellipse for Covariance ---
        cells = self.lut.cell_count
        xcells = cells[0]
        ycells = cells[1]
        zcells = cells[2]
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

        # ------ Spatial Function ------
        # --- Plotting the marginal window ---
        crd_crnrs = self.lut.xyz2coord(self.lut.grid_corners)
        xmin = min(crd_crnrs[:, 0])
        xmax = max(crd_crnrs[:, 0])
        ymin = min(crd_crnrs[:, 1])
        ymax = max(crd_crnrs[:, 1])
        zmin = min(crd_crnrs[:, 2])
        zmax = max(crd_crnrs[:, 2])

        # xy_slice
        grid1, grid2 = np.mgrid[xmin:xmax:(xmax - xmin) / xcells,
                                ymin:ymax:(ymax - ymin) / ycells]
        rect = Rectangle((np.min(grid1), np.min(grid2)),
                         np.max(grid1) - np.min(grid1),
                         np.max(grid2) - np.min(grid2))
        pc = PatchCollection([rect], facecolor="k")
        xy_slice.add_collection(pc)
        xy_slice.pcolormesh(grid1, grid2, map_[:, :, int(loc[2][0])],
                            cmap=self.cmap, edgecolors="face")
        xy_slice.set_xlim([xmin, xmax])
        xy_slice.set_ylim([ymin, ymax])
        xy_slice.axvline(x=crd[0][0], linestyle="--", linewidth=2,
                         color=self.line_station_color)
        xy_slice.axhline(y=crd[0][1], linestyle="--", linewidth=2,
                         color=self.line_station_color)
        xy_slice.scatter(eq["X"], eq["Y"],
                         150, c="green", marker="*",
                         label="Maximum Coalescence")
        xy_slice.scatter(eq["LocalGaussian_X"], eq["LocalGaussian_Y"],
                         150, c="pink", marker="*",
                         label="Local Gaussian Location")
        xy_slice.scatter(eq["GlobalCovariance_X"], eq["GlobalCovariance_Y"],
                         150, c="blue", marker="*",
                         label="Global Covariance Location")
        xy_slice.add_patch(ellipse_XY)
        xy_slice.add_patch(gellipse_XY)
        xy_slice.legend()

        # xz_slice
        grid1, grid2 = np.mgrid[xmin:xmax:(xmax - xmin) / xcells,
                                zmin:zmax:(zmax - zmin) / zcells]
        rect = Rectangle((np.min(grid1), np.min(grid2)),
                         np.max(grid1) - np.min(grid1),
                         np.max(grid2) - np.min(grid2))
        pc = PatchCollection([rect], facecolor="k")
        xz_slice.add_collection(pc)
        xz_slice.pcolormesh(grid1, grid2, map_[:, int(loc[1][0]), :],
                            cmap=self.cmap, edgecolors="face")
        # CS = xz_slice.contour(grid1, grid2, map_[:, int(loc[1][0]), :],
        #                       levels=[0.65, 0.75, 0.95],
        #                       colors=("g", "m", "k"))
        # xz_slice.clabel(CS, inline=1, fontsize=10)
        xz_slice.set_xlim([xmin, xmax])
        xz_slice.set_ylim([zmax, zmin])
        xz_slice.axvline(x=crd[0][0], linestyle="--", linewidth=2,
                         color=self.line_station_color)
        xz_slice.axhline(y=crd[0][2], linestyle="--", linewidth=2,
                         color=self.line_station_color)
        xz_slice.scatter(eq["X"], eq["Z"],
                         150, c="green", marker="*",
                         label="Maximum Coalescence")
        xz_slice.scatter(eq["LocalGaussian_X"], eq["LocalGaussian_Z"],
                         150, c="pink", marker="*")
        xz_slice.scatter(eq["GlobalCovariance_X"], eq["GlobalCovariance_Z"],
                         150, c="blue", marker="*")
        xz_slice.add_patch(ellipse_XZ)
        xz_slice.add_patch(gellipse_XZ)
        xz_slice.invert_yaxis()

        # yz_slice
        grid1, grid2 = np.mgrid[zmin:zmax:(zmax - zmin) / zcells,
                                ymin:ymax:(ymax - ymin) / ycells]
        rect = Rectangle((np.min(grid1), np.min(grid2)),
                         np.max(grid1) - np.min(grid1),
                         np.max(grid2) - np.min(grid2))
        pc = PatchCollection([rect], facecolor="k")
        yz_slice.add_collection(pc)
        yz_slice.pcolormesh(grid1, grid2, map_[int(loc[0][0]), :, :].transpose(),
                            cmap=self.cmap, edgecolors="face")
        # CS = xz_slice.contour(grid1, grid2, map_[int(loc[0][0]), :, :].transpose(),
        #                       levels=[0.65, 0.75, 0.95],
        #                       colors=("g", "m", "k"))
        yz_slice.set_xlim([zmax, zmin])
        yz_slice.set_ylim([ymin, ymax])
        yz_slice.axvline(x=crd[0][2], linestyle="--", linewidth=2,
                         color=self.line_station_color)
        yz_slice.axhline(y=crd[0][1], linestyle="--", linewidth=2,
                         color=self.line_station_color)
        yz_slice.scatter(eq["Z"], eq["Y"],
                         150, c="green", marker="*",
                         label="Maximum Coalescence")
        yz_slice.scatter(eq["LocalGaussian_Z"], eq["LocalGaussian_Y"],
                         150, c="pink", marker="*")
        yz_slice.scatter(eq["GlobalCovariance_Z"], eq["GlobalCovariance_Y"],
                         150, c="blue", marker="*")
        yz_slice.add_patch(ellipse_YZ)
        yz_slice.add_patch(gellipse_YZ)

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

        if output_file is None:
            plt.show()
        else:
            fig.suptitle("Event Origin Time = {}".format(dt_max.datetime))
            plt.savefig("{}_EventSummary.pdf".format(output_file),
                        dpi=400)
            plt.close("all")

    def _plot_coa_trace(self, trace, x, y, st_idx, color):
        if y.any():
            trace.plot(x, y / np.max(abs(y)) * self.trace_scale + (st_idx + 1),
                       color=color, linewidth=0.5, zorder=1)

    def _coalescence_image(self, tslice_idx):
        """
        Plots a frame of a coalescence video at a particular time.

        Parameters
        ----------
        tslice_idx : int
            Index for the current time slice

        """

        tslice = self.times[tslice_idx]
        idx = np.where(self.event["DT"] == tslice)[0][0]
        loc = self.lut.coord2loc(np.array([[self.event["X"].iloc[idx],
                                            self.event["Y"].iloc[idx],
                                            self.event["Z"].iloc[idx]]])
                                 ).astype(int)[0]
        point = np.array([loc[0],
                          loc[1],
                          loc[2]])
        crd = np.array([[self.event["X"].iloc[idx],
                         self.event["Y"].iloc[idx],
                         self.event["Z"].iloc[idx]]])[0, :]

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
        idx0 = np.where(self.times == self.event["DT"].iloc[0])[0][0]

        # --- Defining the stations in alphabetical order ---
        if self.range_order:
            ttp = self.lut.get_value_at("TIME_P", point)[0]
            sidx = np.argsort(ttp)[::-1]
        else:
            sidx = np.argsort(self.data.stations)[::-1]

        for i in range(self.data.signal.shape[1]):
            if not self.filtered_signal:
                self._plot_coa_trace(trace, self.times,
                                     self.data.signal[0, i, :],
                                     sidx[i], color="r")
                self._plot_coa_trace(trace, self.times,
                                     self.data.signal[1, i, :],
                                     sidx[i], color="b")
                self._plot_coa_trace(trace, self.times,
                                     self.data.signal[2, i, :],
                                     sidx[i], color="g")
            else:
                self._plot_coa_trace(trace, self.times,
                                     self.data.filtered_signal[0, i, :],
                                     sidx[i], color="r")
                self._plot_coa_trace(trace, self.times,
                                     self.data.filtered_signal[1, i, :],
                                     sidx[i], color="b")
                self._plot_coa_trace(trace, self.times,
                                     self.data.filtered_signal[2, i, :],
                                     sidx[i], color="g")

        # --- Plotting the Station Travel Times ---
        ttime_range = self.lut.get_value_at("TIME_P", point)[0].shape[0]
        dt_max = self.event["DT"].iloc[np.argmax(self.event["COA"])]
        tps = []
        tss = []
        dt_max = UTCDateTime(dt_max)
        print(dt_max)
        tmp_p = self.lut.get_value_at("TIME_P", point)
        tmp_s = self.lut.get_value_at("TIME_S", point)
        for i in range(ttime_range):

            tps.append((dt_max + tmp_p[0][i]).datetime)
            tss.append((dt_max + tmp_s[0][i]).datetime)

        del tmp_p, tmp_s
        print(tps)
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
        self.coal_trace_vline = trace.axvline(dt_max.datetime, 0, 1000,
                                              linestyle="--", linewidth=2,
                                              color="r")

        # --- Plotting the Coalescence Function ---
        self._plot_coal(coal_val, tslice)

        # --- Plotting the Coalescence Value Slices ---
        crd_crnrs = self.lut.xyz2coord(self.lut.grid_corners)
        cells = self.lut.cell_count
        xmin = min(crd_crnrs[:, 0])
        xmax = max(crd_crnrs[:, 0])
        xcells = cells[0]
        ymin = min(crd_crnrs[:, 1])
        ymax = max(crd_crnrs[:, 1])
        ycells = cells[1]
        zmin = min(crd_crnrs[:, 2])
        zmax = max(crd_crnrs[:, 2])
        zcells = cells[2]

        # xy_slice
        grid1, grid2 = np.mgrid[xmin:xmax:(xmax - xmin) / xcells,
                                ymin:ymax:(ymax - ymin) / ycells]
        self.xy_plot = xy_slice.pcolormesh(grid1, grid2,
                                           (self.map[:, :, int(loc[2]),
                                                     int(tslice_idx - idx0)]
                                            / self.map_max),
                                           vmin=0, vmax=1, cmap=self.cmap)
        xy_slice.set_xlim([xmin, xmax])
        xy_slice.set_ylim([ymin, ymax])
        self.xy_vline = xy_slice.axvline(x=crd[0], linestyle="--", linewidth=2,
                                         color="k")
        self.xy_hline = xy_slice.axhline(y=crd[1], linestyle="--", linewidth=2,
                                         color="k")

        # xz_slice
        grid1, grid2 = np.mgrid[xmin:xmax:(xmax - xmin) / xcells,
                                zmin:zmax:(zmax - zmin) / zcells]
        self.xz_plot = xz_slice.pcolormesh(grid1, grid2,
                                           (self.map[:, int(loc[1]), :,
                                                     int(tslice_idx - idx0)]
                                            / self.map_max),
                                           vmin=0, vmax=1, cmap=self.cmap)
        xz_slice.set_xlim([xmin, xmax])
        xz_slice.set_ylim([zmax, zmin])
        self.xz_vline = xz_slice.axvline(x=crd[0], linestyle="--", linewidth=2,
                                         color="k")
        self.xz_hline = xz_slice.axhline(y=crd[2], linestyle="--", linewidth=2,
                                         color="k")
        xz_slice.invert_yaxis()

        # yz_slice
        grid1, grid2 = np.mgrid[zmin:zmax:(zmax - zmin) / zcells,
                                ymin:ymax:(ymax - ymin) / ycells]

        self.yz_plot = yz_slice.pcolormesh(grid1, grid2,
                                           (np.transpose(self.map[int(loc[0]), :, :,
                                                                  int(tslice_idx - idx0)])
                                            / self.map_max),
                                           vmin=0, vmax=1, cmap=self.cmap)
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
        frame = int(frame)
        idx0 = np.where(self.times == self.event["DT"].iloc[0])[0][0]
        tslice = self.times[int(frame)]
        idx = np.where(self.event["DT"] == tslice)[0][0]
        crd = np.array([[self.event["X"].iloc[idx],
                         self.event["Y"].iloc[idx],
                         self.event["Z"].iloc[idx]]])
        loc = self.lut.coord2loc(crd).astype(int)[0]
        crd = crd[0, :]

        # Updating the Coalescence Value and Trace Lines
        self.coal_trace_vline.set_xdata(tslice)
        self.coal_val_vline.set_xdata(tslice)

        # Updating the Coalescence Maps
        self.xy_plot.set_array((self.map[:, :, loc[2], int(idx0 - frame)]
                                / self.map_max)[:-1, :-1].ravel())
        self.xz_plot.set_array((self.map[:, loc[1], :, int(idx0 - frame)]
                                / self.map_max)[:-1, :-1].ravel())
        self.yz_plot.set_array((np.transpose(self.map[loc[0], :, :, int(idx0 - frame)])
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
        trace.axvline((pick_time - pick_err/2).datetime,
                      linestyle="--")
        trace.axvline((pick_time + pick_err/2).datetime,
                      linestyle="--")
        trace.axvline((pick_time).datetime)

    def _ttime_vlines(self, trace, dt_max, ttime):
        trace.axvline((dt_max + ttime).datetime, color="red")
        trace.axvline((dt_max + 0.9 * ttime - self.marginal_window).datetime,
                      color="red", linestyle="--")
        trace.axvline((dt_max + 1.1 * ttime + self.marginal_window).datetime,
                      color="red", linestyle="--")

    def _plot_xy_files(self, slice_):
        if self.xy_files is not None:
            xy_files = pd.read_csv(self.xy_files,
                                   names=["File", "Color",
                                          "Linewidth", "Linestyle"])
            for i, f in xy_files.iterrows():
                xy_file = pd.read_csv(f["File"], names=["X", "Y"])
                slice_.plot(xy_file["X"], xy_file["Y"],
                            linestyle=xy_file["Linestyle"],
                            linewidth=xy_file["Linewidth"],
                            color=xy_file["Color"])

    def _plot_logo(self, plot, txt, fontsize):
        try:
            plot.axis("off")
            im = mpimg.imread(str(self.logo))
            plot.imshow(im)
            plot.text(150, 200, txt,
                      fontsize=fontsize, style="italic")
        except:
            print("    \tLogo not plotting")

    def _plot_coal(self, plot, tslice):
        # --- Plotting the Coalescence Function ---
        plot.plot(self.event["DT"], self.event["COA"], zorder=10)
        plot.set_ylabel("Coalescence value")
        plot.set_xlabel("Date-Time")
        plot.yaxis.tick_right()
        plot.yaxis.set_label_position("right")
        plot.set_xlim([self.event["DT"].iloc[0], self.event["DT"].iloc[-1]])
        for tick in plot.get_xticklabels():
            tick.set_rotation(45)

        self.coal_val_vline = plot.axvline(tslice, 0, 1000, linestyle="--",
                                           linewidth=2, color="r")
