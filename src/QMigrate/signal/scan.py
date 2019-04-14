# -*- coding: utf-8 -*-
"""
Module to produce gridded traveltime velocity models

"""

__version__ = "0.1"
__author__ = ""

import re
import os
import pathlib
import time
from datetime import datetime, timedelta

import obspy
from obspy import Stream, Trace, UTCDateTime
from obspy.signal.trigger import classic_sta_lta
from obspy.signal.invsim import cosine_taper
import pandas as pd
from scipy import stats
from scipy.signal import butter, lfilter
from scipy.optimize import curve_fit
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
import matplotlib
try:
    os.environ["DISPLAY"]
    matplotlib.use("Qt5Agg")
except KeyError:
    matplotlib.use("Agg")
import matplotlib.pylab as plt
import matplotlib.dates as mdates
from matplotlib import patches
import matplotlib.image as mpimg
import matplotlib.animation as animation

import QMigrate.core.model as cmod
import QMigrate.core.QMigratelib as ilib


def TicTocGenerator():
    # Generator that returns time differences
    ti = 0  # initial time
    tf = time.time()  # final time
    while True:
        ti = tf
        tf = time.time()
        yield tf - ti  # returns the time difference


TicToc = TicTocGenerator()  # create an instance of the TicTocGen generator


# This will be the main function through which we define both tic() and toc()
def toc(tmp_bool=True):
    # Prints the time difference yielded by generator instance TicToc
    tmp_time_interval = next(TicToc)
    if tmp_bool:
        print("Elapsed time: {} seconds.\n".format(tmp_time_interval))


def tic():
    # Records a time in TicToc, marks the beginning of a time interval
    toc(False)


def gaussian_1d(x, a, b, c):
    """
    Create a 1-dimensional Gaussian function

    Parameters
    ----------
    x : 

    a : 

    b : 

    c : 

    Returns
    -------
    f : 

    """

    f = a * np.exp(-1. * ((x - b) ** 2) / (2 * (c ** 2)))
    return f


def gaussian_3d(nx, ny, nz, sgm):
    """
    Create a 3-dimensional Gaussian function

    Parameters
    ----------
    nx : 

    ny : 

    nz : 

    sgm :

    Returns
    -------


    """

    nx2 = (nx - 1) / 2
    ny2 = (ny - 1) / 2
    nz2 = (nz - 1) / 2
    x = np.linspace(-nx2, nx2, nx)
    y = np.linspace(-ny2, ny2, ny)
    z = np.linspace(-nz2, nz2, nz)
    ix, iy, iz = np.meshgrid(x, y, z, indexing='ij')
    if np.isscalar(sgm):
        sgm = np.repeat(sgm, 3)
    sx, sy, sz = sgm
    return np.exp(- (ix * ix) / (2 * sx * sx)
                  - (iy * iy) / (2 * sy * sy)
                  - (iz * iz) / (2 * sz * sz))


def sta_lta_centred(a, nsta, nlta):
    """
    Calculates the ratio of the average signal of a short-term window to a
    long-term window.

    Parameters
    ----------
    a : array-like
        Signal array
    nsta : int
        Number of samples in short-term window
    nlta : int
        Number of samples in long-term window

    """

    nsta = int(nsta)
    nlta = int(nlta)

    # Cumulative sum to calculate moving average
    sta = np.cumsum(a ** 2)
    sta = np.require(sta, dtype=np.float)
    lta = sta.copy()

    # Compute the STA and the LTA
    sta[nsta:] = sta[nsta:] - sta[:-nsta]
    sta[nsta:-nsta] = sta[nsta * 2]
    sta /= nsta

    lta[nlta:] = lta[nlta:] - lta[:-nlta]
    lta /= nlta

    sta[:(nlta - 1)] = 0
    sta[-nsta:] = 0

    # Avoid division by zero by setting zero values to tiny float
    dtiny = np.finfo(0.0).tiny
    idx = lta < dtiny
    lta[idx] - dtiny

    return sta / lta


def onset(sig, stw, ltw, centred=False):
    """
    Define an onset function

    Parameters
    ----------
    sig : 

    stw : 

    ltw : 

    Returns
    -------

    """

    n_channels, n_samples = sig.shape
    onset = np.copy(sig)
    onset_raw = np.copy(sig)
    for i in range(n_channels):
        if np.sum(sig[i, :]) == 0.0:
            onset[i, :] = 0.0
            onset_raw[i, :] = onset[i, :]
        else:
            if centred is True:
                onset[i, :] = sta_lta_centred(sig[i, :], stw, ltw)
            else:
                onset[i, :] = classic_sta_lta(sig[i, :], stw, ltw)
            onset_raw[i, :] = onset[i, :]
            np.clip(1 + onset[i, :], 0.8, np.inf, onset[i, :])
            np.log(onset[i, :], onset[i, :])

    return onset_raw, onset


def filter(sig, srate, lc, hc, order=3):
    """

    """

    b1, a1 = butter(order, [2.0*lc/srate, 2.0*hc/srate], btype='band')
    nchan, nsamp = sig.shape
    fsig = np.copy(sig)
    # sig = detrend(sig)
    for ch in range(0, nchan):
        fsig[ch, :] = fsig[ch, :] - fsig[ch, 0]
        tap = cosine_taper(len(fsig[ch, :]), 0.1)
        fsig[ch, :] = fsig[ch, :] * tap
        fsig[ch, :] = lfilter(b1, a1, fsig[ch, ::-1])[::-1]
        fsig[ch, :] = lfilter(b1, a1, fsig[ch, :])

    return fsig


def _find(obj, name, default=None):
    if isinstance(name, str):
        if name in obj:
            return obj[name]
        else:
            return default
    elif name[0] in obj:
        if len(name) == 1:
            return obj[name[0]]
        else:
            return _find(obj[name[0]], name[1:], default)
    else:
        return default


class SeisOutFile:
    """
    Input / output control class

    Provides the basic methods for input / output of seismic scan files

    Attributes
    ----------
    path : pathlib Path object
        Location of input/output files
    file_sample_rate : float
        Sample rate (units: ms)

    Methods
    -------
    read_scan()
        Parse information from an existing .scn file
    write_scan(daten, dsnr, dloc)
        Create a new .scn file
    del_scan()
        Delete an existing .scn file

    """

    SCAN_COLS = ["DT", "COA", "X", "Y", "Z"]

    def __init__(self, path='', name=None):
        """
        Class initialisation method

        Parameters
        ----------
        path : str, optional

        name: str, optional

        """

        path = pathlib.Path(path)
        if name is None:
            name = datetime.now().strftime('RUN_%Y%m%d_%H%M%S')
        self.path = path / name
        print("Path = {}, Name = {}".format(str(path), name))

        self.name = name

        self.file_sample_rate = None

    def read_scan(self):
        """
        Parse information from an existing .scn file

        Returns
        -------
        data : pandas DataFrame object
            DataFrame containing information from a seismic scan file

        """

        fname = self.path.with_suffix(".scn")
        data = pd.read_csv(fname, names=self.SCAN_COLS)
        data["DT"] = data["DT"].apply(UTCDateTime)
        return data

    def write_scan(self, daten, dsnr, dloc):
        """
        Create a new .scn file

        Parameters
        ----------
        daten : 

        dsnr : 

        dloc : 


        """

        fname = self.path.with_suffix(".scn")

        df_params = {"DT": daten,
                     "COA": dsnr,
                     "X": dloc[:, 0],
                     "Y": dloc[:, 1],
                     "Z": dloc[:, 2]}

        df = pd.DataFrame(df_params)
        df["DT"] = df["DT"].apply(UTCDateTime)

        if self.file_sample_rate is not None:
            df = df.set_index(df['DT'])
            df = df.resample('{}L'.format(self.file_sample_rate)).mean()
            df = df.reset_index()
            df = df.rename(columns={"index": "DT"})

        df['DT'] = df['DT'].astype(str)

        if fname.exists():
            mode = 'a'  # append if already exists
        else:
            mode = 'w'  # make a new file if not

        array = np.array(df)

        with fname.open(mode=mode) as f:
            for i in range(array.shape[0]):
                f.write('{},{},{},{},{}\n'.format(array[i, 0],
                                                  array[i, 1],
                                                  array[i, 2],
                                                  array[i, 3],
                                                  array[i, 4]))

    def del_scan(self):
        """
        Delete an existing .scn file

        """

        fname = self.path.with_suffix(".scn")
        if fname.exists():
            print('Filename {} already exists - deleting.'.format(str(fname)))
            fname.unlink()

    def read_coal4D(self, fname):
        """
        Reads a binary file

        Parameters
        ----------
        fname : str or pathlib.Path object
            Location of file to be read

        """

        map_ = np.load(fname)
        return map_

    def write_coal4D(self, map_, event, start_time, end_time):
        """
        Outputs a binary file

        Parameters
        ----------
        map_ :

        event : 

        start_time :

        end_time : 

        """

        start_time = UTCDateTime(start_time)
        end_time = UTCDateTime(end_time)
        fname = self.path / "{}_{}_{}".format(event, start_time, end_time)
        fname = fname.with_suffix(".coal4D")

        np.save(fname, map_)

    def read_decscan(self):
        """

        """

        fname = self.path.with_suffix(".scnmseed")
        coa = obspy.read(str(fname))
        coa_stats = coa.select(station="COA")[0].stats

        data = pd.DataFrame()

        td = 1 / coa_stats.sampling_rate
        tmp = np.arange(coa_stats.starttime,
                        coa_stats.endtime + td,
                        td)
        data["DT"] = [x.datetime for x in tmp]
        data["COA"] = coa.select(station="COA")[0].data / 1e8
        data["COA_NORM"] = coa.select(station="COA_N")[0].data / 1e8
        data["X"] = coa.select(station="X")[0].data / 1e6
        data["Y"] = coa.select(station="Y")[0].data / 1e6
        data["Z"] = coa.select(station="Z")[0].data

        data["DT"] = data["DT"].apply(UTCDateTime)

        return data

    def write_decscan(self, original_dataset, daten, dsnr, dsnr_norm, dloc, sampling_rate):
        """
        Create a new .scnmseed file

        Parameters
        ----------
        sampling_rate : int
            Sampling rate in hertz

        """

        fname = self.path.with_suffix(".scnmseed")

        data = pd.DataFrame(columns=["DT", "COA", "COA_N", "X", "Y", "Z"])
        data["DT"] = daten
        data["DT"] = data["DT"].apply(UTCDateTime)
        data["COA"] = dsnr
        data["COA_NORM"] = dsnr_norm
        data["X"] = dloc[:, 0]
        data["Y"] = dloc[:, 1]
        data["Z"] = dloc[:, 2]

        npts = len(data)
        starttime = data.iloc[0][0]
        stats_COA = {"network": "NW",
                     "station": "COA",
                     "npts": npts,
                     "sampling_rate": sampling_rate,
                     "starttime": starttime}
        stats_COA_NORM = {"network": "NW",
                          "station": "COA_N",
                          "npts": npts,
                          "sampling_rate": sampling_rate,
                          "starttime": starttime}
        stats_X = {"network": "NW",
                   "station": "X",
                   "npts": npts,
                   "sampling_rate": sampling_rate,
                   "starttime": starttime}
        stats_Y = {"network": "NW",
                   "station": "Y",
                   "npts": npts,
                   "sampling_rate": sampling_rate,
                   "starttime": starttime}
        stats_Z = {"network": "NW",
                   "station": "Z",
                   "npts": npts,
                   "sampling_rate": sampling_rate,
                   "starttime": starttime}

        st = Stream(Trace(data=(np.array(data['COA']) * 1e8).astype(np.int32),
                          header=stats_COA))
        st += Stream(Trace(data=(np.array(data["COA_NORM"]) * 1e8).astype(np.int32),
                           header=stats_COA_NORM))
        st += Stream(Trace(data=(np.array(data['X']) * 1e6).astype(np.int32),
                           header=stats_X))
        st += Stream(Trace(data=(np.array(data['Y']) * 1e6).astype(np.int32),
                           header=stats_Y))
        st += Stream(Trace(data=np.array(data['Z']).astype(np.int32),
                           header=stats_Z))

        if original_dataset is not None:
            original_dataset = original_dataset + st
        else:
            original_dataset = st

        original_dataset.write(str(fname), format="MSEED", encoding=11)

        return original_dataset

    def write_log(self, message):
        """
        Method that tracks the progress of a scanning run through time

        Parameters
        ----------
        message : str
            Information to be saved to the log file

        """

        fname = self.path.with_suffix(".log")
        with fname.open(mode="a") as f:
            f.write(message + '\n')

    def cut_mseed(self, data, event_name):
        """
        Output a mSEED file

        Parameters
        ----------
        data : 

        event_name : str
            Event ID

        """

        fname = (self.path / "_{}".format(event_name)).with_suffix(".mseed")
        st = data.st
        st.write(str(fname), format='MSEED')

    def write_coal_video(self, map_, lut, data, event_coa_val, event_name):
        """
        Create a coalescence video for each event

        Parameters
        ----------
        map_ : 

        lut : 

        data : 

        event_coa_val : 

        event_name : str

        """

        fname = str(self.path)

        SeisPLT = SeisPlot(lut, map_, data, event_coa_val)

        SeisPLT.CoalescenceVideo(output_file='{}_{}'.format(fname,
                                                            event_name))
        SeisPLT.CoalescenceMarginal(output_file='{}_{}'.format(fname,
                                                               event_name))

    def write_station_file(self, station_picks, event_name):
        """
        Create a new .stn file

        Parameters
        ----------
        stations : pandas DataFrame object

        event_name : str


        """

        fname = (self.path / "_{}".format(event_name)).with_suffix(".stn")
        station_picks.to_csv(fname, index=False)

    def write_event(self, event, event_name):
        """
        Create a new .event file

        Parameters
        ----------
        events : pandas DataFrame object

        event_name : str


        """

        fname = (self.path / "_{}".format(event_name)).with_suffix(".event")
        event.to_csv(fname, index=False)

    def read_triggered_events(self, start_time, end_time):
        """

        Parameters
        ----------
        start_time : UTCDateTime object

        end_time : UTCDateTime object

        """
        fname = self.path / "_TriggeredEvents.csv"
        events = pd.read_csv(fname)

        # Trim the events between the start and end times
        events["CoaTime"] = events["CoaTime"].apply(UTCDateTime)
        events = events[(events["CoaTime"] >= start_time) &
                        (events["CoaTime"] <= end_time)]

        events["MinTime"] = events["MinTime"].apply(UTCDateTime)
        events["MaxTime"] = events["MaxTime"].apply(UTCDateTime)

        return events

    def write_triggered_events(self, events):
        """
        Create a new triggered events csv

        Parameters
        ----------
        events : pandas DataFrame object
            Contains information on triggered events

        """

        fname = self.path / "_TriggeredEvents.csv"
        events.to_csv(fname, index=False)


class SeisPlot:
    """
    QuakeMigrate plotting class

    Describes methods for plotting various outputs of QuakeMigrate

    Methods
    -------

    """

    logo = (pathlib.Path(__file__) / "QuakeMigrate").with_suffix(".png")
    '''
         Seismic plotting for QuakeMigrate outputs. Functions include:
            CoalescenceVideo - A script used to generate a coalescence
            video over the period of earthquake location

            CoalescenceLocation - Location plot

            CoalescenceMarginalizeLocation -

    '''
    def __init__(self, lut, map_, coa_map, data, event, station_pick,
                 marginal_window, options=None):
        """
        Class initialisation method

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

        '''
            This is the initial variatiables
        '''
        self.lut = lut
        self.map = map_
        self.data = data
        self.event = event
        self.coa_map = coa_map
        self.stat_pick = station_pick
        self.marginal_window = marginal_window
        self.range_order = True

        if options is None:
            self.trace_scale = 1
            self.cmap = "hot_r"
            self.LineStationColor = "black"
            self.plot_stats = True
            self.filtered_signal = True
            self.xy_files = None
        else:
            try:
                self.trace_scale = options.TraceScaling
                self.CMAP = options.MAPColor
                self.LineStationColor = options.LineStationColor
                self.Plot_Stations = options.Plot_Stations
                self.FilteredSignal = options.FilteredSignal
                self.xy_files = options.xy_files

            except AttributeError:
                msg = "Error - define all plot options."
                print(msg)

        tmp = np.arange(self.data.start_time,
                        self.data.end_time + self.data.sample_size,
                        self.data.sample_size)
        self.times = [x.datetime for x in tmp]
        self.event = self.event[(self.event["DT"] > self.times[0])
                                & (self.event["DT"] < self.event[-1])]

        self.MAPmax = np.max(map_)

        self.CoaTraceVLINE = None
        self.CoaValVLINE = None

        self.CoaXYPlt = None
        self.CoaYZPlt = None
        self.CoaXZPlt = None
        self.CoaXYPlt_VLINE = None
        self.CoaXYPlt_HLINE = None
        self.CoaYZPlt_VLINE = None
        self.CoaYZPlt_HLINE = None
        self.CoaXZPlt_VLINE = None
        self.CoaXZPlt_HLINE = None
        self.CoaArriavalTP = None
        self.CoaArriavalTS = None

    def _plot_coa_trace(self, trace, x, y, st_idx, color):
        trace.plot(x,
                   y / np.max(abs(y)) * self.trace_scale + (st_idx + 1),
                   linecolor=color,
                   linewidth=0.5)

    def _coalescence_image(self, tslice_idx):
        '''
            Takes the outputted coalescence values to plot a video over time
        '''

        tslice = self.times[tslice_idx]
        index = np.where(self.event['DT'] == tslice)[0][0]
        indexVal = self.lut.coord2loc(np.array([[self.event['X'].iloc[index],
                                                 self.event['Y'].iloc[index],
                                                 self.event['Z'].iloc[index]]])
                                      ).astype(int)[0]
        indexCoord = np.array([[self.event['X'].iloc[index],
                                self.event['Y'].iloc[index],
                                self.event['Z'].iloc[index]]])[0, :]

        # ----------------  Defining the Plot Area -------------------
        fig = plt.figure(figsize=(30, 15))
        fig.patch.set_facecolor('white')
        Coa_XYSlice = plt.subplot2grid((3, 5), (0, 0), colspan=2, rowspan=2)
        Coa_YZSlice = plt.subplot2grid((3, 5), (2, 0), colspan=2)
        Coa_XZSlice = plt.subplot2grid((3, 5), (0, 2), rowspan=2)
        trace = plt.subplot2grid((3, 5), (0, 3), colspan=2, rowspan=2)
        logo = plt.subplot2grid((3, 5), (2, 2))
        Coa_CoaVal = plt.subplot2grid((3, 5), (2, 3), colspan=2)

        # ---------------- Plotting the Traces -----------
        STIn = np.where(self.times == self.event['DT'].iloc[0])[0][0]
        ENIn = np.where(self.times == self.event['DT'].iloc[-1])[0][0]

        # ------------ Defining the stations in alphabetical order --------
        if self.range_order:
            ttp = self.lut.get_value_at('TIME_P',
                                        np.array([indexVal[0],
                                                  indexVal[1],
                                                  indexVal[2]]))[0]
            StaInd = np.argsort(ttp)[::-1]
        else:
            StaInd = np.argsort(self.data.stations)[::-1]

        # ------------ Defining the stations in alphabetical order --------
        for i in range(self.data.signal.shape[1]):
            tmp = np.arange(self.data.start_time,
                            self.data.end_time + self.data.sample_size,
                            self.data.sample_size)
            times = [x.datetime for x in tmp]
            if not self.filtered_signal:
                self._plot_coa_trace(trace, times,
                                     self.data.signal[0, i, :],
                                     StaInd[i], color='r')
                self._plot_coa_trace(trace, times,
                                     self.data.signal[1, i, :],
                                     StaInd[i], color='b')
                self._plot_coa_trace(trace, times,
                                     self.data.signal[2, i, :],
                                     StaInd[i], color='g')
            else:
                self._plot_coa_trace(trace, times,
                                     self.data.filtered_signal[0, i, :],
                                     StaInd[i], color='r')
                self._plot_coa_trace(trace, times,
                                     self.data.filtered_signal[1, i, :],
                                     StaInd[i], color='b')
                self._plot_coa_trace(trace, times,
                                     self.data.filtered_signal[2, i, :],
                                     StaInd[i], color='g')

        # ---------------- Plotting the Station Travel Times -----------
        for i in range(self.lut.get_value_at('TIME_P',
                                             np.array([indexVal[0],
                                                       indexVal[1],
                                                       indexVal[2]])
                                             )[0].shape[0]):
            tp = self.event['DT'].iloc[np.argmax(self.event['COA'])] \
                 + self.lut.get_value_at('TIME_P',
                                         np.array([indexVal[0],
                                                   indexVal[1],
                                                   indexVal[2]]))[0][i]

            ts = self.event['DT'].iloc[np.argmax(self.event['COA'])] \
                 + self.lut.get_value_at('TIME_S',
                                         np.array([indexVal[0],
                                                   indexVal[1],
                                                   indexVal[2]]))[0][i]

            if i == 0:
                TP = tp
                TS = ts
            else:
                TP = np.append(TP, tp)
                TS = np.append(TS, ts)

        self.CoaArriavalTP = trace.scatter(TP, (StaInd+1), 40, 'pink', marker='v')
        self.CoaArriavalTS = trace.scatter(TS, (StaInd+1), 40, 'purple', marker='v')

        # trace.set_ylim([0,ii+2])
        trace.set_xlim([self.data.start_time + 1.6, np.max(TS)])
        # trace.get_xaxis().set_ticks([])
        trace.yaxis.tick_right()
        trace.yaxis.set_ticks(StaInd+1)
        trace.yaxis.set_ticklabels(self.data.stations)
        self.CoaTraceVLINE = trace.axvline(self.event['DT'].iloc[np.argmax(self.event['COA'])],0,1000,linestyle='--',linewidth=2,color='r')

        # ------------- Plotting the Coalescence Function -----------
        Coa_CoaVal.plot(self.EVENT['DT'],self.EVENT['COA'])
        Coa_CoaVal.set_ylabel('Coalescence Value')
        Coa_CoaVal.set_xlabel('Date-Time')
        Coa_CoaVal.yaxis.tick_right()
        Coa_CoaVal.yaxis.set_label_position("right")
        Coa_CoaVal.set_xlim([self.EVENT['DT'].iloc[0],self.EVENT['DT'].iloc[-1]])
        Coa_CoaVal.format_xdate = mdates.DateFormatter('%Y-%m-%d')  #FIX - Not working
        for tick in Coa_CoaVal.get_xticklabels():
            tick.set_rotation(45)
        self.CoaValVLINE = Coa_CoaVal.axvline(tslice, 0, 1000,
                                              linestyle='--',
                                              linewidth=2,
                                              color='r')

        #  ------------- Plotting the Coalescence Value Slices -----------
        gridX, gridY = np.mgrid[min(self.lut.xyz2coord(self.lut.get_grid_xyz())[:, 0]):max(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,0]):(max(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,0]) - min(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,0]))/self.LUT.cell_count[0], min(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,1]):max(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,1]):(max(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,1]) - min(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,1]))/self.LUT.cell_count[1]]
        self.CoaXYPlt = Coa_XYSlice.pcolormesh(gridX,gridY,self.MAP[:,:,int(indexVal[2]),int(tslice_idx-STIn)]/self.MAPmax,vmin=0,vmax=1,cmap=self.CMAP)
        Coa_XYSlice.set_xlim([min(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,0]),max(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,0])])
        Coa_XYSlice.set_ylim([min(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,1]),max(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,1])])
        self.CoaXYPlt_VLINE = Coa_XYSlice.axvline(x=indexCoord[0],linestyle='--',linewidth=2,color='k')
        self.CoaXYPlt_HLINE = Coa_XYSlice.axhline(y=indexCoord[1],linestyle='--',linewidth=2,color='k')

        gridX,gridY = np.mgrid[min(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,0]):max(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,0]):(max(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,0]) - min(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,0]))/self.LUT.cell_count[0], min(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,2]):max(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,2]):(max(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,2]) - min(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,2]))/self.LUT.cell_count[2]]
        self.CoaYZPlt = Coa_YZSlice.pcolormesh(gridX,gridY,self.MAP[:,int(indexVal[1]),:,int(tslice_idx-STIn)]/self.MAPmax,vmin=0,vmax=1,cmap=self.CMAP)
        Coa_YZSlice.set_xlim([min(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,0]),max(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,0])])
        Coa_YZSlice.set_ylim([max(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,2]),min(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,2])])
        self.CoaYZPlt_VLINE = Coa_YZSlice.axvline(x=indexCoord[0],linestyle='--',linewidth=2,color='k')
        self.CoaYZPlt_HLINE = Coa_YZSlice.axhline(y=indexCoord[2],linestyle='--',linewidth=2,color='k')
        Coa_YZSlice.invert_yaxis()

        gridX,gridY = np.mgrid[min(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,2]):max(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,2]):(max(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,2]) - min(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,2]))/self.LUT.cell_count[2], min(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,1]):max(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,1]):(max(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,1]) - min(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,1]))/self.LUT.cell_count[1]]
        self.CoaXZPlt = Coa_XZSlice.pcolormesh(gridX,gridY,np.transpose(self.MAP[int(indexVal[0]),:,:,int(tslice_idx-STIn)])/self.MAPmax,vmin=0,vmax=1,cmap=self.CMAP)
        Coa_XZSlice.set_xlim([max(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,2]),min(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,2])])
        Coa_XZSlice.set_ylim([min(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,1]),max(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,1])])
        self.CoaXZPlt_VLINE = Coa_XZSlice.axvline(x=indexCoord[2],linestyle='--',linewidth=2,color='k')
        self.CoaXZPlt_HLINE = Coa_XZSlice.axhline(y=indexCoord[1],linestyle='--',linewidth=2,color='k')

        #  ------------- Plotting the station Locations -----------
        Coa_XYSlice.scatter(self.LUT.station_data['Longitude'],self.LUT.station_data['Latitude'],15,'k',marker='^')
        Coa_YZSlice.scatter(self.LUT.station_data['Longitude'],self.LUT.station_data['Elevation'],15,'k',marker='^')
        Coa_XZSlice.scatter(self.LUT.station_data['Elevation'],self.LUT.station_data['Latitude'],15,'k',marker='<')
        for i,txt in enumerate(self.LUT.station_data['Name']):
            Coa_XYSlice.annotate(txt,[self.LUT.station_data['Longitude'][i],self.LUT.station_data['Latitude'][i]])

        #  ------------- Plotting the xy_files -----------
        if self.xy_files is not None:
            xy_files = pd.read_csv(self.xy_files,
                                   names=['File', 'Color',
                                          'Linewidth', 'Linestyle'])
            c = 0
            for ff in xy_files['File']:
                XYF = pd.read_csv(ff,names=['X', 'Y'])
                Coa_XYSlice.plot(XYF['X'], XYF['Y'],
                                 linestyle=xy_files['Linestyle'].iloc[c],
                                 linewidth=xy_files['Linewidth'].iloc[c],
                                 color=xy_files['Color'].iloc[c])
                c += 1

        #  ------------- Plotting the station Locations -----------
        try:
            logo.axis('off')
            im = mpimg.imread(str(self.logo))
            logo.imshow(im)
            logo.text(150, 200, r'CoalescenceVideo', fontsize=14,style='italic')
        except:
            'Logo not plotting'

        return fig

    def _CoalescenceVideo_update(self,frame):
        frame = int(frame)
        STIn = np.where(self.times == self.EVENT['DT'].iloc[0])[0][0]
        TimeSlice  = self.times[int(frame)]
        index      = np.where(self.EVENT['DT'] == TimeSlice)[0][0]
        indexVal = self.LUT.coord2loc(np.array([[self.EVENT['X'].iloc[index],self.EVENT['Y'].iloc[index],self.EVENT['Z'].iloc[index]]])).astype(int)[0]
        indexCoord = np.array([[self.EVENT['X'].iloc[index],self.EVENT['Y'].iloc[index],self.EVENT['Z'].iloc[index]]])[0,:]

        # Updating the Coalescence Value and Trace Lines
        self.CoaTraceVLINE.set_xdata(TimeSlice)
        self.CoaValVLINE.set_xdata(TimeSlice)

        # Updating the Coalescence Maps
        self.CoaXYPlt.set_array((self.MAP[:,:,indexVal[2],int(STIn-frame)]/self.MAPmax)[:-1,:-1].ravel())
        self.CoaYZPlt.set_array((self.MAP[:,indexVal[1],:,int(STIn-frame)]/self.MAPmax)[:-1,:-1].ravel())
        self.CoaXZPlt.set_array((np.transpose(self.MAP[indexVal[0],:,:,int(STIn-frame)])/self.MAPmax)[:-1,:-1].ravel())

        # Updating the Coalescence Lines
        self.CoaXYPlt_VLINE.set_xdata(indexCoord[0])
        self.CoaXYPlt_HLINE.set_ydata(indexCoord[1])
        self.CoaYZPlt_VLINE.set_xdata(indexCoord[0])
        self.CoaYZPlt_HLINE.set_ydata(indexCoord[2])
        self.CoaXZPlt_VLINE.set_xdata(indexCoord[2])
        self.CoaXZPlt_HLINE.set_ydata(indexCoord[1])

        # Updating the station travel-times
        for i in range(self.LUT.get_value_at('TIME_P',np.array([indexVal]))[0].shape[0]):
            try:
                tp = np.argmin(abs((self.times.astype(datetime) - (TimeSlice.astype(datetime) + timedelta(seconds=self.LUT.get_value_at('TIME_P',np.array([indexVal]))[0][i])))/timedelta(seconds=1)))
                ts = np.argmin(abs((self.times.astype(datetime) - (TimeSlice.astype(datetime) + timedelta(seconds=self.LUT.get_value_at('TIME_S',np.array([indexVal]))[0][i])))/timedelta(seconds=1)))
            except:
                tp=0
                ts=0

            if i == 0:
                TP = tp
                TS = ts
            else:
                TP = np.append(TP,tp)
                TS = np.append(TS,ts)

        self.CoaArriavalTP.set_offsets(np.c_[TP,(np.arange(self.DATA.signal.shape[1])+1)])
        self.CoaArriavalTS.set_offsets(np.c_[TS,(np.arange(self.DATA.signal.shape[1])+1)])

        # # Updating the station travel-times
        # self.CoaArriavalTP.remove()
        # self.CoaArriavalTS.remove()
        # self.CoaArriavalTS = None
        # self.CoaArriavalTP = None
        # for i in range(self.LUT.get_value_at('TIME_P',np.array([indexVal]))[0].shape[0]):
        #     if i == 0:
        #         TP = TimeSlice + timedelta(seconds=self.LUT.get_value_at('TIME_P',np.array([indexVal]))[0][i])
        #         TS = TimeSlice + timedelta(seconds=self.LUT.get_value_at('TIME_S',np.array([indexVal]))[0][i])
        #     else:
        #         TP = np.append(TP,(TimeSlice + timedelta(seconds=self.LUT.get_value_at('TIME_P',np.array([indexVal]))[0][i])))
        #         TS = np.append(TS,(TimeSlice + timedelta(seconds=self.LUT.get_value_at('TIME_S',np.array([indexVal]))[0][i])))

        # self.CoaArriavalTP = trace.scatter(TP,(np.arange(self.DATA.signal.shape[1])+1),15,'r',marker='v')
        # self.CoaArriavalTS = trace.scatter(TS,(np.arange(self.DATA.signal.shape[1])+1),15,'b',marker='v')

    def CoalescenceTrace(self, output_file=None):

        # Determining the maginal window value from the coalescence function
        mMAP = self.CoaMAP
        # mMAP = np.log(np.sum(np.exp(mMAP),axis=-1))
        # mMAP = mMAP/np.max(mMAP)
        # mMAP_Cutoff = np.percentile(mMAP,95)
        # mMAP[mMAP < mMAP_Cutoff] = mMAP_Cutoff
        # mMAP = mMAP - mMAP_Cutoff
        # mMAP = mMAP/np.max(mMAP)
        indexVal = np.where(mMAP == np.max(mMAP))
        indexCoord = self.LUT.xyz2coord(self.LUT.xyz2loc(np.array([[indexVal[0][0],indexVal[1][0],indexVal[2][0]]]), inverse=True))

        # Looping through all stations
        ii = 0
        while ii < self.data.signal.shape[1]:
            fig = plt.figure(figsize=(30, 15))

            # Defining the plot
            fig.patch.set_facecolor('white')
            XTrace_Seis  =  plt.subplot(322)
            YTrace_Seis  =  plt.subplot(324)
            ZTrace_Seis  =  plt.subplot(321)
            P_Onset      =  plt.subplot(323)
            S_Onset      =  plt.subplot(326)

            # --- If trace is blank then remove and don't plot ---

            # Plotting the X-trace
            tmp = np.arange(self.data.start_time,
                            self.data.end_time + self.data.sample_size,
                            self.data.sample_size)
            times = [x.datetime for x in tmp]
            XTrace_Seis.plot(times,
                             (self.data.filtered_signal[0, ii, :]
                              / np.max(abs(self.data.filtered_signal[0, ii, :])))
                             * self.trace_scale, 'r', linewidth=0.5)
            YTrace_Seis.plot(times,
                             (self.data.filtered_signal[1, ii, :]
                              / np.max(abs(self.data.filtered_signal[1, ii, :])))
                             * self.trace_scale, 'b', linewidth=0.5)
            ZTrace_Seis.plot(times,
                             (self.data.filtered_signal[2, ii, :]
                              / np.max(abs(self.data.filtered_signal[2, ii, :])))
                             * self.trace_scale, 'g', linewidth=0.5)
            P_Onset.plot(times, self.data.p_onset[ii, :], 'r', linewidth=0.5)
            S_Onset.plot(times, self.data.s_onset[ii, :], 'b', linewidth=0.5)

            # Defining Pick and Error
            PICKS_df = self.StationPick['Pick']
            STATION_pick = PICKS_df[PICKS_df['Name'] == self.LUT.station_data['Name'][ii]].reset_index(drop=True)
            if len(STATION_pick) > 0:
                STATION_pick = STATION_pick.replace('-1.0',np.nan)

                for jj in range(len(STATION_pick)):
                    if np.isnan(STATION_pick['PickError'].iloc[jj]):
                        continue

                    if STATION_pick['Phase'].iloc[jj] == 'P':
                        ZTrace_Seis.axvline(pd.to_datetime(STATION_pick['PickTime'].iloc[jj])+timedelta(seconds=-STATION_pick['PickError'].iloc[jj]/2),linestyle='--')
                        ZTrace_Seis.axvline(pd.to_datetime(STATION_pick['PickTime'].iloc[jj])+timedelta(seconds=+STATION_pick['PickError'].iloc[jj]/2),linestyle='--')
                        ZTrace_Seis.axvline(pd.to_datetime(STATION_pick['PickTime'].iloc[jj]))

                        # S_Onset.axvline(pd.to_datetime(STATION_pick['PickTime'].iloc[jj])+timedelta(seconds=-STATION_pick['PickError'].iloc[jj]/2),linestyle='--')
                        # S_Onset.axvline(pd.to_datetime(STATION_pick['PickTime'].iloc[jj])+timedelta(seconds=+STATION_pick['PickError'].iloc[jj]/2),linestyle='--')
                        # S_Onset.axvline(pd.to_datetime(STATION_pick['PickTime'].iloc[jj]))

                        yy = gaussian_1d(self.StationPick['GAU_P'][ii]['xdata'],self.StationPick['GAU_P'][ii]['popt'][0],self.StationPick['GAU_P'][ii]['popt'][1],self.StationPick['GAU_P'][ii]['popt'][2])
                        P_Onset.plot(self.StationPick['GAU_P'][ii]['xdata_dt'],yy)
                        P_Onset.axvline(pd.to_datetime(STATION_pick['PickTime'].iloc[jj])+timedelta(seconds=-STATION_pick['PickError'].iloc[jj]/2),linestyle='--')
                        P_Onset.axvline(pd.to_datetime(STATION_pick['PickTime'].iloc[jj])+timedelta(seconds=+STATION_pick['PickError'].iloc[jj]/2),linestyle='--')
                        P_Onset.axvline(pd.to_datetime(STATION_pick['PickTime'].iloc[jj]))

                    else:
                        YTrace_Seis.axvline(pd.to_datetime(STATION_pick['PickTime'].iloc[jj])+timedelta(seconds=-STATION_pick['PickError'].iloc[jj]/2),linestyle='--')
                        YTrace_Seis.axvline(pd.to_datetime(STATION_pick['PickTime'].iloc[jj])+timedelta(seconds=+STATION_pick['PickError'].iloc[jj]/2),linestyle='--')
                        YTrace_Seis.axvline(pd.to_datetime(STATION_pick['PickTime'].iloc[jj]))

                        XTrace_Seis.axvline(pd.to_datetime(STATION_pick['PickTime'].iloc[jj])+timedelta(seconds=-STATION_pick['PickError'].iloc[jj]/2),linestyle='--')
                        XTrace_Seis.axvline(pd.to_datetime(STATION_pick['PickTime'].iloc[jj])+timedelta(seconds=+STATION_pick['PickError'].iloc[jj]/2),linestyle='--')
                        XTrace_Seis.axvline(pd.to_datetime(STATION_pick['PickTime'].iloc[jj]))

                        yy = gaussian_1d(self.StationPick['GAU_S'][ii]['xdata'],self.StationPick['GAU_S'][ii]['popt'][0],self.StationPick['GAU_S'][ii]['popt'][1],self.StationPick['GAU_S'][ii]['popt'][2])
                        S_Onset.plot(self.StationPick['GAU_S'][ii]['xdata_dt'],yy)

                        S_Onset.axvline(pd.to_datetime(STATION_pick['PickTime'].iloc[jj])+timedelta(seconds=-STATION_pick['PickError'].iloc[jj]/2),linestyle='--')
                        S_Onset.axvline(pd.to_datetime(STATION_pick['PickTime'].iloc[jj])+timedelta(seconds=+STATION_pick['PickError'].iloc[jj]/2),linestyle='--')
                        S_Onset.axvline(pd.to_datetime(STATION_pick['PickTime'].iloc[jj]))

            ZTrace_Seis.axvline(self.EVENT['DT'].iloc[np.argmax(self.EVENT['COA'])] + timedelta(seconds=self.LUT.get_value_at('TIME_P',np.array([indexVal[0][0],indexVal[1][0],indexVal[2][0]]))[0][ii] ),color='red')
            ZTrace_Seis.axvline(self.EVENT['DT'].iloc[np.argmax(self.EVENT['COA'])] + timedelta(seconds=(self.LUT.get_value_at('TIME_P',np.array([indexVal[0][0],indexVal[1][0],indexVal[2][0]]))[0][ii] - self.marginal_window - 0.1*self.LUT.get_value_at('TIME_P',np.array([indexVal[0][0],indexVal[1][0],indexVal[2][0]]))[0][ii])),color='red',linestyle='--')
            ZTrace_Seis.axvline(self.EVENT['DT'].iloc[np.argmax(self.EVENT['COA'])] + timedelta(seconds=(self.LUT.get_value_at('TIME_P',np.array([indexVal[0][0],indexVal[1][0],indexVal[2][0]]))[0][ii] + self.marginal_window + 0.1*self.LUT.get_value_at('TIME_P',np.array([indexVal[0][0],indexVal[1][0],indexVal[2][0]]))[0][ii])),color='red',linestyle='--')
            P_Onset.axvline(self.EVENT['DT'].iloc[np.argmax(self.EVENT['COA'])] + timedelta(seconds=self.LUT.get_value_at('TIME_P',np.array([indexVal[0][0],indexVal[1][0],indexVal[2][0]]))[0][ii]),color='red')
            P_Onset.axvline(self.EVENT['DT'].iloc[np.argmax(self.EVENT['COA'])] + timedelta(seconds=(self.LUT.get_value_at('TIME_P',np.array([indexVal[0][0],indexVal[1][0],indexVal[2][0]]))[0][ii] - self.marginal_window - 0.1*self.LUT.get_value_at('TIME_P',np.array([indexVal[0][0],indexVal[1][0],indexVal[2][0]]))[0][ii])),color='red',linestyle='--')
            P_Onset.axvline(self.EVENT['DT'].iloc[np.argmax(self.EVENT['COA'])] + timedelta(seconds=(self.LUT.get_value_at('TIME_P',np.array([indexVal[0][0],indexVal[1][0],indexVal[2][0]]))[0][ii] + self.marginal_window + 0.1*self.LUT.get_value_at('TIME_P',np.array([indexVal[0][0],indexVal[1][0],indexVal[2][0]]))[0][ii])),color='red',linestyle='--')

            YTrace_Seis.axvline(self.EVENT['DT'].iloc[np.argmax(self.EVENT['COA'])] + timedelta(seconds=self.LUT.get_value_at('TIME_S',np.array([indexVal[0][0],indexVal[1][0],indexVal[2][0]]))[0][ii]),color='red')
            YTrace_Seis.axvline(self.EVENT['DT'].iloc[np.argmax(self.EVENT['COA'])] + timedelta(seconds=(self.LUT.get_value_at('TIME_S',np.array([indexVal[0][0],indexVal[1][0],indexVal[2][0]]))[0][ii] - self.marginal_window - 0.1*self.LUT.get_value_at('TIME_S',np.array([indexVal[0][0],indexVal[1][0],indexVal[2][0]]))[0][ii])),color='red',linestyle='--')
            YTrace_Seis.axvline(self.EVENT['DT'].iloc[np.argmax(self.EVENT['COA'])] + timedelta(seconds=(self.LUT.get_value_at('TIME_S',np.array([indexVal[0][0],indexVal[1][0],indexVal[2][0]]))[0][ii] + self.marginal_window + 0.1*self.LUT.get_value_at('TIME_S',np.array([indexVal[0][0],indexVal[1][0],indexVal[2][0]]))[0][ii])),color='red',linestyle='--')

            XTrace_Seis.axvline(self.EVENT['DT'].iloc[np.argmax(self.EVENT['COA'])] + timedelta(seconds=self.LUT.get_value_at('TIME_S',np.array([indexVal[0][0],indexVal[1][0],indexVal[2][0]]))[0][ii]),color='red')
            XTrace_Seis.axvline(self.EVENT['DT'].iloc[np.argmax(self.EVENT['COA'])] + timedelta(seconds=(self.LUT.get_value_at('TIME_S',np.array([indexVal[0][0],indexVal[1][0],indexVal[2][0]]))[0][ii] - self.marginal_window - 0.1*self.LUT.get_value_at('TIME_S',np.array([indexVal[0][0],indexVal[1][0],indexVal[2][0]]))[0][ii])),color='red')
            XTrace_Seis.axvline(self.EVENT['DT'].iloc[np.argmax(self.EVENT['COA'])] + timedelta(seconds=(self.LUT.get_value_at('TIME_S',np.array([indexVal[0][0],indexVal[1][0],indexVal[2][0]]))[0][ii] + self.marginal_window + 0.1*self.LUT.get_value_at('TIME_S',np.array([indexVal[0][0],indexVal[1][0],indexVal[2][0]]))[0][ii])),color='red')

            S_Onset.axvline(self.EVENT['DT'].iloc[np.argmax(self.EVENT['COA'])] + timedelta(seconds=self.LUT.get_value_at('TIME_S',np.array([indexVal[0][0],indexVal[1][0],indexVal[2][0]]))[0][ii]),color='red')
            S_Onset.axvline(self.EVENT['DT'].iloc[np.argmax(self.EVENT['COA'])] + timedelta(seconds=(self.LUT.get_value_at('TIME_S',np.array([indexVal[0][0],indexVal[1][0],indexVal[2][0]]))[0][ii] - self.marginal_window - 0.1*self.LUT.get_value_at('TIME_S',np.array([indexVal[0][0],indexVal[1][0],indexVal[2][0]]))[0][ii])),color='red',linestyle='--')
            S_Onset.axvline(self.EVENT['DT'].iloc[np.argmax(self.EVENT['COA'])] + timedelta(seconds=(self.LUT.get_value_at('TIME_S',np.array([indexVal[0][0],indexVal[1][0],indexVal[2][0]]))[0][ii] + self.marginal_window + 0.1*self.LUT.get_value_at('TIME_S',np.array([indexVal[0][0],indexVal[1][0],indexVal[2][0]]))[0][ii])),color='red',linestyle='--')

            P_Onset.axhline(self.StationPick['GAU_P'][ii]['PickThreshold'])
            S_Onset.axhline(self.StationPick['GAU_S'][ii]['PickThreshold'])

            # Refining the window as around the pick time
            MINT = pd.to_datetime(self.EVENT['DT'].iloc[np.argmax(self.EVENT['COA'])] + timedelta(seconds=0.5*self.LUT.get_value_at('TIME_P',np.array([indexVal[0][0],indexVal[1][0],indexVal[2][0]]))[0][ii]))
            MAXT = pd.to_datetime(self.EVENT['DT'].iloc[np.argmax(self.EVENT['COA'])] + timedelta(seconds=1.5*self.LUT.get_value_at('TIME_S',np.array([indexVal[0][0],indexVal[1][0],indexVal[2][0]]))[0][ii]))

            XTrace_Seis.set_xlim([MINT, MAXT])
            YTrace_Seis.set_xlim([MINT, MAXT])
            ZTrace_Seis.set_xlim([MINT, MAXT])
            P_Onset.set_xlim([MINT, MAXT])
            S_Onset.set_xlim([MINT, MAXT])

            fig.suptitle('Trace for Station {} - PPick = {}, SPick = {}'.format(self.LUT.station_data['Name'][ii],self.StationPick['GAU_P'][ii]['PickValue'],self.StationPick['GAU_S'][ii]['PickValue']))

            if output_file is None:
                plt.show()
            else:
                plt.savefig('{}_CoalescenceTrace_{}.pdf'.format(output_file,self.LUT.station_data['Name'][ii]))
                plt.close("all")

            ii += 1

    def coalescence_video(self, output_file=None):
        idx0 = np.where(self.times == self.EVENT['DT'].iloc[0])[0][0]
        idx1 = np.where(self.times == self.EVENT['DT'].iloc[-1])[0][0]

        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=4, metadata=dict(artist='Ulvetanna'), bitrate=1800)

        fig = self._coalescence_image(idx0)
        ani = animation.FuncAnimation(fig, self._CoalescenceVideo_update,
                                      frames=np.linspace(idx0, idx1 - 1, 200),
                                      blit=False, repeat=False)

        if output_file is None:
            plt.show()
        else:
            ani.save('{}_CoalescenceVideo.mp4'.format(output_file),
                     writer=writer)

    def CoalescenceMarginal(self, output_file=None, earthquake=None):
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

        t_slice_idx = np.where(self.times == self.event['DT'].iloc[np.argmax(self.event['COA'])])[0][0]
        t_slice = self.times[t_slice_idx]
        idx = np.where(self.EVENT['DT'] == t_slice)[0][0]
        indexVal1 = self.lut.coord2loc(np.array([[self.event['X'].iloc[idx],
                                                  self.event['Y'].iloc[idx],
                                                  self.event['Z'].iloc[idx]]])).astype(int)[0]
        idx_coord = np.array([[self.event['X'].iloc[idx],
                               self.event['Y'].iloc[idx],
                               self.event['Z'].iloc[idx]]])[0, :]

        # Determining the maginal window value from the coalescence function
        mMAP = self.coa_map
        indexVal = np.where(mMAP == np.max(mMAP))
        xyz = self.lut.xyz2loc(np.array([[indexVal[0][0],
                                          indexVal[1][0],
                                          indexVal[2][0]]]), inverse=True)
        indexCoord = self.lut.xyz2coord(xyz)

        # Defining the plots to be represented
        fig = plt.figure(figsize=(30, 15))
        fig.patch.set_facecolor('white')
        xy_slice = plt.subplot2grid((3, 5), (0, 0), colspan=2, rowspan=2)
        yz_slice = plt.subplot2grid((3, 5), (2, 0), colspan=2)
        xz_slice = plt.subplot2grid((3, 5), (0, 2), rowspan=2)
        trace = plt.subplot2grid((3, 5), (0, 3), colspan=2, rowspan=2)
        logo = plt.subplot2grid((3, 5), (2, 2))
        Coa_CoaVal = plt.subplot2grid((3, 5), (2, 3), colspan=2)

        # ---------------- Plotting the Traces -----------
        STIn = np.where(self.times == self.EVENT['DT'].iloc[0])[0][0]
        ENIn = np.where(self.times == self.EVENT['DT'].iloc[-1])[0][0]

        # --------------- Ordering by distance to event --------------
        if self.range_order:
            ttp = self.lut.get_value_at('TIME_P',
                                        np.array([indexVal[0][0],
                                                  indexVal[1][0],
                                                  indexVal[2][0]]))[0]
            StaInd = abs(np.argsort(np.argsort(ttp))
                         - np.max(np.argsort(np.argsort(ttp))))
        else:
            StaInd = np.argsort(self.data.stations)[::-1]

        for i in range(self.data.signal.shape[1]):
            tmp = np.arange(self.data.start_time,
                            self.data.end_time + self.data.sample_size,
                            self.data.sample_size)
            times = [x.datetime for x in tmp]
            if not self.filtered_signal:
                self._plot_coa_trace(trace, times,
                                     self.data.signal[0, i, :],
                                     StaInd[i], color='r')
                self._plot_coa_trace(trace, times,
                                     self.data.signal[1, i, :],
                                     StaInd[i], color='b')
                self._plot_coa_trace(trace, times,
                                     self.data.signal[2, i, :],
                                     StaInd[i], color='g')
            else:
                self._plot_coa_trace(trace, times,
                                     self.data.filtered_signal[0, i, :],
                                     StaInd[i], color='r')
                self._plot_coa_trace(trace, times,
                                     self.data.filtered_signal[1, i, :],
                                     StaInd[i], color='b')
                self._plot_coa_trace(trace, times,
                                     self.data.filtered_signal[2, i, :],
                                     StaInd[i], color='g')

        # ---------------- Plotting the Station Travel Times -----------
        for i in range(self.lut.get_value_at('TIME_P',np.array([indexVal[0][0],indexVal[1][0],indexVal[2][0]]))[0].shape[0]):
            tp = self.event['DT'].iloc[np.argmax(self.EVENT['COA'])] + self.lut.get_value_at('TIME_P',np.array([indexVal[0][0],indexVal[1][0],indexVal[2][0]]))[0][i]
            ts = self.event['DT'].iloc[np.argmax(self.EVENT['COA'])] + self.lut.get_value_at('TIME_S',np.array([indexVal[0][0],indexVal[1][0],indexVal[2][0]]))[0][i]

            if i == 0:
                TP = tp
                TS = ts
            else:
                TP = np.append(TP,tp)
                TS = np.append(TS,ts)

        self.CoaArriavalTP = trace.scatter(TP, (StaInd+1), 50, 'pink',
                                           marker='v', zorder=4, linewidth=0.1,
                                           edgecolors='black')
        self.CoaArriavalTS = trace.scatter(TS, (StaInd+1),50,'purple',marker='v',zorder=5,linewidth=0.1,edgecolors='black')

        # trace.set_ylim([0,ii+2])
        trace.set_xlim([self.data.start_time + 1.6, np.max(TS)])
        # trace.get_xaxis().set_ticks([])
        trace.yaxis.tick_right()
        trace.yaxis.set_ticks(StaInd+1)
        trace.yaxis.set_ticklabels(self.data.stations)
        self.CoaTraceVLINE = trace.axvline(self.EVENT['DT'].iloc[np.argmax(self.EVENT['COA'])],0,1000,linestyle='--',linewidth=2,color='r')

        # ------------- Plotting the Coalescence Function ----------- 
        Coa_CoaVal.plot(self.EVENT['DT'],self.EVENT['COA'])
        Coa_CoaVal.set_ylabel('Coalescence Value')
        Coa_CoaVal.set_xlabel('Date-Time')
        Coa_CoaVal.yaxis.tick_right()
        Coa_CoaVal.yaxis.set_label_position("right")
        Coa_CoaVal.set_xlim([self.EVENT['DT'].iloc[0],self.EVENT['DT'].iloc[-1]])
        Coa_CoaVal.format_xdate = mdates.DateFormatter('%Y-%m-%d') #FIX - Not working
        for tick in Coa_CoaVal.get_xticklabels():
                tick.set_rotation(45)
        self.CoaValVLINE   = Coa_CoaVal.axvline(TimeSlice,0,1000,linestyle='--',linewidth=2,color='r')

        # -------------- Determining Error ellipse for Covariance -----------
        dCo = abs(indexCoord - self.LUT.xyz2coord(self.LUT.xyz2loc(np.array([[indexVal[0][0] + int(earthquake['Covariance_ErrX'].iloc[0]/self.LUT.cell_size[0]),indexVal[1][0] + int(earthquake['Covariance_ErrY'].iloc[0]/self.LUT.cell_size[1]),indexVal[2][0] + int(Earthquake['Covariance_ErrZ'].iloc[0]/self.LUT.cell_size[2])]]), inverse=True)))

        ellipse_XY = patches.Ellipse((earthquake['Covariance_X'].iloc[0], earthquake['Covariance_Y'].iloc[0]), 2*dCo[0][0], 2*dCo[0][1], angle=0, linewidth=2, fill=False)
        ellipse_YZ = patches.Ellipse((earthquake['Covariance_X'].iloc[0], earthquake['Covariance_Z'].iloc[0]), 2*dCo[0][0], 2*dCo[0][2], angle=0, linewidth=2, fill=False)
        ellipse_XZ = patches.Ellipse((earthquake['Covariance_Z'].iloc[0], earthquake['Covariance_Y'].iloc[0]), 2*dCo[0][2], 2*dCo[0][1], angle=0, linewidth=2, fill=False)

        # ------------- Spatial Function  ----------- 

        # Plotting the marginal window
        gridX,gridY = np.mgrid[min(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,0]):max(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,0]):(max(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,0]) - min(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,0]))/self.LUT.cell_count[0], min(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,1]):max(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,1]):(max(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,1]) - min(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,1]))/self.LUT.cell_count[1]]
        rect = Rectangle((np.min(gridX), np.min(gridY)), np.max(gridX)-np.min(gridX),np.max(gridY)-np.min(gridY))
        pc = PatchCollection([rect], facecolor='k')
        Coa_XYSlice.add_collection(pc)
        Coa_XYSlice.pcolormesh(gridX,gridY,mMAP[:,:,int(indexVal[2][0])],cmap=self.CMAP,edgecolors='face')
        #CS = Coa_XYSlice.contour(gridX,gridY,mMAP[:,:,int(indexVal[2][0])],levels=[0.65,0.75,0.95],colors=('g','m','k'))
        #Coa_XYSlice.clabel(CS, inline=1, fontsize=10)
        Coa_XYSlice.set_xlim([min(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,0]),max(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,0])])
        Coa_XYSlice.set_ylim([min(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,1]),max(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,1])])
        Coa_XYSlice.axvline(x=indexCoord[0][0],linestyle='--',linewidth=2,color=self.LineStationColor)
        Coa_XYSlice.axhline(y=indexCoord[0][1],linestyle='--',linewidth=2,color=self.LineStationColor)
        Coa_XYSlice.scatter(earthquake['Gaussian_X'].iloc[0],earthquake['Gaussian_Y'].iloc[0],150,c='pink',marker='*')
        Coa_XYSlice.scatter(earthquake['Covariance_X'].iloc[0],earthquake['Covariance_Y'].iloc[0],150,c='blue',marker='*')
        Coa_XYSlice.add_patch(ellipse_XY)


        gridX,gridY = np.mgrid[min(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,0]):max(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,0]):(max(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,0]) - min(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,0]))/self.LUT.cell_count[0], min(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,2]):max(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,2]):(max(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,2]) - min(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,2]))/self.LUT.cell_count[2]]
        rect = Rectangle((np.min(gridX), np.min(gridY)), np.max(gridX)-np.min(gridX),np.max(gridY)-np.min(gridY))
        pc = PatchCollection([rect], facecolor='k')
        Coa_YZSlice.add_collection(pc)
        Coa_YZSlice.pcolormesh(gridX,gridY,mMAP[:,int(indexVal[1][0]),:],cmap=self.CMAP,edgecolors='face')
        #CS = Coa_YZSlice.contour(gridX,gridY,mMAP[:,int(indexVal[1][0]),:], levels=[0.65,0.75,0.95],colors=('g','m','k'))
        #Coa_YZSlice.clabel(CS, inline=1, fontsize=10)
        Coa_YZSlice.set_xlim([min(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,0]),max(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,0])])
        Coa_YZSlice.set_ylim([max(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,2]),min(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,2])])
        Coa_YZSlice.axvline(x=indexCoord[0][0],linestyle='--',linewidth=2,color=self.LineStationColor)
        Coa_YZSlice.axhline(y=indexCoord[0][2],linestyle='--',linewidth=2,color=self.LineStationColor)
        Coa_YZSlice.scatter(earthquake['Gaussian_X'].iloc[0],earthquake['Gaussian_Z'].iloc[0],150,c='pink',marker='*')
        Coa_YZSlice.scatter(earthquake['Covariance_X'].iloc[0],earthquake['Covariance_Z'].iloc[0],150,c='blue',marker='*')
        Coa_YZSlice.add_patch(ellipse_YZ)
        Coa_YZSlice.invert_yaxis()

        gridX,gridY = np.mgrid[min(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,2]):max(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,2]):(max(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,2]) - min(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,2]))/self.LUT.cell_count[2], min(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,1]):max(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,1]):(max(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,1]) - min(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,1]))/self.LUT.cell_count[1]]
        rect = Rectangle((np.min(gridX), np.min(gridY)), np.max(gridX)-np.min(gridX),np.max(gridY)-np.min(gridY))
        pc = PatchCollection([rect], facecolor='k')
        Coa_XZSlice.add_collection(pc)
        Coa_XZSlice.pcolormesh(gridX,gridY,mMAP[int(indexVal[0][0]),:,:].transpose(),cmap=self.CMAP,edgecolors='face')
        #CS = Coa_XZSlice.contour(gridX,gridY,mMAP[int(indexVal[0][0]),:,:].transpose(),levels =[0.65,0.75,0.95],colors=('g','m','k'))
        Coa_XZSlice.set_xlim([max(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,2]),min(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,2])])
        Coa_XZSlice.set_ylim([min(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,1]),max(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,1])])
        Coa_XZSlice.axvline(x=indexCoord[0][2],linestyle='--',linewidth=2,color=self.LineStationColor)
        Coa_XZSlice.axhline(y=indexCoord[0][1],linestyle='--',linewidth=2,color=self.LineStationColor)
        Coa_XZSlice.scatter(earthquake['Gaussian_Z'].iloc[0],earthquake['Gaussian_Y'].iloc[0],150,c='pink',marker='*')
        Coa_XZSlice.scatter(earthquake['Covariance_Z'].iloc[0],earthquake['Covariance_Y'].iloc[0],150,c='blue',marker='*')
        Coa_XZSlice.add_patch(ellipse_XZ)

        # Plotting the station locations
        Coa_XYSlice.scatter(self.LUT.station_data['Longitude'],self.LUT.station_data['Latitude'],15,marker='^',color=self.LineStationColor)
        Coa_YZSlice.scatter(self.LUT.station_data['Longitude'],self.LUT.station_data['Elevation'],15,marker='^',color=self.LineStationColor)
        Coa_XZSlice.scatter(self.LUT.station_data['Elevation'],self.LUT.station_data['Latitude'],15,marker='<',color=self.LineStationColor)
        for i,txt in enumerate(self.LUT.station_data['Name']):
            Coa_XYSlice.annotate(txt,[self.LUT.station_data['Longitude'][i],self.LUT.station_data['Latitude'][i]],color=self.LineStationColor)

        # Plotting the xy_files
        if self.xy_files != None:
           xy_files = pd.read_csv(self.xy_files,names=['File','Color','Linewidth','Linestyle'])
           c=0
           for ff in xy_files['File']:
                XYF = pd.read_csv(ff,names=['X','Y'])       
                Coa_XYSlice.plot(XYF['X'],XYF['Y'],linestyle=xy_files['Linestyle'].iloc[c],linewidth=xy_files['Linewidth'].iloc[c],color=xy_files['Color'].iloc[c])
                c+=1

        # Plotting the logo
        try:
            logo.axis('off')
            im = mpimg.imread(str(self.logo))
            logo.imshow(im)
            logo.text(150, 200, r'Earthquake Location Error', fontsize=10,style='italic')
        except:
            'Logo not plotting'

        if output_file is None:
            plt.show()

        else:
            plt.savefig('{}_EventLocationError.pdf'.format(output_file),dpi=400)
            plt.close('all')


class SeisScanParam:
    """
    Scan parameter class

    Reads in a user defined parameter file, specifying the scan parameters.

    """

    '''
      _set_param - Definition of the path for the Parameter file to be read

    '''

    def __init__(self, param=None):
        self.lookup_table = None
        self.seis_reader = None
        self.p_bp_filter = [2.0, 16.0, 3]
        self.s_bp_filter = [2.0, 12.0, 3]
        self.p_onset_win = [0.2, 1.0]
        self.s_onset_win = [0.2, 1.0]
        self.p_station = None
        self.s_station = None
        self.detection_threshold = 3.0
        self.detection_downsample = 5
        self.detection_window = 3.0
        self.minimum_velocity = 3000.0
        self.marginal_window = [0.5, 3000.0]
        self.location_method = "Mean"
        self.time_step = 10
        self.start_datetime = None
        self.end_datetime = None
        self.decimate = [1, 1, 1]

        if param is not None:
            with open(param, 'r') as f:
                params = json.load(f)
            self._set_param(params)

    def _set_param(self, param):
        # Defining the Model Types to load LUT from
        type_ = _find(param, ("MODEL", "Type"))

        if (type_ == "MATLAB"):
            path = _find(param, ("MODEL", "Path"))
            if path:
                decimate = _find(param, ("MODEL", "Decimate"))
                self.lookup_table = mload.lut02(path, decimate)

        if (type_ == "QuakeMigrate"):
            path = _find(param, ("MODEL", "Path"))
            if path:
                decimate = _find(param, ("MODEL", "Decimate"))
                self.lookup_table = mload.lut02(path, decimate)

        if (type_ == "NLLoc"):
            path = _find(param, ("MODEL", "Path"))
            if path:
                decimate = _find(param, ("MODEL", "Decimate"))
                self.lookup_table = mload.lut02(path, decimate)

        # Defining the Seimsic Data to load the information from
        type_ = _find(param, ("SEISMIC", "Type"))
        if (type_ == 'MSEED'):
            path = _find(param, ("SEISMIC", "Path"))
            if path:
                self.seis_reader = mload.mseed_reader(path)

        # ~ Other possible types to add DAT,SEGY,RAW

        # Defining the Time-Period to scan across
        scn = _find(param, "SCAN")
        if scn:
            self.start_datetime = _find(scn, "Start_DateTime",
                                        self.start_datetime)
            self.end_datetime = _find(scn, "End_DateTime",
                                      self.end_datetime)
            self.start_datetime = datetime.strptime(self.start_datetime,
                                                    '%Y-%m-%dT%H:%M:%S.%f')
            self.end_datetime = datetime.strptime(self.end_datetime,
                                                  '%Y-%m-%dT%H:%M:%S.%f')

        # Defining the Parameters for the Coalescence
        scn = _find(param, ("PARAM"))
        if scn:
            self.time_step = _find(scn, "TimeStep",
                                   self.time_step)
            self.p_station = _find(scn, "StationSelectP",
                                   self.p_station)
            self.s_station = _find(scn, "StationSelectS",
                                   self.s_station)
            self.p_bp_filter = _find(scn, "SigFiltP1Hz",
                                     self.p_bp_filter)
            self.s_bp_filter = _find(scn, "SigFiltS1Hz",
                                     self.s_bp_filter)
            self.p_sta_lta = _find(scn, "OnsetWinP1Sec",
                                   self.p_sta_lta)
            self.s_sta_lta = _find(scn, "OnsetWinS1Sec",
                                   self.s_sta_lta)
            self.detection_downsample = _find(scn, "DetectionDownsample",
                                              self.detection_downsample)
            self.detection_window = _find(scn, "DetectionWindow",
                                          self.detection_window)
            self.minimum_velocity = _find(scn, "MinimumVelocity",
                                          self.minimum_velocity)
            self.marginal_window = _find(scn, "marginal_window",
                                         self.marginal_window)
            self.location_method = _find(scn, "LocationMethod",
                                         self.location_method)


class SeisScan:
    """
    QuakeMigrate scanning class

    Forms the core of the QuakeMigrate method, providing wrapping functions for the
    C-compiled methods.

    Attributes
    ----------
    pre_pad : float

    post_pad : float
        Maximum travel-time from a point in the grid to a station

    Methods
    -------
    detect(start_time, end_time, log=False)
        Core detection method
    trigger(start_time, end_time)
        Core trigger method
    """

    _pre_pad = None
    _time_step = 10.0
    _n_cores = 1
    _min_repeat = 30

    raw_data = {}
    filt_data = {}
    onset_data = {}

    DEFAULT_GAUSSIAN_FIT = {"popt": 0,
                            "xdata": 0,
                            "xdata_dt": 0,
                            "PickValue": -1}

    def __init__(self, data, lookup_table, reader=None, params=None,
                 output_path=None, output_name=None):
        """
        Class initialisation method

        Parameters
        ----------
        data : 

        lut : 

        reader : 

        params : 

        output_path : 

        output_name : 


        """

        self.data = data
        lut = cmod.LUT()
        lut.load(lookup_table)
        self.lut = lut
        ttmax = np.max(lut.fetch_map("TIME_S"))
        self.post_pad = round(ttmax + ttmax*0.05)
        self.sampling_rate = 1000.0
        self.seis_reader = None

        if params is None:
            params = SeisScanParam()

        if output_path is not None:
            self.output = SeisOutFile(output_path, output_name)
        else:
            self.output = None

        self.pick_threshold = 1.0

        self.p_bp_filter = params.p_bp_filter
        self.s_bp_filter = params.s_bp_filter
        self.p_onset_win = params.p_onset_win
        self.s_onset_win = params.s_onset_win
        self.p_station = params.p_station
        self.s_station = params.s_station
        self.detection_threshold = params.detection_threshold

        self.marginal_window = 30
        self.percent_tt = 0.1
        self.plot_coal_grid = False
        self.plot_coal_video = False
        self.plot_coal_picture = False
        self.plot_coal_trace = False
        self.picking_mode = "Gaussian"
        self.location_error = 0.95
        self.onset_centred = False
        self.normalise_coalescence = False
        self.deep_learning = False
        self.output_sampling_rate = None

        # self.plot = SeisPlot(lut)

        self.xy_files = None

        msg = "=" * 126 + "\n"
        msg += "=" * 126 + "\n"
        msg += "   QuakeMigrate - Coalescence Scanning - Path: {} - Name: {}\n"
        msg += "=" * 126 + "\n"
        msg += "=" * 126 + "\n"
        msg += "\n"
        msg = msg.format(self.output.path, self.output.name)
        print(msg)

    @property
    def pre_pad(self):
        return self._pre_pad

    @pre_pad.setter
    def pre_pad(self, value):
        # Add tests for a valid pre-pad
        self._pre_pad = value

    @property
    def post_pad(self):
        return self._post_pad

    @post_pad.setter
    def post_pad(self, value):
        # Add tests for a valid pre-pad
        self._post_pad = value

    @property
    def time_step(self):
        return self._time_step

    @time_step.setter
    def time_step(self, value):
        # Add tests for a valid time_step
        self._time_step = value

    @property
    def n_cores(self):
        return self._n_cores

    @n_cores.setter
    def n_cores(self, value):
        # Add tests for a valid number of cores
        self._n_cores = value

    @property
    def min_repeat(self):
        return self._min_repeat

    def detect(self, start_time, end_time, log=False):
        """
        Searches through continuous data to find earthquakes

        Parameters
        ----------
        start_time : str

        end_time : str

        log : bool, optional
            Output processing to a log file

        """

        # Convert times to UTCDateTime objects
        start_time = UTCDateTime(start_time)
        end_time = UTCDateTime(end_time)

        self.log = log

        # Conduct the continuous compute on the decimated grid
        self.lut = self.lut.decimate(self.decimate)

        # Define pre-pad as a function of the onset windows
        if self.pre_pad is None:
            self.pre_pad = max(self.p_onset_win[1],
                               self.s_onset_win[1]) \
                           + 3 * max(self.p_onset_win[0],
                                     self.s_onset_win[0])

        # Detect the possible events from the decimated grid
        self._continuous_compute(start_time, end_time)

    def trigger(self, start_time, end_time, savefig=True, log=False):
        """

        Parameters
        ----------
        start_time : str
            Start time to perform trigger from
        end_time : str
            End time to perform trigger to
        savefig : bool, optional
            Saves plots if True
        log : bool, optional
            Output processing to a log file

        """

        # Convert times to UTCDateTime objects
        start_time = UTCDateTime(start_time)
        end_time = UTCDateTime(end_time)

        self.log = log

        msg = "=" * 126 + "\n"
        msg += "   TRIGGER - Triggering events from coalescence\n"
        msg += "=" * 126 + "\n"
        msg += "\n"
        msg += "   Parameters specified:\n"
        msg += "         Start time                = {}\n"
        msg += "         End   time                = {}\n"
        msg += "         Number of CPUs            = {}\n"
        msg += "\n"
        msg += "         Marginal window           = {} s\n"
        msg += "         Minimum repeat            = {} s\n"
        msg += "=" * 126 + "\n"
        msg = msg.format(str(start_time), str(end_time), self.n_cores,
                         self.marginal_window, self.min_repeat)
        if self.log:
            self.output.write_log(msg)
        else:
            print(msg)

        # Intial detection of the events from .scn file
        coa_val = self.output.read_decscan()
        events = self._trigger_scn(coa_val, start_time, end_time)

        if self._no_events is True:
            pass
        else:
            self.output.write_triggered_events(events)
            self.plot_scn(events=events, start_time=start_time,
                          end_time=end_time, stations=self.lut.station_data,
                          savefig=savefig)

        self._no_events = False

    def locate(self, start_time, end_time, cut_mseed=False, log=False):
        """

        Parameters
        ----------
        start_time : str
            Start time to perform trigger from
        end_time : str
            End time to perform trigger to
        cut_mseed : bool, optional
            Saves cut mSEED files if True
        log : bool, optional
            Output processing to a log file

        """

        # Convert times to UTCDateTime objects
        start_time = UTCDateTime(start_time)
        end_time = UTCDateTime(end_time)

        self.log = log

        msg = "=" * 126 + "\n"
        msg += "   LOCATE - Determining earthquake location and error\n"
        msg += "=" * 126 + "\n"
        msg += "\n"
        msg += "   Parameters specified:\n"
        msg += "         Start time                = {}\n"
        msg += "         End   time                = {}\n"
        msg += "         Number of CPUs            = {}\n"
        msg += "\n"
        msg += "=" * 126 + "\n"
        msg = msg.format(str(start_time), str(end_time), self.n_cores)
        if self.log:
            self.output.write_log(msg)
        else:
            print(msg)

        events = self.output.read_events(start_time, end_time)

        self.onset_centred = True

        n_evts = len(events)

        # Conduct the continuous compute on the decimated grid
        self.lut = self.lut.decimate(self.decimate)

        if self.pre_pad is None:
            self.pre_pad = max(self.p_onset_win[1],
                               self.s_onset_win[1]) \
                           + 3 * max(self.p_onset_win[0],
                                     self.s_onset_win[0])

        for i, event in events.iterrows():
            evt_id = event["EventID"].astype(str)
            msg = "Processing for Event {} of {} - {}"
            msg = msg.format(i + 1, n_evts, evt_id)
            if self.log:
                self.output.write_log(msg)
            else:
                print(msg)

            tic()
            # Determining the Seismic event location
            w_beg = event['MinTime'] - self.pre_pad
            w_end = event['MaxTime'] + self.post_pad
            self.data.read_mseed(w_beg, w_end, self.sampling_rate)

            daten, dsnr, dloc, map_ = self._compute(w_beg, w_end,
                                                    self.data.signal,
                                                    self.data.availability)

            dcoord = self.lut.xyz2coord(np.array(dloc).astype(int))
            event_coa_val = pd.DataFrame(np.array((daten, dsnr,
                                                   dcoord[:, 0],
                                                   dcoord[:, 1],
                                                   dcoord[:, 2])).transpose(),
                                         columns=['DT', 'COA', 'X', 'Y', 'Z'])
            event_coa_val['DT'] = event_coa_val['DT'].apply(UTCDateTime)
            event = event_coa_val
            event_max = event.iloc[event_coa_val['COA'].idxmax()]

            # Determining the hypocentral location from the maximum over
            # the marginal window.
            picks, GAUP, GAUS = self._arrival_trigger(event_max, evt_id)

            station_pick = {}
            station_pick['Pick'] = picks
            station_pick['GAU_P'] = GAUP
            station_pick['GAU_S'] = GAUS
            toc()

            # Determining earthquake location error
            tic()
            loc, loc_err, loc_cov, loc_err_cov = self._location_error(map_)
            toc()

            evt = pd.DataFrame([np.append(event_max.as_matrix(),
                                          [loc[0], loc[1], loc[2],
                                           loc_err[0], loc_err[1], loc_err[2],
                                           loc_cov[0], loc_cov[1], loc_cov[2],
                                           loc_err_cov[0], loc_err_cov[1],
                                           loc_err_cov[2]])],
                               columns=['DT', 'COA', 'X', 'Y', 'Z',
                                        'Gaussian_X', 'Gaussian_Y',
                                        'Gaussian_Z', 'Gaussian_ErrX',
                                        'Gaussian_ErrY', 'Gaussian_ErrZ',
                                        'Covariance_X', 'Covariance_Y',
                                        'Covariance_Z', 'Covariance_ErrX',
                                        'Covariance_ErrY', 'Covariance_ErrZ'])
            self.output.write_event(evt, event["EventID"])

            if cut_mseed:
                print("Creating cut Mini-SEED")
                tic()
                self.output.cut_mseed(self.data, evt_id)
                toc()

            output_file = self.output_path / "_{}".format(evt_id)
            # Outputting coalescence grids and triggered events
            if self.plot_coal_trace:
                tic()
                print('Creating Station Traces')
                seis_plot = SeisPlot(self.lut,
                                     map_,
                                     self.coa_map,
                                     self.data,
                                     event,
                                     station_pick,
                                     self.marginal_window)
                seis_plot.CoalescenceTrace(output_file=output_file)
                toc()

            if self.plot_coal_grid:
                tic()
                print('Creating 4D Coalescence Grids')
                self.output.write_coal4D(map_, evt_id, w_beg, w_end)
                toc()

            if self.plot_coal_video:
                tic()
                print('Creating Seismic Videos')
                seis_plot = SeisPlot(self.lut,
                                     map_,
                                     self.coa_map,
                                     self.data,
                                     event,
                                     station_pick,
                                     self.marginal_window)
                seis_plot.CoalescenceVideo(output_file=output_file)
                toc()

            if self.plot_coal_picture:
                tic()
                print('Creating Seismic Picture')
                seis_plot = SeisPlot(self.lut,
                                     map_,
                                     self.coa_map,
                                     self.data,
                                     event,
                                     station_pick,
                                     self.marginal_window)
                seis_plot.CoalescenceMarginal(output_file=output_file,
                                              earthquake=evt)
                toc()

            del map_, event, station_pick
            self.coa_map = None

        self.onset_centred = False

    def plot_scn(self, events, start_time, end_time, stations=None, savefig=False):
        """
        Plots the data from a .scn file

        Parameters
        ----------
        events : 

        start_time : UTCDateTime

        end_time : UTCDateTime

        stations : 

        savefig : bool, optional
            Output the plot as a file. The plot is just shown by default.

        TO-DO
        -----
        Plot station availability if requested.

        """

        fname = self.output.path.with_suffix(".scn")

        if fname.exists():
            # Loading the .scn file
            data = pd.read_csv(str(fname), names=["DT", "COA", "COA_NORM",
                                                  "X", "Y", "Z"])
            data['DT'] = data['DT'].apply(UTCDateTime)

            fig = plt.figure(figsize=(30, 15))
            fig.patch.set_facecolor('white')
            coa = plt.subplot2grid((6, 16), (0, 0), colspan=9, rowspan=3)
            coa_norm = plt.subplot2grid((6, 16), (3, 0), colspan=9, rowspan=3,
                                        sharex=coa)
            xy = plt.subplot2grid((6, 16), (0, 10), colspan=4, rowspan=4)
            xz = plt.subplot2grid((6, 16), (4, 10), colspan=4, rowspan=2,
                                  sharex=xy)
            yz = plt.subplot2grid((6, 16), (0, 14), colspan=2, rowspan=4,
                                  sharey=xy)

            coa.plot(data["DT"], data["COA"], color="blue",
                     label="Maximum coalescence", linewidth=1.0)
            coa.get_xaxis().set_ticks([])
            coa_norm.plot(self.data["DT"], self.data["COA_NORM"], color="blue",
                          label="Maximum coalescence", linewidth=1.0)

            for i, event in events.iterrows():
                if i == 0:
                    label1 = "Minimum repeat window"
                    label2 = "Marginal window"
                    label3 = "Detected events"
                else:
                    label1 = ""
                    label2 = ""
                for plot in [coa, coa_norm]:
                    plot.axvspan(event["MinTime"].datetime - self.min_repeat,
                                 event["MaxTime"].datetime + self.min_repeat,
                                 label=label1, alpha=0.5, color='red')
                    plot.axvline(event["CoaTime"].datetime - self.marginal_window,
                                 label=label2, c='m', linestyle='--', linewidth=1.75)
                    plot.axvline(event["CoaTime"].datetime + self.marginal_window,
                                 c='m', linestyle='--', linewidth=1.75)
                    plot.axvline(event["CoaTime"].datetime, label=label3,
                                 c='m', linewidth=1.75)

            props = {"boxstyle": "round",
                     "facecolor": "white",
                     "alpha": 0.5}
            coa.set_xlim(start_time.datetime, end_time.datetime)
            coa.text(.5, .9, "Maximum coalescence", horizontalalignment="center",
                     transform=coa.transAxes, bbox=props)
            coa.legend(loc=2)
            coa.set_ylabel("Maximum coalescence value")
            coa_norm.set_xlim(start_time.datetime, end_time.datetime)
            coa_norm.text(.5, .9, "Normalised maximum coalescence", horizontalalignment="center",
                          transform=coa_norm.transAxes, bbox=props)
            coa_norm.legend(loc=2)
            coa_norm.set_ylabel("Normalised maximum coalescence value")
            coa_norm.set_xlabel("DateTime")

            if self.normalise_coalescence:
                coa_norm.axhline(self.detection_threshold, c="g",
                                 label="Detection threshold")
            else:
                coa_norm.axhline(self.detection_threshold, c="g",
                                 label="Detection threshold")

            # Plotting the scatter of the earthquake locations
            xy.set_title("Decimated coalescence earthquake locations")
            xy.scatter(events["COA_X"], events["COA_Y"], 50, events["COA_V"])
            xy.get_xaxis().set_ticks([])
            xy.get_yaxis().set_ticks([])

            yz.scatter(events["COA_Z"], events["COA_Y"], 50, events["COA_V"])
            yz.yaxis.tick_right()
            yz.yaxis.set_label_position("right")
            yz.invert_xaxis()
            yz.set_ylabel("Latitude (deg)")
            yz.set_xlabel("Depth (m)")

            xz.scatter(events["COA_X"], events["COA_Z"], 50, events["COA_V"])
            xz.yaxis.tick_right()
            xz.yaxis.set_label_position("right")
            xz.set_ylabel("Longitude (deg)")
            xz.set_xlabel("Depth (m)")

            if stations is not None:
                xy.scatter(stations["Longitude"], stations["Latitude"], 20,)

                xy.scatter(stations["Longitude"], stations["Latitude"], 15,
                           marker="^", color="black")
                xz.scatter(stations["Longitude"], stations["Elevation"], 15,
                           marker="^", color="black")
                yz.scatter(stations["Elevation"], stations["Latitude"], 15,
                           marker="<", color="black")
                for i, txt in enumerate(stations["Name"]):
                    xy.annotate(txt, [stations["Longitude"][i], stations["Latitude"][i]],
                                color="black")

            # Saving figure if defined
            if savefig:
                out = self.output.path / "_{}"
                out = out.format("Trigger").with_suffix(".pdf")
                plt.savefig(out)
            else:
                plt.show()

        else:
            msg = "Please run detect to generate a .scn file."
            print(msg)

    def _continuous_compute(self, start_time, end_time):
        """


        Parameters
        ----------
        start_time : 

        end_time : 

        """

        # # Clear existing .scn files
        # self.output.del_scan()

        coalescence_mSEED = None

        msg = "=" * 126 + "\n"
        msg += "   DETECT - Continuous Seismic Processing\n"
        msg += "=" * 126 + "\n"
        msg += "\n"
        msg += "   Parameters specified:\n"
        msg += "         Start time                = {}\n"
        msg += "         End   time                = {}\n"
        msg += "         Time step (s)             = {}\n"
        msg += "         Number of CPUs            = {}\n"
        msg += "\n"
        msg += "         Sampling rate             = {}\n"
        msg += "         Grid decimation [X, Y, Z] = [{}, {}, {}]\n"
        msg += "         Bandpass filter P         = [{}, {}, {}]\n"
        msg += "         Bandpass filter S         = [{}, {}, {}]\n"
        msg += "         Onset P [STA, LTA]        = [{}, {}]\n"
        msg += "         Onset S [STA, LTA]        = [{}, {}]\n"
        msg += "\n"
        msg += "=" * 126
        msg = msg.format(str(start_time), str(end_time), self.time_step,
                         self.n_cores, self.sampling_rate,
                         self.decimate[0], self.decimate[1], self.decimate[2],
                         self.p_bp_filter[0], self.p_bp_filter[1],
                         self.p_bp_filter[2], self.s_bp_filter[0],
                         self.s_bp_filter[1], self.s_bp_filter[2],
                         self.p_onset_win[0], self.p_onset_win[1],
                         self.s_onset_win[0], self.s_onset_win[1])
        if self.log:
            self.output.write_log(msg)
        else:
            print(msg)

        t_length = self.pre_pad + self.post_pad + self.time_step
        self.pre_pad += round(t_length * 0.06)
        self.post_pad += round(t_length * 0.06)

        i = 0
        while start_time + self.time_step * (i + 1) <= end_time:
            w_beg = start_time + self.time_step * i - self.pre_pad
            w_end = start_time + self.time_step * (i + 1) + self.post_pad

            msg = ("~" * 15) + " Processing - {} to {} " + ("~" * 15)
            msg = msg.format(str(w_beg), str(w_end))
            if self.log:
                self.output.write_log(msg)
            else:
                print(msg)

            self.data.read_mseed(w_beg, w_end, self.sampling_rate)

            daten, dsnr, dsnr_norm, dloc, map_ = self._compute(w_beg, w_end,
                                                               self.data.signal,
                                                               self.data.availability)

            dcoord = self.lut.xyz2coord(dloc)

            self.output.file_sample_rate = self.output_sampling_rate
            coalescence_mSEED = self.output.write_decscan(coalescence_mSEED,
                                                          daten[:-1],
                                                          dsnr[:-1],
                                                          dsnr_norm[:-1],
                                                          dcoord[:-1, :],
                                                          self.sampling_rate)

            del daten, dsnr, dsnr_norm, dloc, map_
            i += 1

    def _compute(self, w_beg, w_end, signal, station_availability):

        sampling_rate = self.sampling_rate

        avail_idx = np.where(station_availability == 1)[0]
        sige = signal[0]
        sign = signal[1]
        sigz = signal[2]

        # Demeaning the data
        # sige -= np.mean(sige, axis=1)
        # sign -= np.mean(sign, axis=1)
        # sigz -= np.mean(sigz, axis=1)

        if self.deep_learning:
            msg = "Deep Learning coalescence under development."
            # msg = "Applying deep learning coalescence technique."
            print(msg)
            # dl = DeepLearningPhaseDetection(sige, sign, sigz, sampling_rate)
            # self.data.p_onset = DL.prob_P
            # self.data.s_onset = DL.prob_S
            # self.data.p_onset_raw = DL.prob_P
            # self.data.s_onset_raw = DL.prob_S
        else:
            p_onset_raw, p_onset = self._compute_p_onset(sigz, sampling_rate)
            s_onset_raw, s_onset = self._compute_s_onset(sige, sign, sampling_rate)
            self.data.p_onset = p_onset
            self.data.s_onset = s_onset
            self.data.p_onset_raw = p_onset_raw
            self.data.s_onset_raw = s_onset_raw

        # tmp1, tmp2 = self._gaussian_coalescence(phase="P")
        # self.data.p_gau_onset_num = tmp1
        # self.data.p_gaussian_onset = tmp2
        # tmp1, tmp2 = self._gaussian_coalescence(phase="S")
        # self.data.s_gau_onset_num = tmp1
        # self.data.s_gaussian_onset = tmp2
        # del tmp1, tmp2

        p_s_onset = np.concatenate((self.data.p_onset, self.data.s_onset))
        p_s_onset[np.isnan(p_s_onset)] = 0

        p_ttime = self.lut.fetch_index("TIME_P", sampling_rate)
        s_ttime = self.lut.fetch_index("TIME_S", sampling_rate)
        ttime = np.c_[p_ttime, s_ttime]
        del p_ttime, s_ttime

        nchan, tsamp = p_s_onset.shape

        pre_smp = int(round(self.pre_pad * int(sampling_rate)))
        pos_smp = int(round(self.post_pad * int(sampling_rate)))
        nsamp = tsamp - pre_smp - pos_smp
        daten = 0.0 - pre_smp / sampling_rate

        ncell = tuple(self.lut.cell_count)

        map_ = np.zeros(ncell + (nsamp,), dtype=np.float64)

        dind = np.zeros(nsamp, np.int64)
        dsnr = np.zeros(nsamp, np.double)
        dsnr_norm = np.zeros(nsamp, np.double)

        ilib.scan(p_s_onset, ttime, pre_smp, pos_smp,
                  nsamp, map_, self.n_cores)
        ilib.detect(map_, dsnr, dind, 0, nsamp, self.n_cores)

        tmp = np.arange(w_beg + self.pre_pad,
                        w_end - self.post_pad + (1 / sampling_rate),
                        1 / sampling_rate)
        daten = [x.datetime for x in tmp]
        dsnr = np.exp((dsnr / (len(avail_idx) * 2)) - 1.0)
        dloc = self.lut.xyz2index(dind, inverse=True)

        # Determining the normalised coalescence through time
        sum_coa = np.sum(map_, axis=(0, 1, 2))
        map_ = map_ / sum_coa[np.newaxis, np.newaxis, np.newaxis, :]
        ilib.detect(map_, dsnr, dind, 0, nsamp, self.n_cores)
        dsnr_norm = dsnr_norm * map_.shape[0] * map_.shape[1] * map_.shape[2]

        # Reset map to original coalescence value
        map_ = map_ * sum_coa[np.newaxis, np.newaxis, np.newaxis, :]

        return daten, dsnr, dsnr_norm, dloc, map_

    def _compute_p_onset(self, sig_z, sampling_rate):
        """
        Generates an onset function for the Z-component

        Parameters
        ----------
        sig_z : array-like
            Z-component time-series
        sampling_rate : int
            Sampling rate in hertz

        Returns
        -------


        """

        stw, ltw = self.p_onset_win
        stw = int(stw * sampling_rate) + 1
        ltw = int(ltw * sampling_rate) + 1
        sig_z = self._preprocess_p(sig_z, sampling_rate)
        self.filt_data['sigz'] = sig_z
        p_onset_raw, p_onset = onset(sig_z, stw, ltw,
                                     centred=self.onset_centred)
        self.onset_data['sigz'] = p_onset

        return p_onset_raw, p_onset

    def _preprocess_p(self, sig_z, sampling_rate):
        """
        Pre-processing method for Z-component

        Applies a bandpass filter followed by a butterworth filter

        Parameters
        ----------
        sig_z : array-like
            Z-component time-series
        sampling_rate : int
            Sampling rate in hertz

        Returns
        -------
        A filtered version of the Z-component time-series

        """

        lc, hc, ord_ = self.p_bp_filter
        sig_z = filter(sig_z, sampling_rate, lc, hc, ord_)
        self.data.filtered_signal[2, :, :] = sig_z

        return sig_z

    def _compute_s_onset(self, sig_e, sig_n, sampling_rate):
        """
        Generates onset functions for the N- and E-components

        Parameters
        ----------
        sig_e : array-like
            E-component time-series
        sig_n : array-like
            N-component time-series
        sampling_rate : int
            Sampling rate in hertz

        Returns
        -------
        s_onset_raw : 

        s_onset : 

        """

        stw, ltw = self.s_onset_win
        stw = int(stw * sampling_rate) + 1
        ltw = int(ltw * sampling_rate) + 1
        sig_e, sig_n = self._preprocess_s(sig_e, sig_n, sampling_rate)
        self.filt_data['sige'] = sig_e
        self.filt_data['sign'] = sig_n
        s_e_onset_raw, s_e_onset = onset(sig_e, stw, ltw,
                                         centred=self.onset_centred)
        s_n_onset_raw, s_n_onset = onset(sig_n, stw, ltw,
                                         centred=self.onset_centred)
        self.onset_data['sige'] = s_e_onset
        self.onset_data['sign'] = s_n_onset
        s_onset = np.sqrt((s_e_onset ** 2 + s_n_onset ** 2) / 2.)
        s_onset_raw = np.sqrt((s_e_onset_raw ** 2 + s_n_onset_raw ** 2) / 2.)
        self.onset_data['sigs'] = s_onset

        return s_onset_raw, s_onset

    def _preprocess_s(self, sig_e, sig_n, sampling_rate):
        """
        Pre-processing method for N- and E-components

        Applies a bandpass filter followed by a butterworth filter (DOES IT?)

        Parameters
        ----------
        sig_e : array-like
            E-component time-series
        sig_n : array-like
            N-component time-series
        sampling_rate : int
            Sampling rate in hertz

        Returns
        -------
        A filtered version of the N- and E-components time-series

        """
        lc, hc, ord_ = self.s_bp_filter
        sig_e = filter(sig_e, sampling_rate, lc, hc, ord_)
        sig_n = filter(sig_n, sampling_rate, lc, hc, ord_)
        self.data.filtered_signal[0, :, :] = sig_n
        self.data.filtered_signal[1, :, :] = sig_e

        return sig_e, sig_n

    def _trigger_scn(self, coa_val, start_time, end_time):

        if self.normalise_coalescence is True:
            coa_val["COA"] = coa_val["COA_NORM"]

        coa_val = coa_val[coa_val["COA"] >= self.detection_threshold]
        coa_val = coa_val[(coa_val["DT"] >= start_time) &
                          (coa_val["DT"] <= end_time)]

        coa_val = coa_val.reset_index(drop=True)

        if len(coa_val) == 0:
            msg = "No events triggered at this threshold"
            print(msg)
            self._no_events = True
            return

        event_cols = ['EventNum', 'CoaTime', 'COA_V', 'COA_X', 'COA_Y',
                      'COA_Z', 'MinTime', 'MaxTime']

        ss = 1 / self.sampling_rate

        # Determine the initial triggered events
        init_events = pd.DataFrame(columns=event_cols)
        c = 0
        e = 1
        while c < len(coa_val) - 1:
            # Determining the index when above the level and maximum value
            d = c

            while coa_val["DT"].iloc[d] + ss == coa_val["DT"].iloc[d + 1]:
                d += 1
                if d + 1 >= len(coa_val) - 2:
                    d = len(coa_val) - 2
                    break

            min_idx = c
            max_idx = d
            val_idx = np.argmax(coa_val['COA'].iloc[np.arange(c, d + 1)])

            # Determining the times for min, max and max coalescence value
            t_min = coa_val['DT'].iloc[min_idx]
            t_max = coa_val['DT'].iloc[max_idx]
            t_val = coa_val['DT'].iloc[val_idx]

            COA_V = coa_val['COA'].iloc[val_idx]
            COA_X = coa_val['X'].iloc[val_idx]
            COA_Y = coa_val['Y'].iloc[val_idx]
            COA_Z = coa_val['Z'].iloc[val_idx]

            if (t_val - t_min) < self.min_repeat:
                t_min = coa_val['DT'].iloc[min_idx] - self.min_repeat
            if (t_max - t_val) < self.min_repeat:
                t_max = coa_val['DT'].iloc[max_idx] + self.min_repeat

            tmp = pd.DataFrame([[e, t_val,
                                COA_V, COA_X, COA_Y, COA_Z,
                                t_min, t_max]],
                               columns=event_cols)
            init_events = init_events.append(tmp, ignore_index=True)

            c = d + 1
            e += 1

        n_evts = len(init_events)
        evt_num = np.ones((n_evts), dtype=int)

        count = 1
        for i, event in init_events.iterrows():
            evt_num[i] = count
            if (i + 1 < n_evts) and ((event["MaxTime"]
                                      - init_events["MinTime"].iloc[i + 1]) < 0):
                count += 1
        init_events['EventNum'] = evt_num

        events = pd.DataFrame(columns=event_cols)
        for i in range(1, count + 1):
            tmp = init_events[init_events['EventNum'] == i]
            tmp = tmp.reset_index(drop=True)
            j = np.argmax(tmp["COA_V"])
            event = pd.DataFrame([[i, tmp['CoaTime'].iloc[j],
                                   tmp['COA_V'].iloc[j],
                                   tmp['COA_X'].iloc[j],
                                   tmp['COA_X'].iloc[j],
                                   tmp['COA_X'].iloc[j],
                                   tmp['CoaTime'].iloc[j] - self.marginal_window,
                                   tmp['CoaTime'].iloc[j] + self.marginal_window]],
                                 columns=event_cols)
            events = events.append(event, ignore_index=True)

        evt_id = events["CoaTime"].astype(str)
        for char_ in ["-", ":", ".", " "]:
            evt_id.str.replace(char_, "")
        events["EventID"] = evt_id

        return events

    def _gaussian_coalescence(self, phase):
        """
        Fits a Gaussian for the coalescence function

        """

        if phase == "P":
            onset = self.data.p_onset
        elif phase == "S":
            onset = self.data.s_onset
        x = np.arange(onset.shape[1])

        gauss_threshold = 1.4

        # ---- Selecting only the data above a predefined threshold ----
        # Setting values below threshold to nan
        onset[np.where(onset < gauss_threshold)] = np.nan

        # Defining two blank arrays that gaussian periods should be defined for
        gau_onset_num = np.zeros(onset.shape) * np.nan

        # --- Determing the indexs to fit gaussians about ---
        for i in range(len(onset)):
            c = 0
            e = 1

            idx = np.where(~np.isnan(onset[i, :]))[0]
            while c < len(idx):
                # Determining the index when above the level and maximum value
                d = c
                while idx[d] + 1 == idx[d + 1]:
                    d += 1
                    if d + 1 >= len(idx) - 1:
                        d = len(idx) - 1
                        break

                gau_onset_num[i, idx[c]:idx[d]] = e

                c = d + 1
                e += 1

        # --- Determing the indexs to fit gaussians about ---

        gau_onset = np.zeros(onset.shape)
        for i in range(onset.shape[0]):
            if ~np.isnan(np.nanmax(gau_onset_num[i, :])):
                c = 0
                for j in range(1, int(round(np.nanmax(gau_onset_num[i, :])))):
                    XSig = x[np.where((gau_onset_num[i, :] == j))[0]]
                    YSig = onset[i, np.where((gau_onset_num[i, :] == j))[0]]

                    if len(YSig) > 8:

                        # self.DATA.p_onset =  YSig

                        try:
                            if phase == "P":
                                lowfreq = float(self.p_bp_filter[0])
                            elif phase == "S":
                                lowfreq = float(self.s_bp_filter[0])
                            p0 = [np.max(YSig),
                                  np.argmax(YSig) + np.min(XSig),
                                  1. / (lowfreq / 4.)]

                            # Fitting the gaussian to the function
                            popt, pcov = curve_fit(gaussian_1d,
                                                   XSig,
                                                   YSig,
                                                   p0)
                            tmp_gau = gaussian_1d(XSig.astype(float),
                                                  float(popt[0]),
                                                  float(popt[1]),
                                                  float(popt[2]))

                            if c == 0:
                                onset_gaussian = np.zeros(x.shape)
                                onset_gaussian[np.where((gau_onset_num[i, :] == j))[0]] = tmp_gau
                                c += 1
                            else:
                                onset_gaussian[np.where((gau_onset_num[i, :] == j))[0]] = tmp_gau
                        except:
                            print('Error with {}'.format(j))

                    else:
                        continue

                gau_onset[i, :] = onset_gaussian

        return gau_onset_num, gau_onset

    def _gaussian_trigger(self, onset, phase, start_time, p_arrival, s_arrival,
                          p_ttime, s_ttime):
        """
        Fit a Gaussian to the onset function.

        Uses knowledge of approximate trigger index, the lowest frequency
        within the signal and the signal sampling rate.

        Parameters
        ----------
        onset :
            Onset function
        phase : str
            Phase name ("P" or "S")
        start_time : UTCDateTime object
            Start time of triggered data
        p_arrival : UTCDateTime object
            Time when P-phase is observed to arrive
        s_arrival : UTCDateTime object
            Time when S-phase is observed to arrive
        p_ttime : UTCDateTime object
            Traveltime of P-phase
        s_ttime : UTCDateTime object
            Traveltime of S-phase

        Returns
        -------

        """

        msg = "Fitting Gaussian for {} - {} - {}"
        msg = msg.format(phase, str(start_time), str(p_arrival))
        # print(msg)

        sampling_rate = self.sampling_rate

        # Determine indices of P and S trigger times
        pt_idx = int((p_arrival - start_time) * sampling_rate)
        st_idx = int((s_arrival - start_time) * sampling_rate)

        # Define bounds from which to determine cdf information
        pmin_idx = int(pt_idx - (st_idx - pt_idx) / 2)
        pmax_idx = int(pt_idx + (st_idx - pt_idx) / 2)
        smin_idx = int(st_idx - (st_idx - pt_idx) / 2)
        smax_idx = int(st_idx + (st_idx - pt_idx) / 2)
        for idx in [pmin_idx, pmax_idx, smin_idx, smax_idx]:
            if idx < 0:
                idx = 0
            if idx > len(onset):
                idx = len(onset)

        # Defining the bounds to search for the event over
        pp_ttime = p_ttime * self.percent_tt
        ps_ttime = s_ttime * self.percent_tt
        P_idxmin_new = int(pt_idx - int((self.marginal_window + pp_ttime)
                                        * sampling_rate))
        P_idxmax_new = int(pt_idx + int((self.marginal_window + pp_ttime)
                                        * sampling_rate))
        S_idxmin_new = int(st_idx - int((self.marginal_window + ps_ttime)
                                        * sampling_rate))
        S_idxmax_new = int(st_idx + int((self.marginal_window + ps_ttime)
                                        * sampling_rate))

        # Setting so the search region can't be bigger than P-S/2.
        P_idxmin = np.max([pmin_idx, P_idxmin_new])
        P_idxmax = np.min([pmax_idx, P_idxmax_new])
        S_idxmin = np.max([smin_idx, S_idxmin_new])
        S_idxmax = np.min([smax_idx, S_idxmax_new])

        # Setting parameters depending on the phase
        if phase == "P":
            lowfreq = self.p_bp_filter[0]
            win_min = P_idxmin
            win_max = P_idxmax
        if phase == "S":
            lowfreq = self.s_bp_filter[0]
            win_min = S_idxmin
            win_max = S_idxmax

        max_onset = np.argmax(onset[win_min:win_max]) + win_min
        onset_trim = onset[win_min:win_max]

        onset_threshold = onset.copy()
        onset_threshold[P_idxmin:P_idxmax] = -1
        onset_threshold[S_idxmin:S_idxmax] = -1
        onset_threshold = onset_threshold[onset_threshold > -1]

        threshold = np.percentile(onset_threshold, self.pick_threshold * 100)
        threshold_window = np.percentile(onset_trim, 88)
        threshold = np.max([threshold, threshold_window])

        tmp = (onset_trim - threshold).any() > 0
        if onset[max_onset] >= threshold and tmp:
            exceedence = np.where((onset_trim - threshold).any() > 0)[0]
            exceedence_dist = np.zeros(len(exceedence))

            d = 1
            e = 0
            while e < len(exceedence_dist) - 1:
                if e == len(exceedence_dist):
                    exceedence_dist[e] = d
                else:
                    if exceedence[e + 1] == exceedence[e] + 1:
                        exceedence_dist[e] = d
                    else:
                        exceedence_dist[3] = d
                        d += 1
                e += 1

            tmp = exceedence_dist[np.argmax(onset_trim[exceedence])]
            tmp = np.where(exceedence_dist == tmp)
            gau_idxmin = exceedence[tmp][0] + win_min
            gau_idxmax = exceedence[tmp][-1] + win_min

            data_half_range = int(2 * sampling_rate / lowfreq)
            x_data = np.arange(gau_idxmin, gau_idxmax, dtype=float)
            x_data = x_data / sampling_rate
            y_data = onset[gau_idxmin:gau_idxmax]

            x_data_dt = np.array([])
            for i in range(len(x_data)):
                x_data_dt = np.hstack(x_data_dt, start_time + x_data[i])

            try:
                p0 = [np.max(y_data),
                      float(gau_idxmin + np.argmax(y_data)) / sampling_rate,
                      data_half_range / sampling_rate]
                popt, pcov = curve_fit(gaussian_1d, x_data, y_data, p0)
                sigma = np.absolute(popt[2])

                # Mean is popt[1]. x_data[0] + popt[1] (In seconds)
                mean = start_time + float(popt[1])

                max_onset = popt[0]

                gaussian_fit = {"popt": popt,
                                "xdata": x_data,
                                "xdata_dt": x_data_dt,
                                "PickValue": max_onset,
                                "PickThreshold": threshold}
            except:
                gaussian_fit = self.DEFAULT_GAUSSIAN_FIT
                gaussian_fit["PickThreshold"] = threshold

                sigma = -1
                mean = -1
                max_onset = -1
        else:
            gaussian_fit = self.DEFAULT_GAUSSIAN_FIT
            gaussian_fit["PickThreshold"] = threshold

            sigma = -1
            mean = -1
            max_onset = -1

        return gaussian_fit, max_onset, sigma, mean

    def _arrival_trigger(self, max_coa, event_name):
        """
        Determines arrival times for triggered earthquakes.

        Parameters
        ----------
        max_coa : pandas DataFrame object
            DataFrame containing the maximum coalescence values for a
            given event
        event_name : str
            Event ID - used for saving the stations file

        Returns
        -------

        """

        p_onset = self.data.p_onset
        s_onset = self.data.s_onset
        start_time = self.data.start_time

        max_coa_crd = np.array([max_coa[["X", "Y", "Z"]].values])
        max_coa_xyz = np.array(self.lut.xyz2coord(max_coa_crd,
                                                  inverse=True)).astype(int)[0]

        p_ttime = self.lut.value_at("TIME_P", max_coa_xyz)
        s_ttime = self.lut.value_at("TIME_S", max_coa_xyz)

        # Determining the stations that can be picked on and the phasese
        stations = pd.DataFrame(index=np.arange(0, 2 * len(p_onset)),
                                columns=["Name", "Phase", "ModelledTime",
                                         "PickTime", "PickError"])
        p_gauss = np.array([])
        s_gauss = np.array([])
        idx = 0
        for i in range(len(p_onset)):
            p_arrival = max_coa["DT"] + p_ttime[i]
            s_arrival = max_coa["DT"] + s_ttime[i]

            if self.picking_mode == "Gaussian":
                for phase in ["P", "S"]:
                    if phase == "P":
                        onset = p_onset[i]
                        arrival = p_arrival
                    else:
                        onset = s_onset[i]
                        arrival = s_arrival

                    msg = "Fitting Gaussian for  {}  -  {}  -  {}"
                    msg = msg.format(phase, str(start_time), str(arrival))
                    if self.log:
                        self.output.write_log(msg)
                    else:
                        print(msg)

                    gau, max_onset, err, mn = self._gaussian_trigger(onset,
                                                                     phase,
                                                                     start_time,
                                                                     p_arrival,
                                                                     s_arrival,
                                                                     p_ttime[i],
                                                                     s_ttime[i])

                    if phase == "P":
                        p_gauss = np.hstack(p_gauss, gau)
                    else:
                        s_gauss = np.hstack(s_gauss, gau)

                    stations.loc[idx] = [self.lut.stations[i], phase, arrival,
                                         mn, err, max_onset]

        self.output.write_station_file(stations, event_name)

        return stations, p_gauss, s_gauss

    def _gaufilt3d(self, vol, sgm, shp=None):
        """


        Parameters
        ----------
        vol : 

        sgm : 

        shp : array-like, optional
            Shape of volume

        Returns
        -------


        """

        if shp is None:
            shp = vol.shape
        nx, ny, nz = shp
        flt = gaussian_3d(nx, ny, nz, sgm)
        return fftconvolve(vol, flt, mode='same')

    def _mask3d(self, n, i, win):
        """

 
        Parameters
        ----------
        n : 

        i : 

        win : 

        Returns
        -------
        mask


        """

        n = np.array(n)
        i = np.array(i)
        w2 = (win-1)//2
        x1, y1, z1 = np.clip(i - w2, 0 * n, n)
        x2, y2, z2 = np.clip(i + w2 + 1, 0 * n, n)
        mask = np.zeros(n, dtype=np.bool)
        mask[x1:x2, y1:y2, z1:z2] = True
        return mask

    def _gaufit3d(self, pdf, lx=None, ly=None, lz=None, smooth=None,
                  thresh=0.0, win=3, mask=7):
        """


        Parameters
        ----------
        pdf :

        lx : , optional

        ly : , optional

        lz : , optional

        smooth : , optional

        thresh : , optional

        win : , optional

        mask : , optional

        Returns
        -------
        loc + iloc

        vec

        sgm

        csgm

        val


        """

        nx, ny, nz = pdf.shape
        if smooth:
            pdf = self._gaufilt3d(pdf, smooth, [11, 11, 11])
        mx, my, mz = np.unravel_index(pdf.argmax(), pdf.shape)
        mval = pdf[mx, my, mz]
        flg = np.logical_or(
            np.logical_and(pdf > mval * np.exp(-(thresh * thresh) / 2),
                           self._mask3d([nx, ny, nz], [mx, my, mz], mask)),
            self._mask3d([nx, ny, nz], [mx, my, mz], win))
        ix, iy, iz = np.where(flg)
        msg = "X : {}-{} - Y : {}-{} - Z : {}-{}"
        msg = msg.format(min(ix), max(ix), min(iy), max(iy), min(iz), max(iz))
        if self.log:
            self.output.write_log(msg)
        else:
            print(msg)

        ncell = len(ix)

        if not lx:
            lx = np.arange(nx)
            ly = np.arange(ny)
            lz = np.arange(nz)

        if lx.ndim == 3:
            iloc = [lx[mx, my, mz], ly[mx, my, mz], lz[mx, my, mz]]
            x = lx[ix, iy, iz] - iloc[0]
            y = ly[ix, iy, iz] - iloc[1]
            z = lz[ix, iy, iz] - iloc[2]
        else:
            iloc = [lx[mx], ly[my], lz[mz]]
            x = lx[ix] - iloc[0]
            y = ly[iy] - iloc[1]
            z = lz[iz] - iloc[2]

        X = np.c_[x * x, y * y, z * z,
                  x * y, x * z, y * z,
                  x, y, z, np.ones(ncell)].T
        Y = -np.log(np.clip(pdf.astype(np.float64)[ix, iy, iz],
                            1e-300, np.inf))

        X_inv = np.linalg.pinv(X)
        P = np.matmul(Y, X_inv)
        G = -np.array([2 * P[0], P[3], P[4],
                       P[3], 2 * P[1], P[5],
                       P[4], P[5], 2 * P[2]]).reshape((3, 3))
        H = np.array([P[6], P[7], P[8]])
        loc = np.matmul(np.linalg.inv(G), H)
        cx, cy, cz = loc

        K = P[9]             \
            - P[0] * cx ** 2 \
            - P[1] * cy ** 2 \
            - P[2] * cz ** 2 \
            - P[3] * cx * cy \
            - P[4] * cx * cz \
            - P[5] * cy * cz \

        M = np.array([P[0], P[3] / 2, P[4] / 2,
                      P[3] / 2, P[1], P[5] / 2,
                      P[4] / 2, P[5] / 2, P[2]]).reshape(3, 3)
        egv, vec = np.linalg.eig(M)
        sgm = np.sqrt(0.5 / np.clip(np.abs(egv), 1e-10, np.inf))
        val = np.exp(-K)
        csgm = np.sqrt(0.5 / np.clip(np.abs(M.diagonal()), 1e-10, np.inf))

        return loc + iloc, vec, sgm, csgm, val

    def _location_error(self, map_4d):
        """

        Parameters
        ----------
        map_4d : 


        Returns
        -------
        loc : 

        loc_err : 

        loc_cov : 

        loc_err_cov : 


        """

        # Determining the coalescence 3D map
        coa_map = np.log(np.sum(np.exp(map_4d), axis=-1))
        coa_map = coa_map / np.max(coa_map)
        cutoff = 0.88
        coa_map[coa_map < cutoff] = cutoff
        coa_map = coa_map - cutoff
        coa_map = coa_map / np.max(coa_map)
        self.coa_map = coa_map

        # Determining the location error as an error-ellipse
        loc, loc_err, loc_cov, cov_matrix = self._error_ellipse(coa_map)
        loc_err_cov = np.array([np.sqrt(cov_matrix[0, 0]),
                                np.sqrt(cov_matrix[1, 1]),
                                np.sqrt(cov_matrix[2, 2])])

        # Determining maximum location and error about this point
        # err_vol = np.zeros((coa_map.shape))
        # err_vol[np.where(coa_map > self.location_error)] = 1
        # xmax = np.sum(np.max(np.sum(err_vol, axis=0), axis=1), axis=0) \
        #        * self.lut.cell_size[0]
        # ymax = np.sum(np.max(np.sum(err_vol, axis=1), axis=1), axis=0) \
        #        * self.lut.cell_size[1]
        # zmax = np.sum(np.max(np.sum(err_vol, axis=2), axis=1), axis=0) \
        #        * self.lut.cell_size[2]

        return loc, loc_err, loc_cov, loc_err_cov

    def _error_ellipse(self, coa_3d):
        """
        Function to calculate covariance matrix and expectation hypocentre from
        coalescence array.

        Parameters
        ----------
        coa_3d : array-like
            Coalescence values for a particular time (x, y, z dimensions)

        Returns
        -------
        expect_vector
            x, y, z coordinates of expectation hypocentre
        xyz_err

        crd_cov

        cov_matrix
            Covariance matrix

        """

        # Get point sample coords and weights:
        smp_weights = coa_3d.flatten()

        lc = self.lut.cell_count
        # Ordering below due to handedness of the grid
        ly, lx, lz = np.meshgrid(np.arange(lc[1]),
                                 np.arange(lc[0]),
                                 np.arange(lc[2]))
        x_samples = lx.flatten() * self.lut.cell_size[0]
        y_samples = ly.flatten() * self.lut.cell_size[1]
        z_samples = lz.flatten() * self.lut.cell_size[2]

        ssw = np.sum(smp_weights)

        # Expectation values:
        x_expect = np.sum(smp_weights * x_samples) / ssw
        y_expect = np.sum(smp_weights * y_samples) / ssw
        z_expect = np.sum(smp_weights * z_samples) / ssw

        msg = "Covariance GridXYZ - X : {} - Y : {} - Z : {}"
        msg.format(x_expect, y_expect, z_expect)
        if self.log:
            self.output.write_log(msg)
        else:
            print(msg)

        # Covariance matrix:
        cov_matrix = np.zeros((3, 3))
        cov_matrix[0, 0] = np.sum(smp_weights
                                  * (x_samples - x_expect) ** 2) / ssw
        cov_matrix[1, 1] = np.sum(smp_weights
                                  * (y_samples - y_expect) ** 2) / ssw
        cov_matrix[2, 2] = np.sum(smp_weights
                                  * (z_samples - z_expect) ** 2) / ssw
        cov_matrix[0, 1] = np.sum(smp_weights
                                  * (x_samples - x_expect)
                                  * (y_samples - y_expect)) / ssw
        cov_matrix[1, 0] = cov_matrix[0, 1]
        cov_matrix[0, 2] = np.sum(smp_weights
                                  * (x_samples - x_expect)
                                  * (z_samples - z_expect)) / ssw
        cov_matrix[2, 0] = cov_matrix[0, 2]
        cov_matrix[1, 2] = np.sum(smp_weights
                                  * (y_samples - y_expect)
                                  * (z_samples - z_expect)) / ssw
        cov_matrix[2, 1] = cov_matrix[1, 2]

        # Determining the maximum location, and taking 2xgrid cells positive
        # and negative for location in each dimension
        gau_3d = self._gaufit3d(coa_3d)

        # Converting the grid location to X,Y,Z
        xyz = self.lut.xyz2loc(np.array([[gau_3d[0][0],
                                          gau_3d[0][1],
                                          gau_3d[0][2]]]),
                               inverse=True)
        expect_vector = self.lut.xyz2coord(xyz)[0]

        expect_vector_cov = np.array([x_expect,
                                      y_expect,
                                      z_expect],
                                     dtype=float)
        loc_cov = np.array([[expect_vector_cov[0] / self.lut.cell_size[0],
                             expect_vector_cov[1] / self.lut.cell_size[1],
                             expect_vector_cov[2] / self.lut.cell_size[2]]])
        xyz_cov = self.lut.xyz2loc(loc_cov, inverse=True)
        crd_cov = self.lut.xyz2coord(xyz_cov)[0]

        xyz_err = np.array([gau_3d[2][0] * self.lut.cell_size[0],
                            gau_3d[2][1] * self.lut.cell_size[1],
                            gau_3d[2][2] * self.lut.cell_size[2]])

        return expect_vector, xyz_err, crd_cov, cov_matrix


# class DeepLearningPhaseDetection:

#     def __init__(self, sign, sige, sigz, srate):

#         #####################
#         # Hyperparameters
#         self.freq_min = 2.0
#         self.freq_max = 16.0

#         self.decimate_data = False  # If false, assumes data is already 100 Hz samprate

#         self.n_shift = 10  # Number of samples to shift the sliding window at a time
#         self.n_gpu = 0  # Number of GPUs to use (if any)

#         self.batch_size = 1000*3

#         self.half_dur = 2.00
#         self.only_dt = 0.01
#         self.n_win = int(self.half_dur/self.only_dt)
#         self.n_feat = 2*self.n_win

#         self.sign = sign
#         self.sige = sige
#         self.sigz = sigz
#         self.srate = srate

#         self.prob_S = None
#         self.prob_P = None
#         self.prob_N = None

#         self.models_path = '/raid1/jds70/PhaseLink/generalized-phase-detection/model_pol.json'
#         self.weights_path = '/raid1/jds70/PhaseLink/generalized-phase-detection/model_pol_best.hdf5'

#         self.PhaseProbability()

#     def sliding_window(self, data, size, stepsize=1, padded=False, axis=-1, copy=True):
#         """
#         Calculate a sliding window over a signal

#         Parameters
#         ----------
#         data : numpy array
#             The array to be slided over.
#         size : int
#             The sliding window size
#         stepsize : int
#             The sliding window stepsize. Defaults to 1.
#         axis : int
#             The axis to slide over. Defaults to the last axis.
#         copy : bool
#             Return strided array as copy to avoid sideffects when manipulating
#             the output array.

#         Returns
#         -------
#         data : numpy array
#             A matrix where row in last dimension consists of one instance
#             of the sliding window.
#         Notes
#         -----
#         - Be wary of setting `copy` to `False` as undesired sideffects with the
#           output values may occur.

#         Examples
#         --------
#         >>> a = numpy.array([1, 2, 3, 4, 5])
#         >>> sliding_window(a, size=3)
#         array([[1, 2, 3],
#                [2, 3, 4],
#                [3, 4, 5]])
#         >>> sliding_window(a, size=3, stepsize=2)
#         array([[1, 2, 3],
#                [3, 4, 5]])

#         See Also
#         --------
#         pieces : Calculate number of pieces available by sliding
#         """

#         if axis >= data.ndim:
#             raise ValueError(
#                 "Axis value out of range"
#             )

#         if stepsize < 1:
#             raise ValueError(
#                 "Stepsize may not be zero or negative"
#             )

#         if size > data.shape[axis]:
#             raise ValueError(
#                 "Sliding window size may not exceed size of selected axis"
#             )

#         shape = list(data.shape)
#         shape[axis] = np.floor(data.shape[axis] / stepsize - size / stepsize + 1).astype(int)
#         shape.append(size)

#         strides = list(data.strides)
#         strides[axis] *= stepsize
#         strides.append(data.strides[axis])

#         strided = np.lib.stride_tricks.as_strided(
#             data, shape=shape, strides=strides
#         )

#         if copy:
#             return strided.copy()
#         else:
#             return strided

#     def PhaseProbability(self):

#         # load json and create model
#         json_file = open(self.models_path, 'r')
#         loaded_model_json = json_file.read()
#         json_file.close()
#         model = model_from_json(loaded_model_json, custom_objects={'tf': tf})

#         # load weights into new model
#         model.load_weights(self.weights_path)
#         # print("Loaded model from disk")

#         # Parallelising for GPU usage
#         if self.n_gpu > 1:
#             from keras.utils import multi_gpu_model
#             model = multi_gpu_model(model, gpus=self.n_gpu)

#         # Manipulating the streams so samplerate and station same
#         sr = self.srate
#         dt = 1.0 / sr

#         self.prob_S = np.zeros(self.sign.shape)
#         self.prob_P = np.zeros(self.sign.shape)
#         self.prob_N = np.zeros(self.sign.shape)

#         for ii in range(self.sign.shape[0]):
#             tt = (np.arange(0, self.sign.shape[1], self.n_shift) + self.n_win) * dt
#             tt_i = np.arange(0, self.sign.shape[1], self.n_shift) + self.n_feat

#             sliding_N = self.sliding_window(self.sign[ii, :], self.n_feat,
#                                             stepsize=self.n_shift)
#             sliding_E = self.sliding_window(self.sige[ii, :], self.n_feat,
#                                             stepsize=self.n_shift)
#             sliding_Z = self.sliding_window(self.sigz[ii, :], self.n_feat,
#                                             stepsize=self.n_shift)

#             tr_win = np.zeros((sliding_N.shape[0], self.n_feat, 3))
#             tr_win[:, :, 0] = sliding_N
#             tr_win[:, :, 1] = sliding_E
#             tr_win[:, :, 2] = sliding_Z
#             # tr_win = tr_win / np.max(np.abs(tr_win), axis=(1,2))[:,None,None]
#             tt = tt[:tr_win.shape[0]]
#             tt_i = tt_i[:tr_win.shape[0]]

#             ts = model.predict(tr_win, verbose=False,
#                                batch_size=tr_win.shape[0])

#             SS = np.interp(np.arange(self.sign.shape[1])*dt, tt, ts[:, 1])
#             PP = np.interp(np.arange(self.sign.shape[1])*dt, tt, ts[:, 0])
#             NN = np.interp(np.arange(self.sign.shape[1])*dt, tt, ts[:, 2])
#             PP[np.isnan(PP)] = 0.0
#             SS[np.isnan(SS)] = 0.0
#             NN[np.isnan(NN)] = 0.0

#             # plt.plot(ts[:,0])
#             # plt.savefig('TEST.pdf')

#             self.prob_S[ii, :] = SS
#             self.prob_P[ii, :] = PP
#             self.prob_N[ii, :] = NN
