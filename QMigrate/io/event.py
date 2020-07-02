# -*- coding: utf-8 -*-
"""
Module containing the Event class, which stores information related to an
individual event.

"""

import logging

import numpy as np
from obspy import Trace
import pandas as pd

import QMigrate.util as util


EVENT_FILE_COLS = ["EventID", "DT", "COA", "COA_NORM", "X", "Y", "Z",
                   "LocalGaussian_X", "LocalGaussian_Y", "LocalGaussian_Z",
                   "LocalGaussian_ErrX", "LocalGaussian_ErrY",
                   "LocalGaussian_ErrZ", "GlobalCovariance_X",
                   "GlobalCovariance_Y", "GlobalCovariance_Z",
                   "GlobalCovariance_ErrX", "GlobalCovariance_ErrY",
                   "GlobalCovariance_ErrZ", "TRIG_COA", "DEC_COA",
                   "DEC_COA_NORM", "ML", "ML_Err", "ML_r2"]


class Event:
    """
    Light class to encapsulate information about an event, including waveform
    data (raw, filtered, unfiltered), coalescence information, locations and
    origin times, picks, magnitudes.

    Parameters
    ----------
    triggered_event : `pandas.Series` object
        Contains information on the event output by the trigger stage.
    marginal_window : float
        Estimate of the uncertainty in the earthquake origin time.

    Attributes
    ----------
    coa_data : `pandas.DataFrame` object
        Event coalescence data computed during locate.
        DT : `numpy.ndarray` of `obspy.UTCDateTime` objects, shape(nsamples)
            Timestamp of first sample of coalescence data.
        COA : `numpy.ndarray` of floats, shape(nsamples)
            Coalescence value through time.
        COA_NORM : `numpy.ndarray` of floats, shape(nsamples)
            Normalised coalescence value through time.
        X : `numpy.ndarray` of floats, shape(nsamples)
            X coordinate of maximum coalescence through time in input
            projection space.
        Y : `numpy.ndarray` of floats, shape(nsamples)
            Y coordinate of maximum coalescence through time in input
            projection space.
        Z : `numpy.ndarray` of floats, shape(nsamples)
            Z coordinate of maximum coalescence through time in input
            projection space.
    coa_time : `obspy.UTCDateTime` object
        The peak coalescence time of the triggered event from the (decimated)
        coalescence output by detect.
    df : `pandas.DataFrame` object
        Collects all the information together for an event to be written out to
        a .event file.
    hypocentre : `numpy.ndarray` of floats
        Geographical coordinates of the instantaneous event hypocentre.
    locations : dict
        Information on the various locations and reported uncertainties.
        spline : dict
            The location of the maximum coalescence in the marginalised
            grid, interpolated using a 3-D spline. If no spline fit was able to
            be made, it is just the location in the original grid.
        gaussian : dict
            The location and uncertainty as determined by fitting a 3-D
            Gaussian to the coalescence in a small region around the maximum
            coalescence in the marginalised grid.
        covariance : dict
            The location and uncertainty as determined by calculating the
            covariance of the coalescence values in X, Y, and Z above some
            percentile.
    map4d : `numpy.ndarry`, shape(nx, ny, nz, nsamp), optional
        4-D coalescence map.
    max_coalescence : dict
        Dictionary containing the timestamps of the maximum coalescence, the
        coalescence values, and the normalised coalescence values.
    otime : `obspy.UTCDateTime` object
        Timestamp of the instantaneous peak in the coalescence function.
    trigger_info : dict
        Other useful information about the triggered event to be fed forward.
        TRIG_COA : float
            The peak value of the coalescence stream used to trigger.
        DEC_COA : float
            The peak coalescence value.
        DEC_COA_NORM : float
            The peak normalised coalescence value.
    uid : str
        A unique identifier for the event based on the peak coalescence time.

    Methods
    -------
    add_coalescence(times, max_coa, max_coa_n, coord, map4d)
        Add values returned by QuakeScan._compute to the event.
    add_covariance_location(xyz, xyz_unc)
        Add the covariance location and uncertainty to the event.
    add_gaussian_location(xyz, xyz_unc)
        Add the gaussian location and uncertainty to the event.
    add_spline_location(xyz)
        Add the splined location to the event.
    add_picks(pick_df)
        Add phase picks to the event.
    add_local_magnitude(mag, mag_err, mag_r2)
        Add local magnitude to the event.
    in_marginal_window(marginal_window)
        Simple test to see if event is within the marginal window around the
        triggered event time.
    mw_times(marginal_window, sampling_rate)
        Generates timestamps for data in the marginal window.
    trim2window(marginal_window)
        Trim the coalescence data and map4d to the marginal window.
    write(run)
        Output the event to a .event file.
    get_hypocentre(method)
        Get the event hypocentre estimate calculated by a specific method.

    """

    coa_data = None
    df = None
    map4d = None

    def __init__(self, triggered_event, marginal_window):
        """Instantiate the Event object."""

        self.coa_time = triggered_event["CoaTime"]
        self.uid = triggered_event["EventID"]
        self.marginal_window = marginal_window

        try:
            self.trigger_info = {"TRIG_COA": triggered_event["COA_V"],
                                 "DEC_COA": triggered_event["COA"],
                                 "DEC_COA_NORM": triggered_event["COA_NORM"]}
        except KeyError:
            # --- Backwards compatibility ---
            self.trigger_info = {"TRIG_COA": triggered_event["COA_V"],
                                 "DEC_COA": np.nan,
                                 "DEC_COA_NORM": np.nan}

        self.data = None
        self.locations = {}
        self.picks = {}
        self.localmag = {}

    def add_waveform_data(self, data):
        """
        Add waveform data in the form of a WaveformData object.

        Parameters
        ----------
        data : `QMigrate.io.data.WaveformData` object
            Contains raw cut waveforms, signal data (at a unified sample rate,
            prepared for use in scan), station availability info, onset
            functions calculated from the signal data, pre_processed filtered
            waveforms, et cetera.

        """

        self.data = data

    def add_coalescence(self, times, max_coa, max_coa_n, coord, map4d):
        """
        Append output of _compute from locate() to `obspy.Stream` object.

        Parameters
        ----------
        times : `numpy.ndarray` of `obspy.UTCDateTime` objects, shape(nsamples)
            Timestamp of first sample of coalescence data.
        max_coa : `numpy.ndarray` of floats, shape(nsamples)
            Coalescence value through time.
        max_coa_n : `numpy.ndarray` of floats, shape(nsamples)
            Normalised coalescence value through time.
        coord : `numpy.ndarray` of floats, shape(nsamples)
            Location of maximum coalescence through time in input projection
            space.
        map4d : `numpy.ndarry`, shape(nx, ny, nz, nsamp), optional
            4-D coalescence map.

        """

        self.coa_data = pd.DataFrame({"DT": times,
                                      "COA": max_coa,
                                      "COA_NORM": max_coa_n,
                                      "X": coord[:, 0],
                                      "Y": coord[:, 1],
                                      "Z": coord[:, 2]})
        self.map4d = map4d

    def add_covariance_location(self, xyz, xyz_unc):
        """
        Add the location determined by calculating the 3-D covariance of the
        marginalised coalescence map filtered above a percentile threshold.

        Parameters
        ----------
        xyz : `numpy.ndarray' of floats, shape(3)
            Geographical coordinates (lon/lat/depth) of covariance location.
        xyz_unc : `numpy.ndarray' of floats, shape(3)
            One sigma uncertainties on the covariance location (in m).

        """

        self.locations["covariance"] = {"GlobalCovariance_X": xyz[0],
                                        "GlobalCovariance_Y": xyz[1],
                                        "GlobalCovariance_Z": xyz[2],
                                        "GlobalCovariance_ErrX": xyz_unc[0],
                                        "GlobalCovariance_ErrY": xyz_unc[1],
                                        "GlobalCovariance_ErrZ": xyz_unc[2]}

    def add_gaussian_location(self, xyz, xyz_unc):
        """
        Add the location determined by fitting a 3-D Gaussian to a small window
        around the Gaussian smoothed maximum coalescence location.

        Parameters
        ----------
        xyz : `numpy.ndarray' of floats, shape(3)
            Geographical coordinates (lon/lat/depth) of Gaussian location.
        xyz_unc : `numpy.ndarray' of floats, shape(3)
            One sigma uncertainties on the Gaussian location (in m).

        """

        self.locations["gaussian"] = {"LocalGaussian_X": xyz[0],
                                      "LocalGaussian_Y": xyz[1],
                                      "LocalGaussian_Z": xyz[2],
                                      "LocalGaussian_ErrX": xyz_unc[0],
                                      "LocalGaussian_ErrY": xyz_unc[1],
                                      "LocalGaussian_ErrZ": xyz_unc[2]}

    def add_spline_location(self, xyz):
        """
        Add the location determined by fitting a 3-D spline to a small window
        around the maximum coalescence location and interpolating.

        Parameters
        ----------
        xyz : `numpy.ndarray' of floats, shape(3)
            Geographical coordinates (lon/lat/depth) of best-fitting location.

        """

        self.locations["spline"] = dict(zip(["X", "Y", "Z"], xyz))

    def add_picks(self, pick_df, **kwargs):
        """
        Add phase picks, and a selection of picker outputs and parameters.

        Parameters
        ----------
        pick_df : `pandas.DataFrame` object
            DataFrame that contains the measured picks with columns:
            ["Name", "Phase", "ModelledTime", "PickTime", "PickError", "SNR"]
            Each row contains the phase pick from one station/phase.

        For GaussianPicker:
            gaussfits : dict
                {station : phase{gaussian_fit_params}}
                gaussian fit params: {"popt": popt,
                                      "xdata": x_data,
                                      "xdata_dt": x_data_dt,
                                      "PickValue": max_onset,
                                      "PickThreshold": threshold}
            pick_windows : dict
                {station : phase{window}}
                window: [min_time, max_time]
            pick_threshold : float
                float (between 0 and 1)
                Picks will only be made if the onset function exceeds this
                percentile of the noise level (average amplitude of onset
                function outside pick windows).
            fraction_tt : float
                Defines width of time window around expected phase arrival time
                in which to search for a phase pick as a function of the
                traveltime from the event location to that station -- should be
                an estimate of the uncertainty in the velocity model.

        """

        # DataFrame containing the phase picks
        self.picks["df"] = pick_df

        # Any further information that is useful to store on the Event object
        for key, value in kwargs.items():
            self.picks[key] = value

    def add_local_magnitude(self, mag, mag_err, mag_r2):
        """
        Add outputs from local magnitude calculation to the Event object.

        Parameters
        ----------
        mag : float
            Network-averaged local magnitude estimate for the event.
        mag_err : float
            (Weighted) standard deviation of the magnitude estimates from
            amplitude measurements on individual stations/channels.
        mag_r2 : float
            r-squared statistic describing the fit of the amplitude vs.
            distance curve predicted by the calculated mean_mag and chosen
            attenuation model to the measured amplitude observations. This is
            intended to be used to help discriminate between 'real' events, for
            which the predicted amplitude vs. distance curve should provide a
            good fit to the observations, from artefacts, which in general will
            not.

        """

        self.localmag["ML"] = mag
        self.localmag["ML_Err"] = mag_err
        self.localmag["ML_r2"] = mag_r2

    def in_marginal_window(self):
        """
        Test if triggered event time is within marginal window around
        the maximum coalescence time (origin time).

        Returns
        -------
        cond : bool
            Result of test.

        """

        window_start = self.otime - self.marginal_window
        window_end = self.otime + self.marginal_window
        cond = (self.coa_time > window_start) and (self.coa_time < window_end)
        if not cond:
            logging.info(f"\tEvent {self.uid} is outside marginal window.")
            logging.info("\tDefine more realistic error - the marginal "
                         "window should be an estimate of overall uncertainty")
            logging.info("\tdetermined from expected spatial uncertainty"
                         " and uncertainty in the seismic velocity model.\n")
            logging.info(util.log_spacer)

        return cond

    def mw_times(self, sampling_rate):
        """
        Utility function to generate timestamps between `data.starttime` and
        `data.endtime`, with a sample size of `data.sample_size`

        Returns
        -------
        times : `numpy.ndarray`, shape(nsamples)
            Timestamps for the timeseries data.

        """

        # Utilise the .times() method of `obspy.Trace` objects
        tr = Trace(header={"npts": 4*self.marginal_window*sampling_rate + 1,
                           "sampling_rate": sampling_rate,
                           "starttime": self.coa_time - 2*self.marginal_window})
        return tr.times(type="utcdatetime")

    def trim2window(self):
        """
        Trim the coalescence data to be within the marginal window.

        """

        window_start = self.otime - self.marginal_window
        window_end = self.otime + self.marginal_window

        self.coa_data = self.coa_data[(self.coa_data["DT"] >= window_start) &
                                      (self.coa_data["DT"] <= window_end)]
        self.map4d = self.map4d[:, :, :,
                                self.coa_data.index[0]:self.coa_data.index[-1]]
        self.coa_data.reset_index(drop=True, inplace=True)

    def write(self, run):
        """
        Write event. to a .event file.

        Parameters
        ----------
        run : `QMigrate.io.Run` object
            Light class encapsulating i/o path information for a given run.

        """

        fpath = run.path / "locate" / run.subname / "events"
        fpath.mkdir(exist_ok=True, parents=True)

        out = {"EventID": self.uid, **self.trigger_info, **self.localmag}
        out = {**out, **self.max_coalescence}
        for _, location in self.locations.items():
            out = {**out, **location}

        self.df = pd.DataFrame([out])[EVENT_FILE_COLS]

        fstem = f"{self.uid}"
        file = (fpath / fstem).with_suffix(".event")
        self.df.to_csv(file, index=False)

    def get_hypocentre(self, method="spline"):
        """
        Get an estimate of the hypocentral location.

        Parameters
        ----------
        method : {"spline", "gaussian", "covariance"}, optional
            Which location result to return. (Default "spline")

        Returns
        -------
        ev_loc : ndarray of floats
            [x_coordinate, y_coordinate, z_coordinate] of event hypocentre, in
            the global coordinate system.

        """

        hypocentre = self.locations[method]

        ev_loc = np.array([hypocentre[k] for k in list(hypocentre.keys())[:3]])

        return ev_loc

    hypocentre = property(get_hypocentre)

    @property
    def max_coalescence(self):
        """Get information related to the maximum coalescence."""
        idxmax = self.coa_data["COA"].astype("float").idxmax()
        max_coa = self.coa_data.iloc[idxmax]
        keys = ["DT", "COA", "COA_NORM"]

        return dict(zip(keys, max_coa[keys].values))

    @property
    def otime(self):
        """Get the origin time based on the peak coalescence."""
        idxmax = self.coa_data["COA"].astype(float).idxmax()
        return self.coa_data.iloc[idxmax]["DT"]
