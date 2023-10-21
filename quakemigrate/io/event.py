# -*- coding: utf-8 -*-
"""
Module containing the Event class, which stores information related to an individual
event.

:copyright:
    2020–2023, QuakeMigrate developers.
:license:
    GNU General Public License, Version 3
    (https://www.gnu.org/licenses/gpl-3.0.html)

"""

import logging

import numpy as np
from obspy import Trace
import pandas as pd

import quakemigrate.util as util


EVENT_FILE_COLS = [
    "EventID",
    "DT",
    "X",
    "Y",
    "Z",
    "COA",
    "COA_NORM",
    "GAU_X",
    "GAU_Y",
    "GAU_Z",
    "GAU_ErrX",
    "GAU_ErrY",
    "GAU_ErrZ",
    "COV_ErrX",
    "COV_ErrY",
    "COV_ErrZ",
    "TRIG_COA",
    "DEC_COA",
    "DEC_COA_NORM",
]

XYZ, ERR_XYZ = ["X", "Y", "Z"], ["ErrX", "ErrY", "ErrZ"]


class Event:
    """
    Light class to encapsulate information about an event, including waveform data,
    coalescence information, origin time, locations, picks, magnitudes.

    Parameters
    ----------
    marginal_window : float
        Estimate of the uncertainty in the event origin time; time window over which the
        4-D coalescence image is marginalised around the peak coalescence time (event
        origin time) to produce the 3-D coalescence map.
    triggered_event : `pandas.Series` object, optional
        Contains information on the candidate event identified by
        :func:`~quakemigrate.signal.trigger.Trigger.trigger`

    Attributes
    ----------
    coa_data : `pandas.DataFrame` object
        Event coalescence data computed during locate.\n
        DT : `numpy.ndarray` of `obspy.UTCDateTime` objects, shape(nsamples)
            Timestamps for the coalescence data.
        COA : `numpy.ndarray` of floats, shape(nsamples)
            Max coalescence value in the grid at each timestep.
        COA_NORM : `numpy.ndarray` of floats, shape(nsamples)
            Normalised max coalescence value in the grid at each timestep.
        X : `numpy.ndarray` of floats, shape(nsamples)
            X coordinate of maximum coalescence value in the grid at each timestep, in
            input (geographic) projection coordinates.
        Y : `numpy.ndarray` of floats, shape(nsamples)
            Y coordinate of maximum coalescence value in the grid at each timestep, in
            input (geographic) projection coordinates.
        Z : `numpy.ndarray` of floats, shape(nsamples)
            Z coordinate of maximum coalescence value in the grid at each timestep, in
            input (geographic) projection coordinates.
    data : :class:`~quakemigrate.io.data.WaveformData` object
        Light class encapsulating waveform data returned from an archive query.
    hypocentre : `numpy.ndarray` of floats
        [X, Y, Z]; Geographical coordinates of the event hypocentre (default is
        interpolated peak of a spline function fitted to the marginalised 3-D
        coalescence map).
    locations : dict
        Information on the various locations and reported uncertainties.\n
        spline : dict
            The location of the peak coalescence value in the marginalised 3-D
            coalescence map, interpolated using a 3-D spline. If no spline fit was able
            to be made, it is just the gridded peak location.
        gaussian : dict
            The location and uncertainty as determined by fitting a 3-D Gaussian to the
            marginalised 3-D coalescence map in a small region around the (gridded) peak
            coalescence location.
        covariance : dict
            The location and uncertainty as determined by calculating the covariance of
            the coalescence values in X, Y, and Z above some percentile of the max
            coalescence value in the marginalised 3-D coalescence map.
    map4d : `numpy.ndarray`, shape(nx, ny, nz, nsamp), optional
        4-D coalescence map generated in
        :func:`~quakemigrate.signal.scan.QuakeScan.locate`.
    max_coalescence : dict
        Dictionary containing the raw and normalised maximum coalescence values in the
        3-D grid at the timestamp corresponding to the instantaneous (non-marginalised)
        maximum coalescence value in the 4-D grid (i.e. the event origin time).
    onset_data : :class:`~quakemigrate.signal.onsets.base.OnsetData` object
        Light class encapsulating data generated during onset calculation.
    otime : `obspy.UTCDateTime` object
        Timestamp of the instantaneous peak in the 4-D coalescence function generated in
        :func:`~quakemigrate.signal.scan.QuakeScan.locate` - best estimate of the event
        origin time.
    trigger_info : dict
        Useful information about the triggered event to be fed forward.\n
        TRIG_COA : float
            The peak value of the coalescence stream used to trigger the event.
        DEC_COA : float
            The coalescence value of the "raw" maximum coalsecence stream at the
            `trigger_time`.
        DEC_COA_NORM : float
            The coalescence value of the normalised maximum coalsecence stream at the
            `trigger_time`.
    trigger_time : `obspy.UTCDateTime` object
        The time of the peak in the continuous coalescence stream (output by detect)
        corresponding to the triggered event.
    uid : str
        A unique identifier for the event based on the event trigger time.

    Methods
    -------
    add_compute_output(times, max_coa, max_coa_n, coord, map4d, onset_data)
        Add values returned by :func:`~quakemigrate.signal.scan.QuakeScan._compute` to
        the event.
    add_covariance_location(xyz, xyz_unc)
        Add the covariance location and uncertainty to the event.
    add_gaussian_location(xyz, xyz_unc)
        Add the gaussian location and uncertainty to the event.
    add_spline_location(xyz)
        Add the spline-interpolated location to the event.
    add_picks(pick_df)
        Add phase picks to the event.
    add_local_magnitude(mag, mag_err, mag_r2)
        Add local magnitude to the event.
    add_waveform_data(data)
        Add waveform data read from the archive to the event (as a
        :class:`~quakemigrate.io.data.WaveformData` object).
    in_marginal_window(marginal_window)
        Simple test to see if event is within the marginal window around the event
        origin time (time of max instantaneous coalescence value).
    mw_times(marginal_window, sampling_rate)
        Generates timestamps for data in the window around the event trigger scanned by
        :func:`~quakemigrate.signal.scan.QuakeScan._compute`;
        `trigger_time` +/- 2*`marginal_window`.
    trim2window(marginal_window)
        Trim the coalescence data and `map4d` to the marginal window about the event
        origin time.
    write(run)
        Output the event to a .event file.
    get_hypocentre(method)
        Get the event hypocentre estimate calculated by a specific method;
        {"gaussian", "covariance", "spline"}.

    """

    def __init__(self, marginal_window, triggered_event=None):
        """Instantiate the Event object."""

        self.marginal_window = marginal_window

        if triggered_event is not None:
            self.uid = triggered_event["EventID"]
            self.trigger_time = triggered_event["CoaTime"]
            self.trigger_info = self._parse_triggered_event(triggered_event)

        self.data = None
        self.coa_data = None
        self.map4d = None
        self.onset_data = None
        self.otime = None
        self.locations = {}
        self.picks = {}
        self.localmag = {}

    def add_waveform_data(self, data):
        """
        Add waveform data in the form of a :class:`~quakemigrate.io.data.WaveformData`
        object.

        Parameters
        ----------
        data : :class:`~quakemigrate.io.data.WaveformData` object
            Contains cut waveforms - `raw_waveforms` may be for all stations in the
            archive, and include an additional pre- and post-pad; `waveforms` contains
            data only for the stations and time period required for migration.

        """

        self.data = data

    def add_compute_output(self, times, max_coa, max_coa_n, coord, map4d, onset_data):
        """
        Append outputs of compute to the Event object. This includes time series of the
        maximum coalescence values in the 3-D grid at each timestep, and their
        locations, the full 4-D coalescence map, and the onset data generated for
        migration.

        Parameters
        ----------
        times : `numpy.ndarray` of `obspy.UTCDateTime` objects, shape(nsamples)
            Timestamps for the coalescence data.
        max_coa : `numpy.ndarray` of floats, shape(nsamples)
            Max coalescence value in the grid at each timestep.
        max_coa_n : `numpy.ndarray` of floats, shape(nsamples)
            Normalised max coalescence value in the grid at each timestep.
        coord : `numpy.ndarray` of floats, shape(nsamples, 3)
            [x, y, z] Location of maximum coalescence in the grid at each timestep, in
            input (geographic) projection coordinates
        map4d : `numpy.ndarry`, shape(nx, ny, nz, nsamp)
            4-D coalescence map.
        onset_data : :class:`~quakemigrate.signal.onsets.base.OnsetData` object
            Light class encapsulating data generated during onset calculation.

        """

        self.coa_data = pd.DataFrame(
            {
                "DT": times,
                "COA": max_coa,
                "COA_NORM": max_coa_n,
                "X": coord[:, 0],
                "Y": coord[:, 1],
                "Z": coord[:, 2],
            }
        )
        self.map4d = map4d
        idxmax = self.coa_data["COA"].astype(float).idxmax()
        self.otime = self.coa_data.iloc[idxmax]["DT"]

        self.onset_data = onset_data

    def add_covariance_location(self, xyz, xyz_unc):
        """
        Add the location determined by calculating the 3-D covariance of the
        marginalised coalescence map filtered above a percentile threshold.

        Parameters
        ----------
        xyz : `numpy.ndarray` of floats, shape(3)
            Geographical coordinates (lon/lat/depth) of covariance location.
        xyz_unc : `numpy.ndarray' of floats, shape(3)
            One sigma uncertainties on the covariance location (units determined by the
            LUT projection units).

        """

        self.locations["covariance"] = {
            "X": xyz[0],
            "Y": xyz[1],
            "Z": xyz[2],
            "ErrX": xyz_unc[0],
            "ErrY": xyz_unc[1],
            "ErrZ": xyz_unc[2],
        }

    def add_gaussian_location(self, xyz, xyz_unc):
        """
        Add the location determined by fitting a 3-D Gaussian to a small window around
        the Gaussian smoothed maximum coalescence location.

        Parameters
        ----------
        xyz : `numpy.ndarray` of floats, shape(3)
            Geographical coordinates (lon/lat/depth) of Gaussian location.
        xyz_unc : `numpy.ndarray' of floats, shape(3)
            One sigma uncertainties on the Gaussian location (units determined by the
            LUT projection units).

        """

        self.locations["gaussian"] = {
            "X": xyz[0],
            "Y": xyz[1],
            "Z": xyz[2],
            "ErrX": xyz_unc[0],
            "ErrY": xyz_unc[1],
            "ErrZ": xyz_unc[2],
        }

    def add_spline_location(self, xyz):
        """
        Add the location determined by fitting a 3-D spline to a small window around the
        maximum coalescence location and interpolating.

        Parameters
        ----------
        xyz : `numpy.ndarray` of floats, shape(3)
            Geographical coordinates (lon/lat/depth) of best-fitting location.

        """

        self.locations["spline"] = dict(zip(XYZ, xyz))

    def add_picks(self, pick_df, **kwargs):
        """
        Add phase picks, and a selection of picker outputs and parameters.

        Parameters
        ----------
        pick_df : `pandas.DataFrame` object
            DataFrame that contains the measured picks with columns:
            ["Name", "Phase", "ModelledTime", "PickTime", "PickError", "SNR"]
            Each row contains the phase pick from one station/phase.

        **kwargs
            For :class:`~quakemigrate.signal.pickers.gaussian.GaussianPicker`:\n
                gaussfits : dict of dicts
                    Keys "station"["phase"], each containing:\n
                        "popt" : popt
                        "xdata" : x_data
                        "xdata_dt" : x_data_dt
                        "PickValue" : max_onset
                        "PickThreshold" : threshold
                pick_windows : dict
                    {station : phase{window}}\n
                    window: [min_time, modelled_arrival, max_time] - all ints,
                    referring to indices of the onset function.

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
            (Weighted) standard deviation of the magnitude estimates from amplitude
            measurements on individual stations/channels.
        mag_r2 : float
            r-squared statistic describing the fit of the amplitude vs. distance curve
            predicted by the calculated mean_mag and chosen attenuation model to the
            measured amplitude observations. This is intended to be used to help
            discriminate between 'real' events, for which the predicted amplitude vs.
            distance curve should provide a good fit to the observations, from
            artefacts, which in general will not.

        """

        self.localmag["ML"] = mag
        self.localmag["ML_Err"] = mag_err
        self.localmag["ML_r2"] = mag_r2

    def in_marginal_window(self):
        """
        Test if triggered event time is within marginal window around the maximum
        coalescence time (origin time).

        Returns
        -------
        cond : bool
            Result of test.

        """

        window_start = self.otime - self.marginal_window
        window_end = self.otime + self.marginal_window
        cond = self.trigger_time > window_start and self.trigger_time < window_end
        if not cond:
            logging.info(f"\tEvent {self.uid} is outside marginal window.")
            logging.info(
                "\tDefine more realistic error - the marginal window should be an "
                "estimate of overall uncertainty"
            )
            logging.info(
                "\tdetermined from expected spatial uncertainty and uncertainty in the "
                "seismic velocity model.\n"
            )
            logging.info(util.log_spacer)

        return cond

    def mw_times(self, sampling_rate):
        """
        Utility function to generate timestamps for the time period around the trigger
        time for which the 4-D coalescence function is calculated in
        :func:`~quakemigrate.signal.scan.QuakeScan._compute`.

        Returns
        -------
        times : `numpy.ndarray` of `obspy.UTCDateTime`, shape(nsamples)
            Timestamps for time range `trigger_time` +/- 2 * `marginal_window`.

        """

        # Utilise the .times() method of `obspy.Trace` objects
        tr = Trace(
            header={
                "npts": 4 * self.marginal_window * sampling_rate + 1,
                "sampling_rate": sampling_rate,
                "starttime": self.trigger_time - 2 * self.marginal_window,
            }
        )
        return tr.times(type="utcdatetime")

    def trim2window(self):
        """
        Trim the coalescence data to be within the marginal window.

        """

        window_start = self.otime - self.marginal_window
        window_end = self.otime + self.marginal_window

        self.coa_data = self.coa_data[
            (self.coa_data["DT"] >= window_start) & (self.coa_data["DT"] <= window_end)
        ]
        self.map4d = self.map4d[
            :, :, :, self.coa_data.index[0] : self.coa_data.index[-1]
        ]
        self.coa_data.reset_index(drop=True, inplace=True)

        idxmax = self.coa_data["COA"].astype(float).idxmax()
        self.otime = self.coa_data.iloc[idxmax]["DT"]

    def write(self, run, lut):
        """
        Write event to a .event file.

        Parameters
        ----------
        run : :class:`~quakemigrate.io.core.Run` object
            Light class encapsulating i/o path information for a given run.
        lut : :class:`~quakemigrate.lut.lut.LUT` object
            Contains the traveltime lookup tables for seismic phases, computed for some
            pre-defined velocity model.

        """

        fpath = run.path / "locate" / run.subname / "events"
        fpath.mkdir(exist_ok=True, parents=True)

        out = {"EventID": self.uid, **self.trigger_info, **self.localmag}
        out = {**out, **self.max_coalescence}

        # Rename keys for locations; do not output covariance loc (just err)
        loc = self.locations["spline"]
        gau = dict(
            (f"GAU_{key}", value) for (key, value) in self.locations["gaussian"].items()
        )
        cov = dict(
            (f"COV_{key}", value)
            for (key, value) in list(self.locations["covariance"].items())[3:]
        )
        out = {**out, **loc, **gau, **cov}

        if self.localmag.get("ML") is not None:
            event_file_cols = EVENT_FILE_COLS + ["ML", "ML_Err", "ML_r2"]
        else:
            event_file_cols = EVENT_FILE_COLS

        event_df = pd.DataFrame([out])[event_file_cols]

        # Set floating point precision for COA values
        for col in event_df.filter(like="COA").columns:
            event_df[col] = event_df[col].map(lambda x: f"{x:.4g}", na_action="ignore")

        # Set floating point precision for locations & loc uncertainties
        for axis_precision, axis in zip(lut.precision, XYZ):
            # Sort out which columns to format
            cols = [axis, f"GAU_{axis}"]
            if axis == "Z":
                unit_correction = 3 if lut.unit_name == "km" else 0
                decimals = max((axis_precision + 2), 0 + unit_correction)
                cols.extend(event_df.filter(regex="Err[X,Y,Z]"))
            else:
                decimals = max((axis_precision + 2), 6)
            for col in cols:
                event_df[col] = event_df.loc[:, col].round(decimals=decimals)
                if decimals <= 0:
                    event_df[col] = event_df.loc[:, col].astype(int)

        # Set floating point precision for mags (if applicable)
        if self.localmag.get("ML") is not None:
            for col in ["ML", "ML_Err", "ML_r2"]:
                event_df[col] = event_df[col].map(
                    lambda x: f"{x:.3g}", na_action="ignore"
                )

        fstem = f"{self.uid}"
        file = (fpath / fstem).with_suffix(".event")
        event_df.to_csv(file, index=False)

    def get_hypocentre(self, method="spline"):
        """
        Get an estimate of the event hypocentre location.

        Parameters
        ----------
        method : {"spline", "gaussian", "covariance"}, optional
            Which location result to return. (Default "spline").

        Returns
        -------
        ev_loc : `numpy.ndarray` of floats
            [x_coordinate, y_coordinate, z_coordinate] of event hypocentre, in the
            global (geographic) coordinate system.

        """

        hypocentre = self.locations[method]

        ev_loc = np.array([hypocentre[k] for k in XYZ])

        return ev_loc

    hypocentre = property(get_hypocentre)

    def get_loc_uncertainty(self, method="gaussian"):
        """
        Get an estimate of the hypocentre location uncertainty.

        Parameters
        ----------
        method : {"gaussian", "covariance"}, optional
            Which location result to return. (Default "gaussian").

        Returns
        -------
        ev_loc_unc : `numpy.ndarray` of floats
            [x_uncertainty, y_uncertainty, z_uncertainty] of event hypocentre; units are
            determined by the LUT projection units.

        """

        loc = self.locations[method]

        ev_loc_unc = np.array([loc[k] for k in ERR_XYZ])

        return ev_loc_unc

    loc_uncertainty = property(get_loc_uncertainty)

    @property
    def local_magnitude(self):
        """Get the local magnitude, if it exists."""

        if len(self.localmag) == 0:
            return None
        else:
            return (value for _, value in self.localmag.items())

    @property
    def max_coalescence(self):
        """Get information related to the maximum coalescence."""
        idxmax = self.coa_data["COA"].astype("float").idxmax()
        max_coa = self.coa_data.iloc[idxmax]
        keys = ["DT", "COA", "COA_NORM"]

        return dict(zip(keys, max_coa[keys].values))

    def _parse_triggered_event(self, event_data):
        """
        Parse the information from a triggered event `pandas.Series` object into the
        Event object.

        Parameters
        ----------
        event_data : `~pandas.Series` object
            Contains information on the event output by the trigger stage.

        """

        try:
            trigger_info = {
                "TRIG_COA": event_data["TRIG_COA"],
                "DEC_COA": event_data["COA"],
                "DEC_COA_NORM": event_data["COA_NORM"],
            }
        except KeyError:
            # --- Backwards compatibility ---
            try:
                trigger_info = {
                    "TRIG_COA": event_data["COA_V"],
                    "DEC_COA": event_data["COA"],
                    "DEC_COA_NORM": event_data["COA_NORM"],
                }
            except KeyError:
                trigger_info = {
                    "TRIG_COA": event_data["COA_V"],
                    "DEC_COA": np.nan,
                    "DEC_COA_NORM": np.nan,
                }

        return trigger_info
