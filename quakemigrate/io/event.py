"""
Module containing the Event class, which stores information related to an individual
event.

:copyright:
    2020–2026, QuakeMigrate developers.
:license:
    GNU General Public License, Version 3
    (https://www.gnu.org/licenses/gpl-3.0.html)

"""

from __future__ import annotations

import logging
from typing import Literal, TYPE_CHECKING

import numpy as np
import pandas as pd
from obspy import Trace

import quakemigrate.util as util


if TYPE_CHECKING:
    from obspy import UTCDateTime

    from quakemigrate.io.core import Run
    from quakemigrate.io.data import WaveformData
    from quakemigrate.lut.lut import LUT
    from quakemigrate.signal.onsets.base import OnsetData


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
    "COV_Err_XYZ",
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
    marginal_window:
        Estimate of the uncertainty in the event origin time; time window over which the
        4-D coalescence image is marginalised around the peak coalescence time (event
        origin time) to produce the 3-D coalescence map.
    triggered_event:
        Contains information on the candidate event identified by
        :func:`~quakemigrate.signal.trigger.Trigger.trigger`

    Attributes
    ----------
    coa_data:
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
    data:
        Light class encapsulating waveform data returned from an archive query.
    hypocentre:
        [X, Y, Z]; Geographical coordinates of the event hypocentre (default is
        interpolated peak of a spline function fitted to the marginalised 3-D
        coalescence map).
    locations:
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
    map4d:
        4-D coalescence map generated in
        :func:`~quakemigrate.signal.scan.QuakeScan.locate`.
    max_coalescence:
        Dictionary containing the raw and normalised maximum coalescence values in the
        3-D grid at the timestamp corresponding to the instantaneous (non-marginalised)
        maximum coalescence value in the 4-D grid (i.e. the event origin time).
    onset_data:
        Light class encapsulating data generated during onset calculation.
    otime:
        Timestamp of the instantaneous peak in the 4-D coalescence function generated in
        :func:`~quakemigrate.signal.scan.QuakeScan.locate` - best estimate of the event
        origin time.
    trigger_info:
        Useful information about the triggered event to be fed forward.\n
        TRIG_COA : float
            The peak value of the coalescence stream used to trigger the event.
        DEC_COA : float
            The coalescence value of the "raw" maximum coalsecence stream at the
            `trigger_time`.
        DEC_COA_NORM : float
            The coalescence value of the normalised maximum coalsecence stream at the
            `trigger_time`.
    trigger_time:
        The time of the peak in the continuous coalescence stream (output by detect)
        corresponding to the triggered event.
    uid:
        A unique identifier for the event based on the event trigger time.

    """

    def __init__(
        self, marginal_window: float, triggered_event: pd.Series | None = None
    ):
        """Instantiate the Event object."""

        self.marginal_window = marginal_window

        if triggered_event is not None:
            self.uid: str = triggered_event["EventID"]
            self.trigger_time: UTCDateTime = triggered_event["CoaTime"]
            self.trigger_info: dict = self._parse_triggered_event(triggered_event)

        self.data: WaveformData | None = None
        self.coa_data: pd.DataFrame | None = None
        self.map4d: np.ndarray | None = None
        self.onset_data: OnsetData | None = None
        self.otime: UTCDateTime | None = None
        self.locations: dict = {}
        self.picks: dict = {}
        self.localmag: dict = {}

    def add_waveform_data(self, data: WaveformData) -> None:
        """
        Add waveform data in the form of a :class:`~quakemigrate.io.data.WaveformData`
        object.

        Parameters
        ----------
        data:
            Contains cut waveforms - `raw_waveforms` may be for all stations in the
            archive, and include an additional pre- and post-pad; `waveforms` contains
            data only for the stations and time period required for migration.

        """

        self.data = data

    def add_compute_output(
        self,
        times: np.ndarray[UTCDateTime],
        max_coa: np.ndarray[float],
        max_coa_n: np.ndarray[float],
        coord: np.ndarray[float],
        map4d: np.ndarray[float],
        onset_data: OnsetData,
    ) -> None:
        """
        Append outputs of compute to the Event object. This includes time series of the
        maximum coalescence values in the 3-D grid at each timestep, and their
        locations, the full 4-D coalescence map, and the onset data generated for
        migration.

        Parameters
        ----------
        times:
            Timestamps for the coalescence data.
        max_coa:
            Max coalescence value in the grid at each timestep.
        max_coa_n:
            Normalised max coalescence value in the grid at each timestep.
        coord:
            [x, y, z] Location of maximum coalescence in the grid at each timestep, in
            input (geographic) projection coordinates
        map4d:
            4-D coalescence map.
        onset_data:
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

    def add_covariance_location(self, xyz: np.ndarray, xyz_unc: np.ndarray) -> None:
        """
        Add the location determined by calculating the 3-D covariance of the
        marginalised coalescence map filtered above a percentile threshold.

        Parameters
        ----------
        xyz:
            Geographical coordinates (lon/lat/depth) of covariance location.
        xyz_unc:
            One sigma uncertainties on the covariance location (units determined by the
            LUT projection units).

        """

        # Compute geometric mean of the covariance uncertainties, for use when filtering
        xyz_unc = np.asarray(xyz_unc, dtype=float)
        valid = np.isfinite(xyz_unc) & (xyz_unc > 0)
        n_valid = np.sum(valid)
        cov_err_xyz = (
            np.nan if n_valid == 0 else np.power(np.prod(xyz_unc[valid]), 1.0 / n_valid)
        )

        self.locations["covariance"] = {
            "X": xyz[0],
            "Y": xyz[1],
            "Z": xyz[2],
            "ErrX": xyz_unc[0],
            "ErrY": xyz_unc[1],
            "ErrZ": xyz_unc[2],
            "Err_XYZ": cov_err_xyz,
        }

    def add_gaussian_location(
        self, xyz: np.ndarray, xyz_unc: np.ndarray, gaussian: dict | None = None
    ) -> None:
        """
        Add the location determined by fitting a 3-D Gaussian to a small window around
        the Gaussian smoothed maximum coalescence location.

        Parameters
        ----------
        xyz:
            Geographical coordinates (lon/lat/depth) of Gaussian location.
        xyz_unc:
            One sigma uncertainties on the Gaussian location (units determined by the
            LUT projection units).
        gaussian:
            Various useful intermediary stages used for the calculation of the Gaussian
            location and associated uncertainty ellipsoid.

        """

        self.locations["gaussian"] = {
            "X": xyz[0],
            "Y": xyz[1],
            "Z": xyz[2],
            "ErrX": xyz_unc[0],
            "ErrY": xyz_unc[1],
            "ErrZ": xyz_unc[2],
        }

        if gaussian is not None:
            self.locations["gaussian"].update(
                {
                    "principal_uncertainty": gaussian["principal_uncertainty"],
                    "principal_axes": gaussian["principal_axes"],
                    "covariance_matrix": gaussian["covariance_matrix"],
                    "covariance_matrix_physical": gaussian[
                        "covariance_matrix_physical"
                    ],
                    "precision_matrix": gaussian["precision_matrix"],
                    "amplitude": gaussian["amplitude"],
                    "depth_constrained": gaussian["depth_constrained"],
                    "fit_dims": gaussian["fit_dims"],
                    "grid_location": gaussian["grid_location"],
                }
            )

    def add_spline_location(self, xyz: np.ndarray) -> None:
        """
        Add the location determined by fitting a 3-D spline to a small window around the
        maximum coalescence location and interpolating.

        Parameters
        ----------
        xyz:
            Geographical coordinates (lon/lat/depth) of best-fitting location.

        """

        self.locations["spline"] = dict(zip(XYZ, xyz))

    def add_picks(self, pick_df: pd.DataFrame, **kwargs: dict) -> None:
        """
        Add phase picks, and a selection of picker outputs and parameters.

        Parameters
        ----------
        pick_df:
            DataFrame that contains the measured picks with columns:
            ["Name", "Phase", "ModelledTime", "PickTime", "PickError", "SNR"]
            Each row contains the phase pick from one station/phase.

        **kwargs
            For :class:`~quakemigrate.plugins.pickers.gaussian.GaussianPicker`:\n
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

    def add_local_magnitude(self, mag: float, mag_err: float, mag_r2: float) -> None:
        """
        Add outputs from local magnitude calculation to the Event object.

        Parameters
        ----------
        mag:
            Network-averaged local magnitude estimate for the event.
        mag_err:
            (Weighted) standard deviation of the magnitude estimates from amplitude
            measurements on individual stations/channels.
        mag_r2:
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

    def in_marginal_window(self) -> bool:
        """
        Test if triggered event time is within marginal window around the maximum
        coalescence time (origin time).

        Returns
        -------
        cond:
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

    def mw_times(self, sampling_rate: float) -> np.ndarray[UTCDateTime]:
        """
        Utility function to generate timestamps for the time period around the trigger
        time for which the 4-D coalescence function is calculated in
        :func:`~quakemigrate.signal.scan.QuakeScan._compute`.

        Parameters
        ----------
        sampling_rate:
            Number of samples per second of the coalescence scan data.

        Returns
        -------
        times:
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

    def trim2window(self) -> None:
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

    def write(self, run: Run, lut: LUT) -> None:
        """
        Write event to a .event file.

        Parameters
        ----------
        run:
            Light class encapsulating i/o path information for a given run.
        lut:
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
                cols.extend(["COV_Err_XYZ"])
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

    def get_hypocentre(
        self, method: Literal["spline", "gaussian", "covariance"] = "spline"
    ) -> np.ndarray:
        """
        Get an estimate of the event hypocentre location.

        Parameters
        ----------
        method:
            Which location result to return.

        Returns
        -------
        ev_loc:
            [x_coordinate, y_coordinate, z_coordinate] of event hypocentre, in the
            global (geographic) coordinate system.

        """

        hypocentre = self.locations[method]

        ev_loc = np.array([hypocentre[k] for k in XYZ])

        return ev_loc

    hypocentre = property(get_hypocentre)

    def get_loc_uncertainty(
        self, method: Literal["gaussian", "covariance"] = "gaussian"
    ) -> np.ndarray:
        """
        Get an estimate of the hypocentre location uncertainty.

        Parameters
        ----------
        method:
            Which location result to return.

        Returns
        -------
        ev_loc_unc:
            [x_uncertainty, y_uncertainty, z_uncertainty] of event hypocentre; units are
            determined by the LUT projection units.

        """

        loc = self.locations[method]

        ev_loc_unc = np.array([loc[k] for k in ERR_XYZ])

        return ev_loc_unc

    loc_uncertainty = property(get_loc_uncertainty)

    @property
    def local_magnitude(self) -> tuple[float] | None:
        """Get the local magnitude, if it exists."""

        if len(self.localmag) == 0:
            return None
        else:
            return (value for _, value in self.localmag.items())

    @property
    def max_coalescence(self) -> dict:
        """Get information related to the maximum coalescence."""
        idxmax = self.coa_data["COA"].astype("float").idxmax()
        max_coa = self.coa_data.iloc[idxmax]
        keys = ["DT", "COA", "COA_NORM"]

        return dict(zip(keys, max_coa[keys].values))

    def _parse_triggered_event(self, event_data: pd.Series) -> dict:
        """
        Parse the information from a triggered event `pandas.Series` object into the
        Event object.

        Parameters
        ----------
        event_data:
            Contains information on the event output by the trigger stage.

        Returns
        -------
        trigger_info:
            Information about the triggered event.

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
