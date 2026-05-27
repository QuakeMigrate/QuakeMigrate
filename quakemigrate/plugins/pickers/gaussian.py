"""
The default seismic phase picking class - fits a 1-D Gaussian to the calculated onset
functions.

:copyright:
    2020–2026, QuakeMigrate developers.
:license:
    GNU General Public License, Version 3
    (https://www.gnu.org/licenses/gpl-3.0.html)

"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal, TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

import quakemigrate.util as util
from quakemigrate.exceptions import NoOnsetPeak
from quakemigrate.plot.phase_picks import pick_summary
from .base import PhasePicker


if TYPE_CHECKING:
    from obspy import UTCDateTime

    from quakemigrate.io.core import Run
    from quakemigrate.io.event import Event
    from quakemigrate.lut import LUT
    from quakemigrate.plugins.onsets import Onset, OnsetData


def build_gaussian_picker(
    onset: Onset,
    threshold_method: Literal["MAD", "percentile"] = "MAD",
    mad_multiplier: float = 8.0,
    percentile: float = 1.0,
    plot_picks: bool = False,
    write_seed_ids: bool = False,
) -> GaussianPicker:
    """
    Build a :class:`GaussianPicker` from configuration values.

    Parameters
    ----------
    onset:
        Onset plugin used to calculate onset functions for picking.
    threshold_method:
        Thresholding method used to identify candidate picks. Supported values are
        ``"MAD"`` and ``"percentile"``.
    mad_multiplier:
        Scaling factor applied to the Median Absolute Deviation when
        ``threshold_method="MAD"``.
    percentile:
        Fraction in the range [0, 1] used to define the percentile threshold when
        ``threshold_method="percentile"``. For example, ``0.99`` selects the 99th
        percentile.
    plot_picks:
        Whether to generate phase-pick summary plots.
    write_seed_ids:
        Whether to write the SEED IDs of traces contributing to each phase pick.

    Returns
    -------
    picker:
        Configured Gaussian phase picker.

    Raises
    ------
    ValueError
        If ``threshold_method`` is not one of ``"MAD"`` or ``"percentile"``.

    """

    if threshold_method == "MAD":
        threshold = MADThreshold(multiplier=mad_multiplier)
    elif threshold_method == "percentile":
        threshold = PercentileThreshold(percentile=percentile)
    else:
        raise ValueError(
            f"Invalid threshold_method '{threshold_method}'. "
            "Supported: 'MAD', 'percentile'."
        )

    return GaussianPicker(
        onset=onset,
        threshold=threshold,
        plot_picks=plot_picks,
        write_seed_ids=write_seed_ids,
    )


@dataclass
class MADThreshold:
    """
    Median Absolute Deviation (MAD)-based threshold for onset picking.

    Threshold = median + multiplier * MAD

    where the MAD is computed from the onset data outside the pick windows.

    Attributes
    ----------
    multiplier:
        Scaling factor applied to the MAD to define the detection threshold.

    """

    multiplier: float = 8.0

    def __post_init__(self) -> None:
        if self.multiplier < 0:
            raise ValueError("MAD threshold multiplier must be non-negative.")


@dataclass
class PercentileThreshold:
    """
    Percentile-based threshold for onset picking.

    Threshold = percentile(onset_noise)

    where the percentile is computed from onset data outside the pick windows.

    Attributes
    ----------
    percentile:
        Fraction in range [0, 1] specifying the percentile of the noise distribution to
        use (e.g., 0.99 = 99th percentile).

    """

    percentile: float = 1.0

    def __post_init__(self) -> None:
        if not 0 <= self.percentile <= 1:
            raise ValueError("percentile threshold must be in the range [0, 1].")


PickThreshold = MADThreshold | PercentileThreshold


class GaussianPicker(PhasePicker):
    """
    Automatic phase picking based on the fitting of a 1-D Gaussian function to an onset
    function.

    Attributes
    ----------
    threshold:
        Which method to use to calculate the pick threshold; a percentile of the data
        outside the pick windows (e.g., 0.99 = 99th percentile) or a multiple of the
        Median Absolute Deviation of the signal outside the pick windows.
    plot_picks:
        Toggle plotting of phase picks.
    write_seed_ids:
        Toggle writing the SEED id's of the traces that have contributed to a given phase
        pick within the .picks file.

    Raises
    ------
    ValueError
        If an invalid pick threshold method is selected.

    """

    stage: str = "locate_event"
    order: int = 250
    name: str = "GaussianPicker"

    DEFAULT_GAUSSIAN_FIT = {"popt": 0, "xdata": 0, "xdata_dt": 0, "PickValue": -1}

    def __init__(
        self,
        onset: Onset,
        *,
        threshold: PickThreshold | None = None,
        plot_picks: bool = False,
        write_seed_ids: bool = False,
    ) -> None:
        """Instantiate the GaussianPicker object."""
        super().__init__(plot_picks=plot_picks)

        self.onset = onset
        if threshold is not None and not isinstance(threshold, PickThreshold):
            raise ValueError(
                f"Invalid pick threshold method. "
                "Supported methods are: 'percentile', 'MAD'."
            )
        self.threshold = MADThreshold() if threshold is None else threshold

        self.write_seed_ids = write_seed_ids

    def __str__(self) -> str:
        """Returns a short summary string of the GaussianPicker."""

        str_ = "\tPhase picking by fitting a 1-D Gaussian to onsets\n"
        if isinstance(self.threshold, PercentileThreshold):
            str_ += f"\t\tPercentile threshold  = {self.threshold.percentile}\n"
        elif isinstance(self.threshold, MADThreshold):
            str_ += f"\t\tMAD multiplier  = {self.threshold.multiplier}\n"

        return str_

    @util.timeit("info")
    def run(self, event: Event, lut: LUT, run: Run) -> Event:
        """
        Picks phase arrival times for located events.

        Parameters
        ----------
        event:
            Light class encapsulating waveforms, coalescence information and location
            information for a given event.
        lut:
            Contains the traveltime lookup tables for seismic phases, computed for some
            pre-defined velocity model.
        run:
            Light class encapsulating i/o path information for a given run.

        Returns
        -------
        event:
            Event object provided to pick_phases(), but now with phase picks!

        """

        logging.info("\tMaking phase picks...")

        # Onsets are recalculated without logging
        _, onset_data = self.onset.calculate_onsets(
            event.data, timespan=4 * event.marginal_window
        )

        e_ijk = lut.index2coord(event.hypocentre, inverse=True)[0]

        # Pre-define pick DataFrame and fit params and pick windows dicts
        p_idx = np.arange(sum([len(v) for _, v in onset_data.onsets.items()]))
        columns = [
            "Station",
            "Phase",
            "ModelledTime",
            "PickTime",
            "PickError",
            "SNR",
            "Residual",
        ]
        if self.write_seed_ids:
            columns = [columns[0], "SEED_ids", *columns[1:]]
        picks = pd.DataFrame(index=p_idx, columns=columns)
        gaussfits = {}
        pick_windows = {}
        idx = 0

        for station, onsets in onset_data.onsets.items():
            for phase, onset in onsets.items():
                traveltime = lut.traveltime_to(phase, e_ijk, station)[0]
                pick_windows.setdefault(station, {}).update(
                    {
                        phase: self._determine_window(
                            event, onset_data, traveltime, lut.fraction_tt
                        )
                    }
                )
                n_samples = len(onset)

            self._distinguish_windows(
                pick_windows[station], list(onsets.keys()), n_samples
            )

            for phase, onset in onsets.items():
                # Find threshold from 'noise' part of onset
                pick_threshold = self._find_pick_threshold(onset, pick_windows[station])

                logging.debug(f"\t\tPicking {phase} at {station}...")
                fit, *pick = self._fit_gaussian(
                    onset,
                    onset_data.sampling_rate,
                    self.onset.gaussian_halfwidth(phase),
                    onset_data.starttime,
                    pick_threshold,
                    pick_windows[station][phase],
                )

                gaussfits.setdefault(station, {}).update({phase: fit})

                traveltime = lut.traveltime_to(phase, e_ijk, station)[0]
                model_time = event.otime + traveltime
                if pick[0] == -1:
                    residual = -1
                else:
                    residual = pick[0] - model_time

                if self.write_seed_ids:
                    stream = onset_data.filtered_waveforms.select(
                        station=station,
                        channel=self.onset.channel_maps[phase],
                    )
                    seed_ids = sorted(set([tr.id for tr in stream]))
                    picks.iloc[idx] = [
                        station,
                        seed_ids,
                        phase,
                        model_time,
                        *pick,
                        residual,
                    ]
                else:
                    picks.iloc[idx] = [station, phase, model_time, *pick, residual]
                idx += 1

        event.add_picks(picks, gaussfits=gaussfits, pick_windows=pick_windows)

        self.write(run, event.uid, picks)

        if self.plot_picks:
            logging.info("\t\tPlotting picks...")
            for station, onsets in onset_data.onsets.items():
                traveltimes = [
                    lut.traveltime_to(phase, e_ijk, station)[0]
                    for phase in onsets.keys()
                ]
                self.plot(event, station, onset_data, picks, traveltimes, run)

        return event

    def _determine_window(
        self, event: Event, onset_data: OnsetData, tt: float, fraction_tt: float
    ) -> list[int]:
        """
        Determine phase pick window upper and lower bounds based on the event marginal
        window and a set percentage of the phase travel time.

        Parameters
        ----------
        event:
            Light class to encapsulate information about an event, including origin
            time, location and waveform data.
        onset_data:
            Light class encapsulating data generated during onset calculation.
        tt:
            Traveltime for the requested phase.
        fraction_tt:
            Defines width of time window around expected phase arrival time in which to
            search for a phase pick as a function of the traveltime from the event
            location to that station -- should be an estimate of the uncertainty in the
            velocity model.

        Returns
        -------
        lower_idx:
            Index of lower bound for the phase pick window.
        arrival_idx:
            Index of the modelled phase arrival time.
        upper_idx:
            Index of upper bound for the phase pick window.

        """

        arrival_idx = util.time2sample(
            event.otime + tt - onset_data.starttime, onset_data.sampling_rate
        )

        # Add length of marginal window to this and convert to index
        samples = util.time2sample(
            tt * fraction_tt + event.marginal_window, onset_data.sampling_rate
        )

        return [arrival_idx - samples, arrival_idx, arrival_idx + samples]

    def _distinguish_windows(
        self, windows: dict, phases: list[str], samples: int
    ) -> None:
        """
        Ensure pick windows do not overlap - if they do, set the upper bound of window
        one and the lower bound of window two to be the midpoint index of the two
        modelled phase arrival times.

        Parameters
        ----------
        windows:
            Dictionary of windows with phases as keys.
        phases:
            Phases being migrated.
        samples:
            Total number of samples in the onset function.

        """

        # Handle first key
        first_idx = windows[phases[0]][0]
        windows[phases[0]][0] = 0 if first_idx < 0 else first_idx

        # Handle keys pairwise
        for p1, p2 in util.pairwise(phases):
            p1_window, p2_window = windows[p1], windows[p2]
            mid_idx = int((p1_window[1] + p2_window[1]) / 2)
            windows[p1][2] = min(mid_idx, p1_window[2])
            windows[p2][0] = max(mid_idx, p2_window[0])

        # Handle last key
        last_idx = windows[phases[-1]][2]
        windows[phases[-1]][2] = samples if last_idx > samples else last_idx

    def _find_pick_threshold(
        self,
        onset: np.ndarray[np.double],
        windows: dict,
    ) -> float:
        """
        Determine a pick threshold from the onset data outside the pick windows.

        Parameters
        ----------
        onset:
            Onset (characteristic) function.
        windows:
            Indexes of the lower window bound, the phase arrival, and the upper window
            bound.

        Return
        ------
        pick_threshold:
            The threshold calculated from the onset data outside the pick windows.

        """

        onset_noise = onset.copy()
        for _, window in windows.items():
            onset_noise[window[0] : window[2]] = -1
        # Remove data during pick windows, and data set to 1 (in onset function taper
        # pad windows)
        onset_noise = onset_noise[onset_noise > 1]

        if isinstance(self.threshold, PercentileThreshold):
            pick_threshold = np.percentile(onset_noise, self.threshold.percentile * 100)
        elif isinstance(self.threshold, MADThreshold):
            med = np.median(onset_noise)
            mad = util.calculate_mad(onset_noise)
            pick_threshold = med + (mad * self.threshold.multiplier)

        return pick_threshold

    def _fit_gaussian(
        self,
        onset: np.ndarray[np.double],
        sampling_rate: int,
        halfwidth: float,
        starttime: UTCDateTime,
        pick_threshold: float,
        window: list[int],
    ) -> tuple[dict, float, float, UTCDateTime]:
        """
        Fit a Gaussian to the onset function in order to make a time pick with an
        associated uncertainty.

        Uses the amplitude and timing of the onset function peak and some knowledge of
        the onset function parameters (e.g., short-term average window length, for the
        :class:`~quakemigrate.plugins.onsets.stalta.STALTAOnset`) to make an initial
        estimate of a gaussian fit to the onset function.

        Parameters
        ----------
        onset:
            Onset function.
        sampling_rate:
            Sampling rate of the onset function.
        halfwidth:
            Initial estimate for the Gaussian half-width based on some function of the
            onset function parameters.
        starttime:
            Timestamp for first sample of the onset function.
        pick_threshold:
            Value above which to threshold data based on noise.
        window:
            Indices for the window start, modelled phase arrival, and window end.

        Returns
        -------
        gaussian_fit:
            Gaussian fit parameters: {"popt": popt,
                                      "xdata": x_data,
                                      "xdata_dt": x_data_dt,
                                      "PickValue": max_onset,
                                      "PickThreshold": pick_threshold}
        max_onset:
            Amplitude of Gaussian fit to onset function, i.e., the SNR.
        sigma:
            Sigma of Gaussian fit to onset function, i.e., the pick uncertainty.
        mean:
            Mean of Gaussian fit to onset function, i.e., the pick time.

        """

        # Trim the onset function in the pick window
        onset_signal = onset[window[0] : window[2]]
        logging.debug(f"\t\t    win_min: {window[0]}, win_max: {window[2]}")

        # Identify the peak in the windowed onset that exceeds this threshold
        # AND contains the maximum value in the window (i.e. the 'true' peak).
        try:
            peak_idxs = self._find_peak(onset_signal, pick_threshold)
            # add an extra sample either side for the curve fitting. This makes the
            # fitting more stable, and guarantees at least 3 samples --> avoids an
            # under-constrained optimisation (3 fitting params).
            padded_peak_idxs = [peak_idxs[0] - 1, peak_idxs[1] + 1]
            padded_peak_idxs = [window[0] + p for p in padded_peak_idxs]
            logging.debug(
                f"\t\t    padded_peak_idxmin: {padded_peak_idxs[0]},"
                f" padded_peak_idxmax: {padded_peak_idxs[1]}"
            )
            x_data = np.arange(*padded_peak_idxs) / sampling_rate
            y_data = onset[padded_peak_idxs[0] : padded_peak_idxs[1]]
        except NoOnsetPeak as e:
            logging.debug(e.msg)
            return self._pick_failure(pick_threshold)

        # Try to fit a 1-D Gaussian
        # Initial parameters (p0) are:
        #   height = max value of onset function
        #   mean   = time of max value
        #   sigma  = `halfwidth` - determined from onset function parameters
        p0 = [
            max(y_data),
            (padded_peak_idxs[0] + np.argmax(y_data)) / sampling_rate,
            halfwidth / sampling_rate,
        ]
        try:
            popt, _ = curve_fit(util.gaussian_1d, x_data, y_data, p0)
        except (ValueError, RuntimeError) as e:
            # curve_fit can fail for a number of reasons - primarily if the input data
            # contains nans or if the least-squares minimisation fails. A warning may
            # also be emitted to stdout if the covariance of the parameters could not
            # be estimated - this is suppressed by default in scan.py.
            logging.debug(f"\t\t    Failed curve_fit:\n{e}\n\t\t    Continuing...")
            return self._pick_failure(pick_threshold)
        except TypeError as e:
            logging.debug(
                "\t\t    Failed curve_fit - too few input data?"
                f"{e}\n\t\t    Continuing..."
            )
            return self._pick_failure(pick_threshold)

        # Unpack results:
        #  popt = [height, mean (seconds), sigma (seconds)]
        max_onset = popt[0]
        mean = starttime + float(popt[1])
        sigma = np.absolute(popt[2])

        # Check pick mean is within the pick window.
        if not window[0] < popt[1] * sampling_rate < window[2]:
            logging.debug("\t\t    Pick mean out of bounds - continuing.")
            return self._pick_failure(pick_threshold)

        gaussian_fit = {
            "popt": popt,
            "xdata": x_data,
            "xdata_dt": np.array([starttime + x for x in x_data]),
            "PickValue": max_onset,
            "PickThreshold": pick_threshold,
        }

        return gaussian_fit, mean, sigma, max_onset

    def _pick_failure(self, pick_threshold: float) -> tuple[dict, int, int, int]:
        """
        Short utility function to produce the default values when a pick cannot be made.

        Parameters
        ----------
        pick_threshold:
            Pick threshold value for onset data.

        Returns
        -------
        gaussian_fit:
            The default Gaussian fit dictionary, with relevant pick threshold value.
        mean:
            A default of -1 value to indicate failure.
        sigma:
            A default of -1 value to indicate failure.
        max_onset:
            A default of -1 value to indicate failure.

        """

        gaussian_fit = self.DEFAULT_GAUSSIAN_FIT.copy()
        gaussian_fit["PickThreshold"] = pick_threshold
        mean = sigma = max_onset = -1

        return gaussian_fit, mean, sigma, max_onset

    def _find_peak(
        self, windowed_onset: np.ndarray[np.double], pick_threshold: float
    ) -> tuple[int, int]:
        """
        Identify peaks, if any, within the windowed onset that exceed the specified
        threshold value. Of those peaks, this function seeks the one that contains the
        maximum value within the window, i.e. the 'true' peak - see the diagram below.

                                             v
                                             *
                                            * *
                                    *      *   *
                         |         * *    *     *     |
                         |---------------#-------#----|
                         |        *    **         *   |

        Parameters
        ----------
        windowed_onset:
            The onset function within the picking window.
        pick_threshold:
            Value above which to search for peaks in the onset data.

        Returns
        -------
        true_peak_idx:
            Start and end index values for the 'true' peak, with +1 added to the last
            index so that all of the values above the threshold are returned when
            slicing by index.

        Raises
        ------
        NoOnsetPeak
            If no onset data, or only a single sample, exceeds the pick threshold.

        """

        exceedence = np.where(windowed_onset > pick_threshold)[0]
        if len(exceedence) == 0:
            raise NoOnsetPeak(pick_threshold)

        # Identify all peaks - there are possibly multiple distinct periods of data that
        # exceed the threshold. The following command simply seeks non-consecutive index
        # values in the array of points that exceed the threshold and splits the array
        # at these points into 'peaks'.
        peaks = np.split(exceedence, np.where(np.diff(exceedence) != 1)[0] + 1)

        # Identify the peak that contains the true peak (maximum)
        true_maximum = np.argmax(windowed_onset)
        for i, peak in enumerate(peaks):
            if np.any(peak == true_maximum):
                break

        # Check if there is more than a single sample above the threshold
        if len(peaks[i]) < 2:
            raise NoOnsetPeak(pick_threshold)

        # Grab the peak and return the start/end index values. NOTE: + 1 is required so
        # that the last sample is included when slicing by index
        true_peak_idxs = [peaks[i][0], peaks[i][-1] + 1]

        return true_peak_idxs

    @util.timeit()
    def plot(
        self,
        event: Event,
        station: str,
        onset_data: OnsetData,
        picks_df: pd.DataFrame,
        traveltimes: list[float],
        run: Run,
    ) -> None:
        """
        Plot figure showing the filtered traces for each data component and the onset
        functions calculated from them (P and/or S) for each station. The search window
        to make a phase pick is displayed, along with the dynamic pick threshold, the
        phase pick time and its uncertainty (if made) and the Gaussian fit to the onset
        function.

        Parameters
        ----------
        event:
            Light class to encapsulate information about an event, including origin
            time, location and waveform data.
        station:
            Station name.
        onset_data:
            Light class encapsulating data generated during onset calculation.
        picks_df:
            DataFrame that contains the measured picks with columns:
            ["Name", "Phase", "ModelledTime", "PickTime", "PickError", "SNR"]
            Each row contains the phase pick from one station/phase.
        traveltimes:
            Modelled traveltimes from the event hypocentre to the station for each phase
            to be plotted.
        run:
            Light class encapsulating i/o path information for a given run.

        """

        fpath = run.path / f"locate/{run.subname}/pick_plots/{event.uid}"
        fpath.mkdir(exist_ok=True, parents=True)

        onsets = onset_data.onsets[station]
        channel_maps = onset_data.channel_maps
        waveforms = onset_data.filtered_waveforms.select(station=station)
        # Check if any data available to plot
        if not bool(waveforms):
            return
        picks = picks_df[picks_df["Station"] == station].reset_index(drop=True)
        windows = event.picks["pick_windows"][station]

        # Call subroutine to plot phase pick figure
        fig = pick_summary(
            event, station, waveforms, picks, onsets, channel_maps, traveltimes, windows
        )

        fstem = f"{event.uid}_{station}"
        file = (fpath / fstem).with_suffix(".pdf")
        fig.savefig(file)
        plt.close(fig)
