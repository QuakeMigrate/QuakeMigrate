"""
Base classes for QuakeMigrate onset plugins.

This module defines the common interface that all onset implementations must follow, and
lightweight container for data generated during onset calculation.

:copyright:
    2020–2026, QuakeMigrate developers.
:license:
    GNU General Public License, Version 3
    (https://www.gnu.org/licenses/gpl-3.0.html)

"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

import quakemigrate.util as util


if TYPE_CHECKING:
    from obspy import Stream, UTCDateTime

    from quakemigrate.io.data import WaveformData


class Onset(ABC):
    """
    Abstract base class for QuakeMigrate onset plugins.

    An onset plugin converts waveform data into phase-specific onset functions used by
    migration and picking. Implementations may use classical signal processing,
    machine-learning models, or other characteristic functions, but must return data
    using QuakeMigrate's standard OnsetData contract.

    Subclasses are responsible for:
    - validating plugin-specific configuration,
    - checking or respecting waveform availability requirements,
    - returning one onset function per available station/phase pair,
    - using station IDs consistently in OnsetData.onsets,
    - defining pre_pad and post_pad for waveform read windows,
    - optionally implementing gaussian_halfwidth() if GaussianPicker is used.

    Parameters
    ----------
    sampling_rate:
        Sampling rate, in Hz, at which onset functions are calculated and returned.

    Attributes
    ----------
    sampling_rate:
        Desired sampling rate for input data; sampling rate at which the onset functions
        will be computed.
    pre_pad:
        Option to override the default pre-pad duration of data to read before computing
        4-D coalescence in detect() and locate().
    post_pad:
        Option to override the default post-pad duration of data to read before
        computing 4-D coalescence in detect() and locate().

    """

    def __init__(self, sampling_rate: int) -> None:
        """Instantiate the Onset object."""

        self.sampling_rate = sampling_rate

        self._pre_pad: float = 0.0
        self._post_pad: float = 0.0

    def __str__(self) -> str:
        """Return a short summary string describing the onset plugin."""

        return f"{self.__class__.__name__}(sampling_rate={self.sampling_rate})"

    def pad(self, timespan: float) -> tuple[float, float]:
        """
        Determine the number of samples needed to pre- and post-pad the timespan.

        Parameters
        ----------
        timespan:
            The time window to pad.

        Returns
        -------
        pre_pad:
            Duration, in seconds, to read before the requested time window.
        post_pad:
            Duration, in seconds, to read after the requested time window.

        """

        # Add additional padding for any tapering applied to data
        timespan += self.pre_pad + self.post_pad
        pre_pad = util.trim2sample(
            self.pre_pad + np.ceil(timespan * 0.06), self.sampling_rate
        )
        post_pad = util.trim2sample(
            self.post_pad + np.ceil(timespan * 0.06), self.sampling_rate
        )

        return pre_pad, post_pad

    def gaussian_halfwidth(self, phase: str) -> float:
        """
        Return a phase-specific Gaussian half-width estimate in samples.

        Plugins that support GaussianPicker should override this method.

        Parameters
        ----------
        phase:
            Phase for which to return the Gaussian half-width estimate.

        Returns
        -------
        halfwidth:
            Gaussian half-width estimate, in samples.

        Raises
        ------
        AttributeError
            If the onset plugin does not provide a Gaussian half-width
            estimate.

        """

        raise AttributeError(
            "In order to use the GaussianPicker module with a custom Onset, you need "
            "to provide a 'gaussian_halfwidth' method."
        )

    @abstractmethod
    def calculate_onsets(
        self,
        data: WaveformData,
        timespan: float | None = None,
    ) -> tuple[np.ndarray, OnsetData]:
        """
        Calculate onset functions for the requested stations and phases.

        Parameters
        ----------
        data:
            Waveform data returned by an archive query.
        timespan:
            If the timespan for which the onsets are being generated is provided, this
            will be used to calculate the tapered window of data at the start and end of
            the onset function which should be disregarded. This is necessary to
            accurately set the pick threshold in GaussianPicker, for example.

        Returns
        -------
        onsets:
            Stacked onset functions served up for migration, shape(nonsets, nsamples).
        onset_data:
            Light class encapsulating data generated during onset calculation.

        Raises
        ------
        AllDataRejected
            If no data passes the specified criteria.

        """

        ...

    @property
    @abstractmethod
    def pre_pad(self) -> float:
        """
        Duration in seconds to read before the requested time window.

        This should include any algorithm-specific context required before the first
        requested onset sample, such as STA/LTA history or model context.

        """

        return self._pre_pad

    @pre_pad.setter
    @abstractmethod
    def pre_pad(self, value: float) -> None:
        """Set the algorithm-specific pre-pad duration, in seconds."""

        self._pre_pad = value

    @property
    @abstractmethod
    def post_pad(self) -> float:
        """
        Duration in seconds to read after the requested time window.

        This should include any algorithm-specific context required after the final
        requested onset sample, such as travel-time padding, LTA windows, or model
        context.

        """

        return self._post_pad

    @post_pad.setter
    @abstractmethod
    def post_pad(self, value: float) -> None:
        """Set the algorithm-specific post-pad duration, in seconds."""

        self._post_pad = value


@dataclass
class OnsetData:
    """
    The OnsetData class encapsulates the onset functions calculated by transforming
    seismic data using the chosen onset detection algorithm (characteristic function).

    This includes a dictionary describing which onset functions are available for each
    station and phase, and the intermediary filtered or otherwise pre-processed waveform
    data used to calculate the onset function.

    Attributes
    ----------
    onsets:
        Keys "station", each of which contains keys for each phase, e.g., "P" and "S".
        {"station": {"P": `p_onset`, "S": `s_onset`}}. Onset functions are calculated by
        transforming the raw seismic data using some characteristic function designed to
        highlight phase arrivals.
    phases:
        Phases for which onsets have been calculated. (e.g., ["P", "S"])
    channel_maps:
        Data component maps - keys are phases. (e.g., {"P": "Z"})
    filtered_waveforms:
        Filtered and/or resampled and otherwise processed seismic data generated during
        onset function generation. Only contains waveforms that have passed the quality
        control criteria, at a unified sampling rate - see `sampling_rate`.
    availability:
        Dictionary with keys "station_phase", containing 1's or 0's corresponding to
        whether an onset function is available for that station and phase - determined
        by data availability and quality checks.
    starttime:
        Start time of onset functions.
    endtime:
        End time of onset functions.
    sampling_rate:
        Sampling rate of filtered waveforms and onset functions.

    """

    onsets: dict
    phases: list[str]
    channel_maps: dict
    filtered_waveforms: Stream
    availability: dict
    starttime: UTCDateTime
    endtime: UTCDateTime
    sampling_rate: int
