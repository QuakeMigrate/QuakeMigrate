"""
Base waveform client.

Provides an abstract base class implementing the common features of the suite of
waveform clients available in QuakeMigrate.

:copyright:
    2020–2026, QuakeMigrate developers.
:license:
    GNU General Public License, Version 3
    (https://www.gnu.org/licenses/gpl-3.0.html)

"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

import quakemigrate.util as util
from quakemigrate.exceptions import DataGap
from quakemigrate.io.data import WaveformData

if TYPE_CHECKING:
    from obspy import Stream, UTCDateTime
    from obspy.core.inventory import Inventory

    from quakemigrate.io.station import Station


@dataclass
class BaseWaveformClient(ABC):
    """
    Abstract base class for waveform data clients.

    This class defines the common interface and shared processing logic for all
    waveform clients used by QuakeMigrate, including local archive clients and remote
    service wrappers.

    Subclasses are responsible only for implementing :meth:`_fetch_stream`, which
    retrieves waveform data as an ObsPy :class:`~obspy.Stream` for a given station/time
    request. The public :meth:`read_waveform_data` method then applies the common
    QuakeMigrate post-processing steps and packages the result into a
    :class:`~quakemigrate.io.data.WaveformData` object.

    Parameters
    ----------
    inventory:
        ObsPy response inventory containing instrument response information for each
        channel of each station of each network.
    resample:
        If true, perform resampling of data which cannot be decimated directly to the
        desired sampling rate. See :func:`~quakemigrate.util.resample`
    upfactor:
        Factor by which to upsample the data to enable it to be decimated to the desired
        sampling rate, e.g., 40 Hz -> 50 Hz requires upfactor = 5.
        See :func:`~quakemigrate.util.resample`
    interpolate:
        If data is timestamped "off-sample" (i.e., a non-integer number of samples after
        midnight), whether to interpolate the data to apply the necessary correction.
        Default behaviour is to just alter the metadata, resulting in a sub-sample
        timing offset. See :func:`~quakemigrate.util.shift_to_sample`.
    response_removal_params:
        Optional dictionary of response-removal settings. Recognized keys are
        "water_level", "pre_filt", and "remove_full_response".
    water_level:
        Water level to use in instrument response removal.
    pre_filt:
        Pre-filter to apply during the instrument response removal, e.g.,
        (0.03, 0.05, 30., 35.) - all in Hz.
    remove_full_response:
        Whether to remove the full response (including the effect of digital FIR
        filters) or just the instrument transform function (as defined by the PolesZeros
        Response Stage). Significantly slower.

    """

    inventory: Inventory | None = None

    # Resampling parameters
    resample: bool = False
    upfactor: int | None = None
    interpolate: bool = False

    # Response removal parameters
    response_removal_params: dict | None = None
    water_level: float = 60.0
    pre_filt: tuple[float, float, float, float] | None = None
    remove_full_response: bool = False

    def __post_init__(self) -> None:
        """
        Initialise common response-removal settings.

        If an instrument response inventory is provided, response-removal parameters are
        loaded from response_removal_params where present. Otherwise the default values
        defined on the dataclass are retained.

        """

        if self.inventory:
            params = self.response_removal_params or {}
            if self.response_removal_params is None:
                print(
                    "Warning: 'water level' for instrument correction not specified. "
                    "Set to default: 60"
                )
            self.water_level = params.get("water_level", 60.0)
            self.pre_filt = _as_pre_filt(params.get("pre_filt"))
            self.remove_full_response = params.get("remove_full_response", False)

    @util.timeit("debug")
    def read_waveform_data(
        self,
        stations: list[Station],
        starttime: UTCDateTime,
        endtime: UTCDateTime,
        pre_pad: float = 0.0,
        post_pad: float = 0.0,
    ) -> WaveformData:
        """
        Fetch waveform data from a client.

        This is the public client API shared by all waveform client implementations.
        The method normalises padding, requests the raw stream from the subclass,
        applies common post-processing, and returns the result in a
        :class:`~quakemigrate.io.data.WaveformData` container.

        Parameters
        ----------
        stations:
            List of stations for which waveform data should be requested.
        starttime:
            First timestamp of data to be requested from the Client.
        endtime:
            Final timestamp of data to be requested from the Client.
        pre_pad:
            Optional time-padding, in seconds, to account for potential tapering.
        post_pad:
            Optional time-padding, in seconds, to account for potential tapering.

        Returns
        -------
        data:
            Waveform data read from the client that satisfies the query.

        Raises
        ------
        DataGapException
            Raised if no usable waveform data remain after post-processing.

        """

        logging.debug(f"\tRequesting waveform data for {starttime} to {endtime}")

        pre_pad = max(0.0, pre_pad)
        post_pad = max(0.0, post_pad)

        data = WaveformData(
            starttime=starttime,
            endtime=endtime,
            stations=stations,
            resample=self.resample,
            upfactor=self.upfactor,
            inventory=self.inventory,
            water_level=self.water_level,
            pre_filt=self.pre_filt,
            remove_full_response=self.remove_full_response,
            pre_pad=pre_pad,
            post_pad=post_pad,
        )

        read_start = starttime - pre_pad
        read_end = endtime + post_pad

        st = self._fetch_stream(stations, read_start, read_end)
        waveforms, raw_waveforms = self._postprocess_stream(
            st=st,
            stations=stations,
            starttime=starttime,
            endtime=endtime,
            pre_pad=pre_pad,
            post_pad=post_pad,
        )

        data.waveforms = waveforms
        data.raw_waveforms = raw_waveforms

        return data

    @abstractmethod
    def _fetch_stream(
        self,
        stations: list[Station],
        starttime: UTCDateTime,
        endtime: UTCDateTime,
    ) -> Stream:
        """
        Fetch waveform data for a station/time request.

        Subclasses must implement this method to retrieve waveform data from their
        underlying source, such as a local archive, an FDSN service, or another remote
        backend.

        Parameters
        ----------
        stations:
            List of stations for which waveform data should be loaded.
        starttime:
            First timestamp of data to be loaded.
        endtime:
            Final timestamp of data to be loaded.

        Returns
        -------
        st:
            Stream containing waveform data returned by the client backend.

        """

        raise NotImplementedError

    def _postprocess_stream(
        self,
        st: Stream,
        stations: list[Station],
        starttime: UTCDateTime,
        endtime: UTCDateTime,
        pre_pad: float,
        post_pad: float,
    ) -> tuple[Stream, Stream]:
        """
        Apply common post-processing to a waveform stream.

        The processing steps performed are:

        - populate default network and location codes when none were supplied
        - merge traces channel-by-channel
        - copy the merged stream as raw_waveforms
        - shift traces onto the sample grid if required
        - remove stations marked as read-only
        - trim padded data back to the requested time window
        - verify that usable waveform data remain

        Parameters
        ----------
        st:
            Input stream of waveform data.
        stations:
            List of requested stations.
        starttime:
            First timestamp of data to be requested from the Client.
        endtime:
            Final timestamp of data to be requested from the Client.
        pre_pad:
            Optional time-padding, in seconds, to account for potential tapering.
        post_pad:
            Optional time-padding, in seconds, to account for potential tapering.

        Returns
        -------
        waveforms:
            Post-processed waveform stream.
        raw_waveforms:
            Copy of the merged waveform stream before timing correction and trimming.

        Raises
        ------
        DataGapException
            Raised if the stream is empty after post-processing.

        """

        if not any(station.network for station in stations):
            for tr in st:
                tr.stats.network = "XX"

        if not any(station.location for station in stations):
            for tr in st:
                tr.stats.location = ""

        # Merge waveforms channel-by-channel with no-clobber merge
        st = util.merge_stream(st)

        # Make copy of raw waveforms to output if requested
        raw_waveforms = st.copy()

        # Ensure data is timestamped "on-sample" (i.e., an integer number of samples
        # after midnight). Otherwise the data will be implicitly shifted when it is used
        # to calculate the onset function / migrated.
        st = util.shift_to_sample(st, interpolate=self.interpolate)

        if any(station.read_only for station in stations):
            st_selected = Stream()
            for station in stations:
                if not station.read_only:
                    st_selected += st.select(station=station.station)
            st = st_selected.copy()

        if pre_pad != 0.0 or post_pad != 0.0:
            # Trim data between start and end time
            for tr in list(st):
                tr.trim(starttime=starttime, endtime=endtime, nearest_sample=True)
                if not bool(tr):
                    st.remove(tr)

        # Test if the stream is completely empty
        # (see __nonzero__ for `obspy.Stream` object)
        if not bool(st):
            raise DataGap()

        return st, raw_waveforms

    def __str__(self) -> str:
        """Return a human-readable summary of the waveform client."""

        out = [f"QuakeMigrate {self.__class__.__name__} object"]
        out.extend(self._client_description())
        out.append(f"\tResampling\t:\t{self.resample}")
        if self.upfactor:
            out.append(f"\tUpfactor\t:\t{self.upfactor}")
        out.append(self._response_summary().rstrip())
        return "\n".join(out)

    def _client_description(self) -> list[str]:
        """Subclass hook for describing client-specific configuration."""

        return []

    def _response_summary(self) -> str:
        """Return a summary of the instrument response configuration."""

        if self.inventory:
            out = (
                "\tResponse removal parameters:\n"
                f"\t\tWater level  = {self.water_level}\n"
            )
            if self.pre_filt is not None:
                out += f"\t\tPre-filter   = {self.pre_filt} Hz\n"
            out += (
                "\t\tRemove full response (inc. FIR stages) = "
                f"{self.remove_full_response}\n"
            )
        else:
            out = "\tNo instrument response inventory provided!\n"

        return out


def _as_pre_filt(value: list | tuple) -> tuple[float, float, float, float] | None:
    """
    Convert a response-removal pre-filter specification to a four-value tuple.

    Parameters
    ----------
    value:
        Pre-filter specification. Expected format is [f1, f2, f3, f4] with frequencies
        in Hz. If None, no pre-filter is applied.

    Returns
    -------
    pre_filt:
        Four frequency corners as (f1, f2, f3, f4), or None if no pre-filter was
        specified.

    Raises
    ------
    TypeError
        If value is not a list or tuple, or if any entry cannot be converted to a float.
    ValueError
        If value does not contain exactly four entries.

    """

    if value is None:
        return None

    if not isinstance(value, (list, tuple)):
        raise TypeError("pre_filt must be an array of four numbers.")

    if len(value) != 4:
        raise ValueError("pre_filt must contain exactly four values.")

    try:
        a, b, c, d = (float(x) for x in value)
    except (TypeError, ValueError) as e:
        raise TypeError("pre_filt must contain only numbers.") from e

    return a, b, c, d
