"""
Module for processing waveform files stored in a data archive.

:copyright:
    2020–2026, QuakeMigrate developers.
:license:
    GNU General Public License, Version 3
    (https://www.gnu.org/licenses/gpl-3.0.html)

"""

from __future__ import annotations

import logging
import pathlib
import string
from dataclasses import dataclass, field
from itertools import chain
from typing import TYPE_CHECKING

from obspy import read, Stream, UTCDateTime

import quakemigrate.util as util
from quakemigrate.exceptions import (
    ArchiveEmpty,
    DataGap,
    InvalidArchivePathStructure,
    ResponseNotFoundError,
    ResponseRemovalError,
)


if TYPE_CHECKING:
    from collections.abc import Iterator

    from obspy import Trace
    from obspy.core.inventory import Inventory

    from quakemigrate.io.station import Station


ARCHIVE_FORMATS = {
    "SeisComp3": (
        "{year}/{network}/{station}/{channels}.D/"
        "{network}.{station}.{location}.{channels}.D.{year}.{jday:03d}"
    ),
    "YEAR/JD/*_STATION_*": "{year}/{jday:03d}/*_{station}_*",
    "YEAR/JD/STATION": "{year}/{jday:03d}/{station}*",
    "STATION.YEAR.JULIANDAY": "*{station}.*.{year}.{jday:03d}",
    "/STATION/STATION.YearMonthDay": "{station}/{station}.{year}{month:02d}{day:02d}",
    "YEAR_JD/STATION*": "{year}_{jday:03d}/{station}*",
    "YEAR_JD/STATION_*": "{year}_{jday:03d}/{station}_*",
}

_ALLOWED_FIELDS = {
    "year",
    "month",
    "day",
    "jday",
    "network",
    "station",
    "location",
    "channels",
    "dtime",
}


def _root_field(field_name: str) -> str:
    """
    Return the root field for a format field name.

    Parameters
    ----------
    field_name:
        Format field name extracted from a Python format string.

    Returns
    -------
    root_field:
        The root field, e.g., `dtime` from `dtime.year`.

    """

    i = field_name.find(".")
    if i != -1:
        return field_name[:i]

    return field_name


def _validate_path_format(archive_format: str) -> str:
    """
    Validate the path format, ensuring it does not contain any placeholders that are
    not supported by the `_load_from_path` method.

    Parameters
    ----------
    archive_format:
        The template archive path string to be formatted during data queries.

    Returns
    -------
    archive_format:
        The template archive path string to be formatted during data queries.

    Raises
    ------
    InvalidArchivePathStructure
        Raised if there is a placeholder that is not supported.

    """

    # Deprecation handling
    if archive_format in ARCHIVE_FORMATS:
        print(
            "This method of setting the archive format has been deprecated.\n"
            f"Mapping {archive_format} -> {ARCHIVE_FORMATS[archive_format]}"
        )
        archive_format = ARCHIVE_FORMATS[archive_format]

    allowed = set(_ALLOWED_FIELDS)

    for _, field_name, _, _ in string.Formatter().parse(archive_format):
        if field_name is None:
            continue

        if field_name == "":
            raise InvalidArchivePathStructure(archive_format)

        root = _root_field(field_name)
        if root not in allowed:
            raise InvalidArchivePathStructure(root)

    return archive_format


@dataclass
class Archive:
    """
    The Archive class handles the reading of archived waveform data.

    It is capable of handling any regular archive structure. Requests to read waveform
    data are served up as a :class:`~quakemigrate.io.data.WaveformData` object.

    If provided, a response inventory for the archive will be stored with the waveform
    data for response removal, if needed (e.g., for local magnitude calculation, or to
    output real cut waveforms).

    By default, data with mismatched sampling rates will only be decimated. If
    necessary, and if the user specifies `resample = True` and an upfactor to upsample
    by `upfactor = int` for the waveform archive, data can also be upsampled and then,
    if necessary, subsequently decimated to achieve the desired sampling rate.

    For example, for raw input data sampled at a mix of 40, 50 and 100 Hz, to achieve a
    unified sampling rate of 50 Hz, the user would have to specify an upfactor of 5;
    40 Hz x 5 = 200 Hz, which can then be decimated to 50 Hz - see
    :func:`~quakemigrate.util.resample`.

    Parameters
    ----------
    archive_path:
        Location of seismic data archive: e.g.: "./DATA_ARCHIVE".
    archive_format:
        Sets directory structure and file naming format for different archive formats.
        See :func:`~quakemigrate.io.data.Archive.path_structure`
    kwargs:
        See Archive Attributes for details.

    Attributes
    ----------
    path:
        Location of seismic data archive: e.g.: ./DATA_ARCHIVE.
    format:
        Directory structure and file naming format of data archive.
    resample:
        If true, perform resampling of data which cannot be decimated directly to the
        desired sampling rate. See :func:`~quakemigrate.util.resample`
    inventory:
        ObsPy response inventory for this waveform archive, containing response
        information for each channel of each station of each network.
    pre_filt:
        Pre-filter to apply during the instrument response removal, e.g.,
        (0.03, 0.05, 30., 35.) - all in Hz.
    water_level:
        Water level to use in instrument response removal.
    remove_full_response:
        Whether to remove the full response (including the effect of digital FIR
        filters) or just the instrument transform function (as defined by the PolesZeros
        Response Stage). Significantly slower.
    upfactor:
        Factor by which to upsample the data to enable it to be decimated to the desired
        sampling rate, e.g., 40 Hz -> 50 Hz requires upfactor = 5.
        See :func:`~quakemigrate.util.resample`
    interpolate:
        If data is timestamped "off-sample" (i.e. a non-integer number of samples after
        midnight), whether to interpolate the data to apply the necessary correction.
        Default behaviour is to just alter the metadata, resulting in a sub-sample
        timing offset. See :func:`~quakemigrate.util.shift_to_sample`.

    """

    path: str | pathlib.Path
    format: str
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
        self.path = pathlib.Path(self.path)
        self.format = _validate_path_format(self.format)

        if self.inventory:
            if self.response_removal_params is None:
                self.response_removal_params = {}
                print(
                    "Warning: 'water level' for instrument correction not "
                    "specified. Set to default: 60"
                )
            self.water_level = self.response_removal_params.get("water_level", 60.0)
            self.pre_filt = self.response_removal_params.get("pre_filt")
            self.remove_full_response = self.response_removal_params.get(
                "remove_full_response", False
            )

    def __str__(self, response_only: bool = False) -> str:
        """
        Returns a short summary string of the Archive object.

        Parameters
        ----------
        response_only:
            Whether to just output the a string describing the instrument response
            parameters.

        Returns
        -------
        out:
            Summary string.

        """

        if self.inventory:
            response_str = (
                "\tResponse removal parameters:\n"
                f"\t\tWater level  = {self.water_level}\n"
            )
            if self.pre_filt is not None:
                response_str += f"\t\tPre-filter   = {self.pre_filt} Hz\n"
            response_str += (
                "\t\tRemove full response (inc. FIR stages) = "
                f"{self.remove_full_response}\n"
            )
        else:
            response_str = "\tNo instrument response inventory provided!\n"

        if not response_only:
            out = (
                "QuakeMigrate Archive object"
                f"\n\tArchive root\t:\t{self.path}"
                f"\n\tPath structure\t:\t{self.format}"
                f"\n\tResampling\t:\t{self.resample}"
            )
            if self.upfactor:
                out += f"\n\tUpfactor\t:\t{self.upfactor}"
            out += f"\n{response_str}"
        else:
            out = response_str

        return out

    def read_waveform_data(
        self,
        stations: list[Station],
        starttime: UTCDateTime,
        endtime: UTCDateTime,
        pre_pad: float = 0.0,
        post_pad: float = 0.0,
    ) -> WaveformData:
        """
        Read in waveform data from the archive between two times.

        Supports all formats currently supported by ObsPy, including: "MSEED", "SAC",
        "SEGY", "GSE2".

        Optionally, read data with some pre- and post-pad, and for all stations in the
        archive - this will be stored in `data.raw_waveforms`, while `data.waveforms`
        will contain only data for selected stations between `starttime` and `endtime`.

        Parameters
        ----------
        stations:
            List of Station objects for which to read waveform data.
        starttime:
            Timestamp from which to read waveform data.
        endtime:
            Timestamp up to which to read waveform data.
        pre_pad:
            Additional pre pad of data to read.
        post_pad:
            Additional post pad of data to read.

        Returns
        -------
        data:
            Waveform data read from the archive that satisfies the query.

        Raises
        ------
        ArchiveEmpty
            If no data files are found in the archive for this day(s).
        DataGap
            If no data is found in the archive for the specified stations within the
            specified time window.

        """

        # Ensure pre-pad and post-pad are not negative.
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

        read_start, read_end = starttime - pre_pad, endtime + post_pad
        files = self._load_from_path(stations, read_start, read_end)

        st = Stream()
        try:
            first = next(files)
            files = chain([first], files)
            for file in files:
                file = str(file)
                try:
                    st += read(
                        file,
                        starttime=read_start,
                        endtime=read_end,
                        nearest_sample=True,
                    )
                except TypeError:
                    logging.info(f"File not compatible with ObsPy - {file}")
                    continue

            if not any(station.network for station in stations):
                for tr in st:
                    tr.stats.network = "XX"

            if not any(station.location for station in stations):
                for tr in st:
                    tr.stats.location = ""

            # Merge waveforms channel-by-channel with no-clobber merge
            st = util.merge_stream(st)

            # Make copy of raw waveforms to output if requested
            data.raw_waveforms = st.copy()

            # Ensure data is timestamped "on-sample" (i.e. an integer number of samples
            # after midnight). Otherwise the data will be implicitly shifted when it is
            # used to calculate the onset function / migrated.
            st = util.shift_to_sample(st, interpolate=self.interpolate)

            if any(station.read_only for station in stations):
                st_selected = Stream()
                for station in stations:
                    if not station.read_only:
                        st_selected += st.select(station=station.station)
                st = st_selected.copy()
                del st_selected

            if pre_pad != 0.0 or post_pad != 0.0:
                # Trim data between start and end time
                for tr in st:
                    tr.trim(starttime=starttime, endtime=endtime, nearest_sample=True)
                    if not bool(tr):
                        st.remove(tr)

            # Test if the stream is completely empty
            # (see __nonzero__ for `obspy.Stream` object)
            if not bool(st):
                raise DataGap()

            # Add cleaned stream to `waveforms`
            data.waveforms = st

        except StopIteration:
            raise ArchiveEmpty()

        return data

    def _load_from_path(
        self, stations: list[Station], starttime: UTCDateTime, endtime: UTCDateTime
    ) -> Iterator[pathlib.Path]:
        """
        Retrieves available files between two times.

        Parameters
        ----------
        stations:
            List of Station objects to be read from archive.
        starttime:
            Timestamp from which to read waveform data.
        endtime:
            Timestamp up to which to read waveform data.

        Returns
        -------
        files:
            Iterator object of available waveform data files.

        """

        # Loop through time period by day adding files to list
        # NOTE! This assumes the archive structure is split into days.
        files = []
        loadstart = UTCDateTime(starttime.date)
        while loadstart <= endtime:
            for station in stations:
                file_format = self.format.format(
                    year=loadstart.year,
                    month=loadstart.month,
                    day=loadstart.day,
                    jday=loadstart.julday,
                    network=station.network if station.network is not None else "*",
                    station=station.station,
                    location=station.location if station.location is not None else "*",
                    channels=station.channels if station.channels is not None else "*",
                    dtime=loadstart,
                )
                files = chain(files, self.path.glob(file_format))
            loadstart += 86400

        return files

    # --- Deprecation/Future handling ---
    def path_structure(
        self, archive_format: str = "YEAR/JD/STATION", channels: str = "*"
    ) -> None:
        """Deprecation handling for old method."""
        if archive_format in ARCHIVE_FORMATS.keys():
            print(
                "This method of setting the archive format has been deprecated.\n"
                f"Mapping {archive_format} -> {ARCHIVE_FORMATS[archive_format]}"
            )
            self.format = ARCHIVE_FORMATS[archive_format]
        else:
            raise KeyError


@dataclass
class WaveformData:
    """
    The WaveformData class encapsulates the waveform data returned by an Archive query.

    It also provides a number of utility functions. These include removing instrument
    response and checking data availability against a flexible set of data quality
    criteria.

    Parameters
    ----------
    starttime:
        Timestamp of first sample of waveform data requested from the archive.
    endtime:
        Timestamp of last sample of waveform data requested from the archive.
    stations:
        Iterable of Station objects.
    resample:
        If true, allow resampling of data which cannot be decimated directly to the
        desired sampling rate. See :func:`~quakemigrate.util.resample`
    upfactor:
        Factor by which to upsample the data to enable it to be decimated to the desired
        sampling rate, e.g., 40Hz -> 50Hz requires upfactor = 5.
        See :func:`~quakemigrate.util.resample`
    inventory:
        ObsPy response inventory for this waveform data, containing response information
        for each channel of each station of each network.
    pre_filt:
        Pre-filter to apply during the instrument response removal, e.g.,
        (0.03, 0.05, 30., 35.) - all in Hz.
    water_level:
        Water level to use in instrument response removal.
    remove_full_response:
        Whether to remove the full response (including the effect of digital FIR
        filters) or just the instrument transform function (as defined by the PolesZeros
        Response Stage). Significantly slower.
    pre_pad:
        Additional pre pad of data included in `raw_waveforms`.
    post_pad:
        Additional post pad of data included in `raw_waveforms`.

    Attributes
    ----------
    raw_waveforms:
        Raw seismic data read in from the archive. This may be for all stations in the
        archive, or only those specified by the user. It may also cover the time period
        between `starttime` and `endtime`, or feature an additional pre- and post-pad.
        See `pre_pad` and `post_pad`.
    waveforms:
        Seismic data read in from the archive for the specified list of stations,
        between `starttime` and `endtime`.

    Raises
    ------
    NotImplementedError
        If the user attempts to use the get_real_waveform() method.

    """

    starttime: UTCDateTime
    endtime: UTCDateTime
    stations: list[Station] | None = None
    inventory: Inventory | None = None
    water_level: float = 60.0
    pre_filt: tuple[float, float, float, float] | None = None
    remove_full_response: bool = False
    resample: bool = False
    upfactor: int | None = None
    pre_pad: float = 0.0
    post_pad: float = 0.0

    waveforms: Stream = field(default_factory=Stream)
    raw_waveforms: Stream | None = None
    wa_waveforms: Stream | None = None
    real_waveforms: Stream | None = None

    def check_availability(
        self,
        st: Stream,
        all_channels: bool = False,
        n_channels: int | None = None,
        allow_gaps: bool = False,
        full_timespan: bool = True,
        check_sampling_rate: bool = False,
        sampling_rate: int | None = None,
        check_start_end_times: bool = False,
    ) -> tuple[int, dict]:
        """
        Check waveform availability against data quality criteria.

        There are a number of hard-coded checks: for whether any data is present; for
        whether the data is a flatline (all samples have the same value); and for
        whether the data contains overlaps. There are a selection of additional optional
        checks which can be specified according to the onset function / user preference.

        Parameters
        ----------
        st:
            Stream containing the waveform data to check against the availability
            criteria.
        all_channels:
            Whether all supplied channels (distinguished by SEED id) need to meet the
            availability criteria to mark the data as 'available'.
        n_channels:
            If `all_channels=True`, this argument is required (in order to specify the
            number of channels expected to be present).
        allow_gaps:
            Whether to allow gaps.
        full_timespan:
            Whether to ensure the data covers the entire timespan requested; note that
            this implicitly requires that there be no gaps. Checks the number of samples
            in the trace, not the start and end times; for that see
            `check_start_end_times`.
        check_sampling_rate:
            Check that all channels are at the desired sampling rate.
        sampling_rate:
            If `check_sampling_rate=True`, this argument is required to specify the
            sampling rate that the data should be at.
        check_start_end_times:
            A stricter alternative to `full_timespan`; checks that the first and last
            sample of the trace have exactly the requested timestamps.

        Returns
        -------
        available:
            0 if data doesn't meet the availability requirements; 1 if it does.
        availability:
            Dict of {tr_id : available} for each unique SEED ID in the input stream
            (available is again 0 or 1).

        Raises
        ------
        TypeError
            If `check_sampling_rate` is requested, but no `sampling_rate` provided.
        TypeError
            If the user specifies `all_channels=True` but does not specify `n_channels`.

        """

        availability = {}
        available = 0
        timespan = self.endtime - self.starttime

        # Check if any channels in stream
        if bool(st):
            # Loop through channels with unique SEED id's
            for tr_id in sorted(set([tr.id for tr in st])):
                st_id = st.select(id=tr_id)
                availability[tr_id] = 0

                # Check it's not flatlined
                if any(tr.data.max() == tr.data.min() for tr in st_id):
                    continue
                # Check for overlaps
                overlaps = st_id.get_gaps(max_gap=-0.000001)
                if len(overlaps) != 0:
                    continue
                # Check for gaps (if requested)
                if not allow_gaps:
                    gaps = st_id.get_gaps()  # Overlaps already dealt with
                    if len(gaps) != 0:
                        continue
                # Check sampling rate
                if check_sampling_rate:
                    if not sampling_rate:
                        raise TypeError(
                            "Please specify sampling_rate if you wish to check all "
                            "channels are at the correct sampling rate."
                        )
                    if any(tr.stats.sampling_rate != sampling_rate for tr in st_id):
                        continue
                # Check data covers full timespan (if requested) - this
                # strictly checks the *timespan*, so uses the trace sampling
                # rate as provided. To check that as well, use
                # `check_sampling_rate=True` and specify a sampling rate.
                if full_timespan:
                    n_samples = int(round(timespan * st_id[0].stats.sampling_rate + 1))
                    if len(st_id) > 1:
                        continue
                    elif st_id[0].stats.npts < n_samples:
                        logging.debug("Trace has too few samples.")
                        logging.debug(
                            "(n_samples, trace_npts) : "
                            f"({n_samples}, {st_id[0].stats.npts})"
                        )
                        continue
                # Check start and end times of trace are exactly correct
                if check_start_end_times:
                    if len(st_id) > 1:
                        continue
                    elif (
                        st_id[0].stats.starttime != self.starttime
                        or st_id[0].stats.endtime != self.endtime
                    ):
                        continue

                # If passed all tests, set availability to 1
                availability[tr_id] = 1

            # Return availability based on "all_channels" setting
            if all(ava == 1 for ava in availability.values()):
                if all_channels:
                    # If all_channels requested, must also check that the
                    # expected number of channels are present
                    if not n_channels:
                        raise TypeError(
                            "Please specify n_channels if you wish to check all "
                            "channels meet the availability criteria."
                        )
                    elif len(availability) == n_channels:
                        available = 1
                else:
                    available = 1
            elif not all_channels and any(ava == 1 for ava in availability.values()):
                available = 1

        return available, availability

    def get_real_waveform(self, tr: Trace, velocity: bool = True) -> Trace:
        """
        Calculate the real waveform for a Trace by removing the instrument response.

        Parameters
        ----------
        tr:
            Trace containing the waveform for which to remove the instrument response.
        velocity:
            Output velocity waveform (as opposed to displacement).

        Returns
        -------
        tr_out:
            Trace with instrument response removed.

        Raises
        ------
        AttributeError
            If no response inventory has been supplied.
        ResponseNotFoundError
            If the response information for a trace can't be found in the supplied
            response inventory.
        ResponseRemovalError
            If the deconvolution of the instrument response is unsuccessful.

        """

        if not self.inventory:
            raise AttributeError("No response inventory provided!")

        tr_out = tr.copy()
        tr_out.detrend("linear")

        if not self.remove_full_response:
            # Just remove the response encapsulated in the instrument transfer function
            # (stored as a PolesAndZeros response). NOTE: this does not account for the
            # effect of the digital FIR filters applied to the recorded waveforms.
            # However, due to this it is significantly faster to compute.
            try:
                response = self.inventory.get_response(
                    tr_out.id, tr_out.stats.starttime
                )
            except Exception as e:
                raise ResponseNotFoundError(str(e), tr_out.id)

            # Get the instrument transfer function as a PAZ dictionary
            paz = response.get_paz()

            if not velocity:
                paz.zeros.extend([0j])

            paz_dict = {
                "poles": paz.poles,
                "zeros": paz.zeros,
                "gain": paz.normalization_factor,
                "sensitivity": response.instrument_sensitivity.value,
            }

            try:
                tr_out.simulate(
                    paz_remove=paz_dict,
                    pre_filt=self.pre_filt,
                    water_level=self.water_level,
                    taper=True,
                    sacsim=True,  # To replicate remove_response()
                    pitsasim=False,  # To replicate remove_response()
                )
            except ValueError as e:
                raise ResponseRemovalError(e, tr_out.id)
        else:
            # Use remove_response(), which removes the effect of _all_ response stages,
            # including the FIR stages. Considerably slower.
            output = "VEL" if velocity else "DISP"

            try:
                tr_out.remove_response(
                    inventory=self.inventory,
                    output=output,
                    pre_filt=self.pre_filt,
                    water_level=self.water_level,
                    taper=True,
                )
            except ValueError as e:
                raise ResponseRemovalError(e, tr_out.id)

        if self.real_waveforms is None:
            self.real_waveforms = Stream()
        self.real_waveforms.append(tr_out.copy())

        return tr_out

    def get_wa_waveform(self, tr: Trace, velocity: bool = False) -> Trace:
        """
        Calculate simulated Wood-Anderson displacement waveform for a Trace.

        NOTE: all attenuation functions provided in QuakeMigrate are calculated for
        displacement seismograms.

        Parameters
        ----------
        tr:
            Trace containing the waveform to be corrected to a Wood-Anderson response.
        velocity:
            Output velocity waveform, instead of displacement.

        Returns
        -------
        tr_out:
            Trace corrected to Wood-Anderson response.

        """

        tr_out = tr.copy()
        tr_out.detrend("linear")

        # Remove instrument response
        tr_out = self.get_real_waveform(tr_out, velocity)

        # Simulate Wood-Anderson response
        tr_out.simulate(
            paz_simulate=util.wa_response(obspy_def=True),
            pre_filt=self.pre_filt,
            water_level=self.water_level,
            taper=True,
            sacsim=True,  # To replicate remove_response()
            pitsasim=False,  # To replicate remove_response()
        )

        if self.wa_waveforms is None:
            self.wa_waveforms = Stream()
        self.wa_waveforms.append(tr_out.copy())

        return tr_out
