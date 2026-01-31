"""
Custom exceptions used by QuakeMigrate.

:copyright:
    2020–2026, QuakeMigrate developers.
:license:
    GNU General Public License, Version 3
    (https://www.gnu.org/licenses/gpl-3.0.html)

"""

from __future__ import annotations

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    import pathlib

    from obspy import Trace, UTCDateTime


class QMException(Exception):
    """Base class for QuakeMigrate exceptions."""


class CLIError(QMException):
    """User-facing CLI error."""

    exit_code: int = 1

    def __init__(self, message: str, exit_code: int | None = None) -> None:
        super().__init__(message)
        if exit_code is not None:
            self.exit_code = exit_code


class ConfigError(CLIError):
    exit_code = 2


class ProjectError(CLIError):
    exit_code = 3


class InvalidStationFileHeader(QMException):
    """Raised when the station file header is missing required columns."""

    def __init__(self, found: list[str] | None = None) -> None:
        expected = ["Latitude", "Longitude", "Elevation", "Name"]

        msg = f"Invalid station file header.\nExpected columns: {', '.join(expected)}"

        if found is not None:
            msg += f"\nFound columns:    {', '.join(found)}"

        super().__init__(msg)


class InvalidVelocityModelHeader(QMException):
    """Raised when the velocity model file header is missing required columns."""

    def __init__(self, missing: str) -> None:
        super().__init__(
            f"Invalid velocity model header: missing required column '{missing}'."
        )


class ArchiveFormatError(QMException):
    """Raised when `Archive.format` is not set."""

    def __init__(self) -> None:
        super().__init__(
            "Archive format has not been set.\n"
            "Specify `archive_format=...` when creating the Archive."
        )


class InvalidArchivePathStructure(QMException):
    """Raised when an invalid archive path structure option is selected."""

    def __init__(self, archive_format: str) -> None:
        super().__init__(
            f"Invalid archive path structure: '{archive_format}'.\n"
            "Select a supported structure via `Archive.path_structure(...)` or provide "
            "a custom format string via `archive.format = '...'`."
        )


class ArchiveEmpty(QMException):
    """Raised when the archive is completely empty for a requested time period."""

    def __init__(self) -> None:
        super().__init__("No data was available for this timestep.")

        # Additional message printed to log
        self.msg = "\t\tNo files found in archive for this time period."


class NoScanmSEEDData(QMException):
    """Raised when no scanmSEED data can be found to trigger on."""

    def __init__(
        self, fpath: pathlib.Path, readstart: UTCDateTime, readend: UTCDateTime
    ) -> None:
        super().__init__(
            "No .scanmseed data found.\n"
            f"Path: {fpath}\n"
            f"Time window: {readstart}-{readend}"
        )


class NoStationAvailabilityData(QMException):
    """Raised when no .StationAvailability files can be found."""

    def __init__(
        self, fpath: pathlib.Path, readstart: UTCDateTime, readend: UTCDateTime
    ) -> None:
        super().__init__(
            "No .StationAvailability files found.\n"
            f"Path: {fpath}\n"
            f"Time window: {readstart}-{readend}"
        )


class AllDataRejected(QMException):
    """Raised when no data passes the sequence of data quality validations."""

    def __init__(self) -> None:
        super().__init__(
            "No data for this timestep passed the specified data quality criteria."
        )

        # Additional message printed to log
        self.msg = (
            "\t\tAll data for this timestep failed to pass the"
            "\n\t\tspecified data quality criteria. This includes the"
            "\n\t\tpresence of gaps or overlaps, or the data not"
            "\n\t\tspanning the full time window."
        )


class DataGap(QMException):
    """Raised when no data found for selected stations for a given timestep."""

    def __init__(self) -> None:
        super().__init__(
            "No data present in the archive for the selected stations for this time "
            "window."
        )

        # Additional message printed to log
        self.msg = (
            "\t\tNo data for the selected stations was found in the"
            "\n\t\tarchive for this time window."
        )


class NoOnsetPeak(QMException):
    """Raised when no values in the onset function exceed threshold used for picking."""

    def __init__(self, pick_threshold: float) -> None:
        self.msg = (
            "\t\t    No onset signal exceeding pick threshold "
            f"({pick_threshold:5.3f}) - continuing."
        )
        super().__init__(self.msg)


class LUTMissingPhaseTables(QMException):
    """Raised when the LUT does not contain traveltimes for a required phase."""

    def __init__(self, missing: str, phases: list[str]) -> None:
        super().__init__(
            f"LUT is missing traveltimes for phase '{missing}'. "
            f"Requested phases: {phases}. "
            "Rebuild the LUT including this phase."
        )


class PickOrderError(QMException):
    """Raised when the pick for the P phase is later than for the S phase."""

    def __init__(self, event_uid: str, station: str, p_pick: str, s_pick: str) -> None:
        super().__init__(
            "The P-phase arrival-time pick is later than the S-phase arrival pick! "
            f"Something has gone wrong.\nEvent: {event_uid}, station: {station}, "
            f"p_pick: {p_pick}, s_pick: {s_pick}. There is probably a bug with the "
            "picker."
        )


class NoTriggeredEventsData(QMException):
    """
    Raised when no trigger files are found during locate.

    This can occur for one of two reasons:
        1. an entirely invalid time period was used, i.e., one that does not overlap
           at all with a period of time for which there exists TriggeredEvents.csv files
        2. an invalid run name was provided.

    """

    def __init__(self) -> None:
        super().__init__(
            "Double check you have supplied a valid run name and a time period for "
            "which you have run detect."
        )


class ResponseNotFoundError(QMException):
    """
    Raised when the provided response inventory does not contain the response
    information for a trace.

    Parameters
    ----------
    e:
        Error message from ObsPy `Inventory.get_response()`.
    tr_id:
        ID string for the Trace for which the response cannot be found.

    """

    def __init__(self, e: str, tr_id: str) -> None:
        super().__init__(f"{e} -- skipping {tr_id}")


class ResponseRemovalError(QMException):
    """
    Raised when the response removal is not successful.

    Parameters
    ----------
    e:
        Error message from ObsPy `Trace.remove_response()` or `Trace.simulate()`.
    tr_id:
        ID string for the Trace for which the response cannot be removed.

    """

    def __init__(self, e: str, tr_id: str) -> None:
        super().__init__(f"{e} -- skipping {tr_id}")


class NyquistException(QMException):
    """
    Raised when the specified filter has a lowpass corner above the signal Nyquist
    frequency.

    Parameters
    ----------
    freqmax:
        Specified lowpass frequency for filter.
    f_nyquist:
        Nyquist frequency for the relevant waveform data.
    tr_id:
        ID string for the Trace.

    """

    def __init__(self, freqmax: float, f_nyquist: float, tr_id: str) -> None:
        super().__init__(
            f"    Selected bandpass_highcut {freqmax} Hz is at or above the Nyquist "
            f"frequency ({f_nyquist} Hz) for trace {tr_id}. "
        )


class PeakToTroughError(QMException):
    """Raised when peak-to-trough amplitude cannot be computed for a window."""

    def __init__(self, err: str) -> None:
        super().__init__(err)
