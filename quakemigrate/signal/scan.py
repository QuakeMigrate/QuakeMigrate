"""
Module to perform core QuakeMigrate functions: detect() and locate().

:copyright:
    2020–2026, QuakeMigrate developers.
:license:
    GNU General Public License, Version 3
    (https://www.gnu.org/licenses/gpl-3.0.html)

"""

from __future__ import annotations

import logging
import warnings
from datetime import time
from typing import Literal, TYPE_CHECKING

import numpy as np
import pandas as pd
from obspy import UTCDateTime

import quakemigrate.util as util
from quakemigrate.core import find_max_coa, migrate
from quakemigrate.exceptions import (
    ArchiveEmpty,
    AllDataRejected,
    DataGap,
    LUTMissingPhaseTables,
)
from quakemigrate.io import (
    Event,
    Run,
    ScanmSEED,
    read_triggered_events,
    write_availability,
    write_cut_waveforms,
    write_coalescence,
)
from quakemigrate.plugins import call_by_signature
from quakemigrate.plugins.onsets.base import Onset
from quakemigrate.signal.location_uncertainty import (
    covariance_fit,
    spline_location,
    gaussian_filter,
    gaussian_fit,
)


if TYPE_CHECKING:
    from quakemigrate.io.data import Archive, WaveformData
    from quakemigrate.io.station import Station
    from quakemigrate.lut import LUT


# Filter warnings
warnings.filterwarnings(
    "ignore", message=("Covariance of the parameters could not be estimated")
)


class QuakeScan:
    """
    QuakeMigrate scanning class.

    Provides an interface for the wrapped compiled C functions, used to perform the
    continuous scan (detect) or refined event migrations (locate).

    Parameters
    ----------
    archive:
        Details the structure and location of a data archive and provides methods for
        reading data from file.
    lut:
        Contains the traveltime lookup tables for seismic phases, computed for some
        pre-defined velocity model.
    onset:
        Provides callback methods for calculation of onset functions.
    run_path:
        Points to the top level directory containing all input files, under which the
        specific run directory will be created.
    run_name:
        Name of the current QuakeMigrate run.
    kwargs:
        See QuakeScan Attributes for details. In addition to these:

    Attributes
    ----------
    continuous_scanmseed_write:
        Option to continuously write the .scanmseed file output by detect() at the end
        of every time step. Default behaviour is to write in day chunks where possible.
    cut_waveform_format:
        File format used when writing waveform data. We support any format also
        supported by ObsPy - "MSEED" (default), "SAC", "SEGY", "GSE2".
    log:
        Toggle for logging. If True, will output to stdout and generate a log file.
        Default is to only output to stdout.
    loglevel:
        Toggle to set the logging level: "debug" will print out additional diagnostic
        information to the log and stdout.
    mags:
        Provides methods for calculating local magnitudes, performed during locate.
    marginal_window:
        Half-width of window centred on the maximum coalescence time. The 4-D
        coalescence functioned is marginalised over time across this window such that
        the earthquake location and associated uncertainty can be appropriately
        calculated. It should be an estimate of the time uncertainty in the earthquake
        origin time, which itself is some combination of the expected spatial
        uncertainty and uncertainty in the seismic velocity model used.
    picker:
        Provides callback methods for phase picking, performed during locate.
    plot_all_stations:
        If true, plot all stations in the LUT. Otherwise, only plot stations which were
        used for migration (i.e. omitting stations for which there was no data, or data
        did not pass the specified quality checks).
    plot_event_summary:
        Plot event summary figure - see `quakemigrate.plot` for more details.
    plot_event_video:
        Plot coalescence video for each located earthquake.
    post_pad:
        Additional amount of data to read in after the timestep, used to ensure the
        correct coalescence is calculated at every sample.
    pre_pad:
        Additional amount of data to read in before the timestep, used to ensure the
        correct coalescence is calculated at every sample.
    real_waveform_units:
        Units to output real cut waveforms.
    run:
        Light class encapsulating i/o path information for a given run.
    scan_rate:
        Sampling rate at which the 4-D coalescence map will be calculated. Currently
        fixed to be the same as the onset function sampling rate (not
        user-configurable).
    threads:
        The number of threads for the C functions to use on the executing host.
    timestep:
        Length (in seconds) of timestep used in detect(). Note: total detect run
        duration should be divisible by timestep. Increasing timestep will increase RAM
        usage during detect, but will slightly speed up overall detect run.
    wa_waveform_units:
        Units to output Wood-Anderson simulated cut waveforms.
    write_cut_waveforms:
        Write raw cut waveforms for all data read from the archive for each event
        located by locate(). See `~quakemigrate.io.data.Archive` parameter
        `read_all_stations`.
        NOTE: this data has not been processed or quality-checked!
    write_marginal_coalescence:
        Write the marginalised 3-D coalescence map to file (in .npy format).
    write_coalescence:
        Write the raw 4-D coalescence map from locate to file (in .npy format).
    write_real_waveforms:
        Write real cut waveforms for all data read from the archive for each event
        located by locate(). See `~quakemigrate.io.data.Archive` parameter
        `read_all_stations`.
        NOTE: the units of this data (displacement or velocity) are controlled by
        `real_waveform_units`.
        NOTE: this data has not been processed or quality-checked!
        NOTE: no padding has been added to take into account the taper applied during
        response removal.
    write_wa_waveforms:
        Write Wood-Anderson simulated cut waveforms for all data read from the archive
        for each event located by locate(). See `~quakemigrate.io.data.Archive`
        parameter `read_all_stations`.
        NOTE: the units of this data (displacement or velocity) are controlled by
        `wa_waveform_units`.
        NOTE: this data has not been processed or quality-checked!
        NOTE: no padding has been added to take into account the taper applied during
        response removal.
    xy_files:
        Path to comma-separated value file (.csv) containing a series of coordinate
        files to plot. Columns: ["File", "Color", "Linewidth", "Linestyle"], where
        "File" is the absolute path to the file containing the coordinates to be
        plotted. E.g: "/home/user/volcano_outlines.csv,black,0.5,-". Each .csv
        coordinate file should contain coordinates only, with columns: ["Longitude",
        "Latitude"]. E.g.: "-17.5,64.8". Lines pre-pended with ``#`` will be treated as
        a comment - this can be used to include references. See the
        Volcanotectonic_Iceland example XY_files for a template.\n
        .. note:: Do not include a header line in either file.

    +++ TO BE REMOVED TO ARCHIVE CLASS +++
    pre_cut : float, optional
        Specify how long before the event origin time to cut the waveform data from.
    post_cut : float, optional
        Specify how long after the event origin time to cut the waveform.
        data to
    +++ TO BE REMOVED TO ARCHIVE CLASS +++

    Raises
    ------
    TypeError
        If an object is passed in through the `onset` argument that is not derived from
        the :class:`~quakemigrate.plugins.base.Onset` base class.
    TypeError
        If an object is passed in through the `picker` argument that is not derived from
        the :class:`~quakemigrate.plugins.base.PhasePicker` base class.
    TypeError
        If an object is passed in through the `mags` argument that is not derived from
        the :class:`~quakemigrate.plugins.magnitudes.LocalMag` base class.
    RuntimeError
        If the user does not supply the locate function with valid arguments.

    """

    def __init__(
        self,
        archive: Archive,
        lut: LUT,
        onset: Onset,
        run_path: str,
        run_name: str,
        **kwargs: dict,
    ) -> None:
        """Instantiate the QuakeScan object."""

        self.archive = archive
        self.lut = lut

        if not isinstance(onset, Onset):
            raise TypeError(
                f"onset must inherit from quakemigrate.plugins.onsets.Onset "
                f"(got {type(onset).__name__})."
            )
        self.onset = onset
        self.onset.post_pad = lut.max_traveltime

        self.pre_pad = 0.0
        self.post_pad = 0.0

        # --- Set up i/o ---
        loglevel: Literal["info", "debug"] = kwargs.get("loglevel", "info")
        self.run: Run = Run(
            run_path,
            run_name,
            kwargs.get("run_subname", ""),
            loglevel=loglevel,
        )
        self.log: bool = kwargs.get("log", False)

        self.plugins = kwargs.get("plugins", [])

        # --- Grab QuakeScan parameters or set defaults ---
        # Parameters related specifically to Detect
        self.timestep: float = kwargs.get("timestep", 120.0)

        # Parameters related specifically to Locate
        self.marginal_window: float = kwargs.get("marginal_window", 2.0)

        # General QuakeScan parameters
        self.threads: int = kwargs.get("threads", 1)
        self.scan_rate: int = self.onset.sampling_rate

        self.plot_event_video: bool = kwargs.get("plot_event_video", False)

        # File writing toggles
        self.continuous_scanmseed_write: bool = kwargs.get(
            "continuous_scanmseed_write", False
        )
        self.write_cut_waveforms: bool = kwargs.get("write_cut_waveforms", False)
        self.write_real_waveforms: bool = kwargs.get("write_real_waveforms", False)
        self.real_waveform_units: Literal["displacement", "velocity"] = kwargs.get(
            "real_waveform_units", "displacement"
        )
        self.write_wa_waveforms: bool = kwargs.get("write_wa_waveforms", False)
        self.wa_waveform_units: Literal["displacement", "velocity"] = kwargs.get(
            "wa_waveform_units", "displacement"
        )
        self.cut_waveform_format: str = kwargs.get("cut_waveform_format", "MSEED")
        self.write_marginal_coalescence: bool = kwargs.get(
            "write_marginal_coalescence", False
        )
        self.write_coalescence: bool = kwargs.get("write_coalescence", False)

        # +++ TO BE REMOVED TO ARCHIVE CLASS +++
        self.pre_cut = None
        self.post_cut = None
        # +++ TO BE REMOVED TO ARCHIVE CLASS +++

    def __str__(self) -> str:
        """Return short summary string of the QuakeScan object."""

        out = (
            "\tScan parameters:\n"
            f"\t\tScan sampling rate = {self.scan_rate} Hz\n"
            f"\t\tThread count       = {self.threads}\n"
        )
        if self.run.stage == "detect":
            out += f"\t\tTime step          = {self.timestep} s\n"
        elif self.run.stage == "locate":
            out += f"\t\tMarginal window    = {self.marginal_window} s\n"

        return out

    def detect(self, stations: list[Station], starttime: str, endtime: str) -> None:
        """
        Scans through data calculating coalescence in a (decimated) 3-D grid by
        continuously migrating onset functions.

        Note: if the time interval between starttime and endtime is not divisible by
        the specified timestep, the endtime will be extended to accommodate. If the
        endtime is set to midnight, then it will be automatically adjusted to one sample
        prior.

        Parameters
        ----------
        stations:
            Iterable of Station objects to be used in detect.
        starttime:
            Timestamp from which to run continuous scan.
        endtime:
            Timestamp up to which to run continuous scan.

        Raises
        ------
        ValueError
            If `starttime` is later than `endtime`.

        """

        # Configure logging
        self.run.stage = "detect"
        self.run.logger(self.log)

        starttime, endtime = UTCDateTime(starttime), UTCDateTime(endtime)
        if starttime > endtime:
            raise ValueError("starttime must be <= endtime")
        # Shift endtime one sample earlier if it is at midnight (not necessary for
        # typical combinations of starttimes and timesteps, but here to cover edge
        # cases)
        if endtime.time == time(0, 0):
            endtime -= 1 / self.scan_rate

        # Number of steps to break run duration into
        n_steps = int(np.ceil((endtime - starttime) / self.timestep))

        # Check if chosen start & endtimes and timestep are compatible
        calc_endtime = starttime + n_steps * self.timestep - 1 / self.scan_rate
        if calc_endtime - endtime > 1 / self.scan_rate:
            logging.info(
                f"Warning: chosen run duration {endtime - starttime} s is not "
                f"divisible by the specified timestep {self.timestep} s. Detect will "
                f"instead compute up to {calc_endtime}\n"
            )

        logging.info(util.log_spacer)
        logging.info("\tDETECT - Continuous coalescence scan")
        logging.info(util.log_spacer)
        logging.info(f"\n\tScanning from {starttime} to {calc_endtime}\n")
        logging.info(self)
        logging.info(self.onset)
        logging.info(util.log_spacer)

        self._continuous_compute(stations, starttime, n_steps)

        logging.info(util.log_spacer)

    def locate(
        self,
        stations: list[Station],
        starttime: UTCDateTime | None = None,
        endtime: UTCDateTime | None = None,
        trigger_file: str | None = None,
    ) -> None:
        """
        Re-computes the coalescence on an undecimated grid for a short time window
        around each candidate earthquake triggered from the (decimated) continuous
        detect scan. Calculates event location and uncertainties, makes phase arrival
        picks, plus multiple optional plotting / data outputs for further analysis and
        processing.

        Parameters
        ----------
        stations:
            Iterable of Station objects to be used in locate.
        starttime:
            Timestamp from which to include events in the locate scan.
        endtime:
            Timestamp up to which to include events in the locate scan. Note: if the
            endtime is set to midnight, then only events during the previous day will
            be included.
        trigger_file:
            File containing triggered events to be located.

        Raises
        ------
        ValueError
            If `starttime` is later than `endtime`.
        RuntimeError
            If none of `trigger_file`, `starttime`, or `endtime` are provided.
        RuntimeError
            If only one of `starttime` or `endtime` are provided.

        """

        # Configure logging
        self.run.stage = "locate"
        self.run.logger(self.log)

        if not (starttime is None and endtime is None):
            starttime, endtime = UTCDateTime(starttime), UTCDateTime(endtime)
            if starttime > endtime:
                raise ValueError("starttime must be <= endtime")
        if trigger_file is None and starttime is None and endtime is None:
            raise RuntimeError("must supply an input argument.")
        if (starttime is None) ^ (endtime is None):
            raise RuntimeError("must supply a starttime AND an endtime.")

        logging.info(util.log_spacer)
        logging.info("\tLOCATE - Determining event location and uncertainty")
        logging.info(util.log_spacer)
        if trigger_file is not None:
            logging.info(f"\n\tLocating events in {trigger_file}")
        else:
            logging.info(f"\n\tLocating events from {starttime} to {endtime}\n")
        logging.info(self)
        logging.info(self.onset)
        for plugin in sorted(self.plugins, key=lambda p: p.order):
            logging.info(str(plugin))
        logging.info(util.log_spacer)

        if trigger_file is not None:
            self._locate_events(stations, trigger_file=trigger_file)
        else:
            self._locate_events(stations, starttime=starttime, endtime=endtime)

        logging.info(util.log_spacer)

    def _continuous_compute(
        self, stations: list[Station], starttime: UTCDateTime, n_steps: int
    ) -> None:
        """
        Compute coalescence between two timestamps, divided into increments of
        `timestep`. Outputs coalescence and station availability data to file.

        Parameters
        ----------
        stations:
            List of Station objects for which to perform continuous compute.
        starttime:
            Timestamp from which to compute continuous coalescence.
        n_steps:
            Number of timesteps (of length `timestep`) to compute.

        """

        coalescence = ScanmSEED(
            self.run, self.continuous_scanmseed_write, self.scan_rate
        )

        self.pre_pad, self.post_pad = self.onset.pad(self.timestep)
        availability_cols = np.array(
            [
                [f"{station.id}_{phase}" for station in stations]
                for phase in self.onset.phases
            ]
        ).flatten()
        availability = pd.DataFrame(index=range(n_steps), columns=availability_cols)

        for i in range(n_steps):
            w_beg = starttime + self.timestep * i - self.pre_pad
            w_end = (
                starttime + self.timestep * (i + 1) - 1 / self.scan_rate + self.post_pad
            )
            logging.debug(f" Processing : {w_beg}-{w_end} ".center(110, "~"))
            logging.info(
                (
                    f" Processing : {w_beg + self.pre_pad}-{w_end - self.post_pad} "
                ).center(110, "~")
            )

            try:
                data = self.archive.read_waveform_data(stations, w_beg, w_end)
                time, max_coa, max_coa_n, coord, onset_data = self._compute(data)
                logging.debug(f"1-D con shape : {max_coa.shape}")
                coalescence.append(
                    time, max_coa, max_coa_n, coord, self.lut.unit_conversion_factor
                )
                availability.loc[i] = onset_data.availability
            except (ArchiveEmpty, AllDataRejected, DataGap) as e:
                coalescence.empty(
                    starttime, self.timestep, i, e.msg, self.lut.unit_conversion_factor
                )
                availability.loc[i] = np.zeros(len(availability_cols), dtype=int)

            availability.rename(
                index={i: str(starttime + self.timestep * i)}, inplace=True
            )

        if not coalescence.written:
            coalescence.write()
        write_availability(self.run, availability)

    def _locate_events(self, stations: list[Station], **kwargs: dict) -> None:
        """
        Loop through list of earthquakes read in from trigger results and re-compute
        coalescence; output phase picks, event location and uncertainty, plus optional
        plots and outputs.

        Parameters
        ----------
        stations:
            List of Station objects to be used for event location.
        kwargs:
            Can contain:
            starttime : `obspy.UTCDateTime` object, optional
                Timestamp from which to include events in the locate scan.
            endtime : `obspy.UTCDateTime` object, optional
                Timestamp up to which to include events in the locate scan.
            trigger_file : str, optional
                File containing triggered events to be located.

        """

        triggered_events = read_triggered_events(self.run, **kwargs)
        n_events = len(triggered_events.index)

        self.pre_pad, self.post_pad = self.onset.pad(4 * self.marginal_window)

        for i, triggered_event in triggered_events.iterrows():
            event = Event(self.marginal_window, triggered_event)
            w_beg = event.trigger_time - 2 * self.marginal_window - self.pre_pad
            w_end = event.trigger_time + 2 * self.marginal_window + self.post_pad
            logging.info(util.log_spacer)
            logging.info(f"\tEVENT - {i + 1} of {n_events} - {event.uid}")
            logging.info(util.log_spacer)

            try:
                logging.info("\tReading waveform data...")
                event.add_waveform_data(
                    self._read_event_waveform_data(stations, w_beg, w_end)
                )
                logging.info("\tComputing 4-D coalescence function...")
                event.add_compute_output(*self._compute(event.data, event))
            except (ArchiveEmpty, AllDataRejected, DataGap) as e:
                logging.info(e.msg)
                continue

            if self.write_coalescence:
                logging.info("\tSaving full coalescence map...")
                write_coalescence(self.run, event.map4d, event)

            # --- Trim coalescence map to marginal window ---
            if event.in_marginal_window():
                event.trim2window()
            else:
                del event
                continue

            logging.info("\tDetermining event location and uncertainty...")
            marginalised_coa_map = self._calculate_location(event)

            if self.write_marginal_coalescence:
                logging.info("\tSaving marginalised coalescence map...")
                write_coalescence(
                    self.run, marginalised_coa_map, event, marginalised=True
                )

            plugin_context = {
                "event": event,
                "lut": self.lut,
                "run": self.run,
                "marginalised_coa_map": marginalised_coa_map,
            }
            for plugin in sorted(self.plugins, key=lambda p: p.order):
                out = call_by_signature(plugin.run, plugin_context)
                if isinstance(out, dict):
                    plugin_context.update(out)

            event.write(self.run, self.lut)

            if self.plot_event_video:
                logging.info("Support for event videos coming soon.")

            if self.write_cut_waveforms:
                write_cut_waveforms(
                    self.run,
                    event,
                    self.cut_waveform_format,
                    pre_cut=self.pre_cut,
                    post_cut=self.post_cut,
                )
            if self.write_real_waveforms:
                write_cut_waveforms(
                    self.run,
                    event,
                    self.cut_waveform_format,
                    pre_cut=self.pre_cut,
                    post_cut=self.post_cut,
                    waveform_type="real",
                    units=self.real_waveform_units,
                )
            if self.write_wa_waveforms:
                write_cut_waveforms(
                    self.run,
                    event,
                    self.cut_waveform_format,
                    pre_cut=self.pre_cut,
                    post_cut=self.post_cut,
                    waveform_type="wa",
                    units=self.wa_waveform_units,
                )

            del event, marginalised_coa_map
            logging.info(util.log_spacer)

    @util.timeit("info")
    def _compute(
        self,
        data: WaveformData,
        event: Event | None = None,
    ) -> tuple[
        np.ndarray[UTCDateTime],
        np.ndarray[float],
        np.ndarray[float],
        np.ndarray[float],
        np.ndarray[float],
    ]:
        """
        Compute 3-D coalescence between two time stamps.

        Parameters
        ----------
        data:
            Light class encapsulating data returned by an archive query.
        event:
            Light class encapsulating waveforms, coalescence information, picks and
            location information for a given event.

        Returns
        -------
        times:
            Timestamps for the coalescence data.
        max_coa:
            Coalescence value through time.
        max_coa_n:
            Normalised coalescence value through time.
        coord:
            Location of maximum coalescence through time in input projection space.
        map4d:
            4-D coalescence map.

        Raises
        ------
        LUTMissingPhaseTables
            If traveltime tables for a specific phase are not available.

        """

        # --- Calculate continuous coalescence within 3-D volume ---
        onsets, onset_data = self.onset.calculate_onsets(data)
        try:
            traveltimes = self.lut.serve_traveltimes(
                onset_data.sampling_rate, onset_data.availability
            )
        except KeyError as e:
            missing = e.args[0] if e.args else "<unknown>"
            raise LUTMissingPhaseTables(missing, list(onset_data.phase))

        # Here fsmp and lsmp are used to calculate the length of map4d from the shape of
        # the onset functions --> need to use onset sampling_rate, not scan rate.
        fsmp = util.time2sample(self.pre_pad, onset_data.sampling_rate)
        lsmp = util.time2sample(self.post_pad, onset_data.sampling_rate)
        avail = np.sum([value for _, value in onset_data.availability.items()])
        map4d = migrate(onsets, traveltimes, fsmp, lsmp, avail, self.threads)

        # --- Find continuous peak coalescence in 3-D volume ---
        max_coa, max_coa_n, max_idx = find_max_coa(map4d, self.threads)
        coord = self.lut.index2coord(max_idx, unravel=True)

        if self.run.stage == "detect":
            del map4d
            time = data.starttime + self.pre_pad
            return time, max_coa, max_coa_n, coord, onset_data
        else:
            times = event.mw_times(self.scan_rate)
            return times, max_coa, max_coa_n, coord, map4d, onset_data

    @util.timeit("info")
    def _read_event_waveform_data(
        self, stations: list[Station], w_beg: UTCDateTime, w_end: UTCDateTime
    ) -> WaveformData:
        """
        Read waveform data for a triggered event.

        Parameters
        ----------
        stations:
            List of Station objects to be used for event location.
        w_beg:
            Timestamp from which to read waveform data.
        w_end:
            Timestamp up to which to read waveform data.

        Returns
        -------
        data:
            Light class encapsulating data returned by an archive query.

        """

        # Extra pre- and post-pad default to 0.
        pre_pad = post_pad = 0.0

        # If calculating magnitudes, read in padding required for amplitude measurements
        mag_plugin = next(
            (p for p in self.plugins if p.kind == "magnitudes"),
            None,
        )

        if mag_plugin is not None:
            pre_pad, post_pad = mag_plugin.amp.pad(
                self.marginal_window,
                self.lut.max_traveltime,
                self.lut.fraction_tt,
            )

        # If a specific pre / post cut has been requested by the user,
        # check which is bigger.
        if self.pre_cut:
            pre_pad = max(pre_pad, self.pre_cut)
        if self.post_cut:
            post_pad = max(post_pad, self.post_cut)

        # Trim the pre_pad and post_pad to avoid cutting more data than we need; only
        # subtract 1*marginal_window so that if the event otime moves by this much (the
        # maximum allowed) from the triggered event time, we still have the correct
        # window of data to apply the pre_cut.
        pre_pad = max(0.0, pre_pad - self.marginal_window - self.pre_pad)
        post_pad = max(0.0, post_pad - self.marginal_window - self.post_pad)

        logging.debug(f"{w_beg}, {w_end}, {pre_pad}, {post_pad}")
        return self.archive.read_waveform_data(
            stations, w_beg, w_end, pre_pad, post_pad
        )

    @util.timeit("info")
    def _calculate_location(self, event: Event) -> np.ndarray:
        """
        Marginalise the 4-D coalescence grid and calculate a set of locations and
        associated uncertainties by:
            (1) fitting a spline function to a region around the maximum coalescence
                location in the marginalised coalescence map;
            (2) smoothing and fitting a Gaussian function to a region around the maximum
                coalescence location in the marginalised coalescence map;
            (3) calculating the covariance of the entire marginalised coalescence map.

        Parameters
        ----------
        event:
            Light class encapsulating waveforms, coalescence information, picks and
            location information for a given event.

        Returns
        -------
        marginal_coalescence:
            Spatial coalescence map, marginalised over time.

        """

        # --- Marginalise and normalise the coalescence grid ---
        marginal_coalescence = np.sum(event.map4d, axis=-1)
        marginal_coalescence /= np.nanmax(marginal_coalescence)

        # --- Determine best-fitting interpolated spline location ---
        event.add_spline_location(
            spline_location(self.lut, np.copy(marginal_coalescence))
        )

        # --- Determine best-fitting Gaussian location and uncertainty ---
        event.add_gaussian_location(
            *gaussian_fit(self.lut, gaussian_filter(np.copy(marginal_coalescence)))
        )

        # --- Determine global covariance location and uncertainty ---
        event.add_covariance_location(
            *covariance_fit(self.lut, np.copy(marginal_coalescence))
        )

        return marginal_coalescence

    # --- Deprecation/Future handling ---
    @property
    def scan_rate(self):
        """Get scan_rate"""
        return self._scan_rate

    @scan_rate.setter
    def scan_rate(self, value):
        if value is None:
            return
        elif value == self.onset.sampling_rate:
            self._scan_rate = value
            return
        print(
            "Warning: Parameter not yet user-configurable. Currently\n"
            "the scan sampling rate must be the same as the onset sampling\n"
            f"rate, which you have set to {self.scan_rate} Hz. Please\n"
            "contact the QuakeMigrate developers for further info."
        )

    @property
    def sampling_rate(self):
        """Get sampling_rate"""
        return self.scan_rate

    @sampling_rate.setter
    def sampling_rate(self, value):
        if value is None:
            return
        print(
            "Warning: Parameter name has changed - continuing. Currently\n"
            "the scan sampling rate must be the same as the onset sampling\n"
            f"rate, which you have set to {self.scan_rate} Hz. Please\n"
            "contact the QuakeMigrate developers for further info."
        )

    @property
    def time_step(self):
        """Handler for deprecated attribute name 'time_step'"""
        return self.timestep

    @time_step.setter
    def time_step(self, value):
        if value is None:
            return
        print(
            "FutureWarning: Parameter name has changed - continuing.\n"
            "To remove this message, change:\n"
            "\t'time_step' -> 'timestep'"
        )
        self.timestep = value

    @property
    def n_cores(self):
        """Handler for deprecated attribute name 'n_cores'"""
        return self.threads

    @n_cores.setter
    def n_cores(self, value):
        if value is None:
            return
        print(
            "FutureWarning: Parameter name has changed - continuing.\n"
            "To remove this message, change:\n"
            "\t'n_cores' -> 'threads'"
        )
        self.threads = value
