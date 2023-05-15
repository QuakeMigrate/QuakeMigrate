# -*- coding: utf-8 -*-
"""
Module for processing waveform files stored in a data archive.

:copyright:
    2020–2023, QuakeMigrate developers.
:license:
    GNU General Public License, Version 3
    (https://www.gnu.org/licenses/gpl-3.0.html)

"""

from itertools import chain
import logging
import pathlib

from obspy import read, Stream, UTCDateTime

import quakemigrate.util as util


class Archive:
    """
    The Archive class handles the reading of archived waveform data.

    It is capable of handling any regular archive structure. Requests to read waveform
    data are served up as a :class:`~quakemigrate.io.data.WaveformData` object.

    If provided, a response inventory for the archive will be stored with the waveform
    data for response removal, if needed (e.g. for local magnitude calculation, or to
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
    archive_path : str
        Location of seismic data archive: e.g.: "./DATA_ARCHIVE".
    stations : `pandas.DataFrame` object
        Station information.
        Columns ["Latitude", "Longitude", "Elevation", "Name"].
        See :func:`~quakemigrate.io.core.read_stations`
    archive_format : str, optional
        Sets directory structure and file naming format for different archive formats.
        See :func:`~quakemigrate.io.data.Archive.path_structure`
    kwargs : **dict
        See Archive Attributes for details.

    Attributes
    ----------
    archive_path : `pathlib.Path` object
        Location of seismic data archive: e.g.: ./DATA_ARCHIVE.
    stations : `pandas.Series` object
        Series object containing station names.
    format : str
        Directory structure and file naming format of data archive.
    read_all_stations : bool, optional
        If True, read all stations in archive for that time period. Else, only read
        specified stations.
    resample : bool, optional
        If true, perform resampling of data which cannot be decimated directly to the
        desired sampling rate. See :func:`~quakemigrate.util.resample`
    response_inv : `obspy.Inventory` object, optional
        ObsPy response inventory for this waveform archive, containing response
        information for each channel of each station of each network.
    pre_filt : tuple of floats
        Pre-filter to apply during the instrument response removal. E.g.
        (0.03, 0.05, 30., 35.) - all in Hz. (Default None)
    water_level : float
        Water level to use in instrument response removal. (Default 60.)
    remove_full_response : bool
        Whether to remove the full response (including the effect of digital FIR
        filters) or just the instrument transform function (as defined by the PolesZeros
        Response Stage). Significantly slower. (Default False)
    upfactor : int, optional
        Factor by which to upsample the data to enable it to be decimated to the desired
        sampling rate, e.g. 40Hz -> 50Hz requires upfactor = 5.
        See :func:`~quakemigrate.util.resample`
    interpolate : bool, optional
        If data is timestamped "off-sample" (i.e. a non-integer number of samples after
        midnight), whether to interpolate the data to apply the necessary correction.
        Default behaviour is to just alter the metadata, resulting in a sub-sample
        timing offset. See :func:`~quakemigrate.util.shift_to_sample`.

    Methods
    -------
    path_structure(archive_format, channels="*")
        Set the directory structure and file naming format of the data archive.
    read_waveform_data(starttime, endtime)
        Read in waveform data between two times.

    """

    def __init__(self, archive_path, stations, archive_format=None, **kwargs):
        """Instantiate the Archive object."""

        self.archive_path = pathlib.Path(archive_path)
        self.stations = stations["Name"]
        if archive_format:
            channels = kwargs.get("channels", "*")
            self.path_structure(archive_format, channels)
        else:
            self.format = kwargs.get("format")

        self.read_all_stations = kwargs.get("read_all_stations", False)
        # Resampling parameters
        self.resample = kwargs.get("resample", False)
        self.upfactor = kwargs.get("upfactor")
        self.interpolate = kwargs.get("interpolate", False)
        # Response removal parameters
        self.response_inv = kwargs.get("response_inv")
        response_removal_params = kwargs.get("response_removal_params", {})
        if self.response_inv and "water_level" not in response_removal_params.keys():
            print(  # Logger not yet spun up
                "Warning: 'water level' for instrument correction not "
                "specified. Set to default: 60"
            )
        self.water_level = response_removal_params.get("water_level", 60.0)
        self.pre_filt = response_removal_params.get("pre_filt")
        self.remove_full_response = response_removal_params.get(
            "remove_full_response", False
        )

    def __str__(self, response_only=False):
        """
        Returns a short summary string of the Archive object.

        Parameters
        ----------
        response_only : bool, optional
            Whether to just output the a string describing the instrument response
            parameters.

        Returns
        -------
        out : str
            Summary string.

        """

        if self.response_inv:
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
                f"\n\tArchive path\t:\t{self.archive_path}"
                f"\n\tPath structure\t:\t{self.format}"
                f"\n\tResampling\t:\t{self.resample}"
            )
            if self.upfactor:
                out += f"\n\tUpfactor\t:\t{self.upfactor}"
            out += "\n\tStations:"
            for station in self.stations:
                out += f"\n\t\t{station}"
            out += f"\n{response_str}"
        else:
            out = response_str

        return out

    def path_structure(self, archive_format="YEAR/JD/STATION", channels="*"):
        """
        Define the directory structure and file naming format of the data archive.

        Parameters
        ----------
        archive_format : str, optional
            Directory structure and file naming format of the data archive. This may be
            the name of a generic archive format (e.g. SeisComp3), or one of a selection
            of additional formats built into QuakeMigrate.
        channels : str, optional
            Channel codes to include. E.g. channels="[B,H]H*". (Default "*")

        Raises
        ------
        ArchivePathStructureError
            If the `archive_format` specified by the user is not a valid option.

        """

        if archive_format == "SeisComp3":
            self.format = (
                "{year}/*/{station}/" + channels + "/*.{station}.*.*.D."
                "{year}.{jday:03d}"
            )
        elif archive_format == "YEAR/JD/*_STATION_*":
            self.format = "{year}/{jday:03d}/*_{station}_*"
        elif archive_format == "YEAR/JD/STATION":
            self.format = "{year}/{jday:03d}/{station}*"
        elif archive_format == "STATION.YEAR.JULIANDAY":
            self.format = "*{station}.*.{year}.{jday:03d}"
        elif archive_format == "/STATION/STATION.YearMonthDay":
            self.format = "{station}/{station}.{year}{month:02d}{day:02d}"
        elif archive_format == "YEAR_JD/STATION*":
            self.format = "{year}_{jday:03d}/{station}*"
        elif archive_format == "YEAR_JD/STATION_*":
            self.format = "{year}_{jday:03d}/{station}_*"
        else:
            raise util.ArchivePathStructureError(archive_format)

    def read_waveform_data(self, starttime, endtime, pre_pad=0.0, post_pad=0.0):
        """
        Read in waveform data from the archive between two times.

        Supports all formats currently supported by ObsPy, including: "MSEED", "SAC",
        "SEGY", "GSE2".

        Optionally, read data with some pre- and post-pad, and for all stations in the
        archive - this will be stored in `data.raw_waveforms`, while `data.waveforms`
        will contain only data for selected stations between `starttime` and `endtime`.

        Parameters
        ----------
        starttime : `obspy.UTCDateTime` object
            Timestamp from which to read waveform data.
        endtime : `obspy.UTCDateTime` object
            Timestamp up to which to read waveform data.
        pre_pad : float, optional
            Additional pre pad of data to read. Defaults to 0.
        post_pad : float, optional
            Additional post pad of data to read. Defaults to 0.

        Returns
        -------
        data : :class:`~quakemigrate.io.data.WaveformData` object
            Object containing the waveform data read from the archive that satisfies the
            query.

        Raises
        ------
        ArchiveEmptyException
            If no data files are found in the archive for this day(s).
        DataAvailabilityException
            If no data is found in the archive for the specified stations within the
            specified time window.

        """

        # Ensure pre-pad and post-pad are not negative.
        pre_pad = max(0.0, pre_pad)
        post_pad = max(0.0, post_pad)

        data = WaveformData(
            starttime=starttime,
            endtime=endtime,
            stations=self.stations,
            read_all_stations=self.read_all_stations,
            resample=self.resample,
            upfactor=self.upfactor,
            response_inv=self.response_inv,
            water_level=self.water_level,
            pre_filt=self.pre_filt,
            remove_full_response=self.remove_full_response,
            pre_pad=pre_pad,
            post_pad=post_pad,
        )

        files = self._load_from_path(starttime - pre_pad, endtime + post_pad)

        st = Stream()
        try:
            first = next(files)
            files = chain([first], files)
            for file in files:
                file = str(file)
                try:
                    read_start = starttime - pre_pad
                    read_end = endtime + post_pad
                    st += read(
                        file,
                        starttime=read_start,
                        endtime=read_end,
                        nearest_sample=True,
                    )
                except TypeError:
                    logging.info(f"File not compatible with ObsPy - {file}")
                    continue

            # Merge waveforms channel-by-channel with no-clobber merge
            st = util.merge_stream(st)

            # Make copy of raw waveforms to output if requested
            data.raw_waveforms = st.copy()

            # Ensure data is timestamped "on-sample" (i.e. an integer number of samples
            # after midnight). Otherwise the data will be implicitly shifted when it is
            # used to calculate the onset function / migrated.
            st = util.shift_to_sample(st, interpolate=self.interpolate)

            if self.read_all_stations:
                # Re-populate st with only stations in station file
                st_selected = Stream()
                for station in self.stations:
                    st_selected += st.select(station=station)
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
                raise util.DataGapException

            # Add cleaned stream to `waveforms`
            data.waveforms = st

        except StopIteration:
            raise util.ArchiveEmptyException

        return data

    def _load_from_path(self, starttime, endtime):
        """
        Retrieves available files between two times.

        Parameters
        ----------
        starttime : `obspy.UTCDateTime` object
            Timestamp from which to read waveform data.
        endtime : `obspy.UTCDateTime` object
            Timestamp up to which to read waveform data.

        Returns
        -------
        files : generator
            Iterator object of available waveform data files.

        Raises
        ------
        ArchiveFormatException
            If the Archive.format attribute has not been set.

        """

        if self.format is None:
            raise util.ArchiveFormatException

        # Loop through time period by day adding files to list
        # NOTE! This assumes the archive structure is split into days.
        files = []
        loadstart = UTCDateTime(starttime)
        while loadstart < endtime:
            temp_format = self.format.format(
                year=loadstart.year,
                month=loadstart.month,
                day=loadstart.day,
                jday=loadstart.julday,
                station="{station}",
                dtime=loadstart,
            )
            if self.read_all_stations is True:
                file_format = temp_format.format(station="*")
                file_format = file_format.replace("**", "*")
                files = chain(files, self.archive_path.glob(file_format))
            else:
                for station in self.stations:
                    file_format = temp_format.format(station=station)
                    files = chain(files, self.archive_path.glob(file_format))
            loadstart = UTCDateTime(loadstart.date) + 86400

        return files


class WaveformData:
    """
    The WaveformData class encapsulates the waveform data returned by an Archive query.

    It also provides a number of utility functions. These include removing instrument
    response and checking data availability against a flexible set of data quality
    criteria.

    Parameters
    ----------
    starttime : `obspy.UTCDateTime` object
        Timestamp of first sample of waveform data requested from the archive.
    endtime : `obspy.UTCDateTime` object
        Timestamp of last sample of waveform data requested from the archive.
    stations : `pandas.Series` object, optional
        Series object containing station names.
    read_all_stations : bool, optional
        If True, `raw_waveforms` contain all stations in archive for that time period.
        Else, only selected stations will be included.
    resample : bool, optional
        If true, allow resampling of data which cannot be decimated directly to the
        desired sampling rate. See :func:`~quakemigrate.util.resample`
        Default: False
    upfactor : int, optional
        Factor by which to upsample the data to enable it to be decimated to the desired
        sampling rate, e.g. 40Hz -> 50Hz requires upfactor = 5.
        See :func:`~quakemigrate.util.resample`
    response_inv : `obspy.Inventory` object, optional
        ObsPy response inventory for this waveform data, containing response information
        for each channel of each station of each network.
    pre_filt : tuple of floats
        Pre-filter to apply during the instrument response removal. E.g.
        (0.03, 0.05, 30., 35.) - all in Hz. (Default None)
    water_level : float
        Water level to use in instrument response removal. (Default 60.)
    remove_full_response : bool
        Whether to remove the full response (including the effect of digital FIR
        filters) or just the instrument transform function (as defined by the PolesZeros
        Response Stage). Significantly slower.
        (Default False)
    pre_pad : float, optional
        Additional pre pad of data included in `raw_waveforms`.
    post_pad : float, optional
        Additional post pad of data included in `raw_waveforms`.

    Attributes
    ----------
    starttime : `obspy.UTCDateTime` object
        Timestamp of first sample of waveform data requested from the archive.
    endtime : `obspy.UTCDateTime` object
        Timestamp of last sample of waveform data requested from the archive.
    stations : `pandas.Series` object
        Series object containing station names.
    read_all_stations : bool
        If True, `raw_waveforms` contain all stations in archive for that time period.
        Else, only selected stations will be included.
    raw_waveforms : `obspy.Stream` object
        Raw seismic data read in from the archive. This may be for all stations in the
        archive, or only those specified by the user. See `read_all_stations`. It may
        also cover the time period between `starttime` and `endtime`, or feature an
        additional pre- and post-pad. See `pre_pad` and `post_pad`.
    waveforms : `obspy.Stream` object
        Seismic data read in from the archive for the specified list of stations,
        between `starttime` and `endtime`.
    pre_pad : float
        Additional pre pad of data included in `raw_waveforms`.
    post_pad : float
        Additional post pad of data included in `raw_waveforms`.

    Methods
    -------
    check_availability(stream, **data_quality_params)
        Check data availability against a set of data quality criteria.
    get_wa_waveform(trace, **response_removal_params)
        Calculate the Wood-Anderson corrected waveform for a `obspy.Trace` object.

    Raises
    ------
    NotImplementedError
        If the user attempts to use the get_real_waveform() method.

    """

    def __init__(
        self,
        starttime,
        endtime,
        stations=None,
        response_inv=None,
        water_level=60.0,
        pre_filt=None,
        remove_full_response=False,
        read_all_stations=False,
        resample=False,
        upfactor=None,
        pre_pad=0.0,
        post_pad=0.0,
    ):
        """Instantiate the WaveformData object."""

        self.starttime = starttime
        self.endtime = endtime
        self.stations = stations
        self.response_inv = response_inv
        self.water_level = water_level
        self.pre_filt = pre_filt
        self.remove_full_response = remove_full_response

        self.read_all_stations = read_all_stations
        self.resample = resample
        self.upfactor = upfactor
        self.pre_pad = pre_pad
        self.post_pad = post_pad

        self.raw_waveforms = None
        self.waveforms = Stream()
        self.wa_waveforms = None
        self.real_waveforms = None

    def check_availability(
        self,
        st,
        all_channels=False,
        n_channels=None,
        allow_gaps=False,
        full_timespan=True,
        check_sampling_rate=False,
        sampling_rate=None,
        check_start_end_times=False,
    ):
        """
        Check waveform availability against data quality criteria.

        There are a number of hard-coded checks: for whether any data is present; for
        whether the data is a flatline (all samples have the same value); and for
        whether the data contains overlaps. There are a selection of additional optional
        checks which can be specified according to the onset function / user preference.

        Parameters
        ----------
        st : `obspy.Stream` object
            Stream containing the waveform data to check against the availability
            criteria.
        all_channels : bool, optional
            Whether all supplied channels (distinguished by SEED id) need to meet the
            availability criteria to mark the data as 'available'.
        n_channels : int, optional
            If `all_channels=True`, this argument is required (in order to specify the
            number of channels expected to be present).
        allow_gaps : bool, optional
            Whether to allow gaps.
        full_timespan : bool, optional
            Whether to ensure the data covers the entire timespan requested; note that
            this implicitly requires that there be no gaps. Checks the number of samples
            in the trace, not the start and end times; for that see
            `check_start_end_times`.
        check_sampling_rate : bool, optional
            Check that all channels are at the desired sampling rate.
        sampling_rate : float, optional
            If `check_sampling_rate=True`, this argument is required to specify the
            sampling rate that the data should be at.
        check_start_end_times : bool, optional
            A stricter alternative to `full_timespan`; checks that the first and last
            sample of the trace have exactly the requested timestamps.

        Returns
        -------
        available : int
            0 if data doesn't meet the availability requirements; 1 if it does.
        availability : dict
            Dict of {tr_id : available} for each unique SEED ID in the input stream
            (available is again 0 or 1).

        Raises
        ------
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
                    n_samples = timespan * st_id[0].stats.sampling_rate + 1
                    if len(st_id) > 1:
                        continue
                    elif st_id[0].stats.npts < n_samples:
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

    def get_real_waveform(self, tr, velocity=True):
        """
        Calculate the real waveform for a Trace by removing the instrument response.

        Parameters
        ----------
        tr : `obspy.Trace` object
            Trace containing the waveform for which to remove the instrument response.
        velocity : bool, optional
            Output velocity waveform (as opposed to displacement).
            Default: True.

        Returns
        -------
        tr : `obspy.Trace` object
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

        if not self.response_inv:
            raise AttributeError("No response inventory provided!")

        # Copy the Trace before operating on it
        tr = tr.copy()
        tr.detrend("linear")

        if not self.remove_full_response:
            # Just remove the response encapsulated in the instrument transfer function
            # (stored as a PolesAndZeros response). NOTE: this does not account for the
            # effect of the digital FIR filters applied to the recorded waveforms.
            # However, due to this it is significantly faster to compute.
            try:
                response = self.response_inv.get_response(tr.id, tr.stats.starttime)
            except Exception as e:
                raise util.ResponseNotFoundError(str(e), tr.id)

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
                tr.simulate(
                    paz_remove=paz_dict,
                    pre_filt=self.pre_filt,
                    water_level=self.water_level,
                    taper=True,
                    sacsim=True,  # To replicate remove_response()
                    pitsasim=False,  # To replicate remove_response()
                )
            except ValueError as e:
                raise util.ResponseRemovalError(e, tr.id)
        else:
            # Use remove_response(), which removes the effect of _all_ response stages,
            # including the FIR stages. Considerably slower.
            output = "VEL" if velocity else "DISP"

            try:
                tr.remove_response(
                    inventory=self.response_inv,
                    output=output,
                    pre_filt=self.pre_filt,
                    water_level=self.water_level,
                    taper=True,
                )
            except ValueError as e:
                raise util.ResponseRemovalError(e, tr.id)

        try:
            self.real_waveforms.append(tr.copy())
        except AttributeError:
            self.real_waveforms = Stream()
            self.real_waveforms.append(tr.copy())

        return tr

    def get_wa_waveform(self, tr, velocity=False):
        """
        Calculate simulated Wood Anderson displacement waveform for a Trace.

        Parameters
        ----------
        tr : `obspy.Trace` object
            Trace containing the waveform to be corrected to a Wood-Anderson response.
        velocity : bool, optional
            Output velocity waveform, instead of displacement. Default: False.
            NOTE: all attenuation functions provided within the QM local_mags module are
            calculated for displacement seismograms.

        Returns
        -------
        tr : `obspy.Trace` object
            Trace corrected to Wood-Anderson response.

        """

        # Copy the Trace before operating on it
        tr = tr.copy()
        tr.detrend("linear")

        # Remove instrument response
        tr = self.get_real_waveform(tr, velocity)

        # Simulate Wood-Anderson response
        tr.simulate(
            paz_simulate=util.wa_response(obspy_def=True),
            pre_filt=self.pre_filt,
            water_level=self.water_level,
            taper=True,
            sacsim=True,  # To replicate remove_response()
            pitsasim=False,  # To replicate remove_response()
        )

        try:
            self.wa_waveforms.append(tr.copy())
        except AttributeError:
            self.wa_waveforms = Stream()
            self.wa_waveforms.append(tr.copy())

        return tr
