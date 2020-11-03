# -*- coding: utf-8 -*-
"""
Module for processing waveform files stored in a data archive.

"""

from itertools import chain
import logging
import pathlib

from obspy import read, Stream, Trace, UTCDateTime

import quakemigrate.util as util


class Archive:
    """
    The Archive class handles the reading of archived waveform data.

    It is capable of handling any regular archive structure. Requests to read
    waveform data are served up as a `quakemigrate.data.WaveformData` object.

    If provided, a response inventory provided for the archive will be stored
    with the waveform data for response removal, if needed.

    By default, data with mismatched sampling rates will only be decimated.
    If necessary, and if the user specifies `resample = True` and an
    upfactor to upsample by `upfactor = int` for the waveform archive, data
    can also be upsampled and then, if necessary, subsequently decimated to
    achieve the desired sampling rate.

    For example, for raw input data sampled at a mix of 40, 50 and 100 Hz,
    to achieve a unified sampling rate of 50 Hz, the user would have to
    specify an upfactor of 5; 40 Hz x 5 = 200 Hz, which can then be
    decimated to 50 Hz.

    NOTE: data will be detrended and a cosine taper applied before
    decimation, in order to avoid edge effects when applying the lowpass
    filter.

    Parameters
    ----------
    archive_path : str
        Location of seismic data archive: e.g.: ./DATA_ARCHIVE.
    stations : `pandas.DataFrame` object
        Station information.
        Columns ["Latitude", "Longitude", "Elevation", "Name"]
    archive_format : str, optional
        Sets path type for different archive formats.
    kwargs : **dict
        See Archive Attributes for details.

    Attributes
    ----------
    archive_path : `pathlib.Path` object
        Location of seismic data archive: e.g.: ./DATA_ARCHIVE.
    stations : `pandas.Series` object
        Series object containing station names.
    format : str
        File naming format of data archive.
    read_all_stations : bool, optional
        If True, read all stations in archive for that time period. Else, only
        read specified stations.
    resample : bool, optional
        If true, perform resampling of data which cannot be decimated directly
        to the desired sampling rate.
    response_inv : `obspy.Inventory` object, optional
        ObsPy response inventory for this waveform archive, containing
        response information for each channel of each station of each network.
    upfactor : int, optional
        Factor by which to upsample the data to enable it to be decimated to
        the desired sampling rate, e.g. 40Hz -> 50Hz requires upfactor = 5.

    Methods
    -------
    path_structure(path_type="YEAR/JD/STATION")
        Set the file naming format of the data archive.
    read_waveform_data(starttime, endtime, sampling_rate)
        Read in all waveform data between two times, decimate / resample if
        required to reach desired sampling rate. Return all raw data as an
        obspy Stream object and processed data for specified stations as an
        array for use by QuakeScan to calculate onset functions for migration.

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
        self.response_inv = kwargs.get("response_inv")
        self.resample = kwargs.get("resample", False)
        self.upfactor = kwargs.get("upfactor")

    def __str__(self):
        """Returns a short summary string of the Archive object."""

        out = ("QuakeMigrate Archive object"
               f"\n\tArchive path\t:\t{self.archive_path}"
               f"\n\tPath structure\t:\t{self.format}"
               f"\n\tResampling\t:\t{self.resample}")
        if self.upfactor:
            out += f"\n\tUpfactor\t:\t{self.upfactor}"
        out += "\n\tStations:"
        for station in self.stations:
            out += f"\n\t\t{station}"

        return out

    def path_structure(self, archive_format="YEAR/JD/STATION", channels="*"):
        """
        Define the path structure of the data archive.

        Parameters
        ----------
        archive_format : str, optional
            Sets path type for different archive formats.
        channels : str, optional
            Channel codes to include. E.g. channels="[B,H]H*". (Default "*")

        Raises
        ------
        ArchivePathStructureError
            If the archive_format specified by the user is not a valid option.

        """

        if archive_format == "SeisComp3":
            self.format = ("{year}/*/{station}/"+channels+"/*.{station}.*.*.D."
                           "{year}.{jday:03d}")
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

    def read_waveform_data(self, starttime, endtime, pre_pad=0., post_pad=0.):
        """
        Read in the waveform data for all stations in the archive between two
        times.

        Supports all formats currently supported by ObsPy, including: "MSEED"
        (default), "SAC", "SEGY", "GSE2" .

        Parameters
        ----------
        starttime : `obspy.UTCDateTime` object, optional
            Timestamp from which to read waveform data.
        endtime : `obspy.UTCDateTime` object, optional
            Timestamp up to which to read waveform data.
        pre_pad : float, optional
            Additional pre pad of data to cut based on user-defined pre_cut
            parameter. Defaults to none: pre_pad calculated in QuakeScan will
            be used (included in starttime).
        post_pad : float, optional
            Additional post pad of data to cut based on user-defined post_cut
            parameter. Defaults to none: post_pad calculated in QuakeScan will
            be used (included in endtime).

        Returns
        -------
        data : :class:`~quakemigrate.io.data.WaveformData` object
            Object containing the archived data that satisfies the query.

        """

        data = WaveformData(starttime=starttime, endtime=endtime,
                            stations=self.stations,
                            read_all_stations=self.read_all_stations,
                            resample=self.resample, upfactor=self.upfactor,
                            response_inv=self.response_inv,
                            pre_pad=pre_pad, post_pad=post_pad)

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
                    st += read(file, starttime=read_start, endtime=read_end)
                except TypeError:
                    logging.info(f"File not compatible with ObsPy - {file}")
                    continue

            # Merge all traces with contiguous data, or overlapping data which
            # exactly matches (== st._cleanup(); i.e. no clobber)
            st.merge(method=-1)

            # Make copy of raw waveforms to output if requested
            data.raw_waveforms = st.copy()

            # Re-populate st with only stations in station file, and only
            # data between start and end time needed for QuakeScan
            st_selected = Stream()
            for station in self.stations:
                st_selected += st.select(station=station)
            st = st_selected.copy()
            for tr in st:
                tr.trim(starttime=starttime, endtime=endtime)
                if not bool(tr):
                    st.remove(tr)
            del st_selected

            # Test if the stream is completely empty
            # (see __nonzero__ for obspy Stream object)
            if not bool(st):
                raise util.DataGapException

            # Pass stream to be processed and added to data.signal. This
            # processing includes resampling and determining the availability
            # of the desired stations.
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
            temp_format = self.format.format(year=loadstart.year,
                                             month=loadstart.month,
                                             day=loadstart.day,
                                             jday=loadstart.julday,
                                             station="{station}",
                                             dtime=loadstart)
            if self.read_all_stations is True:
                file_format = temp_format.format(station="*")
                files = chain(files, self.archive_path.glob(file_format))
            else:
                for station in self.stations:
                    file_format = temp_format.format(station=station)
                    files = chain(files, self.archive_path.glob(file_format))
            loadstart = UTCDateTime(loadstart.date) + 86400

        return files


class WaveformData:
    """
    The WaveformData class encapsulates the waveform data returned by an
    Archive query.

    It also provides a framework in which to store processed and/or
    filtered data generated during onset function calculation, and a number
    of utility functions. These include removing instrument response and
    checking data availability.

    Parameters
    ----------
    starttime : `obspy.UTCDateTime` object
        Timestamp of first sample of waveform data.
    endtime : `obspy.UTCDateTime` object
        Timestamp of last sample of waveform data.
    stations : `pandas.Series` object, optional
        Series object containing station names.
    read_all_stations : bool, optional
        If True, raw_waveforms contain all stations in archive for that time
        period. Else, only selected stations will be included.
    resample : bool, optional
        If true, allow resampling of data which cannot be decimated directly
        to the desired sampling rate.
    upfactor : int, optional
        Factor by which to upsample the data to enable it to be decimated to
        the desired sampling rate, e.g. 40Hz -> 50Hz requires upfactor = 5.
    response_inv : `obspy.Inventory` object, optional
        ObsPy response inventory for this waveform archive, containing
        response information for each channel of each station of each network.
    pre_pad : float, optional
        Additional pre pad of data cut based on user-defined pre_cut
        parameter.
    post_pad : float, optional
        Additional post pad of data cut based on user-defined post_cut
        parameter.

    Attributes
    ----------
    starttime : `obspy.UTCDateTime` object
        Timestamp of first sample of waveform data.
    endtime : `obspy.UTCDateTime` object
        Timestamp of last sample of waveform data.
    stations : `pandas.Series` object
        Series object containing station names.
    read_all_stations : bool
        If True, raw_waveforms contain all stations in archive for that time
        period. Else, only selected stations will be included.
    raw_waveforms : `obspy.Stream` object
        Raw seismic data found and read in from the archive within the
        specified time period. This may be for all stations in the archive,
        or only those specified by the user. See `read_all_stations`.
    waveforms : `obspy.Stream` objexct
        Seismic data found and read in from the archive within the specified
        time period from the specified list of stations.
    filtered_waveforms : `obspy.Stream` object
        Filtered and/or resampled and otherwise processed seismic data
        generated during onset function generation. This may have been further
        sorted based on additional quality control criteria.
    pre_pad : float
        Additional pre pad of data cut based on user-defined pre_cut
        parameter.
    post_pad : float
        Additional post pad of data cut based on user-defined post_cut
        parameter.
    availability : dict
        Dictionary with keys "station.phase", containing 1's or 0's
        corresponding to whether data is available to calculate an onset
        function according to the criteria set out (trace availability,
        sampling frequency, presence of gaps, etc.)

    Methods
    -------
    check_availability(stream, **data_quality_params)
        Check data availability against a set of data quality criteria.
    get_wa_waveform(trace, **response_removal_params)
        Calculate the Wood-Anderson corrected waveform for a `obspy.Trace`
        object.
    times
        Utility function to generate the corresponding timestamps for the
        waveform and coalescence data.

    Raises
    ------
    NotImplementedError
        If the user attempts to use the get_real_waveform() method.

    """

    def __init__(self, starttime, endtime, stations=None, response_inv=None,
                 read_all_stations=False, resample=False, upfactor=None,
                 pre_pad=0., post_pad=0.):
        """Instantiate the WaveformData object."""

        self.starttime = starttime
        self.endtime = endtime
        self.stations = stations
        self.response_inv = response_inv

        self.read_all_stations = read_all_stations
        self.resample = resample
        self.upfactor = upfactor
        self.pre_pad = pre_pad
        self.post_pad = post_pad

        self.raw_waveforms = None
        self.waveforms = Stream()
        self.filtered_waveforms = Stream()
        self.onsets = {}
        self.availability = None
        self.wa_waveforms = None

    def check_availability(self, st, all_channels=False, n_channels=None,
                           allow_gaps=False, full_timespan=True):
        """
        Check waveform availability against data quality criteria. There are a
        number of hard-coded checks: for whether any data is present; for
        whether the data is a flatline (all samples have the same value); and
        for whether the data contains overlaps. There are a selection of
        additional optional checks which can be specified according to the
        onset function / user preference.

        Parameters
        ----------
        st : `obspy.Stream` object
            Stream containing the waveform data to check against the
            availability criteria.
        all_channels : bool, optional
            Whether all supplied channels (distinguished by SEED id) need to
            meet the availability criteria to mark the data as 'available'.
        n_channels : int, optional
            If `all_channels=True`, this argument is required (in order to
            specify the number of channels expected to be present).
        allow_gaps : bool, optional
            Whether to allow gaps.
        full_timespan : bool, optional
            Whether to ensure the data covers the entire timespan requested;
            note that this implicitly requires that there be no gaps.

        Returns
        -------
        available : int
            0 if data doesn't meet the availability requirements; 1 if it does.
        availability : dict
            Dict of {tr_id : available} for each unique SEED ID in the input
            stream (available is again 0 or 1).

        Raises
        ------
        TypeError
            If the user specifies `all_channels=True` but does not specify
            `n_channels`.

        """

        availability = {}
        available = 0

        # Check if any channels in stream
        if bool(st):
            # Loop through channels with unique SEED id's
            for tr_id in sorted([tr.id for tr in st]):
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
                # Check data covers full timespan (if requested)
                if full_timespan:
                    if len(st_id) > 1:
                        continue
                    elif st_id[0].stats.starttime != self.starttime or \
                        st_id[0].stats.endtime != self.endtime:
                        continue
                # If passed all tests, set availability to 1
                availability[tr_id] = 1

            # Return availability based on "all_channels" setting
            if all(ava == 1 for ava in availability.values()):
                if all_channels:
                    # If all_channels requested, must also check that the
                    # expected number of channels are present
                    if not n_channels:
                        raise TypeError("Please specify n_channels if you wish"
                                        " to check all channels meet the "
                                        "availability criteria.")
                    elif len(availability) == n_channels:
                            available = 1
                else:
                    available = 1
            elif not all_channels \
                and any(ava == 1 for ava in availability.values()):
                available = 1

        return available, availability

    def get_real_waveforms(self, tr, remove_full_response=False, velocity=True):
        """
        Coming soon.

        """

        raise NotImplementedError("Coming soon. Please contact the "
                                  "QuakeMigrate developers.")

    def get_wa_waveform(self, tr, water_level, pre_filt,
                        remove_full_response=False, velocity=False):
        """
        Calculate simulated Wood Anderson displacement waveform for a Trace.

        Parameters
        ----------
        tr : `obspy.Trace` object
            Trace containing the waveform to be corrected to a Wood-Anderson
            response
        water_level : float
            Water-level to be used in the instrument correction.
        pre_filt : tuple of floats, or None
            Filter corners describing filter to be applied to the trace before
            deconvolution. E.g. (0.05, 0.06, 30, 35) (in Hz)
        remove_full_response : bool, optional
            Remove all response stages, inc. FIR (st.remove_response()), not
            just poles-and-zero response stage. Default: False.
        velocity : bool, optional
            Output velocity waveform, instead of displacement. Default: False.

        Returns
        -------
        tr : `obspy.Trace` object
            Trace corrected to Wood-Anderson response.

        Raises
        ------
        AttributeError
            If no response inventory has been supplied.
        ResponseNotFoundError
            If the response information for a trace can't be found in the
            supplied response inventory.
        ResponseRemovalError
            If the deconvolution of the instrument response and simulation of
            the Wood-Anderson response is unsuccessful.
        NotImplementedError
            If the user selects velocity=True.

        """

        if not self.response_inv:
            raise AttributeError("No response inventory provided!")

        # Copy the Trace before operating on it
        tr = tr.copy()
        tr.detrend('linear')

        if velocity is True:
            msg = ("Only displacement WA waveforms are currently available. "
                   "Please contact the QuakeMigrate developers.")
            raise NotImplementedError(msg)

        if not remove_full_response:
            # Just remove the response encapsulated in the instrument transfer
            # function (stored as a PolesAndZeros response). NOTE: this does
            # not account for the effect of the digital FIR filters applied to
            # the recorded waveforms. However, due to this it is significantly
            # faster to compute.
            try:
                response = self.response_inv.get_response(tr.id,
                                                          tr.stats.starttime)
            except Exception as e:
                raise util.ResponseNotFoundError(str(e), tr.id)

            # Get the instrument transfer function as a PAZ dictionary
            paz = response.get_paz()

            # if not velocity:
            paz.zeros.extend([0j])
            paz_dict = {'poles': paz.poles,
                        'zeros': paz.zeros,
                        'gain': paz.normalization_factor,
                        'sensitivity': response.instrument_sensitivity.value}
            try:
                tr.simulate(paz_remove=paz_dict,
                            pre_filt=pre_filt,
                            water_level=water_level,
                            taper=True,
                            sacsim=True,  # To replicate remove_response()
                            pitsasim=False,  # To replicate remove_response()
                            paz_simulate=util.wa_response())
            except ValueError as e:
                raise util.ResponseRemovalError(e, tr.id)
        else:
            # Use remove_response(), which removes the effect of _all_ response
            # stages, including the FIR stages. Considerably slower.
            try:
                tr.remove_response(inventory=self.response_inv,
                                   output='DISP',
                                   pre_filt=pre_filt,
                                   water_level=water_level,
                                   taper=True)
                tr.simulate(paz_simulate=util.wa_response())
            except ValueError as e:
                raise util.ResponseRemovalError(e, tr.id)

        try:
            self.wa_waveforms.append(tr)
        except AttributeError:
            self.wa_waveforms = Stream()
            self.wa_waveforms.append(tr)

        return tr
