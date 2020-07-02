# -*- coding: utf-8 -*-
"""
Module for processing waveform files stored in a data archive.

"""

from itertools import chain
import logging
import pathlib

import numpy as np
from obspy import read, Stream, Trace, UTCDateTime

import QMigrate.util as util


class Archive:
    """
    The Archive class handles the reading of archived waveform data. It is
    capable of handling any regular archive structure. Requests to read
    waveform data are served up as a `QMigrate.data.WaveformData` object. Data
    will be checked for availability within the requested time period, and
    optionally resampled to meet a unified sampling rate. The raw data read
    from the archive will also be retained.

    If provided, a response inventory provided for the archive will be stored
    with the waveform data for response removal, if needed.

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
            self.path_structure(archive_format)
        else:
            self.format = kwargs.get("format", "")

        self.read_all_stations = kwargs.get("read_all_stations", False)
        self.response_inv = kwargs.get("response_inv")
        self.resample = kwargs.get("resample", False)
        self.upfactor = kwargs.get("upfactor")

    def __str__(self):
        """Returns a short summary string of the Archive object."""

        out = ("QuakeMigrate Archive object"
               f"\n\tArchive path\t:\t{self.archive_path}"
               f"\n\tPath structure\t:\t{self.format}"
               f"\n\tResampling\t:\t{self.resample}"
               "\n\tStations:")
        for station in self.stations:
            out += f"\n\t\t{station}"

        return out

    def path_structure(self, archive_format="YEAR/JD/STATION"):
        """
        Define the path structure of the data archive.

        Parameters
        ----------
        archive_format : str, optional
            Sets path type for different archive formats.

        """

        if archive_format == "SeisComp3":
            self.format = "{year}/*/{station}/BH*/*.{station}.*.*.D.{year}.{jday}"
        elif archive_format == "YEAR/JD/*_STATION_*":
            self.format = "{year}/{jday}/*_{station}_*"
        elif archive_format == "YEAR/JD/STATION":
            self.format = "{year}/{jday}/{station}*"
        elif archive_format == "STATION.YEAR.JULIANDAY":
            self.format = "*{station}.*.{year}.{jday}"
        elif archive_format == "/STATION/STATION.YearMonthDay":
            self.format = "{station}/{station}.{year}{month:02d}{day:02d}"
        elif archive_format == "YEAR_JD/STATION*":
            self.format = "{year}_{jday}/{station}*"
        elif archive_format == "YEAR_JD/STATION_*":
            self.format = "{year}_{jday}/{station}_*"

    def read_waveform_data(self, starttime, endtime, sampling_rate, pre_pad=0.,
                           post_pad=0.):
        """
        Read in the waveform data for all stations in the archive between two
        times and return station availability of the stations specified in the
        station file during this period. Decimate / resample (optional) this
        data if required to reach desired sampling rate.

        Output both processed data for stations in station file and all raw
        data in an obspy Stream object.

        By default, data with mismatched sampling rates will only be decimated.
        If necessary, and if the user specifies `resample = True` and an
        upfactor to upsample by `upfactor = int`, data can also be upsampled
        and then, if necessary, subsequently decimated to achieve the desired
        sampling rate.

        For example, for raw input data sampled at a mix of 40, 50 and 100 Hz,
        to achieve a unified sampling rate of 50 Hz, the user would have to
        specify an upfactor of 5; 40 Hz x 5 = 200 Hz, which can then be
        decimated to 50 Hz.

        NOTE: data will be detrended and a cosine taper applied before
        decimation, in order to avoid edge effects when applying the lowpass
        filter. Otherwise, data for migration will be added tp data.signal with
        no processing applied.

        Supports all formats currently supported by ObsPy, including: "MSEED"
        (default), "SAC", "SEGY", "GSE2" .

        Parameters
        ----------
        starttime : `obspy.UTCDateTime` object, optional
            Timestamp from which to read waveform data.
        endtime : `obspy.UTCDateTime` object, optional
            Timestamp up to which to read waveform data.
        sampling_rate : int
            Desired sampling rate for data to be added to signal. This will
            be achieved by resampling the raw waveform data. By default, only
            decimation will be applied, but data can also be upsampled if
            specified by the user when creating the Archive object.
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
        data : `QMigrate.io.data.WaveformData` object
            Object containing the archived data that satisfies the query.

        """

        data = WaveformData(starttime=starttime, endtime=endtime,
                            sampling_rate=sampling_rate,
                            stations=self.stations,
                            read_all_stations=self.read_all_stations,
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

            # Remove stations which have gaps in at least one trace
            gaps = st.get_gaps()
            if gaps:
                stations = np.unique(np.array(gaps)[:, 1]).tolist()
                for station in stations:
                    traces = st.select(station=station)
                    for trace in traces:
                        st.remove(trace)

            # Test if the stream is completely empty
            # (see __nonzero__ for obspy Stream object)
            if not bool(st):
                raise util.DataGapException

            # Pass stream to be processed and added to data.signal. This
            # processing includes resampling and determining the availability
            # of the desired stations.
            data.add_stream(st, self.resample, self.upfactor)

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

        if self.format is None or self.format == "":
            raise util.ArchiveFormatException

        start_day = UTCDateTime(starttime.date)

        dy = 0
        files = []
        # Loop through time period by day adding files to list
        # NOTE! This assumes the archive structure is split into days.
        while start_day + (dy * 86400) <= endtime:
            now = starttime + (dy * 86400)
            if self.read_all_stations is True:
                file_format = self.format.format(year=now.year,
                                                 month=f"{now.month:02d}",
                                                 day=f"{now.day:02d}",
                                                 jday=f"{now.julday:03d}",
                                                 station="*")
                files = chain(files, self.archive_path.glob(file_format))
            else:
                for station in self.stations:
                    file_format = self.format.format(year=now.year,
                                                     month=f"{now.month:02d}",
                                                     day=f"{now.day:02d}",
                                                     jday=f"{now.julday:03d}",
                                                     station=station)
                    files = chain(files, self.archive_path.glob(file_format))
            dy += 1

        return files


class WaveformData:
    """
    The WaveformData class encapsulates the waveform data returned by an`
    Archive query.
    
    This includes the waveform data which has been pre-processed to a unified
    sampling rate, and checked for gaps, ready for use to calculate onset
    functions.

    Parameters
    ----------
    starttime : `obspy.UTCDateTime` object
        Timestamp of first sample of waveform data.
    endtime : `obspy.UTCDateTime` object
        Timestamp of last sample of waveform data.
    sampling_rate : int
        Desired sampling rate of signal data.
    stations : `pandas.Series` object, optional
        Series object containing station names.
    read_all_stations : bool, optional
        If True, raw_waveforms contain all stations in archive for that time
        period. Else, only selected stations will be included.
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
    sampling_rate : int
        Sampling rate of signal data.
    stations : `pandas.Series` object
        Series object containing station names.
    read_all_stations : bool
        If True, raw_waveforms contain all stations in archive for that time
        period. Else, only selected stations will be included.
    raw_waveforms : `obspy.Stream` object
        Raw seismic data found and read in from the archive within the
        specified time period. This may be for all stations in the archive,
        or only those specified by the user. See `read_all_stations`.
    pre_pad : float
        Additional pre pad of data cut based on user-defined pre_cut
        parameter.
    post_pad : float
        Additional post pad of data cut based on user-defined post_cut
        parameter.
    signal : `numpy.ndarray`, shape(3, nstations, nsamples)
        3-component seismic data at the desired sampling rate; only for
        desired stations, which have continuous data on all 3 components
        throughout the desired time period and where (if necessary) the data
        could be successfully resampled to the desired sampling rate.
    availability : `np.ndarray` of ints, shape(nstations)
        Array containing 0s (no data) or 1s (data), corresponding to whether
        data for each station met the requirements outlined in `signal`
    filtered_signal : `numpy.ndarray`, shape(3, nstations, nsamples)
        Filtered data originally from signal.

    Methods
    -------
    add_stream(stream, resample, upfactor)
        Function to add data supplied in the form of an `obspy.Stream` object.
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

    def __init__(self, starttime, endtime, sampling_rate, stations=None,
                 response_inv=None, read_all_stations=False, pre_pad=0.,
                 post_pad=0.):
        """Instantiate the WaveformData object."""

        self.starttime = starttime
        self.endtime = endtime
        self.sampling_rate = sampling_rate
        self.stations = stations
        self.response_inv = response_inv

        self.read_all_stations = read_all_stations
        self.pre_pad = pre_pad
        self.post_pad = post_pad

        self.raw_waveforms = None
        self.signal = None
        self.availability = None
        self.p_onset = None
        self.s_onset = None
        self.filtered_signal = None
        self.wa_waveforms = None

    def add_stream(self, stream, resample, upfactor):
        """
        Add signal data supplied in an `obspy.Stream` object. Perform
        resampling if necessary (decimation and/or upsampling), and determine
        availability of selected stations.

        Parameters:
        -----------
        stream : `obspy.Stream` object
            Contains list of `obspy.Trace` objects containing the waveform
            data to add.
        resample : bool, optional
            If true, perform resampling of data which cannot be decimated
            directly to the desired sampling rate.
        upfactor : int, optional
            Factor by which to upsample the data to enable it to be decimated
            to the desired sampling rate, e.g. 40Hz -> 50Hz requires
            upfactor = 5.

        """

        # Decimate and/or upsample stream if required to achieve the specified
        # sampling rate
        stream = self._resample(stream, resample, upfactor)

        # Combine the data into an array and determine station availability
        self.signal, self.availability = self._station_availability(stream)

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
                            sacsim=True, pitsasim=False,  # To replicate remove_response()
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

    def times(self, **kwargs):
        """
        Utility function to generate timestamps between `data.starttime` and
        `data.endtime`, with a sample size of `data.sample_size`

        Returns
        -------
        times : `numpy.ndarray`, shape(nsamples)
            Timestamps for the timeseries data.

        """

        # Utilise the .times() method of `obspy.Trace` objects
        tr = Trace(header={"npts": self.signal.shape[-1],
                           "sampling_rate": self.sampling_rate,
                           "starttime": self.starttime})
        return tr.times(**kwargs)

    def _resample(self, stream, resample, upfactor):
        """
        Resample the stream to the specified sampling rate.

        By default, this function will only perform decimation of the data. If
        necessary, and if the user specifies `resample = True` and an upfactor
        to upsample by `upfactor = int`, data can also be upsampled and then,
        if necessary, subsequently decimated to achieve the desired sampling
        rate.

        For example, for raw input data sampled at a mix of 40, 50 and 100 Hz,
        to achieve a unified sampling rate of 50 Hz, the user would have to
        specify an upfactor of 5; 40 Hz x 5 = 200 Hz, which can then be
        decimated to 50 Hz.

        NOTE: data will be detrended and a cosine taper applied before
        decimation, in order to avoid edge effects when applying the lowpass
        filter.

        Parameters
        ----------
        stream : `obspy.Stream` object
            Contains list of `obspy.Trace` objects to be decimated / resampled.
        resample : bool
            If true, perform resampling of data which cannot be decimated
            directly to the desired sampling rate.
        upfactor : int or None
            Factor by which to upsample the data to enable it to be decimated
            to the desired sampling rate, e.g. 40Hz -> 50Hz requires
            upfactor = 5.

        Returns
        -------
        stream : `obspy.Stream` object
            Contains list of resampled `obspy.Trace` objects at the chosen
            sampling rate `sr`.

        """

        sr = self.sampling_rate
        for trace in stream:
            if sr != trace.stats.sampling_rate:
                if (trace.stats.sampling_rate % sr) == 0:
                    stream.remove(trace)
                    trace = util.decimate(trace, sr)
                    stream += trace
                elif resample and upfactor is not None:
                    # Check the upsampled sampling rate can be decimated to sr
                    if int(trace.stats.sampling_rate * upfactor) % sr != 0:
                        raise util.BadUpfactorException(trace)
                    stream.remove(trace)
                    trace = util.upsample(trace, upfactor)
                    if trace.stats.sampling_rate != sr:
                        trace = util.decimate(trace, sr)
                    stream += trace
                else:
                    logging.info("Mismatched sampling rates - cannot decimate "
                                 "data - to resample data, set .resample "
                                 "= True and choose a suitable upfactor")

        return stream

    def _station_availability(self, stream):
        """
        Determine whether continuous data exists between two times for a given
        station.

        Parameters
        ----------
        stream : `obspy.Stream` object
            Stream containing 3-component data for stations in station file.
        samples : int
            Number of samples expected in the signal.

        Returns
        -------
        signal : `numpy.ndarray`, shape(3, nstations, nsamples)
            3-component seismic data only for stations with continuous data
            on all 3 components throughout the desired time period.
        availability : `np.ndarray` of ints, shape(nstations)
            Array containing 0s (no data) or 1s (data).

        """

        samples = int(round((self.endtime - self.starttime) \
            * self.sampling_rate + 1))

        availability = np.zeros(len(self.stations)).astype(int)
        signal = np.zeros((3, len(self.stations), int(samples)))

        for i, station in enumerate(self.stations):
            tmp_st = stream.select(station=station)
            if len(tmp_st) == 3:
                # Check traces are the correct number of samples and not filled
                # by a constant value (i.e. not flatlines)
                if (tmp_st[0].stats.npts == samples and
                        tmp_st[1].stats.npts == samples and
                        tmp_st[2].stats.npts == samples and
                        tmp_st[0].data.max() != tmp_st[0].data.min() and
                        tmp_st[1].data.max() != tmp_st[1].data.min() and
                        tmp_st[2].data.max() != tmp_st[2].data.min()):

                    # Defining the station as available
                    availability[i] = 1

                    for tr in tmp_st:
                        # Check channel name has 3 characters
                        try:
                            channel = tr.stats.channel[2]
                            # Assign data to signal array by component
                            if channel == "E" or channel == "2":
                                signal[1, i, :] = tr.data
                            elif channel == "N" or channel == "1":
                                signal[0, i, :] = tr.data
                            elif channel == "Z":
                                signal[2, i, :] = tr.data
                            else:
                                raise util.ChannelNameException(tr)

                        except IndexError:
                            raise util.ChannelNameException(tr)

        # Check to see if no traces were continuously active during this period
        if not np.any(availability):
            raise util.DataGapException

        return signal, availability

    @property
    def sample_size(self):
        """Get the size of a sample (units: s)."""

        return 1 / self.sampling_rate
