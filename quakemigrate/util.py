# -*- coding: utf-8 -*-
"""
Module that supplies various utility functions and classes.

:copyright:
    2020 - 2021, QuakeMigrate developers.
:license:
    GNU General Public License, Version 3
    (https://www.gnu.org/licenses/gpl-3.0.html)

"""

import logging
import sys
import time
from datetime import datetime
from functools import wraps
from itertools import tee

import matplotlib.ticker as ticker
import numpy as np
from obspy import Trace


log_spacer = "="*110


def make_directories(run, subdir=None):
    """
    Make run directory, and optionally make subdirectories within it.

    Parameters
    ----------
    run : `pathlib.Path` object
        Location of parent output directory, named by run name.
    subdir : str, optional
        subdir to make beneath the run level.

    """

    run.mkdir(exist_ok=True)

    if subdir:
        new_dir = run / subdir
        new_dir.mkdir(exist_ok=True, parents=True)


def gaussian_1d(x, a, b, c):
    """
    Create a 1-dimensional Gaussian function.

    Parameters
    ----------
    x : array-like
        Array of x values
    a : float / int
        Amplitude (height of Gaussian)
    b : float / int
        Mean (centre of Gaussian)
    c : float / int
        Sigma (width of Gaussian)

    Returns
    -------
    f : function
        1-dimensional Gaussian function

    """

    f = a * np.exp(-1. * ((x - b) ** 2) / (2 * (c ** 2)))

    return f


def gaussian_3d(nx, ny, nz, sgm):
    """
    Create a 3-dimensional Gaussian function.

    Parameters
    ----------
    nx : array-like
        Array of x values
    ny : array-like
        Array of y values
    nz : array-like
        Array of z values
    sgm : float / int
        Sigma (width of gaussian in all directions)

    Returns
    -------
    f : function
        3-dimensional Gaussian function

    """

    nx2 = (nx - 1) / 2
    ny2 = (ny - 1) / 2
    nz2 = (nz - 1) / 2
    x = np.linspace(-nx2, nx2, nx)
    y = np.linspace(-ny2, ny2, ny)
    z = np.linspace(-nz2, nz2, nz)
    ix, iy, iz = np.meshgrid(x, y, z, indexing="ij")

    if np.isscalar(sgm):
        sgm = np.repeat(sgm, 3)
    sx, sy, sz = sgm

    f = np.exp(- (ix * ix) / (2 * sx * sx)
               - (iy * iy) / (2 * sy * sy)
               - (iz * iz) / (2 * sz * sz))

    return f


def logger(logstem, log, loglevel="info"):
    """
    Simple logger that will output to both a log file and stdout.

    Parameters
    ----------
    logstem : str
        Filestem for log file.
    log : bool
        Toggle for logging - default is to only print information to stdout.
        If True, will also create a log file.
    loglevel : str, optional
        Toggle for logging level - default is to print only "info" messages to
        log. To print more detailed "debug" messages, set to "debug".

    """

    level = logging.DEBUG if loglevel == "debug" else logging.INFO

    if log:
        now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        logfile = logstem.parent / f"{logstem.name}_{now}"
        logfile.parent.mkdir(exist_ok=True, parents=True)
        handlers = [logging.FileHandler(str(logfile.with_suffix(".log"))),
                    logging.StreamHandler(sys.stdout)]
    else:
        handlers = [logging.StreamHandler(sys.stdout)]

    logging.basicConfig(level=level,
                        format="%(message)s",
                        handlers=handlers)
                        # format="%(asctime)s [%(levelname)s] %(message)s",


def time2sample(time, sampling_rate):
    """
    Utility function to convert from seconds and sampling rate to number of
    samples.

    Parameters
    ----------
    time : float
        Time to convert
    sampling_rate : int
        Sampling rate of input data/sampling rate at which to compute
        the coalescence function.

    Returns
    -------
    out : int
        Time that correpsonds to an integer number of samples at a specific
        sampling rate.

    """

    return int(round(time*int(sampling_rate)))


def calculate_mad(x, scale=1.4826):
    """
    Calculates the Median Absolute Deviation (MAD) of the input array x.

    Parameters
    ----------
    x : array-like
        Input data.
    scale : float, optional
        A scaling factor for the MAD output to make the calculated MAD factor
        a consistent estimation of the standard deviation of the distribution.

    Returns
    -------
    scaled_mad : array-like
        Array of scaled mean absolute deviation values for the input array, x,
        scaled to provide an estimation of the standard deviation of the
        distribution.

    """

    x = np.asarray(x)

    if not x.size:
        return np.nan

    if np.isnan(np.sum(x)):
        return np.nan

    # Calculate median and mad values:
    med = np.apply_over_axes(np.median, x, 0)
    mad = np.median(np.abs(x - med), axis=0)

    return scale * mad


class DateFormatter(ticker.Formatter):
    """
    Extend the `matplotlib.ticker.Formatter` class to allow for millisecond
    precision when formatting a tick (in days since the epoch) with a
    `datetime.datetime.strftime` format string.

    """

    def __init__(self, fmt, precision=3):
        """
        Parameters
        ----------
        fmt : str
            `datetime.datetime.strftime` format string.
        precision : int
            Degree of precision to which to report sub-second time intervals.

        """

        from matplotlib.dates import num2date

        self.num2date = num2date
        self.fmt = fmt
        self.precision = precision

    def __call__(self, x, pos=0):
        if x == 0:
            raise ValueError("DateFormatter found a value of x=0, which is "
                             "an illegal date; this usually occurs because "
                             "you have not informed the axis that it is "
                             "plotting dates, e.g., with ax.xaxis_date()")

        dt = self.num2date(x)
        ms = dt.strftime("%f")[:self.precision]

        return dt.strftime(self.fmt).format(ms=ms)


def trim2sample(time, sampling_rate):
    """
    Utility function to ensure time padding results in a time that is an
    integer number of samples.

    Parameters
    ----------
    time : float
        Time to trim.
    sampling_rate : int
        Sampling rate of input data/sampling rate at which to compute
        the coalescence function.

    Returns
    -------
    out : int
        Time that correpsonds to an integer number of samples at a specific
        sampling rate.

    """

    return int(np.ceil(time * sampling_rate) / sampling_rate * 1000) / 1000


def wa_response(convert='DIS2DIS', obspy_def=True):
    """
    Generate a Wood Anderson response dictionary.

    Parameters
    ----------
    convert : {'DIS2DIS', 'VEL2VEL', ‘VEL2DIS'}
        Type of output to convert between; determines the number of complex
        zeros used.
    obspy_def : bool, optional
        Use the ObsPy definition of the Wood Anderson response (Default).
        Otherwise, use the IRIS/SAC definition.

    Returns
    -------
    WOODANDERSON : dict
        Poles, zeros, sensitivity and gain of the Wood-Anderson torsion
        seismograph.

    """

    if obspy_def:
        # Create Wood-Anderson response - ObsPy values
        woodanderson = {"poles": [-6.283185 - 4.712j,
                                  -6.283185 + 4.712j],
                        "zeros": [0j],
                        "sensitivity": 2080,
                        "gain": 1.}
    else:
        # Create Wood Anderson response - different to the ObsPy values
        # http://www.iris.washington.edu/pipermail/sac-help/2013-March/001430.html
        woodanderson = {"poles": [-5.49779 + 5.60886j,
                                  -5.49779 - 5.60886j],
                        "zeros": [0j],
                        "sensitivity": 2080,
                        "gain": 1.}

    if convert in ('DIS2DIS', 'VEL2VEL'):
        # Add an extra zero to go from disp to disp or vel to vel.
        woodanderson['zeros'].extend([0j])

    return woodanderson


def shift_to_sample(stream, interpolate=False):
    """
    Check whether any data in an `obspy.Stream` object is "off-sample" - i.e.
    the data timestamps are *not* an integer number of samples after midnight.
    If so, shift data to be "on-sample".

    This can either be done by shifting the timestamps by a sub-sample time
    interval, or interpolating the trace to the "on-sample" timestamps. The
    latter has the benefit that it will not affect the timing of the data, but
    will require additional computation time and some inevitable edge effects -
    though for onset calculation these should be contained within the pad
    windows. If you are using a sampling rate < 10 Hz, contact the QuakeMigrate
    developers.

    Parameters
    ----------
    stream : `obspy.Stream` object
        Contains list of `obspy.Trace` objects for which to check the timing.
    interpolate : bool, optional
        Whether to interpolate the data to correct the "off-sample" timing.
        Otherwise, the metadata will simply be altered to shift the timestamps
        "on-sample"; this will lead to a sub-sample timing offset.

    Returns
    -------
    stream : `obspy.Stream` object
        Waveform data with all timestamps "on-sample".

    """

    # work on a copy
    stream = stream.copy()

    for tr in stream:
        # Check if microsecond is divisible by sampling rate; only guaranteed
        # to work for sampling rates of 1 Hz or less
        delta = tr.stats.starttime.microsecond \
            % (1e6 / tr.stats.sampling_rate)
        if delta == 0:
            if tr.stats.sampling_rate < 1.:
                logging.warning(f"Trace\n\t{tr}\nhas a sampling rate less than"
                                " 1 Hz, so off-sample data might not be "
                                "corrected!")
            continue
        else:
            # Calculate time shift to closest "on-sample" timestamp
            time_shift = round(delta / 1e6 * tr.stats.sampling_rate) \
                / tr.stats.sampling_rate - delta / 1e6
            if not interpolate:
                logging.info(f"Trace\n\t{tr}\nhas off-sample data. Applying "
                             f"{time_shift:+f} s shift to timing.")
                tr.stats.starttime += time_shift
                logging.debug(f"Shifted trace: {tr}")
            else:
                logging.info(f"Trace\n\t{tr}\nhas off-sample data. "
                             f"Interpolating to apply a {time_shift:+f} s "
                             "shift to timing.")
                # Interpolate can only map between values contained within the
                # original array. For negative time shift, shift by one sample
                # so new starttime is within original array, and add constant
                # value pad after interpolation.
                new_starttime = tr.stats.starttime + time_shift
                if time_shift < 0.:
                    new_starttime += tr.stats.delta
                tr.interpolate(sampling_rate=tr.stats.sampling_rate,
                               method="lanczos", a=20, starttime=new_starttime)
                # Add constant-value pad at end if time_shift is positive,
                # (last sample is dropped when interpolating for positive time
                # shifts), else at start. If adding at start, also adjust start
                # time.
                if time_shift > 0.:
                    tr.data = np.append(tr.data, tr.data[-1])
                else:
                    tr.data = np.append(tr.data[0], tr.data)
                    tr.stats.starttime -= tr.stats.delta
                logging.debug(f"Interpolated tr:\n\t{tr}")

    return stream


def resample(stream, sampling_rate, resample, upfactor, starttime, endtime):
    """
    Resample data in an `obspy.Stream` object to the specified sampling rate.

    By default, this function will only perform decimation of the data. If
    necessary, and if the user specifies `resample = True` and an upfactor
    to upsample by `upfactor = int`, data can also be upsampled and then,
    if necessary, subsequently decimated to achieve the desired sampling
    rate.

    For example, for raw input data sampled at a mix of 40, 50 and 100 Hz,
    to achieve a unified sampling rate of 50 Hz, the user would have to
    specify an upfactor of 5; 40 Hz x 5 = 200 Hz, which can then be
    decimated to 50 Hz.

    NOTE: assumes any data with off-sample timing has been corrected with
    :func:`~quakemigrate.util.shift_to_sample`. If not, the resulting traces
    may not all contain the correct number of samples.

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

    # Work on a copy of the stream
    stream = stream.copy()

    for trace in stream:
        trace_sampling_rate = trace.stats.sampling_rate
        if sampling_rate != trace_sampling_rate:
            if (trace_sampling_rate % sampling_rate) == 0:
                stream.remove(trace)
                trace = decimate(trace, sampling_rate)
                stream += trace
            elif resample and upfactor is not None:
                # Check the upsampled sampling rate can be decimated to sr
                if int(trace_sampling_rate * upfactor) % sampling_rate != 0:
                    raise BadUpfactorException(trace)
                stream.remove(trace)
                trace = upsample(trace, upfactor, starttime, endtime)
                if trace_sampling_rate != sampling_rate:
                    trace = decimate(trace, sampling_rate)
                stream += trace
            else:
                logging.info("Mismatched sampling rates - cannot decimate "
                             f"data from\n\t{trace}.\n..to resample data, set "
                             "resample = True and choose a suitable upfactor")

    # Trim as a general safety net. NOTE: here we are using
    # nearest_sample=False, as all data in the stream should now be at the
    # desired sampling rate, and with any off-sample data having had it's
    # timing shifted.
    stream.trim(starttime=starttime-0.00001, endtime=endtime+0.00001,
                nearest_sample=False)

    return stream


def decimate(trace, sampling_rate):
    """
    Decimate a trace to achieve the desired sampling rate, sr.

    NOTE: data will be detrended and a cosine taper applied before
    decimation, in order to avoid edge effects when applying the lowpass
    filter before decimating.

    Parameters:
    -----------
    trace : `obspy.Trace` object
        Trace to be decimated.
    sampling_rate : int
        Output sampling rate.

    Returns:
    --------
    trace : `obspy.Trace` object
        Decimated trace.

    """

    # Work on a copy of the trace
    trace = trace.copy()

    # Detrend and apply cosine taper
    trace.detrend('linear')
    trace.detrend('demean')
    trace.taper(type='cosine', max_percentage=0.05)

    # Zero-phase Butterworth-lowpass filter at Nyquist frequency
    trace.filter("lowpass", freq=float(sampling_rate) / 2.000001, corners=2,
                 zerophase=True)
    trace.decimate(factor=int(trace.stats.sampling_rate / sampling_rate),
                   strict_length=False, no_filter=True)

    return trace


def upsample(trace, upfactor, starttime, endtime):
    """
    Upsample a data stream by a given factor, prior to decimation. The
    upsampling is carried out by linear interpolation.

    NOTE: assumes any data with off-sample timing has been corrected with
    :func:`~quakemigrate.util.shift_to_sample`. If not, the resulting traces
    may not all contain the correct number of samples (and desired start
    and end times).

    Parameters
    ----------
    trace : `obspy.Trace` object
        Trace to be upsampled.
    upfactor : int
        Factor by which to upsample the data in trace.

    Returns
    -------
    out : `obpsy.Trace` object
        Upsampled trace.

    """

    data = trace.data
    # Fenceposts
    dnew = np.zeros((len(data) - 1) * upfactor + 1)
    dnew[::upfactor] = data
    for i in range(1, upfactor):
        dnew[i::upfactor] = float(i)/upfactor*data[1:] \
                        + float(upfactor - i)/upfactor*data[:-1]

    # Check if start needs pad - if so pad with constant value (start value
    # of original trace). Use inequality here to only apply padding to data at
    # the start and end of the requested time window; not for other traces
    # floating in the middle (in the case that there are gaps).
    if 0. < trace.stats.starttime - starttime < trace.stats.delta:
        logging.debug(f"Mismatched starttimes: {trace.stats.starttime}, "
                      f"{starttime}")
        # Calculate how many additional samples are needed
        start_pad = np.round((trace.stats.starttime - starttime) \
            * trace.stats.sampling_rate * upfactor)
        logging.debug(f"Start pad = {start_pad}")
        # Add padding data (constant value)
        start_fill = np.full(np.int(start_pad), trace.data[0], dtype=int)
        dnew = np.append(start_fill, dnew)
        # Calculate new starttime of trace
        new_starttime = trace.stats.starttime - start_pad \
            / (trace.stats.sampling_rate * upfactor)
        logging.debug(f"New starttime = {new_starttime}")
    else:
        new_starttime = trace.stats.starttime

    # Ditto for end of trace
    if 0. < endtime - trace.stats.endtime < trace.stats.delta:
        logging.debug(f"Mismatched endtimes: {trace.stats.endtime}, {endtime}")
        # Calculate how many additional samples are needed
        end_pad = np.round((endtime - trace.stats.endtime) \
            * trace.stats.sampling_rate * upfactor)
        logging.debug(f"End pad = {end_pad}")
        # Add padding data (constant value)
        end_fill = np.full(np.int(end_pad), trace.data[-1], dtype=int)
        dnew = np.append(dnew, end_fill)

    out = Trace()
    out.data = dnew
    out.stats = trace.stats.copy()
    out.stats.npts = len(out.data)
    out.stats.starttime = new_starttime
    out.stats.sampling_rate = int(upfactor * trace.stats.sampling_rate)
    logging.debug(f"Raw upsampled trace:\n\t{out}")

    # Trim to remove additional padding left from reading with
    # nearest_sample=True at a variety of sampling rates.
    # NOTE: here we are using nearest_sample=False, as all data in the stream
    # should now be at a *multiple* of the desired sampling rate, and with any
    # off-sample data having had it's timing shifted.
    out.trim(starttime=starttime-0.00001, endtime=endtime+0.00001,
             nearest_sample=False)
    logging.debug(f"Trimmed upsampled trace:\n\t{out}")

    return out


def pairwise(iterable):
    """Utility to iterate over an iterable pairwise."""

    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def timeit(*args_, **kwargs_):
    """Function wrapper that measures the time elapsed during its execution."""
    def inner_function(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            ts = time.time()
            result = func(*args, **kwargs)
            msg = " "*21 + f"Elapsed time: {time.time() - ts:6f} seconds."
            try:
                if args_[0] == "info":
                    logging.info(msg)
            except IndexError:
                logging.debug(msg)
            return result
        return wrapper
    return inner_function


class StationFileHeaderException(Exception):
    """Custom exception to handle incorrect header columns in station file"""

    def __init__(self):
        msg = ("StationFileHeaderException: incorrect station file header - "
               "use:\nLatitude, Longitude, Elevation, Name")
        super().__init__(msg)


class InvalidVelocityModelHeader(Exception):
    """Custom exception to handle incorrect header columns in station file"""

    def __init__(self, key):
        super().__init__(f"Must include at least '{key}' in header.")


class ArchiveFormatException(Exception):
    """Custom exception to handle case where Archive.format is not set."""

    def __init__(self):
        msg = ("ArchiveFormatException: Archive format has not been set. Set "
               "when making the Archive object with the kwarg "
               "'archive_format=<path_structure>', or afterwards by using the "
               "command 'Archive.path_structure(<path_structure>)'. To set a "
               "custom format, use 'Archive.format = "
               "custom/archive_{year}_{jday}/{day:02d}.{station}_structure' ")
        super().__init__(msg)


class ArchivePathStructureError(Exception):
    """Custom exception to handle case where an invalid Archive path structure
    is selected."""

    def __init__(self, archive_format):
        msg = ("ArchivePathStructureError: The archive path structure you have"
               f" selected: '{archive_format}' is not a valid option! See the "
               "documentation for quakemigrate.data.Archive.path_structure for"
               " a complete list, or specify a custom format with Archive.form"
               "at = custom/archive_{year}_{jday}/{day:02d}.{station}_structur"
               "e")
        super().__init__(msg)


class ArchiveEmptyException(Exception):
    """Custom exception to handle empty archive"""

    def __init__(self):
        msg = "ArchiveEmptyException: No data was available for this timestep."
        super().__init__(msg)

        # Additional message printed to log
        self.msg = ("\t\tNo files found in archive for this time period.")


class NoScanMseedDataException(Exception):
    """
    Custom exception to handle case when no .scanmseed files can be found by
    read_coastream()

    """

    def __init__(self):
        msg = "NoScanMseedDataException: No .scanmseed data found."
        super().__init__(msg)


class NoStationAvailabilityDataException(Exception):
    """
    Custom exception to handle case when no .StationAvailability files can be
    found by read_availability()

    """

    def __init__(self):
        msg = ("NoStationAvailabilityDataException: No .StationAvailability "
               "files found.")
        super().__init__(msg)


class DataGapException(Exception):
    """
    Custom exception to handle case when all data has gaps for a given timestep

    """

    def __init__(self):
        msg = ("DataGapException: All available data had gaps for this "
               "timestep.\n    OR: no data present in the archive for the"
               "selected stations.")
        super().__init__(msg)

        # Additional message printed to log
        self.msg = ("\t\tAll available data for this time period contains gaps"
                    "\n\t\tor data not available at start/end of time period")


class ChannelNameException(Exception):
    """
    Custom exception to handle case when waveform data header has channel names
    which do not conform to the IRIS SEED standard.

    """

    def __init__(self, trace):
        msg = ("ChannelNameException: Channel name header does not conform "
               "to\nthe IRIS SEED standard - 3 characters; ending in 'Z' for\n"
               "vertical and ending either 'E' & 'N' or '1' & '2' for\n"
               "horizontal components.\n    Working on trace: {}".format(trace))
        super().__init__(msg)


class NoOnsetPeak(Exception):
    """
    Custom exception to handle case when no values in the onset function exceed
    the threshold used for picking.

    """

    def __init__(self, pick_threshold):
        self.msg = ("\t\t    No onset signal exceeding pick threshold "
                    f"({pick_threshold:5.3f}) - continuing.")
        super().__init__(self.msg)


class BadUpfactorException(Exception):
    """
    Custom exception to handle case when the chosen upfactor does not create a
    trace with a sampling rate that can be decimated to the target sampling
    rate

    """

    def __init__(self, trace):
        msg = ("BadUpfactorException: chosen upfactor cannot be decimated to\n"
               "target sampling rate.\n    Working on trace: {}".format(trace))
        super().__init__(msg)


class OnsetTypeError(Exception):
    """
    Custom exception to handle case when the onset object passed to QuakeScan
    is not of the default type defined in QuakeMigrate.

    """

    def __init__(self):
        msg = ("OnsetTypeError: The Onset object you have created does not "
               "inherit from the required base class - see manual.")
        super().__init__(msg)


class PickerTypeError(Exception):
    """
    Custom exception to handle case when the phase picker object passed to
    QuakeScan is not of the default type defined in QuakeMigrate.

    """

    def __init__(self):
        msg = ("PickerTypeError: The PhasePicker object you have created does "
               "not inherit from the required base class - see manual.")
        super().__init__(msg)


class LUTPhasesException(Exception):
    """
    Custom exception to handle the case when the look-up table does not
    contain the traveltimes for the phases necessary for a given function.

    """

    def __init__(self, message):
        msg = (f"LUTPhasesException: {message}")
        super().__init__(msg)


class PickOrderException(Exception):
    """
    Custom exception to handle the case when the pick for the P phase is later
    than the pick for the S phase.

    """

    def __init__(self, event_uid, station, p_pick, s_pick):
        msg = ("PickOrderException: The P-phase arrival-time pick is later "
               "than the S-phase arrival pick! Something has gone wrong. "
               f"Event: {event_uid}, station: {station}, p_pick: {p_pick}, "
               f"s_pick: {s_pick}. There is probably a bug with the picker.")
        super().__init__(msg)


class MagsTypeError(Exception):
    """
    Custom exception to handle case when an object has been provided to
    calculate magnitudes during locate, but it isn't supported.

    """

    def __init__(self):
        msg = ("MagsTypeError: The Mags object you have specified is not "
               "supported: currently only "
               "`quakemigrate.signal.local_mag.LocalMag` - see manual.")
        super().__init__(msg)


class NoTriggerFilesFound(Exception):
    """
    Custom exception to handle case when no trigger files are found during
    locate. This can occur for one of two reasons - an entirely invalid time
    period was used (i.e. one that does not overlap at all with a period of
    time for which there exists TriggeredEvents.csv files) or an invalid run
    name was provided.

    """

    def __init__(self):
        msg = ("NoTriggerFilesFound: Double check you have supplied a valid "
               "run name and a time period for which you have run detect.")
        super().__init__(msg)


class ResponseNotFoundError(Exception):
    """
    Custom exception to handle the case where the provided response inventory
    doesn't contain the response information for a trace.

    Parameters
    ----------
    e : str
        Error message from ObsPy `Inventory.get_response()`
    tr_id : str
        ID string for the Trace for which the response cannot be found

    """

    def __init__(self, e, tr_id):
        msg = (f"ResponseNotFoundError: {e} -- skipping {tr_id}")
        super().__init__(msg)


class ResponseRemovalError(Exception):
    """
    Custom exception to handle the case where the response removal was not
    successful.

    Parameters
    ----------
    e : str
        Error message from ObsPy `Trace.remove_response()` or
        `Trace.simulate()`
    tr_id : str
        ID string for the Trace for which the response cannot be removed

    """

    def __init__(self, e, tr_id):
        msg = (f"ResponseRemovalError: {e} -- skipping {tr_id}")
        super().__init__(msg)


class NyquistException(Exception):
    """
    Custom exception to handle the case where the specified filter has a
    lowpass corner above the signal Nyquist frequency.

    Parameters
    ----------
    freqmax : float
        Specified lowpass frequency for filter
    f_nyquist : float
        Nyquist frequency for the relevant waveform data
    tr_id : str
        ID string for the Trace

    """

    def __init__(self, freqmax, f_nyquist, tr_id):
        msg = (f"    NyquistException: Selected bandpass_highcut {freqmax} "
               f"Hz is at or above the Nyquist frequency ({f_nyquist} Hz) "
               f"for trace {tr_id}. ")
        super().__init__(msg)


class PeakToTroughError(Exception):
    """
    Custom exception to handle case when amplitude._peak_to_trough_amplitude
    encounters an anomalous set of peaks and troughs, so can't calculate an
    amplitude.

    """

    def __init__(self, err):
        msg = (f"PeakToTroughError: {err}")
        super().__init__(msg)

        # Additional message printed to log
        self.msg = (err)


class TimeSpanException(Exception):
    """
    Custom exception to handle case when the user has submitted a start time
    that is after the end time.

    """

    def __init__(self):
        msg = ("TimeSpanException: The start time specified is after the end"
               " time.")
        super().__init__(msg)


class InvalidTriggerThresholdMethodException(Exception):
    """
    Custom exception to handle case when the user has not selected a valid
    trigger threshold method.

    """

    def __init__(self):
        msg = ("InvalidTriggerThresholdMethodException: Only 'static' or "
               "'dynamic' thresholds are supported.")
        super().__init__(msg)


class InvalidPickThresholdMethodException(Exception):
    """
    Custom exception to handle case when the user has not selected a valid
    pick threshold method.

    """

    def __init__(self):
        msg = ("InvalidPickThresholdMethodException: Only 'percentile' or "
               "'MAD' thresholds are supported.")
        super().__init__(msg)
