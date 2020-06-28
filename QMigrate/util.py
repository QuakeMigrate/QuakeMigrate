# -*- coding: utf-8 -*-
"""
Module that supplies various utility functions and classes.

"""

import sys

from functools import wraps
import logging
import time

import numpy as np
from obspy import Trace


log_spacer = "="*110


def make_directories(run, subdir=None):
    """
    Make run directory, and optionally make subdirectories within it.

    Parameters
    ----------
    run : pathlib Path object
        Location of parent output directory, named by run name.
    subdir : string, optional
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


def logger(logstem, log):
    """
    Simple logger that will output to both a log file and stdout.

    Parameters
    ----------
    logstem : str
        Filestem for log file.
    log : bool
        Toggle for logging - default is to only print information to stdout.
        If True, will also create a log file.

    """

    if log:
        logstem.parent.mkdir(exist_ok=True, parents=True)
        handlers = [logging.FileHandler(str(logstem.with_suffix(".log"))),
                    logging.StreamHandler(sys.stdout)]
    else:
        handlers = [logging.StreamHandler(sys.stdout)]

    logging.basicConfig(level=logging.INFO,
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
    convert : str, optional
        Type of output to convert between; determines the number of complex
        zeros used. Options are: 'DIS2DIS', 'VEL2VEL', 'VEL2DIS'
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


def decimate(trace, sr):
    """
    Decimate a trace to achieve the desired sampling rate, sr.

    NOTE: data will be detrended and a cosine taper applied before
    decimation, in order to avoid edge effects when applying the lowpass
    filter.

    Parameters:
    -----------
    trace : `obspy.Trace` object
        Trace to be decimated.
    sr : int
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

    # Zero-phase lowpass filter at Nyquist frequency
    trace.filter("lowpass", freq=float(sr) / 2.000001, corners=2,
                 zerophase=True)
    trace.decimate(factor=int(trace.stats.sampling_rate / sr),
                   strict_length=False, no_filter=True)

    return trace


def upsample(trace, upfactor):
    """
    Upsample a data stream by a given factor, prior to decimation. The
    upsampling is done using a linear interpolation.

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
    dnew = np.zeros(len(data)*upfactor - (upfactor - 1))
    dnew[::upfactor] = data
    for i in range(1, upfactor):
        dnew[i::upfactor] = float(i)/upfactor*data[1:] \
                        + float(upfactor - i)/upfactor*data[:-1]

    out = Trace()
    out.data = dnew
    out.stats = trace.stats
    out.stats.npts = len(out.data)
    out.stats.starttime = trace.stats.starttime
    out.stats.sampling_rate = int(upfactor * trace.stats.sampling_rate)

    return out


def timeit(f):
    """Function wrapper that measures the time elapsed during its execution."""
    @wraps(f)
    def wrap(*args, **kw):
        ts = time.time()
        result = f(*args, **kw)
        logging.info(" "*21 + f"Elapsed time: {time.time() - ts:6f} seconds.")
        return result
    return wrap


class StationFileHeaderException(Exception):
    """Custom exception to handle incorrect header columns in station file"""

    def __init__(self):
        msg = ("StationFileHeaderException: incorrect station file header - "
               "use:\nLatitude, Longitude, Elevation, Name")
        super().__init__(msg)


class VelocityModelFileHeaderException(Exception):
    """Custom exception to handle incorrect header columns in station file"""

    def __init__(self):
        msg = ("VelocityModelFileHeaderException: incorrect velocity model "
               "file header - use:\nDepth, Vp, Vs")
        super().__init__(msg)


class ArchiveFormatException(Exception):
    """Custom exception to handle case where Archive.format is not set."""

    def __init__(self):
        msg = ("ArchiveFormatException: Archive format has not been set. Set "
               "with Archive.path_structure() or Archive.format = '' ")
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


class NoMagObjectError(Exception):
    """
    Custom exception to handle case when calc_magnitudes has been selected for
    the QuakeScan run, but no magnitudes object has been provided. (e.g.
    mags=LocalMags() )

    """

    def __init__(self):
        msg = ("NoMagObjectError: You have selected to calculate magnitudes "
               "but have not provided a mags object - see manual.")
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
               f"Hz is at or above the Nyquist frequency ({f_nyquist} Hz)"
               f"\n\t\tfor trace {tr_id}. ")
        super().__init__(msg)


class TimeSpanException(Exception):
    """
    Custom exception to handle case when the user has submitted a start time
    that is after the end time.

    """

    def __init__(self):
        msg = ("TimeSpanException: The start time specified is after the end"
               " time.")
        super().__init__(msg)


class InvalidThresholdMethodException(Exception):
    """
    Custom exception to handle case when the user has not selected a valid
    threshold method.

    """

    def __init__(self):
        msg = ("InvalidThresholdMethodException: Only 'static' or 'dynamic' "
               "thresholds are supported.")
        super().__init__(msg)
