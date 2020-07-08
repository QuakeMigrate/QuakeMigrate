Reading waveform archives
=========================
This tutorial will provide instructions on how to direct QuakeMigrate to a
waveform archive and how to specify its structure. It is in theory possible to use a waveform archive with any arbitrary structure. Additional requirements can be handled on request - contact us at quakemigrate.developers@gmail.com or submit
an Issue on our GitHub.

The Archive class
-----------------
The :class:`Archive` class provides methods for querying a waveform archive on a local system. It is capable of handling any regular archive structure, as well as any data file format that is compatible with `ObsPy`. Waveform data and an overview of the data availability (see :ref:`rejection-criteria-label`) are returned by a query to the archive.

It requires two pieces of information on instantiation:

- ``archive_path``: the path to seismic data archive
- ``stations``: a DataFrame containing station information. There is one required (case-sensitive) column header - "Name".

Here we create a new instance of :class:`Archive`.

::

    from QMigrate.io import Archive, read_stations

    # --- Read in station file ---
    stations = read_stations(station_file)

    # --- Create new Archive and set path structure ---
    archive = Archive(archive_path=data_in, stations=stations)

Specifying the archive structure
--------------------------------
Once the :class:`Archive` object has been instantiated, we need to specify the regular archive structure. There are some standard formats, which can be accessed through the :func:`path_structure` method, including SeisComp3 and the standard structure used by SeisUK. These map to a formattable string used when querying the waveform archive:

::

    archive.path_structure(archive_format="SeisComp3")

It is also possible to override with a custom archive structure:

::

    archive.format = "{year}/{jday:03d}/{station}_{year}_{jday:03d}_{channels}.*"

The full list of keyword arguments that are passed into this formattable string when the archive is queried is:

- ``year``: ``UTCDateTime.year`` for the time period of the query
- ``month``: ``UTCDateTime.month`` for the time period of the query
- ``day``: ``UTCDateTime.day`` for the time period of the query
- ``jday``: ``UTCDateTime.julday`` for the time period of the query 
- ``station``: the station name (replaced with ``"*"`` if reading all)
- ``dtime``: ``UTCDateTime`` for the time period of the query

The inclusion of ``dtime`` allows for incredible flexibility, with most of the other arguments just providing shorthand.

Resampling waveforms
--------------------
It is common for a data archive to contain stations with differing sampling rates. In order to run QuakeMigrate, these sampling rates need to be unified. We have bundled routines for accomplishing this automatically. These routines aim to minimally alter the values of the waveforms by retaining as much of the original data as possible. Downsampling from 100 Hz to 50 Hz, for example, is accomplished by decimating the waveforms by a factor of two - skipping every other sample. If the unified sampling rate is not an integer divisor of the input waveform sampling rate, there is (limited) scope to linearly interpolate the waveform data to a sampling rate that does divide into the unified sampling rate an integer number of times, then decimate down.

Resampling can be toggled on with ``archive.resample = True``, and a single factor by which to linearly interpolate data when resampling with ``archive.upfactor = 2``. We hope to de-restrict this in the future to allow for automatic identification of a suitable `upfactor` (within reason).

.. _rejection-criteria-label:

Rejection criteria
------------------
We currently impose fairly strict criteria on the data to be used in QuakeMigrate, which are detailed below.

Channel names
#############
We currently only support vertical components with ``Z``, and horizontal components with ``E/2``/``N/1`` as the last character in their channel names. We hope to de-restrict this in a similar way detailed below

Missing components
##################
We currently have zero tolerance for missing components. We aim to relax this criteria in version 1.1, whereby the user can stack onsets for an arbitrary number of seismic phase. We also aim to allow for stacking just P phases if the horizontal components are missing (and vice-versa).

Gap tolerance
#############
We currently have zero tolerance for gappy data - the possible causes of gaps in data are numerous, and it is recommended the user makes an effort to identify the root cause. We hope in future to allow for some gap tolerance, for which gap will be spanned by linearly interpolating between the samples either side of the gap.

This also applies to data missing at the start/end of a `timestep`.

Flatlines
#########
Some archives will choose to fill any gaps in their waveform data with flatline values. If, for a given `timestep`, the data all have the same value, they are rejected.
