Reading waveform archives
=========================
This tutorial will provide instructions on how to direct QuakeMigrate to a local waveform archive and how to specify its structure. QuakeMigrate can handle any regularly structured waveform archive. Additional requirements can be handled on request - contact us at quakemigrate.developers@gmail.com or submit an Issue on our GitHub.

The Archive class
-----------------
The :class:`Archive` class provides methods for querying a waveform archive on a local system. It is capable of handling any regular archive structure, as well as any data file format that is compatible with `ObsPy`. Waveform data and an overview of the data availability (see :ref:`rejection-criteria-label`) are returned by a query to the archive.

It requires two pieces of information on instantiation:

- ``archive_path``: the path to seismic data archive
- ``stations``: a DataFrame containing station information. There is one required (case-sensitive) column header - "Name".

All other parameters can either be provided as arguments on instantiation, or set once the :class:`Archive` has been instantiated (see the section on specifying the archive structure below for an example).

Here we create a new instance of :class:`Archive`.

::

    from quakemigrate.io import Archive, read_stations

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
It is not uncommon for a data archive to contain stations with differing sampling rates. QuakeMigrate, however, performs the core migration and stacking routine at a single, unified sampling rate. As such, we have bundled methods for accomplishing this automatically, resampling the waveform data to the specified sampling rate as it is read in. These routines aim to minimally alter the values of the waveforms by retaining as much of the original data as possible. Downsampling from 100 Hz to 50 Hz, for example, is accomplished by decimating the waveforms by a factor of two - skipping every other sample. If the unified sampling rate is not an integer divisor of the input waveform sampling rate, there is (limited) scope to linearly interpolate the waveform data to a sampling rate that does divide into the unified sampling rate an integer number of times, then decimate down.

Resampling can be toggled on with ``archive.resample = True``, and a single factor by which to linearly interpolate data when resampling with ``archive.upfactor = 2``. We hope to de-restrict this in the future to allow for automatic identification of a suitable `upfactor` (within reason).

Instrument response
-------------------
While we do not need to remove the instrument response for the core migration and stacking routine - the default STA/LTA onset function implicitly handles this - if the user wishes to make use of the local magnitude calculation module, they must provide an inventory of instrument response functions. The :func:`quakemigrate.io.read_response_inv` function is a light wrapper for the :mod:`ObsPy` :func:`read_inventory` function. See their documentation for details of compatible formats.

In addition to the inventory of instrument response functions, the user can also set the water level, a pre-filter, and choose to remove the full response. 

.. _rejection-criteria-label:

Rejection criteria
------------------
We currently impose fairly strict criteria on the data to be used in QuakeMigrate, which are detailed below.

Gap tolerance
#############
It is possible to allow QuakeMigrate to use gappy data. We do not recommend using this without first assessing the waveform data and understanding the common causes of data gaps. This is currently set by toggling the `allow_gaps` parameter of the :class:`quakemigrate.signal.onsets.STALTAOnset` object to ``True``.

This also applies to data missing at the start/end of a `timestep`.

Flatlines
#########
Some archives will choose to fill any gaps in their waveform data with flatline values. If, for a given `timestep`, the data all have the same value, they are rejected.

Overlaps
########
If there is overlapping waveform data for a particular station component, it is not used.
