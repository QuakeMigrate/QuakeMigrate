Running detect
==============

With the pre-requisite data and inputs prepared, we can proceed with the first stage of QuakeMigrate—Detect. For a more in-depth discussion of the various options for this stage, please refer to the dedicated section of the documentation.

The purpose of this initial stage is to perform an exhaustive, but low-resolution, search through an entire archive of waveform data. As with the other stages, the first portions of a run script concern the reading in and preparation of the inputs to the scan, including:

- a definition of the time period over which to scan;
- specification of the location of the waveform archive (more info here);
- and defining the "onset" function to be used (more info is available in the dedicated section of the documentation).

The time period for the search is defined to span that for which we produced synthetic waveforms in the previous stage. The important thing here is to note that these variables are converted into :class:`obspy.UTCDateTime` objects, and must therefore take some compatible form.

.. code-block:: python

    starttime = "2021-02-18T12:03:50.0"
    endtime = "2021-02-18T12:06:10.0"

When specifying the waveform archive, we can make use of one of the in-built archive formats (since we used this when writing the waveform files out in the previous tutorial). However, it is likely that, in many use cases, the waveform archive may take some other form—in such cases, it is possible to override the archive format directly by setting the ``.format`` attribute on the :class:`Archive` object directly using a Python f-string—that is, a formattable string:

.. code-block:: python

    archive = Archive(archive_path=data_in, stations=stations)
    archive.format = "{year}/{jday:03d}/{station}_*.m"

The above code block is equivalent to:

.. code-block:: python

    archive = Archive(
        archive_path=data_in, stations=stations, archive_format="YEAR/JD/STATION"
    )

We now read in the traveltime lookup table we created in an earlier step, before decimating—that is, reducing the resolution in the X, Y, and Z axes—in order to reduce the number of nodes in the search volume and hence reducing the computational resources (memory and compute time) used during the detect stage. This is possible because we only need to perform this initial search in a coarse manner. The seismic events of interest will still roughly coalesce above the background noise level at this resolution, and we return later to refine the estimated locations during the locate stage. In this instance, we reduce the resolution in X and Y by a factor of 4 (that is, 0.5 km -> 2 km node spacing), while leaving the Z resolution the same.

.. code-block:: python

    # --- Load the LUT ---
    lut = read_lut(lut_file=lut_out)
    lut.decimate([4, 4, 1], inplace=True)

The next section of the run script is where we specify the Onset function to be used. The Onset function, as is explored in greater detail elsewhere in our documentation and the paper accompaniment, is a function that transforms the input waveform data into some representation that amplifies the presence of the component of information of interest. In our case, we use a short-term to long-term average ratio (or STA/LTA) onset function that is sensitive to sharp changes in waveform amplitudes, such as one would expect to see for the impulsive phase arrivals. This function has been shown to possess Gaussian qualities, making the resultant output an effective, continuous description of the probability of a phase arrival at a given time.

We can tune such an Onset function to our specific dataset via a few parameters. Firstly, we specify the phases of interest. In the simplest case this could just be a single phase, such as the P-phase arrival. However, combining the P- and S-phase arrivals will often provide a greater degree of constraint on the earthquake location and is preferable when possible. For each selected phase, you must have a corresponding traveltime lookup table. After that, we tune the STA/LTA onset function to be maximally sensitive to the likely dominant frequency for each phase of interest. This is achieved using two parameters: a bandpass filter (here defined using a lowcut, highcut, and number of corners); and the lengths (in seconds) of the short-term and long-term windows.

.. code-block:: python

    # --- Create new Onset ---
    onset = STALTAOnset(position="centred", sampling_rate=50)
    onset.phases = ["P", "S"]
    onset.bandpass_filters = {"P": [1, 10, 2], "S": [1, 10, 2]}
    onset.sta_lta_windows = {"P": [0.2, 1.5], "S": [0.2, 1.5]}

Finally, we combine these definitions into a :class:`QuakeScan` object. We can also adjust the scan to ensure each stage fits within the available computational resources (e.g., if you are memory-limited) by adjusting the timestep—that is, the length of time used for each chunk of the migration and stacking stage—and the number of threads to be used in multiprocessing.

.. code-block:: python

    # --- Create new QuakeScan ---
    scan = QuakeScan(
        archive,
        lut,
        onset=onset,
        run_path=run_path,
        run_name=run_name,
        log=True,
        loglevel="info",
    )

    # --- Set detect parameters ---
    scan.timestep = 120
    scan.threads = 4  # NOTE: increase as your system allows to increase speed!

The run is started using:

.. code-block:: python

    # --- Run detect ---
    scan.detect(starttime, endtime)

As this runs, we output status information simultaneously to the standard output stream and a log file.

The full script looks like this:

.. code-block:: python

    """
    This script runs the detect stage for the synthetic example described in the tutorial
    in the online documentation. 

    :copyright:
        2020–2024, QuakeMigrate developers.
    :license:
        GNU General Public License, Version 3
        (https://www.gnu.org/licenses/gpl-3.0.html)

    """

    from quakemigrate import QuakeScan
    from quakemigrate.io import Archive, read_lut, read_stations
    from quakemigrate.signal.onsets import STALTAOnset


    # --- i/o paths ---
    station_file = "./inputs/synthetic_stations.sta"
    data_in = "./inputs/mSEED"
    lut_out = "./outputs/lut/example.LUT"
    run_path = "./outputs/runs"
    run_name = "example_run"

    # --- Set time period over which to run detect ---
    starttime = "2021-02-18T12:03:50.0"
    endtime = "2021-02-18T12:06:10.0"

    # --- Read in station file ---
    stations = read_stations(station_file)

    # --- Create new Archive and set path structure ---
    archive = Archive(
        archive_path=data_in, stations=stations, archive_format="YEAR/JD/STATION"
    )

    # --- Load the LUT ---
    lut = read_lut(lut_file=lut_out)
    lut.decimate([4, 4, 1], inplace=True)

    # --- Create new Onset ---
    onset = STALTAOnset(position="centred", sampling_rate=50)
    onset.phases = ["P", "S"]
    onset.bandpass_filters = {"P": [1, 10, 2], "S": [1, 10, 2]}
    onset.sta_lta_windows = {"P": [0.2, 1.5], "S": [0.2, 1.5]}

    # --- Create new QuakeScan ---
    scan = QuakeScan(
        archive,
        lut,
        onset=onset,
        run_path=run_path,
        run_name=run_name,
        log=True,
        loglevel="info",
    )

    # --- Set detect parameters ---
    scan.timestep = 120
    scan.threads = 4  # NOTE: increase as your system allows to increase speed!

    # --- Run detect ---
    scan.detect(starttime, endtime)
