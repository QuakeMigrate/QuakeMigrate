The Locate Stage
================
This tutorial covers the basic ideas and definitions underpinning the final stage of a QuakeMigrate run—Locate.

In this stage, the candidate events identified during the Trigger stage are used to re-migrate waveform data for a short period of time around the estimated event origin time into a finer resolution search volume. The resulting 4-D coalescence function is then used to estimate the spatial uncertainty in the event location. A number of additional 'plug-in' stages can be simultaneously performed to extract, e.g., event magnitude estimates and refined automatic phase arrival picks.

Before you start
----------------
The requirements are much the same as for the Detect stage—you will need an archive of waveform files organised into a regular structure (see the `Archive tutorial <https://quakemigrate.readthedocs.io/en/master/tutorials/archive.html>`_), a traveltime LUT (as generated in the earlier tutorial), and a station file (as used to generate the LUT).

.. note:: Your ``run_path`` and ``run_name`` variables should be identical to those used in the Detect run. QuakeMigrate will automatically generate suitable output directories for the output from Trigger.

We proceed by defining these parameters as variables.

::

    from quakemigrate.io import read_stations


    archive_path = "/path/to/archived/data"
    lut_file = "/path/to/lut_file"
    station_file = "/path/to/station_file"

    run_path = "/path/to/output"
    run_name = "name_of_run"
    
    stations = read_stations(station_file)

Locate can be run in two ways: either focussing on all events between two specified timestamps (``starttime`` and ``endtime``), or by providing an event file (which must be in the same format as the triggered event file output by the Trigger stage).

The waveform archive is defined using an :class:`Archive` object (`see Archive tutorial <https://quakemigrate.readthedocs.io/en/master/tutorials/archive.html>`_) and the saved LUT can be loaded using the :func:`quakemigrate.io.read_lut` function.

::

    from quakemigrate.io import Archive, read_lut

    
    archive = Archive(
        archive_path=archive_path,
        stations=stations,
        archive_format="YEAR/JD/STATION",
    )
    lut = read_lut(lut_file=lut_out)

Onset Functions in Locate
-------------------------
As with the Detect stage, it is necessary to specify an Onset function to use to transform the raw waveform data into a reduced form. For further details, please refer to the `Detect tutorial <https://quakemigrate.readthedocs.io/en/master/tutorials/detect.html>`_. When using the default :class:`STALTAOnset` in the Locate stage, it is recommended to use the ``centred`` position (as opposed to the ``classic`` position).

Locate parameters
-----------------
The Locate stage of QuakeMigrate can be run with relatively few parameters. There are, however, a number of add-on stages that can take some refinement—more later on that.

::

    from quakemigrate import QuakeScan


    scan = QuakeScan(
        archive,
        lut,
        onset=onset,
        run_name=run_name,
        log=True,
        loglevel="info",
    )
    scan.marginal_window = 0.1
    scan.threads = 12

The ``marginal_window`` parameter is used to define the time period over which the 4-D coalescence function should be 'marginalised'—that is, stacked over the time dimension. The resulting 3-D marginalised coalescence function is then used to estimate the spatial uncertainty in the event hypocentre. The marginal window should be an estimate of the error in the origin time due to the expected spatial error of your seismicity and error in the velocity model, which can be roughly assessed by looking at the width of the peak in the 1-D maximum coalescence function.

The ``threads`` parameter controls the number of CPU threads you wish to make available for detect to use when migrating and stacking the waveform data. If you wish to use your computer for other work while running QuakeMigrate, you may find it useful to leave some of your cores free.

Starting your Locate run
------------------------

::

    scan.locate(starttime, endtime)
    # or
    scan.locate(trigger_file="path_to_triggered_events_file")

A log will be printed to ``STDOUT`` which summarises the chosen parameters for your run. As the computation proceeds, the event being analysed will be printed to the terminal along with some additional information, including timing etc.
