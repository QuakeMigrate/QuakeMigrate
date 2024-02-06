Running trigger
===============

The primary output of the Detect stage is a ``.scanmseed`` file, which contains 5 streams of data:

1. the 1-D coalescence function, which is the maximum coalescence value—resulting from the migration and stacking of all computed onset functions—in the 3-D search volume at each time increment;
2. the normalised 1-D coalescence function, which is the above time-series divided by the average coalescence value in the 3-D search volume at each time increment;
3. and the X, Y, and Z coordinate of the maximum coalescence value at each time increment.

The Trigger stage is, in essence, a peak-finding task, for each spike in the (normalised) 1-D coalescence function indicates a point in time when the onset functions all stack coherently, i.e., an earthquake source. More details can be found in the dedicated Trigger section of the documentation and paper accompaniment. For our synthetic example (for which we know we only have one event), many of the parameters choices are relatively inconsequential, but additional annotations (indicated by the lines starting with ``#!``) have been added in the following:

.. code-block:: python

    # --- Set trigger parameters ---
    #! The marginal window specifies a length of time around the peak, over which the
    #! coalescence will be time-normalised
    trig.marginal_window = 1.0
    #! The minimum time between peaks for them to be considered distinct events.
    trig.min_event_interval = 6.0
    #! Specify whether to trigger on the normalised or non-normalised coalescence function
    trig.normalise_coalescence = True

    # --- Static threshold ---
    #! Here, a static threshold means peaks are only detected if they exceed a fixed
    #! coalescence value, in this instance 3.5. It's often informative to run trigger
    #! with a high value for the threshold first (10+), then inspect the output summary
    #! plot. With this, you can refine the threshold. The lower the threshold, the more
    #! likely you will trigger 'events' on noise.
    trig.threshold_method = "static"
    trig.static_threshold = 3.5

As with the previous steps, there are a few inputs that must be set, but these are entirely similar to inputs used previously in this synthetic example: the traveltime lookup table, and a starttime and endtime between which to seek peaks:

.. code-block:: python

    # --- i/o paths ---
    lut_file = "./outputs/lut/example.LUT"

    # --- Set time period over which to run trigger ---
    starttime = "2021-02-18T12:03:50.0"
    endtime = "2021-02-18T12:06:10.0"

    # --- Load the LUT ---
    lut = read_lut(lut_file=lut_file)

Some status information is output to the standard output stream and a log file. Since we have seeded the waveform data with a single synthetic event, we should expect to see a single candidate event reported by the Trigger stage. An overview of the approximate locations of all candidate events can be viewed in the corresponding trigger summary plot. For larger datasets, there will be a separate summary plot for each day of data.

The full script looks like this:

.. code-block:: python

    """
    This script runs the trigger stage for the synthetic example described in the tutorial
    in the online documentation. 

    :copyright:
        2020–2024, QuakeMigrate developers.
    :license:
        GNU General Public License, Version 3
        (https://www.gnu.org/licenses/gpl-3.0.html)

    """

    from quakemigrate import Trigger
    from quakemigrate.io import read_lut


    # --- i/o paths ---
    lut_file = "./outputs/lut/example.LUT"
    run_path = "./outputs/runs"
    run_name = "example_run"

    # --- Set time period over which to run trigger ---
    starttime = "2021-02-18T12:03:50.0"
    endtime = "2021-02-18T12:06:10.0"

    # --- Load the LUT ---
    lut = read_lut(lut_file=lut_file)

    # --- Create new Trigger ---
    trig = Trigger(lut, run_path=run_path, run_name=run_name, log=True, loglevel="info")

    # --- Set trigger parameters ---
    trig.marginal_window = 1.0
    trig.min_event_interval = 6.0
    trig.normalise_coalescence = True

    # --- Static threshold ---
    trig.threshold_method = "static"
    trig.static_threshold = 3.5

    # --- Run trigger ---
    trig.trigger(starttime, endtime, interactive_plot=False)
