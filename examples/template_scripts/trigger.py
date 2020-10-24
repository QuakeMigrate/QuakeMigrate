# -*- coding: utf-8 -*-
"""
This script will run the trigger stage of QuakeMigrate.

For more details, please see the manual and read the docs.

"""

from quakemigrate.io import read_lut
from quakemigrate.signal import Trigger

# --- i/o paths ---
lut_file = "/path/to/lut_file"
run_path = "/path/to/output"
run_name = "name_of_run"

# --- Set time period over which to run trigger ---
starttime = "2018-001T00:00:00.0"
endtime = "2018-002T00:00:00.00"

# --- Load the LUT ---
lut = read_lut(lut_file=lut_file)

# --- Create new Trigger ---
trig = Trigger(lut, run_path=run_path, run_name=run_name, log=True,
               loglevel="info")

# --- Set trigger parameters ---
# For a complete list of parameters and guidance on how to choose them, please
# see the manual and read the docs.
trig.marginal_window = 1.
trig.min_event_interval = 30.
trig.normalise_coalescence = True

# --- Static threshold ---
trig.threshold_method = "static"
trig.static_threshold = 1.75

# --- Dynamic (Median Absolute Deviation) threshold ---
# trig.threshold_method = "dynamic"
# trig.mad_window_length = 7200.
# trig.mad_multiplier = 8.

# --- Toggle plotting options ---
trig.plot_trigger_summary = True

# --- Run trigger ---
trig.trigger(starttime, endtime, interactive_plot=True)
