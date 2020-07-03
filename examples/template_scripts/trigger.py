# -*- coding: utf-8 -*-
"""
This script will run the trigger stage of QuakeMigrate.

For more details, please see the manual and read the docs.

"""

from QMigrate.lut import LUT
from QMigrate.signal import Trigger

# --- i/o paths ---
lut_file = "/path/to/lut_file"
run_path = "/path/to/output"
run_name = "name_of_run"

# --- Set time period over which to run trigger ---
starttime = "2018-001T00:00:00.0"
endtime = "2018-002T00:00:00.00"

# --- Load the LUT ---
lut = LUT(lut_file=lut_file)

# --- Create new Trigger ---
trig = Trigger(lut, run_path=run_path, run_name=run_name)

# --- Set trigger parameters ---
# For a complete list of parameters and guidance on how to choose them, please
# see the manual and read the docs.
trig.marginal_window = 1.
trig.minimum_repeat = 30.
trig.normalise_coalescence = True

# --- Static threshold ---
trig.threshold_method = "static"
trig.static_threshold = 1.75

# --- Dynamic (Median Absolute Deviation) threshold ---
# trig.threshold_method = "dynamic"
# trig.mad_window_length = 7200.
# trig.mad_multiplier = 8.

# --- Run trigger ---
trig.trigger(starttime, endtime, savefig=False)
