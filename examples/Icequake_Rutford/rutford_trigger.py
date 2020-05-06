# -*- coding: utf-8 -*-
"""
Trigger stage for the Rutford icequake example.

"""

from QMigrate.lut import LUT
from QMigrate.signal import Trigger

# --- i/o paths ---
lut_file = "./outputs/lut/icequake.LUT"
run_path = "./outputs/runs"
run_name = "icequake_example"

# --- Set time period over which to run trigger ---
starttime = "2009-01-21T04:00:05.0"
endtime = "2009-01-21T04:00:15.0"

# --- Load the LUT ---
lut = LUT(lut_file=lut_file)

# --- Create new Trigger ---
trig = Trigger(lut, run_path=run_path, run_name=run_name)

# --- Set trigger parameters ---
# For a complete list of parameters and guidance on how to choose them, please
# see the manual and read the docs.
trig.marginal_window = 0.1
trig.minimum_repeat = 0.5
trig.normalise_coalescence = True

# --- Static threshold ---
trig.threshold_method = "static"
trig.static_threshold = 2.75

# --- Dynamic (Median Absolute Deviation) threshold ---
# trig.threshold_method = "dynamic"
# trig.mad_window_length = 7200.
# trig.mad_multiplier = 8.

# --- Run trigger ---
trig.trigger(starttime, endtime, savefig=False)
