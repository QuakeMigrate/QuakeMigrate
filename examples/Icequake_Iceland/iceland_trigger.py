# -*- coding: utf-8 -*-
"""
Trigger stage for the Iceland icequake example.

"""

from QMigrate.io import read_lut
from QMigrate.signal import Trigger

# --- i/o paths ---
lut_file = "./outputs/lut/example.LUT"
run_path = "./outputs/runs"
run_name = "example_run"

# --- Set time period over which to run trigger ---
starttime = "2014-06-29T18:41:55.0"
endtime = "2014-06-29T18:42:20.0"

# --- Load the LUT ---
lut = read_lut(lut_file=lut_file)

# --- Create new Trigger ---
trig = Trigger(lut, run_path=run_path, run_name=run_name, log=True)

# --- Set trigger parameters ---
# For a complete list of parameters and guidance on how to choose them, please
# see the manual and read the docs.
trig.marginal_window = 1.
trig.minimum_repeat = 6.
trig.normalise_coalescence = True

# --- Static threshold ---
trig.threshold_method = "static"
trig.static_threshold = 1.8

# --- Dynamic (Median Absolute Deviation) threshold ---
# trig.threshold_method = "dynamic"
# trig.mad_window_length = 7200.
# trig.mad_multiplier = 8.

# --- Run trigger ---
trig.trigger(starttime, endtime, savefig=False)
