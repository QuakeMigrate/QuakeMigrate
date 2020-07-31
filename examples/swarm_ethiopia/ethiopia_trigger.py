# -*- coding: utf-8 -*-
"""
Trigger stage for the Iceland icequake example.

"""

from quakemigrate.io import read_lut
from quakemigrate.signal import Trigger

# --- i/o paths ---
lut_file = "./outputs/lut/ethiopia.LUT"
run_path = "./outputs/runs"
run_name = "example_run"

# --- Set time period over which to run trigger ---
starttime = "2016-097T18:31:00"
endtime = "2016-097T18:44:00"

# --- Load the LUT ---
lut = read_lut(lut_file=lut_file)

# --- Create new Trigger ---
trig = Trigger(lut, run_path=run_path, run_name=run_name, log=True,
               loglevel="info")

# --- Set trigger parameters ---
# For a complete list of parameters and guidance on how to choose them, please
# see the manual and read the docs.
trig.marginal_window = 1.
trig.min_event_interval = 10.
trig.normalise_coalescence = False

# --- Static threshold ---
trig.threshold_method = "static"
trig.static_threshold = 2.65

# --- Dynamic (Median Absolute Deviation) threshold ---
# trig.threshold_method = "dynamic"
# trig.mad_window_length = 7200.
# trig.mad_multiplier = 8.

# --- Run trigger ---
trig.trigger(starttime, endtime, savefig=False)
