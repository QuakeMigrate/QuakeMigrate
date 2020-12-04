# -*- coding: utf-8 -*-
"""
This script runs the trigger stage for the Rutford icequake example.

"""

# Stop numpy using all available threads (these environment variables must be
# set before numpy is imported for the first time).
import os
os.environ.update(OMP_NUM_THREADS="1",
                  OPENBLAS_NUM_THREADS="1",
                  NUMEXPR_NUM_THREADS="1",
                  MKL_NUM_THREADS="1")

from quakemigrate import Trigger
from quakemigrate.io import read_lut

# --- i/o paths ---
lut_file = "./outputs/lut/icequake.LUT"
run_path = "./outputs/runs"
run_name = "icequake_example"

# --- Set time period over which to run trigger ---
starttime = "2009-01-21T04:00:05.0"
endtime = "2009-01-21T04:00:15.0"

# --- Load the LUT ---
lut = read_lut(lut_file=lut_file)

# --- Create new Trigger ---
trig = Trigger(lut, run_path=run_path, run_name=run_name, log=True,
               loglevel="info")

# --- Set trigger parameters ---
# For a complete list of parameters and guidance on how to choose them, please
# see the manual and read the docs.
trig.marginal_window = 0.1
trig.min_event_interval = 0.5
trig.normalise_coalescence = True

# --- Static threshold ---
trig.threshold_method = "static"
trig.static_threshold = 2.75

# --- Dynamic (Median Absolute Deviation) threshold ---
# trig.threshold_method = "dynamic"
# trig.mad_window_length = 7200.
# trig.mad_multiplier = 8.

# --- Run trigger ---
trig.trigger(starttime, endtime, interactive_plot=True)
