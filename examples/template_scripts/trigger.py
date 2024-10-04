# -*- coding: utf-8 -*-
"""
This script demonstrates how to run the trigger stage of QuakeMigrate.

For more details, please see the manual and read the docs.

"""

# Stop numpy using all available threads (these environment variables must be
# set before numpy is imported for the first time).
import os

os.environ.update(
    OMP_NUM_THREADS="1",
    OPENBLAS_NUM_THREADS="1",
    NUMEXPR_NUM_THREADS="1",
    MKL_NUM_THREADS="1",
)

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
trig = Trigger(lut, run_path=run_path, run_name=run_name, log=True, loglevel="info")

# --- Set trigger parameters ---
# For a complete list of parameters and guidance on how to choose them, please
# see the manual and read the docs.
trig.marginal_window = 1.0
trig.min_event_interval = 2.0
trig.normalise_coalescence = True

# --- Static threshold ---
trig.threshold_method = "static"
trig.static_threshold = 1.75

# --- Dynamic (Median Absolute Deviation) threshold ---
# trig.threshold_method = "mad"
# trig.mad_window_length = 7200.0
# trig.mad_multiplier = 8.0

# --- Dynamic (Median Ratio) threshold ---
# trig.threshold_method = "median_ratio"
# trig.median_window_length = 300.0
# trig.median_multiplier = 1.2

# --- Apply smoothing to coalescence trace before triggering ---
trig.smooth_coa = True
trig.smoothing_kernel_sigma = 0.2
trig.smoothing_kernel_width = 2

# --- Toggle plotting options ---
trig.plot_trigger_summary = True
# It is possible to supply xy files to enhance and give context to the
# trigger summary map plots. See the volcano-tectonic example from Iceland
# for details.
# trig.xy_files = "/path/to/xy_csv"
# trig.plot_all_stns = False

# --- Run trigger ---
# NOTE: It is possible to specify an optional spatial filter to restrict the
# triggered events to a geographic region. Only candidate events that fall
# within this geographic area will be retained. This is useful for removing
# clear artefacts; for example at the very edges of the grid. See the
# volcano-tectonic example from Iceland for details.
trig.trigger(starttime, endtime, interactive_plot=False)
