"""
This script runs Trigger for the Askja volcano (Iceland) Volcanotectonic (VT)
& Deep-Long-Period (DLP) event example.

:copyright:
    2020–2025, QuakeMigrate developers.
:license:
    GNU General Public License, Version 3
    (https://www.gnu.org/licenses/gpl-3.0.html)

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

from quakemigrate import Trigger
from quakemigrate.io import read_lut

# --- i/o paths ---
lut_file = "./outputs/lut/askja.LUT"
run_path = "./outputs/runs"
run_name = "example_run"

# --- Set time period over which to run trigger ---
starttime = "2011-10-26T17:35:00.0"
endtime = "2011-10-26T18:05:00.0"

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
trig.static_threshold = 1.45

# --- Dynamic (Median Absolute Deviation) threshold ---
# trig.threshold_method = "dynamic"
# trig.mad_window_length = 300.
# trig.mad_multiplier = 5.

# --- Apply smoothing to coalescence trace before triggering ---
trig.smooth_coa = True
trig.smoothing_kernel_sigma = 0.25
trig.smoothing_kernel_width = 2

# --- Toggle plotting options ---
trig.plot_trigger_summary = True
trig.xy_files = "./inputs/XY_FILES/askja_xyfiles.csv"

# --- Run trigger ---
# NOTE: here we use the optional "region" keyword argument to specify a spatial
# filter for the triggered events. Only candidate events that fall within this
# geographic area will be retained. This is useful for removing clear
# artefacts; for example at the very edges of the grid.
trig.trigger(
    starttime,
    endtime,
    interactive_plot=False,
    region=[-17.1, 64.95, -3.0, -16.0, 65.30, 30.0],
)
