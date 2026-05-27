"""
This script demonstrates how to run the locate stage of QuakeMigrate.

For more details, please see the manual and read the docs.

:copyright:
    2020–2026, QuakeMigrate developers.
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

from obspy.core import AttribDict

from quakemigrate import QuakeScan
from quakemigrate.clients import make_waveform_client
from quakemigrate.io import ARCHIVE_FORMATS, read_lut, read_stations
from quakemigrate.plugins.onsets.stalta import STALTAOnset
from quakemigrate.plugins.pickers import GaussianPicker
from quakemigrate.plugins.magnitudes import (
    AmplitudeConfig,
    MagnitudeConfig,
    build_local_magnitude_plugin,
)
from quakemigrate.plugins.visualisation import EventSummary3DPlugin


# --- i/o paths ---
archive_path = "/path/to/archived/data"
lut_file = "/path/to/lut"
station_file = "/path/to/station_file"
response_file = "/path/to/response_file"

run_path = "/path/to/output"
run_name = "name_of_run"

# --- Set time period over which to run locate ---
starttime = "2018-001T00:00:00.0"
endtime = "2018-002T00:00:00.0"

# --- Read in station file ---
stations = read_stations(station_file)

# --- Specify parameters for response removal ---
# All parameters are optional, though one of `water_level` and `pre_filt` is
# recommended - see the documentation for a complete guide.
response_params = AttribDict()
response_params.pre_filt = (0.05, 0.06, 30, 35)
response_params.water_level = 600
response_params.remove_full_response = False

# --- Create new waveform client ---
client_config = {
    "client": "local",
    "path": archive_path,
    "format": ARCHIVE_FORMATS["YEAR/JD/STATION"],
    "inventory_path": response_file,
    "response_removal_params": response_params,
}
waveform_client = make_waveform_client(client_config)

# --- Resample data with mismatched sampling rates ---
# archive.resample = True
# archive.upfactor = 2

# --- Load the LUT ---
lut = read_lut(lut_file=lut_file)

# --- Decimate the lookup table ---
# lut = lut.decimate([1, 1, 1])

# --- Create new Onset function ---
onset = STALTAOnset(position="centred", sampling_rate=50)
onset.phases = ["P", "S"]
onset.bandpass_filters = {"P": [2, 16, 2], "S": [2, 14, 2]}
onset.sta_lta_windows = {"P": [0.2, 1.0], "S": [0.2, 1.0]}

# --- Create new PhasePicker ---
# For a complete list of parameters and guidance on how to choose them, please
# see the manual and read the docs.
# Optionally, define a new onset function for the picker. You can also re-use
# the onset used for migration (as is done below).
# gausspicker_onset = STALTAOnset(position="centred", sampling_rate=100)
# gausspicker_onset.phases = ["P", "S"]
# gausspicker_onset.bandpass_filters = {
#     "P": [2, 16, 2],
#     "S": [2, 14, 2]}
# gausspicker_onset.sta_lta_windows = {
#     "P": [0.2, 1.0],
#     "S": [0.2, 1.0]}

picker = GaussianPicker(onset=onset)
picker.plot_picks = True

# --- Create new LocalMag object ---
# All parameters are optional: see the documentation for a complete guide.
amp_params = AmplitudeConfig(
    signal_window=5.0,
    bandpass_filter=True,
    bandpass_lowcut=2.0,
    bandpass_highcut=20.0,
)

# A0 attenuation function is required: see the documentation for several
# built-in options, or specify your own function. All other parameters are
# optional - see the documentation for a complete guide.
mag_params = MagnitudeConfig(
    A0="Hutton-",
    use_hyp_dist=False,
    amp_feature="S_amp",
    station_corrections=dict(),
    trace_filter=".[BH]H[NE]$",
    noise_filter=3.0,
    station_filter=["KVE", "LIND"],  # List of stations to exclude.
    dist_filter=False,
)

mags = build_local_magnitude_plugin(
    amp_params=amp_params, mag_params=mag_params, plot_amplitudes=True
)

# It is possible to supply xy files to enhance and give context to the
# event summary map plots. See the volcano-tectonic example from Iceland
# for details.
# scan.xy_files = "/path/to/xy_csv"
event_summary = EventSummary3DPlugin(
    # overlay_manifest="/path/to/overlay_manifest.csv"
)

plugins = [picker, mags, event_summary]

# --- Create new QuakeScan ---
# If you do not want to calculate local magnitudes, specify `mags=None`
scan = QuakeScan(
    waveform_client,
    lut,
    onset=onset,
    plugins=plugins,
    run_path=run_path,
    run_name=run_name,
    log=True,
    loglevel="info",
)

# --- Set locate parameters ---
# For a complete list of parameters and guidance on how to choose them, please
# see the manual and read the docs.
scan.marginal_window = 1
# NOTE: increase the thread-count as your system allows. The core migration
# routines are compiled against OpenMP, so the compute time (particularly for
# detect), will decrease roughly linearly with the number of threads used.
scan.threads = 4

# --- Toggle writing of cut waveforms ---
scan.write_cut_waveforms = True
scan.pre_cut = 20.0
scan.post_cut = 60.0

# --- Run locate ---
# Between two timestamps
scan.locate(stations, starttime=starttime, endtime=endtime)
# From a triggered events file.
# scan.locate(stations, trigger_file="filename_of_triggered_events")
