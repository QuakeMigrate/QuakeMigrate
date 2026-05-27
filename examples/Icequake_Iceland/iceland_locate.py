"""
This script runs the locate stage for the Iceland icequake example.

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

from quakemigrate import QuakeScan
from quakemigrate.clients import make_waveform_client
from quakemigrate.io import ARCHIVE_FORMATS, read_lut, read_stations
from quakemigrate.plugins.onsets.stalta import STALTAOnset
from quakemigrate.plugins.pickers import GaussianPicker
from quakemigrate.plugins.visualisation import EventSummary3DPlugin


# --- i/o paths ---
station_file = "./inputs/iceland_stations.txt"
data_in = "./inputs/mSEED"
lut_out = "./outputs/lut/example.LUT"
run_path = "./outputs/runs"
run_name = "example_run"

# --- Set time period over which to run locate ---
starttime = "2014-06-29T18:41:55.0"
endtime = "2014-06-29T18:42:20.0"

# --- Read in station file ---
stations = read_stations(station_file)

# --- Create new waveform client ---
client_config = {
    "client": "local",
    "path": data_in,
    "format": ARCHIVE_FORMATS["YEAR/JD/STATION"],
}
waveform_client = make_waveform_client(client_config)

# --- Load the LUT ---
lut = read_lut(lut_file=lut_out)

# --- Create new Onset ---
onset = STALTAOnset(position="centred", sampling_rate=250)
onset.phases = ["P", "S"]
onset.bandpass_filters = {"P": [10, 124, 4], "S": [10, 124, 4]}
onset.sta_lta_windows = {"P": [0.01, 0.25], "S": [0.05, 0.5]}

# --- Create new PhasePicker ---
picker = GaussianPicker(onset=onset)
picker.plot_picks = True

plugins = [picker, EventSummary3DPlugin(plot_all_stations=False)]

# --- Create new QuakeScan ---
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
scan.marginal_window = 0.06
scan.threads = 4  # NOTE: increase as your system allows to increase speed!

# --- Toggle writing of waveforms ---
scan.write_cut_waveforms = True

# --- Run locate ---
scan.locate(stations, starttime=starttime, endtime=endtime)
