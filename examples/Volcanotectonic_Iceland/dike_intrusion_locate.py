"""
This script runs the locate stage for the Iceland dike intrusion example.

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
from quakemigrate.plugins.onsets import STALTAOnset
from quakemigrate.plugins.pickers import GaussianPicker
from quakemigrate.plugins.magnitudes import LocalMag
from quakemigrate.plugins.visualisation import EventSummary3DPlugin


# --- i/o paths ---
station_file = "./inputs/iceland_stations.txt"
response_file = "./inputs/DATALESS/Z7*.xml"
data_in = "./inputs/mSEED"
lut_file = "./outputs/lut/dike_intrusion.LUT"
run_path = "./outputs/runs"
run_name = "example_run"

# --- Set time period over which to run locate ---
starttime = "2014-08-24T00:01:00.0"
endtime = "2014-08-24T00:11:00.0"

# --- Read in station file ---
stations = read_stations(station_file)

# --- Specify parameters for response removal ---
response_params = AttribDict()
response_params.pre_filt = (0.05, 0.06, 30, 35)
response_params.water_level = 60.0
response_params.remove_full_response = False

# --- Create new waveform client ---
client_config = {
    "client": "local",
    "path": data_in,
    "format": ARCHIVE_FORMATS["YEAR/JD/STATION"],
    "inventory_path": response_file,
    "response_removal_params": response_params,
}
waveform_client = make_waveform_client(client_config)

# --- Specify parameters for amplitude measurement ---
amp_params = AttribDict()
amp_params.noise_window = 5.0
amp_params.noise_measure = "ENV"
amp_params.signal_window = 1.0
amp_params.bandpass_filter = True
amp_params.bandpass_lowcut = 2.0
amp_params.bandpass_highcut = 20.0
amp_params.filter_corners = 4

# --- Specify parameters for magnitude calculation ---
mag_params = AttribDict()
mag_params.A0 = "Greenfield2018_bardarbunga"
mag_params.use_hyp_dist = True
mag_params.amp_feature = "S_amp"
mag_params.trace_filter = ".*H[NE]$"
mag_params.noise_filter = 3.0

mags = LocalMag(amp_params=amp_params, mag_params=mag_params, plot_amplitudes=True)

# --- Load the LUT ---
lut = read_lut(lut_file=lut_file)

# --- Create new Onset ---
onset = STALTAOnset(
    position="centred", sampling_rate=50, signal_transform="env_squared"
)
onset.phases = ["P", "S"]
onset.bandpass_filters = {"P": [2, 16, 2], "S": [2, 16, 2]}
onset.sta_lta_windows = {"P": [0.2, 1.0], "S": [0.2, 1.0]}

# --- Create new PhasePicker ---
picker = GaussianPicker(onset=onset)
picker.plot_picks = False

event_summary = EventSummary3DPlugin(
    overlay_manifest="./inputs/XY_FILES/dike_xyfiles.csv"
)

plugins = [picker, mags, event_summary]

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
scan.marginal_window = 1.0
scan.threads = 4  # NOTE: increase as your system allows to increase speed!

# --- Toggle writing of waveforms ---
scan.write_cut_waveforms = True

# --- Run locate ---
scan.locate(stations, starttime=starttime, endtime=endtime)
