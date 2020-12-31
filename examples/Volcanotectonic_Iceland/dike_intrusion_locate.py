# -*- coding: utf-8 -*-
"""
This script runs the locate stage for the Iceland dike intrusion example.

"""

# Stop numpy using all available threads (these environment variables must be
# set before numpy is imported for the first time).
import os
os.environ.update(OMP_NUM_THREADS="1",
                  OPENBLAS_NUM_THREADS="1",
                  NUMEXPR_NUM_THREADS="1",
                  MKL_NUM_THREADS="1")

from obspy.core import AttribDict

from quakemigrate import QuakeScan
from quakemigrate.io import Archive, read_lut, read_stations, read_response_inv
from quakemigrate.signal.onsets import STALTAOnset
from quakemigrate.signal.pickers import GaussianPicker
from quakemigrate.signal.local_mag import LocalMag

# --- i/o paths ---
station_file = "./inputs/iceland_stations.txt"
response_file = "./inputs/Z7_dataless.xml"
data_in = "./inputs/mSEED"
lut_file = "./outputs/lut/dike_intrusion.LUT"
run_path = "./outputs/runs"
run_name = "example_run"

# --- Set time period over which to run locate ---
starttime = "2014-08-24T00:01:00.0"
endtime = "2014-08-24T00:11:00.0"

# --- Read in station file ---
stations = read_stations(station_file)

# --- Read in response inventory ---
response_inv = read_response_inv(response_file)

# --- Specify parameters for response removal ---
response_params = AttribDict()
response_params.pre_filt = (0.05, 0.06, 30, 35)
response_params.water_level = 600

# --- Create new Archive and set path structure ---
archive = Archive(archive_path=data_in, stations=stations,
                  archive_format="YEAR/JD/STATION", response_inv=response_inv,
                  response_removal_params=response_params)

# --- Specify parameters for amplitude measurement ---
amp_params = AttribDict()
amp_params.signal_window = 5.0
amp_params.highpass_filter = True
amp_params.highpass_freq = 2.0

# --- Specify parameters for magnitude calculation ---
mag_params = AttribDict()
mag_params.A0 = "Greenfield2018_bardarbunga"
mag_params.amp_feature = "S_amp"

mags = LocalMag(amp_params=amp_params, mag_params=mag_params,
                plot_amplitudes=True)

# --- Load the LUT ---
lut = read_lut(lut_file=lut_file)

# --- Create new Onset ---
onset = STALTAOnset(position="centred", sampling_rate=50)
onset.phases = ["P", "S"]
onset.bandpass_filters = {
    "P": [2, 16, 2],
    "S": [2, 16, 2]}
onset.sta_lta_windows = {
    "P": [0.2, 1.0],
    "S": [0.2, 1.0]}

# --- Create new PhasePicker ---
picker = GaussianPicker(onset=onset)
picker.plot_picks = True

# --- Create new QuakeScan ---
scan = QuakeScan(archive, lut, onset=onset, picker=picker, mags=mags,
                 run_path=run_path, run_name=run_name, log=True,
                 loglevel="info")

# --- Set locate parameters ---
# For a complete list of parameters and guidance on how to choose them, please
# see the manual and read the docs.
scan.marginal_window = 1.0
scan.threads = 12

# --- Toggle plotting options ---
scan.plot_event_summary = True
scan.xy_files = "./inputs/XY_FILES/dike_xyfiles.csv"

# --- Toggle writing of waveforms ---
scan.write_cut_waveforms = True

# --- Run locate ---
scan.locate(starttime=starttime, endtime=endtime)
