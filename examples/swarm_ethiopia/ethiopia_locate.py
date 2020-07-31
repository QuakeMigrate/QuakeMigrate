# -*- coding: utf-8 -*-
"""
Locate stage for the Ethiopia swarm example.

"""

from quakemigrate.io import Archive, read_lut, read_stations
from quakemigrate.signal import QuakeScan
from quakemigrate.signal.onset import STALTAOnset
from quakemigrate.signal.pick import GaussianPicker
from quakemigrate.signal.local_mag import LocalMag

# --- i/o paths ---
station_file = "./inputs/ethiopia_stations_TM.csv"
data_in = "./inputs/mSEED"
lut_out = "./outputs/lut/ethiopia.LUT"
run_path = "./outputs/runs"
run_name = "example_run"

# --- Set time period over which to run locate ---
starttime = "2016-097T18:31:00"
endtime = "2016-097T18:44:00"

# --- Read in station file ---
stations = read_stations(station_file)

# --- Create new Archive and set path structure ---
archive = Archive(archive_path=data_in, stations=stations,
                  format='{year}/{jday:03d}/{network}.{station}*',
                  catch_network=True)

# --- Load the LUT ---
lut = read_lut(lut_file=lut_out)

# --- Create new Onset ---
onset = STALTAOnset(position="centred")
onset.p_bp_filter = [1., 10., 2]
onset.s_bp_filter = [1., 10., 2]
onset.p_onset_win = [0.7, 5.]
onset.s_onset_win = [0.7, 5.]

# --- Create new PhasePicker ---
picker = GaussianPicker(onset=onset)
picker.marginal_window = 1.
picker.plot_picks = True

# --- Create new LocalMag object ---
amp_params = {
    "signal_window" : 2.,
    "water_level" : 60.
}
mag_params = {
    "A0" : 'keir2006',
    'use_hyp_dist' : True,
}
localmag = LocalMag(amp_params, mag_params)

# --- Create new QuakeScan ---
scan = QuakeScan(archive, lut, onset=onset, picker=picker,
                 mags=localmag, run_path=run_path, 
                 run_name=run_name, log=True, loglevel="info")

# --- Set locate parameters ---
# For a complete list of parameters and guidance on how to choose them, please
# see the manual and read the docs.
scan.marginal_window = 1.
scan.threads = 20
scan.sampling_rate = 50

# --- Toggle plotting options ---
scan.plot_event_video = False
scan.plot_event_summary = True

# --- Toggle writing of waveforms ---
scan.write_cut_waveforms = False

# --- Run locate ---
# scan.locate(starttime=starttime, endtime=endtime)
scan.locate(trigger_file='1event.csv')
