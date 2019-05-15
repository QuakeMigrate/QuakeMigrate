# ~~~~~ Core Packages ~~~~~~~~~
import QMigrate.signal.scan as cscan
import QMigrate.io.mseed as cmseed
import pandas as pd
import sys


DATA    = '../../seisloc_fiddling/ethiopia_example/mseed/'

OUTPUT='out'

NAME    = 'RiftVolc'

START   = '2017-01-27T00:00:00.0'
END     = '2017-01-28T00:00:00.0'

PROCS = 12

LUT = 'riftvolc.lut'

print('------- Reading MSEED Structure ------')
DATA = cmseed.MSEED(LUT,HOST_PATH=DATA)
DATA.path_structure(path_type='YEAR/JD/STATION')

scn = cscan.SeisScan(DATA,LUT,output_path=OUTPUT,output_name=NAME)

scn.sampling_rate            = 50
scn.p_bp_filter              = [2., 12., 4]
scn.s_bp_filter              = [2., 12., 4]
scn.p_onset_win              = [0.2, 3.]
scn.s_onset_win              = [0.3, 4.5]

# ~~~~~ Trigger Specific ~~~~~~~~~
scn.decimate             = [1, 1, 1]
scn.detection_threshold   = 0.0005
scn.marginal_window       = 1.0
scn.min_repeat        = 30.0
scn.picking_mode          = 'Gaussian'
scn.pick_threshold        = 0.95
scn.normalise_coalescence    = False
scn.n_cores                  = PROCS

# Outputs
scn.plot_coal_grid      = False
scn.plot_coal_video     = False
scn.plot_coal_picture   = True
scn.plot_coal_trace     = True

scn.trigger(START, END, savefig=False)
scn.trigger(START,END)
scn.locate(START,END, cut_mseed=False)
