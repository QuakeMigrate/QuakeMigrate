# ~~~~~ Core Packages ~~~~~~~~~
import QMigrate.signal.scan as cscan
import QMigrate.io.mseed as cmseed
import pandas as pd
import sys
import os

DATA    = '../../seisloc_fiddling/ethiopia_example/mseed/'

OUTPUT = 'out'

if not os.path.exists(OUTPUT):
    print('Output directory does not exist... making dir {}'.format(OUTPUT))
    os.makedirs(OUTPUT)


NAME    = 'RiftVolc'

#START   = '2017-01-25T00:00:00.0'
#END     = '2017-02-03T23:59:59.9'

START   = '2017-01-27T00:00:00.0'
END     = '2017-01-28T00:00:00.0'

PROCS = 18

LUT = 'riftvolc.lut'

print('------- Reading MSEED Structure ------')
DATA = cmseed.MSEED(LUT,HOST_PATH=DATA)
DATA.path_structure(path_type='YEAR/JD/STATION')

scn = cscan.SeisScan(DATA,LUT,output_path=OUTPUT,output_name=NAME)


# CONOR ##
scn.sampling_rate            = 50
scn.p_bp_filter              = [2., 12., 4]
scn.s_bp_filter              = [2., 12., 4]
scn.p_onset_win              = [0.2, 3.]
scn.s_onset_win              = [0.3, 4.5]

scn.normalise_coalescence    = True
scn.n_cores                  = PROCS

# ~~~~~ Detect Specific ~~~~~~~~~
scn.decimate        = [5, 5, 4]
scn.time_step       = 1200

scn.detect(START,END)

# ### JONNY
# scn.sample_rate              = 50
# scn.bp_filter_p1              = [2., 12., 4]
# scn.bp_filter_s1              = [2., 12., 4]
# scn.onset_win_p1              = [0.2, 3.]
# scn.onset_win_s1              = [0.3, 4.5]

# scn.NormaliseCoalescence    = True
# scn.NumberOfCores                  = PROCS

# # ~~~~~ Detect Specific ~~~~~~~~~
# scn.Decimate        = [5, 5, 4]
# scn.time_step       = 1200

# scn.detect(START, END)
