#!/usr/bin/env python
# coding: utf-8

# # QuakeMigrate - Example - Icequake detection

# ## Overview:

# This notebook shows how to run QuakeMigrate for icequake detection, using a 2 minute window of continuous seismicity from Hudson et al (2019). Please refer to this paper for details and justification of the settings used.
# 
# Here, we detail how to:
# 1. Create a travel-times lookup table for the example seismometer network
# 2. Run the detect stage to coalesce energy through time
# 3. Run the trigger stage to determine events above a threshold value
# 4. Run the locate stage to refine the earthquake location
# 
# We also provide an outline of some of the key outputs

# In[3]:


# Import necessary modules:
import QMigrate.core.model as cmod
import QMigrate.signal.scan as cscan
import QMigrate.io.mseed as cmseed


# In[ ]:


# Set i/o paths:
stat_in = "./inputs/stations.txt"
data_in = "./inputs/mSEED"
lut_out = "./outputs/lut/icequake.LUT"
output  = "./outputs/runs/icequake"


# ## 1. Create a travel-times lookup table (LUT)

# In[7]:


# Set the parameters for the travel-times lookup table (LUT)
# Cell count (x,y,z); cell size (x,y,z in metres)
lut = cmod.LUT(cell_count=[20, 20, 140], cell_size=[100, 100, 20])
lut.lonlat_centre(-17.224, 64.328)

# Set the LUT projection (here we use the Lambert Conformal Conic projection)
lut.lcc_standard_parallels = (64.32, 64.335)
lut.projections(grid_proj="LCC")

# Add stations to LUT
lut.stations(path=stat_in, units="lat_lon_elev")

# Compute for a homogeneous velocity model
v_p_homo_model = 3630
v_s_homo_model = 1833
lut.compute_homogeneous_vmodel(v_p_homo_model, v_s_homo_model)

# Save the LUT
lut.save(lut_out)


# ## 2. Coalesce the seismic energy through time

# In[11]:


# Create a new instance of the MSEED class and set path structure
data = cmseed.MSEED(lut_out, HOST_PATH=data_in)
data.path_structure(path_type="YEAR/JD/*_STATION")

# Create a new instance of the SeisScan class
scn = cscan.SeisScan(data, lut_out, output_path=output, output_name="icequake_example")


# In[ ]:


# Set detect parameters
scn.sampling_rate = 500           # Sampling rate of data, in Hz
scn.p_bp_filter   = [10, 125, 4]  # The band-pass filter parameters for the P-phase (10 to 125 Hz, with 4th order corners)
scn.s_bp_filter   = [10, 125, 4]  # The band-pass filter parameters for the P-phase (10 to 125 Hz, with 4th order corners)
scn.p_onset_win   = [0.01, 0.25]  # Length of the STA and LTA time windows for the P-phase
scn.s_onset_win   = [0.05, 0.5]   # Length of the STA and LTA time windows for the S-phase
scn.time_step     = 0.75          # The length of the time-step
scn.decimate      = [1,1,1]       # Decimation factors in x,y,z (no decimation here)
scn.n_cores       = 12            # Number of cores/processors to use

# Defining the start and end times 
starttime = "2014-06-29T18:41:55.0"
endtime   = "2014-06-29T18:42:20.0"


# In[13]:


# Run the detect stage to find the coalescence of energy through time:
scn.detect(starttime, endtime)


# ## 3. Run the trigger stage, to detect and output individual icequakes
# 
# nb: We can use the same SeisScan object here because we are not using a different decimation. If running trigger and locate on grids with different levels of decimation, a new SeisScan object must be initialised.

# In[15]:


# Set trigger parameters:
scn.detection_threshold   = 1.5   # SNR threshold for the coalescence through time. Will detect an event if the coalescence goes above this for a given timestep
scn.marginal_window       = 2.75  # The length of the time-step window, + pre and post padding (i.e. 0.75 sec time-step window + 1s padding either side)
scn.minimum_repeat        = 5.0
scn.normalise_coalescence = True

# Turn on plotting features
scn.plot_coal_video      = False
scn.plot_coal_grid       = False
scn.plot_coal_picture    = True
scn.plot_coal_trace      = False


# In[16]:


# Run the trigger stage to find any events
scn.trigger(starttime, endtime)

# Run the locate stage to determine the location of any triggered events
scn.locate(starttime, endtime)


# ## 4. Some of the key outputs

# In[21]:


# Show the .event file, containing event origin time and location:
icequake_event_fname = "outputs/runs/icequake/icequake_example_20140629T184210208000.event"
with open(icequake_event_fname) as f:
    lines = f.readlines()
for line in lines:
    print(line)


# In[23]:


# Show the .stn file, containing station time picks:
icequake_stn_fname = "outputs/runs/icequake/icequake_example_20140629T184210208000.stn"
with open(icequake_stn_fname) as f:
    lines = f.readlines()
for line in lines:
    print(line)


# In[4]:


# Show the coalescence pdf file, containing event origin time and location:
icequake_coal_image_fname = "outputs/runs/icequake/icequake_example_20140629T184210208000_EventLocationError.pdf"
from IPython.display import IFrame # For plotting pdf
IFrame(icequake_coal_image_fname, width=800, height=400) # Plot pdf


# References:
# 
# Hudson, T.S., Smith, J., Brisbourne, A.M., and White R.S. (2019). Automated detection of basal icequakes and discrimination from surface crevassing. Annals of Glaciology, 79
