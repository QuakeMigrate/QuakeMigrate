
# coding: utf-8

# # QuakeMigrate - Example - Icequake detection

# ## Overview:

# This notebook shows how to run QuakeMigrate for icequake detection, using a 2 minute window of continuous seismicity from Hudson et al (2019). Please refer to this paper for details and justification of the settings used.
# 
# Here, we detail how to:
# 1. Create a travel-times lookup table for the example seismometer network
# 2. Run a stage to coalesce energy through time
# 3. Run a trigger stage determining events above a threshold value
# 4. Refine the earthquake location by running locate.
# 4. Outline of some of the key outputs

# ## 1. Create a travel-times lookup table (LUT)

# In[4]:


# Import neccessary modules:
import QMigrate.core.model  as cmod   # Velocity model generation functions
import QMigrate.signal.scan as cscan  # Detection and location algorithms
import QMigrate.io.mseed    as cmseed # MSEED data processing 
import pandas as pd


# In[7]:


# Set the parameters for the travel-times lookup table (LUT):
lut = cmod.LUT(center=[0.0,0.0,0.0], cell_count=[20,20,140], cell_size=[100,100,20], azimuth=0.0) # Create an empty LUT with a centre, cell count (x,y,z) and cell size (x,y,z in metres) specified
lut.set_lonlat(-17.224,64.328) # Set the lat and lon of the centre of the LUT
lut.lcc_standard_parallels=(64.32,64.335) # Set the LUT standard parallels
lut.setproj_wgs84('LCC') # Set the LUT projection
STATIONS = pd.read_csv('INPUTS/Stations.txt',delimiter=',') # Read in a file containing the station information
lut.set_station(STATIONS.as_matrix(),units='lat_lon_elev') # Set the station parameters for the LUT
lut_path = 'OUTPUTS/LUT/Icequake.LUT' # Set the path to save the LUT to
v_p_homo_model = 3630
v_s_homo_model = 1833


# In[9]:


# And compute and save the LUT:
lut.compute_Homogeous(v_p_homo_model,v_s_homo_model) # Compute for a homogeneous velocity model
lut.save(lut_path)


# ## 2. Coalesce the seismic energy through time

# In[10]:


# Read in the continuous seismic data:
DATA = cmseed.MSEED(lut_path,HOST_PATH='INPUTS/MSEED/Icequake') # Imports the continuous seismic data in
DATA.path_structure(TYPE='YEAR/JD/STATION')


# In[11]:


# Set the parameters for running the coalescence through time:
# Setup the coalescence object:
scn = cscan.SeisScan(DATA,lut_path,output_path='OUTPUTS/RUNS/Icequake',output_name='Icequake_example')
# Specify key detect/trigger parameters:
scn.sample_rate     = 500 # Sampling rate of data, in Hz
scn.bp_filter_p1    = [10, 125, 4] # The band-pass filter parameters for the P-phase (10 to 125 Hz, with 4th order corners)
scn.bp_filter_s1    = [10, 125, 4] # The band-pass filter parameters for the P-phase (10 to 125 Hz, with 4th order corners)
scn.onset_win_p1    = [0.01, 0.25] # Length of the STA and LTA time windows for the P-phase
scn.onset_win_s1    = [0.05, 0.5] # Length of the STA and LTA time windows for the S-phase
scn.time_step       = 0.75 # The length of the time-step
scn.CoalescenceGrid = False
scn.Decimate        = [1,1,1] # Decimation factors in x,y,z (no decimation here)
scn.NumberOfCores   = 12 # Number of cores/processors to use

# Defining the start and end times 
START = '2014-06-29T18:41:55.0'
END   = '2014-06-29T18:42:20.0'


# In[13]:


# Run SeisLoc to find the coalescence of energy through time:
# (Note: Outputs a .scn file with the overall coalesence value for each timestep)
scn.Detect(START,END) # Finds the coalescence of energy over the start and end times specified


# ## 3. Run the trigger stage, to detect and output individual icequakes

# In[15]:


# Set any trigger parameters that may be different/additional to the initial coalescence stage:
scn.DetectionThreshold    = 1.5 # SNR threshold for the coalescence through time. Will detect an event if the coalescence goes above this for a given timestep
scn.MarginalWindow        = 2.75 # The length of the time-step window, + pre and post padding (i.e. 0.75 sec time-step window + 1s padding either side)
scn.MinimumRepeat         = 5.0
scn.NormalisedCoalescence = True 

# Various output boolian switches:
scn.CoalescenceVideo      = False
scn.CoalescenceGrid       = False
scn.CoalescencePicture    = True
scn.CoalescenceTrace      = False


# In[16]:


scn.Trigger(START,END)# Triggers events, outputing .event, .stn and .pdf for each event in the directory SeisLoc_outputs/RUNS/Icequake
scn.Locate(START,END)


# ## 4. Some of the key outputs

# In[21]:


# Show the .event file, containing event origin time and location:
icequake_event_fname = "OUTPUTS/RUNS/Icequake/Icequake_example_20140629184210336.event"
with open(icequake_event_fname) as f:
    lines = f.readlines()
for line in lines:
    print(line)


# In[23]:


# Show the .stn file, containing station time picks:
icequake_stn_fname = "OUTPUTS/RUNS/Icequake/Icequake_example_20140629184210336.stn"
with open(icequake_stn_fname) as f:
    lines = f.readlines()
for line in lines:
    print(line)


# In[31]:


# Show the coalescence pdf file, containing event origin time and location:
icequake_coal_image_fname = "OUTPUTS/RUNS/Icequake/Icequake_example_20140629184210336_EventLocationError.pdf"
from IPython.display import IFrame # For plotting pdf
IFrame(icequake_coal_image_fname, width=800, height=400) # Plot pdf


# References:
# 
# Hudson, T.S., Smith, J., Brisbourne, A.M., and White R.S. (2019). Automated detection of basal icequakes and discrimination from surface crevassing. Annals of Glaciology, 79
