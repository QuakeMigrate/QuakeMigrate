import QMigrate.core.model as cmod   # Velocity model generation functions
from pyproj import Proj
import pandas as pd

# Define the cartesian projection as a pyrpj instance
p0 = Proj('+proj=latlong +ellps=WGS84')
p1 = Proj('+proj=lcc +lat_0=6.7 +lon_0=38.2 +ellps=WGS84 +lat_1=7.2 +lat2=7.8')

# define the upper, bottom left and the lower, upper right corners and the grid spacing of the final SeisLoc travel time grid. Note that the Z coords and the grid spacing are in metres.
gridspec = [(38.7, 7.4, -5000), # lon0, lat0, z0
            (38.9, 7.65, 30000), # lon1, lat1, z1
            (500, 500, 500)] # dx, dy, dz

# read in the 1D velocity model
vmodel = pd.read_csv('vmodel.csv', delimiter=',')

# Set the parameters for the travel-times lookup table (LUT):
lut = cmod.LUT() # Create an empty LUT with a centre, cell count (x,y,z) and cell size (x,y,z in metres) specified

lut.stations('riftvolc.csv', delimiter=',',
                units='lat_lon_elev') # Set the station parameters for the LUT
                
lut.compute_1DVelocity(p0, p1, gridspec, vmodel, block_model=True)

lut_path = 'riftvolc.lut' # Set the path to save the LUT to
lut.save(lut_path)
