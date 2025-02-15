"""
This script generates the traveltime look-up table (LUT) for the Ethiopia
swarm example.

:copyright:
    2020–2025, QuakeMigrate developers.
:license:
    GNU General Public License, Version 3
    (https://www.gnu.org/licenses/gpl-3.0.html)

"""

from obspy.core import AttribDict
from pyproj import Proj

from quakemigrate.io import read_stations, read_vmodel
from quakemigrate.lut import compute_traveltimes

station_file = "./inputs/ethiopia_stations_TM.csv"
velocity_file = "./inputs/ethiopia_vel.csv"
lut_out = "./outputs/lut/ethiopia.LUT"

# --- Read in the station information file ---
stations = read_stations(station_file)

# --- Define the input and grid projections ---
gproj = Proj(
    proj="tmerc",
    units="km",
    lon_0=39,
    lat_0=8,
    datum="WGS84",
    ellps="WGS84",
    no_defs=True,
)
cproj = Proj(proj="longlat", datum="WGS84", ellps="WGS84", no_defs=True)

# --- Define the grid specifications ---
# AttribDict behaves like a Python dict, but also has '.'-style access.
grid_spec = AttribDict()
# grid_spec.ll_corner = [38.7, 7.5, -5.]
# grid_spec.ur_corner = [38.9, 7.7, 20.]
grid_spec.ll_corner = [38.95, 8.05, -5.0]
grid_spec.ur_corner = [39.2, 8.25, 15.0]
grid_spec.node_spacing = [0.5, 0.5, 0.5]
grid_spec.grid_proj = gproj
grid_spec.coord_proj = cproj

# --- 1D model LUT generation ---
# This example uses NonLinLoc to calculate the LUT. Ensure you have installed
# NonLinLoc and placed the "Grid2Time" executable in your PATH
vmod = read_vmodel(velocity_file)
lut = compute_traveltimes(
    grid_spec, stations, method="1dsweep", vmod=vmod, log=True, save_file=lut_out
)
