"""
This script generates the traveltime look-up table (LUT) for the Askja
volcano (Iceland) Volcanotectonic (VT) & Deep-Long-Period (DLP) event example.

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

# --- i/o paths ---
station_file = "./inputs/askja_stations.txt"
vmodel_file = "./inputs/askja_vmodel.txt"
lut_out = "./outputs/lut/askja.LUT"

# --- Read in the station information file ---
stations = read_stations(station_file)

# --- Read in the velocity model file ---
vmodel = read_vmodel(vmodel_file, comment="#")

# --- Define the input and grid projections ---
gproj = Proj(
    proj="lcc",
    units="km",
    lon_0=-16.6,
    lat_0=65.1,
    lat_1=64.9,
    lat_2=65.3,
    datum="WGS84",
    ellps="WGS84",
    no_defs=True,
)
cproj = Proj(proj="longlat", datum="WGS84", ellps="WGS84", no_defs=True)

# --- Define the grid specifications ---
# AttribDict behaves like a Python dict, but also has '.'-style access.
grid_spec = AttribDict()
grid_spec.ll_corner = [-17.3, 64.85, -3.0]
grid_spec.ur_corner = [-15.8, 65.4, 37.0]
grid_spec.node_spacing = [1.0, 1.0, 1.0]
grid_spec.grid_proj = gproj
grid_spec.coord_proj = cproj

# --- 1-D velocity model LUT generation (using NonLinLoc eikonal solver) ---
lut = compute_traveltimes(
    grid_spec,
    stations,
    method="1dnlloc",
    vmod=vmodel,
    phases=["P", "S"],
    log=True,
    save_file=lut_out,
)
