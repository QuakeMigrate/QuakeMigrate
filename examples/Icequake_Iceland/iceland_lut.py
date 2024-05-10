# -*- coding: utf-8 -*-
"""
This script generates the traveltime look-up table (LUT) for the Iceland
icequake example.

"""

from obspy.core import AttribDict
from pyproj import Proj

from quakemigrate.io import read_stations
from quakemigrate.lut import compute_traveltimes

station_file = "./inputs/iceland_stations.txt"
lut_out = "./outputs/lut/example.LUT"

# --- Read in the station information file ---
stations = read_stations(station_file)

# --- Define the input and grid projections ---
gproj = Proj(
    proj="lcc",
    units="km",
    lon_0=-17.222,
    lat_0=64.329,
    lat_1=64.323,
    lat_2=64.335,
    datum="WGS84",
    ellps="WGS84",
    no_defs=True,
)
cproj = Proj(proj="longlat", datum="WGS84", ellps="WGS84", no_defs=True)

# --- Define the grid specifications ---
# AttribDict behaves like a Python dict, but also has '.'-style access.
grid_spec = AttribDict()
grid_spec.ll_corner = [-17.24, 64.322, -1.390]
grid_spec.ur_corner = [-17.204, 64.336, 0.0]
grid_spec.node_spacing = [0.025, 0.025, 0.025]
grid_spec.grid_proj = gproj
grid_spec.coord_proj = cproj

# --- Homogeneous LUT generation ---
lut = compute_traveltimes(
    grid_spec,
    stations,
    method="homogeneous",
    phases=["P", "S"],
    vp=3.630,
    vs=1.833,
    log=True,
    save_file=lut_out,
)
