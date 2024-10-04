# -*- coding: utf-8 -*-
"""
This script generates the traveltime look-up table (LUT) for the Rutford
icequake example.

"""

from obspy.core import AttribDict
from pyproj import Proj

from quakemigrate.io import read_stations
from quakemigrate.lut import compute_traveltimes

station_file = "./inputs/rutford_stations.txt"
lut_out = "./outputs/lut/icequake.LUT"

# --- Read in the station information file ---
stations = read_stations(station_file)

# --- Define the input and grid projections ---
gproj = Proj(
    proj="lcc",
    units="km",
    lon_0=-83.925,
    lat_0=-78.145,
    lat_1=-78.16,
    lat_2=-78.13,
    datum="WGS84",
    ellps="WGS84",
    no_defs=True,
)
cproj = Proj(proj="longlat", datum="WGS84", ellps="WGS84", no_defs=True)

# --- Define the grid specifications ---
# AttribDict behaves like a Python dict, but also has '.'-style access.
grid_spec = AttribDict()
grid_spec.ll_corner = [-84.1, -78.17, 1.0]
grid_spec.ur_corner = [-83.75, -78.12, 3.0]
grid_spec.node_spacing = [0.025, 0.025, 0.025]
grid_spec.grid_proj = gproj
grid_spec.coord_proj = cproj

# --- Homogeneous LUT generation ---
lut = compute_traveltimes(
    grid_spec,
    stations,
    method="homogeneous",
    phases=["P", "S"],
    vp=3.841,
    vs=1.970,
    log=True,
    save_file=lut_out,
)
