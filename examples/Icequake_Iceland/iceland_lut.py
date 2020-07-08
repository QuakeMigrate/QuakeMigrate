# -*- coding: utf-8 -*-
"""
Creation of LUT for the Iceland icequake example.

"""

from pyproj import Proj

from QMigrate.io import read_stations
from QMigrate.lut import compute_traveltimes

station_file = "./inputs/iceland_stations.txt"
lut_out = "./outputs/lut/example.LUT"

# --- Read in the station information file ---
stations = read_stations(station_file)

# --- Define the grid specifications ---
grid_spec = {"ll_corner": [-17.24363934275664, 64.31947715407385, -1390.],
             "ur_corner": [-17.204348515198255, 64.3365202025144, 1390],
             "cell_size": [100., 100., 20.],
             "grid_proj": Proj(proj="lcc", lon_0=-17.224, lat_0=64.328,
                               lat_1=64.32, lat_2=64.335, datum="WGS84",
                               ellps="WGS84", units="m", no_defs=True),
             "coord_proj": Proj(proj="longlat", ellps="WGS84", datum="WGS84",
                                no_defs=True)}

# --- Homogeneous LUT generation ---
lut = compute_traveltimes(grid_spec, stations, method="homogeneous", vp=3630,
                          vs=1833, log=True, save_file=lut_out)
