# -*- coding: utf-8 -*-
"""
Creation of LUT for the Rutford icequake example.

"""

from pyproj import Proj

from QMigrate.io import read_stations
from QMigrate.lut import compute, LUT

station_file = "./inputs/rutford_stations.txt"
lut_out = "./outputs/lut/icequake.LUT"

# --- Read in the station information file ---
stations = read_stations(station_file)

# --- Define projections ---
cproj = Proj(proj="longlat", ellps="WGS84", datum="WGS84", no_defs=True)
gproj = Proj(proj="lcc", lon_0=-83.932, lat_0=-78.144, lat_1=-78.1, lat_2=-77.9,
             datum="WGS84", ellps="WGS84", units="m", no_defs=True)

# --- Create new LUT ---
# Cell count (x,y,z); cell size (x,y,z in metres)
lut = LUT(ll_corner=[-84.14853353566141, -78.18825429331356, -350.],
          ur_corner=[-83.71921885073093, -78.10003166259442, 3550],
          cell_size=[100., 100., 100.], grid_proj=gproj, coord_proj=cproj)

# --- Homogeneous LUT generation ---
compute(lut, stations, method="homogeneous", vp=3841, vs=1970, log=True,
        save_file=lut_out)
