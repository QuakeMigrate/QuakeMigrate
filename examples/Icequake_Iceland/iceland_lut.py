# -*- coding: utf-8 -*-
"""
Creation of LUT for the Iceland icequake example.

"""

from pyproj import Proj

from QMigrate.io import stations
from QMigrate.lut import compute, LUT

station_file = "./inputs/iceland_stations.txt"
data_in = "./inputs/mSEED"
lut_out = "./outputs/lut/icequake.LUT"
out_path = "./outputs/runs"
run_name = "icequake_example"

# --- Read in the station information file ---
stations = stations(station_file)

# --- Define projections ---
cproj = Proj(proj="longlat", ellps="WGS84", datum="WGS84", no_defs=True)
gproj = Proj(proj="lcc", lon_0=-17.224, lat_0=64.328, lat_1=64.32, lat_2=64.335,
             datum="WGS84", ellps="WGS84", units="m", no_defs=True)

# --- Create new LUT ---
# Cell count (x,y,z); cell size (x,y,z in metres)
lut = LUT(ll_corner=[-17.24363934275664, 64.31947715407385, -1390.],
          ur_corner=[-17.204348515198255, 64.3365202025144, 1390],
          cell_size=[100., 100., 20.], grid_proj=gproj, coord_proj=cproj)

# --- Homogeneous LUT generation ---
vp = 3630
vs = 1833
compute(lut, stations, method="homogeneous", vp=vp, vs=vs)

# --- Save LUT ---
lut.save(lut_out)
