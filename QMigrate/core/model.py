# -*- coding: utf-8 -*-
"""
Module to produce gridded traveltime velocity models

"""

import math
import warnings
import pickle
import struct
from copy import copy
import os

import skfmm
import pyproj
import numpy as np
import matplotlib
from scipy.interpolate import RegularGridInterpolator, griddata, interp1d
try:
    os.environ["DISPLAY"]
    matplotlib.use("Qt5Agg")
except KeyError:
    matplotlib.use("Agg")
import matplotlib.pylab as plt


def _coord_transform_np(p1, p2, loc):
    xyz = np.zeros(loc.shape)
    if loc.ndim == 1:
        xyz[0], xyz[1], xyz[2] = pyproj.transform(p1, p2,
                                                  loc[0],
                                                  loc[1],
                                                  loc[2])
    else:
        xyz[:, 0], xyz[:, 1], xyz[:, 2] = pyproj.transform(p1, p2,
                                                           loc[:, 0],
                                                           loc[:, 1],
                                                           loc[:, 2])
    return xyz


def _proj(**kwargs):
    projection = kwargs.get("projection")
    units = kwargs.get("units")
    if projection == "WGS84":
        proj = "+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs"  # "+init=EPSG:4326"
    if projection == "NAD27":
        proj = "+proj=longlat +ellps=clrk66 +datum=NAD27 +no_defs"  # "+init=EPSG:4267"
    if projection == "UTM":
        zone = _utm_zone(kwargs.get("longitude"))
        proj = "+proj=utm +zone={0:d} +datum=WGS84 +units={} +no_defs"
        proj = proj.format(zone, units)
    if projection == "LCC":
        lon0 = kwargs.get("lon0")
        lat0 = kwargs.get("lat0")
        parallel_1 = kwargs.get("parallel_1")
        parallel_2 = kwargs.get("parallel_2")
        proj = "+proj=lcc +lon_0={} +lat_0={} +lat_1={} +lat_2={} +datum=WGS84 +units={} +no_defs"
        proj = proj.format(float(lon0), float(lat0),
                           float(parallel_1), float(parallel_2), units)
    if projection == "TM":
        lon = kwargs.get("lon")
        lat = kwargs.get("lat")
        proj = "+proj=tmerc +lon_0={} +lat_0={} +datum=WGS84 +units={} +no_defs"
        proj = proj.format(float(lon), float(lat), units)

    return pyproj.Proj(proj, preserve_units=True)


def _utm_zone(longitude):
    return (int(1 + math.fmod((longitude + 180.0) / 6.0, 60)))


def bilinear_interp(pos, gridspec, grid):

    if len(pos) == 2:
        x = pos[0]
        z = pos[1]
    else:
        x = pos[:, 0]
        z = pos[:, 1]
    _, _ = gridspec[0]
    x0, z0 = gridspec[1]
    dx, dz = gridspec[2]
    # nzy = nx * nz

    # get the position of the nearest node
    i = np.floor((x - x0)/dx).astype(np.int)
    k = np.floor((z - z0)/dz).astype(np.int)

    # get fractional distance of earthquake along each axis
    xd = (x / dx) - np.floor(x / dx)
    zd = (z / dz) - np.floor(z / dz)

    if np.all(zd == 0):
        # there is no interpolation in Z
        c0 = grid[i, k]
        c1 = grid[i+1, k]

        # do the interpolation along x
        c = c0*(1 - xd) + c1 * xd
    else:

        # Do bi-linear interpolation
        # get the 4 data points of the surrounding square
        c00 = grid[i, k]
        c10 = grid[i + 1, k]
        # k[(k==nz-1) & (zd==0.)] -=1
        c11 = grid[i + 1, k + 1]
        c01 = grid[i, k + 1]

        # do the interpolation along y
        c0 = c00 * (1 - xd) + c10 * xd
        c1 = c01 * (1 - xd) + c11 * xd

        # do the interpolation along z
        c = c0*(1 - zd) + c1 * zd

    return c


def read_2d_nlloc(froot):
    """
    Read the NonLinLoc travel time grids

    HEADER
    nx ny nz x0 y0 z0 dx dy dz
    2 101 111  0.000000 0.000000 -2.000000  0.200000 0.200000 0.200000 SLOW_LEN FLOAT
    TRANSFORM  TRANS_MERC RefEllipsoid WGS-84  LatOrig 8.000000  LongOrig 38.000000  RotCW 0.000000
    """

    with open(froot + ".hdr", "r") as fid:
        line = fid.readline().split()
        nx = int(line[0])
        ny = int(line[1])
        nz = int(line[2])
        x0 = float(line[3])
        y0 = float(line[4])
        z0 = float(line[5])
        dx = float(line[6])
        dy = float(line[7])
        dz = float(line[8])
        line = fid.readline().split()
        # st_name = line[0]
        st_x = float(line[1])
        st_y = float(line[2])
        st_z = float(line[3])

    npts = nx * ny * nz
    with open(froot + ".buf", "rb") as fid:
        buf = fid.read(npts * 4)
        data = struct.unpack("f" * npts, buf)

    data = np.reshape(data, (nx, ny, nz), order="C")
    # print(data.shape)

    distance_x = x0 + (np.linspace(0, nx - 1, nx) * dx)
    distance_y = y0 + (np.linspace(0, ny - 1, ny) * dy)
    distance_z = z0 + (np.linspace(0, nz - 1, nz) * dz)

    X, Y, Z = np.meshgrid(distance_x, distance_y, distance_z, indexing="ij")

    distance = np.sqrt(np.square(X) + np.square(Y) + np.square(Z))

    return data, (X, Y, Z, distance), (st_x, st_y, st_z), \
        [[nx, ny, nz], [x0, y0, z0], [dx, dy, dz]]


def grid_string(max_dist, max_depth, min_depth, dx):
    max_x = int(np.ceil(max_dist / dx)) + 5
    max_z = int(np.ceil((max_depth - min_depth) / dx)) + 5

    string = "2 {0:d} {1:d} 0.0 0.0 {2:f} {3:f} {3:f} {3:f}"
    return string.format(max_x, max_z, min_depth, dx)


def vgradient(i, vmodel):
    d_depth = vmodel["depth"][i+1] - vmodel["depth"][i]
    d_vel_p = vmodel["vp"][i+1] - vmodel["vp"][i]
    d_vel_s = vmodel["vs"][i+1] - vmodel["vs"][i]

    return d_vel_p / d_depth, d_vel_s / d_depth


def vmodel_string(vmodel, block):

    string = "LAYER  {0:f} {1:f} {3:f} {2:f} {4:f} 0.0 0.0"

    out = []

    nlayer = len(vmodel)
    i = 0
    while i < nlayer:
        if not block:
            try:
                gradientp, gradients = vgradient(i, vmodel)
            except KeyError:
                gradientp, gradients = 0., 0.
        else:
            gradientp = 0.
            gradients = 0.
        out.append(string.format(vmodel["depth"][i] / 1000.,
                                 vmodel["vp"][i] / 1000.,
                                 vmodel["vs"][i] / 1000.,
                                 gradientp, gradients))
        i += 1

    return "\n".join(out)


def write_control_file(x, y, z, name, max_dist,
                       vmodel, depth_limits, phase="P",
                       dx=0.2, block_model=True):
    control_string = """CONTROL 0 54321
TRANS NONE
#TRANS LAMBERT WGS-84 8.0 38.0 8.2 8.4 0.0
#TRANS TRANS_MERC WGS-84 8.0 38.0 0.0

VGOUT ./model/layer
VGTYPE {phase:s}

#VGGRID 2 101 111 0.0 0.0 -2.0 0.2 0.2 0.2 SLOW_LEN
VGGRID {grid:s} SLOW_LEN

{vmodel:s}
#LAYER  0.0 3.0 0.0 0.0 0.0 0.0 0.0
#LAYER  2.0 4.0 0.0 0.0 0.0 0.0 0.0
#LAYER 10.0 6.0 0.0 0.0 0.0 0.0 0.0
#LAYER 15.0 6.5 0.0 0.0 0.0 0.0 0.0
#LAYER  5.0 5.0 0.0 0.0 0.0 0.0 0.0
#LAYER 20.0 7.3 0.0 0.0 0.0 0.0 0.0

GTFILES ./model/layer ./time/layer {phase:s}
GTMODE GRID2D ANGLES_NO

#GTSRCE ST01 LATLON 8.1 38.1 0.0 0.0
GTSRCE {name:s} XYZ {x:f} {y:f} {z:f} 0.0

GT_PLFD 1.0E-3 0
                    """
    outstring = control_string.format(phase=phase,
                                      grid=grid_string(max_dist,
                                                       depth_limits[1],
                                                       depth_limits[0], dx),
                                      vmodel=vmodel_string(vmodel,
                                                           block_model),
                                      name=name, y=y, x=x, z=z)

    with open("./control.in", "w") as fid:
        fid.write(outstring)

    # print(outstring)
    return


def eikonal(ix, iy, iz, dxi, dyi, dzi, V, S):
    """
    Travel-Time formulation using a simple eikonal method.

    Requires the skifmm python package.

    Parameters
    ----------
    ix : array-like
        Number of cells in X-direction
    iy : array-like
        Number of cells in Y-direction
    iz : array-like
        Number of cells in Z-direction
    dxi :
        Cell length in X-direction
    dyi :
        Cell length in Y-direction
    dzi :
        Cell length in Z-direction
    V : array-like
        Contains the speed of interface propagation at each point in the domain
    S : array-like
        ???

    Returns
    -------
    t : array-like, same shape as phi
        Contains the travel time from the zero contour (zero level set) of phi
        to each point in the array given the scalar velocity field speed. If
        the input array speed has values less than or equal to zero the return
        value will be a masked array.

    """

    phi = -np.ones(ix.shape)
    indx = np.argmin(abs((ix - S[:, 0]))
                     + abs((iy - S[:, 1]))
                     + abs((iz - S[:, 2])))
    phi[np.unravel_index(indx, ix.shape)] = 1.0

    t = skfmm.travel_time(phi, V, dx=[dxi, dyi, dzi])
    return t


class Grid3D(object):
    """
    3D grid class

    Attributes
    ----------
    cell_count : array-like
        Number of cells in each dimension of the grid
    cell_size : array-like
        Size of a cell in each dimension of the grid
    azimuth : float
        Angle between northing vertical plane and grid y-z plane
    dip : float
        Angle between horizontal plane and grid x-y plane

    Methods
    -------
    lonlat_centre(longitude, latitude)
        Define the longitude and latitude of the centre of the grid
    nlloc_grid_centre(origin_lon, origin_lat)
        Define the centre of the grid from NonLinLoc file parameters

    """

    def __init__(self, cell_count, cell_size, azimuth, dip, sort_order="C"):
        """
        Class initialisation

        Parameters
        ----------
        cell_count : array-like
            Number of cells in each dimension of the grid
        cell_size : array-like
            Size of a cell in each dimension of the grid
        azimuth : float
            Angle between northing vertical plane and grid y-z plane
        dip : float
            Angle between horizontal plane and grid x-y plane
        sort_order : str
            Determines whether the multi-index should be viewed as indexing in
            row-major (C-style) or column-major (Fortran-style) order.
        longitude : float
            Longitude coordinate of the grid centre
        latitude : float
            Latitude coordinate of the grid centre
        elevation : float
            Elevation coordinate of the top grid layer (units: m)
        grid_centre : array-like
            Array containing coordinates of the grid centre
        grid_proj : pyproj object
            Grid space projection
        coord_proj : pyproj object
            Coordinate space projection

        """

        self._coord_proj = None
        self._grid_proj = None
        self._longitude = None
        self._latitude = None
        self._grid_origin = [0.0, 0.0, 0.0]

        self.cell_count = cell_count
        self.cell_size = cell_size
        self.elevation = 0
        self.azimuth = azimuth
        self.dip = dip

    def projections(self, grid_proj, coord_proj=None):
        if coord_proj and self._coord_proj is None:
            self.coord_proj = _proj(projection=coord_proj)
        elif self._coord_proj is None:
            self.coord_proj = _proj(projection="WGS84")

        if grid_proj == "UTM":
            self.grid_proj = _proj(projection=grid_proj,
                                   longitude=self.longitude)
        elif grid_proj == "LCC":
            self.grid_proj = _proj(projection=grid_proj, lon0=self.longitude,
                                   lat0=self.latitude,
                                   parallel_1=self.lcc_standard_parallels[0],
                                   parallel_2=self.lcc_standard_parallels[1])
        elif grid_proj == "TM":
            self.grid_proj = _proj(projection=grid_proj, lon=self.longitude,
                                   lat=self.latitude)
        else:
            msg = "Projection type must be specified.\n"
            msg += "SeisLoc currently supports:\n"
            msg += "        UTM\n"
            msg += "        LCC (Lambert Conical Conformic)\n"
            msg += "        TM (Transverse Mercator"
            raise Exception(msg)

    @property
    def cell_count(self):
        """Get and set the number of cells in each dimension of the grid."""

        return self._cell_count

    @cell_count.setter
    def cell_count(self, value):
        value = np.array(value, dtype="int32")
        if value.size == 1:
            value = np.repeat(value, 3)
        else:
            assert (value.shape == (3,)), "Cell count must be an n by 3 array."
        assert (np.all(value > 0)), "Cell count must be greater than [0]"
        self._cell_count = value

    @property
    def cell_size(self):
        """
        Get and set the size of a cell in each dimension of the grid.

        """

        return self._cell_size

    @cell_size.setter
    def cell_size(self, value):
        value = np.array(value, dtype="float64")
        if value.size == 1:
            value = np.repeat(value, 3)
        else:
            assert (value.shape == (3,)), "Cell size must be an n by 3 array."
        assert (np.all(value > 0)), "Cell size must be greater than [0]"
        self._cell_size = value

    @property
    def longitude(self):
        """Get and set the longitude of the grid centre"""

        return self._longitude

    @longitude.setter
    def longitude(self, value):
        # Add tests for suitable longitude
        self._longitude = value

    @property
    def latitude(self):
        """Get and set the latitude of the grid centre"""

        return self._latitude

    @latitude.setter
    def latitude(self, value):
        # Add tests for suitable latitude
        self._latitude = value

    @property
    def elevation(self):
        """
        Get the elevation of the grid centre

        """

        return self._elevation

    @elevation.setter
    def elevation(self, value):
        # Add tests for suitable elevation
        self._elevation = value

    @property
    def grid_proj(self):
        """
        Get and set the grid projection (defaults to WGS84)

        """

        if self._grid_proj is None:
            msg = "Grid projection has not been set: assuming WGS84"
            warnings.warn(msg)
            return _proj(projection="UTM", longitude=self.longitude)
        else:
            return self._grid_proj

    @grid_proj.setter
    def grid_proj(self, value):
        self._grid_proj = value

    @property
    def coord_proj(self):
        """Get and set the coordinate projection"""
        return self._coord_proj

    @coord_proj.setter
    def coord_proj(self, value):
        self._coord_proj = value

    def xy2lonlat(self, x, y, inverse=False):
        x = np.asarray(x)
        y = np.asarray(y)
        dummy_z = np.ones_like(x)

        out = self.xyz2lonlatdep(x, y, dummy_z, inverse=inverse)

        return out[0], out[1]

    def lonlatdep2index(self, loc, inverse=False):

        if inverse:
        
            xyz = self.xyz2index(loc, inverse=True)
            lonlatdep = self.xyz2lonlatdep(xyz[:, 0], xyz[:, 1], xyz[:, 2], inverse=False)

            return np.asarray(lonlatdep).T

        else:
            xyz = self.xyz2lonlatdep(loc[:, 0], loc[:, 1], loc[:, 2], inverse=True)
            xyz = np.vstack(xyz).T
            index = self.xyz2index(xyz, inverse=False)

            return index

    def xyz2lonlatdep(self, x, y, z, inverse=False, clip=False):
        x = np.asarray(x)
        y = np.asarray(y)
        z = np.asarray(z)
        if inverse:
            x, y = pyproj.transform(self.coord_proj,
                                    self.grid_proj,
                                    x, y)
            
            if clip:
                corners = self.grid_corners
                xmin, ymin, zmin = np.min(corners, axis=0)
                xmax, ymax, zmax = np.max(corners, axis=0)

                x[x < xmin] = xmin + self.cell_size[0] / 2
                y[y < ymin] = ymin + self.cell_size[1] / 2
                z[z < zmin] = zmin + self.cell_size[2] / 2
                x[x > xmax] = xmax - self.cell_size[0] / 2
                y[y > ymax] = ymax - self.cell_size[1] / 2
                z[z > zmax] = zmax - self.cell_size[2] / 2
            
            return x, y, z
        else:
            lon, lat = pyproj.transform(self.grid_proj,
                                    self.coord_proj,
                                    x, y)
            return lon, lat, z

    def xyz2index(self, loc, inverse=False, unravel=False):
        """
        Transforms from grid coordinates (i.e. x, y z) to 
        the grid index
        """

        if unravel:
            loc = np.asarray(np.unravel_index(loc, self.cell_count)).T

        if not type(loc) == np.ndarray:
            loc = np.asarray(loc)
        ndim = len(loc.shape)
        if ndim == 1:
            x = loc[0]
            y = loc[1]
            z = loc[2]
        else:
            x = loc[:, 0]
            y = loc[:, 1]
            z = loc[:, 2]
        dx, dy, dz = self.cell_size
        x0, y0, z0 = self.grid_origin

        if inverse:
            xx = x0 + (x * dx)
            yy = y0 + (y * dy)
            zz = z0 + (z * dz)
            
            if not type(xx) == np.ndarray:
                return np.asarray([xx, yy, zz])
            else:
                return np.vstack((xx, yy, zz)).T
        else:
            # get position of closest node
            i = (x - x0) / dx
            j = (y - y0) / dy
            k = (z - z0) / dz

            return np.asarray([i, j, k]).T

    @property
    def grid_corners(self):
        """
        Get the xyz positions of the cells on the edge of the grid

        """

        lc = self.cell_count - 1
        ly, lx, lz = np.meshgrid([0, lc[1]], [0, lc[0]], [0, lc[2]])
        loc = np.c_[lx.flatten(), ly.flatten(), lz.flatten()]
        return self.xyz2index(loc, inverse=True)

    @property
    def grid_xyz(self):
        """
        Get the xyz positions of all of the cells in the grid

        """

        lc = self.cell_count
        ly, lx, lz = np.meshgrid(np.arange(lc[1]),
                                 np.arange(lc[0]),
                                 np.arange(lc[2]))
        loc = np.c_[lx.flatten(), ly.flatten(), lz.flatten()]
        coord = self.xyz2loc(loc, inverse=True)
        lx = coord[:, 0].reshape(lc)
        ly = coord[:, 1].reshape(lc)
        lz = coord[:, 2].reshape(lc)
        return lx, ly, lz

class LUT(Grid3D):
    """
    Look-Up Table class

    Inherits from Grid3D and NonLinLoc classes

    Attributes
    ----------
    maps : dict
        Contains traveltime tables for P- and S-phases.

    Methods
    -------
    stations(path, units, delimiter=",")
        Read in station files
    station_xyz(station=None)
        Returns the xyz position of a specific station relative to the origin
        (default returns all locations)
    decimate(ds, inplace=False)
        Downsample the initial velocity model tables that are loaded before
        processing


    TO-DO
    -----
    Weighting of the stations with distance (allow the user to define their own
    tables or define a fixed weighting for the problem)
    Move maps from being stored in RAM (use JSON or HDF5)


        _select_station - Selecting the stations to be used in the LUT
        set_station     - Defining the station locations to be used

    """

    def __init__(self, cell_count=[51, 51, 31], cell_size=[30.0, 30.0, 30.0],
                 azimuth=0.0, dip=0.0):
        """
        Class initialisation method

        Parameters
        ----------
        cell_count : array-like
            Number of cells in each dimension of the grid
        cell_size : array-like
            Size of a cell in each dimension of the grid
        azimuth : float
            Angle between northing vertical plane and grid y-z plane
        dip : float
            Angle between horizontal plane and grid x-y plane

        """

        Grid3D.__init__(self, cell_count, cell_size, azimuth, dip)

        self.velocity_model = None
        self.station_data = None
        self._maps = {}
        self.data = None

    def stations(self, path, units, delimiter=","):
        """
        Reads station information from file

        Parameters
        ----------
        path : str
            Location of file containing station information
        delimiter : char, optional
            Station file delimiter, defaults to ","
        units : str

        """

        import pandas as pd

        stats = pd.read_csv(path, delimiter=delimiter).values

        stn_data = {}
        if units == "offset":
            stn_lon, stn_lat = self.xy2lonlat(stats[:, 0].astype("float")
                                              + self.grid_centre[0],
                                              stats[:, 1].astype("float")
                                              + self.grid_centre[1])
        elif units == "xyz":
            stn_lon, stn_lat = self.xy2lonlat(stats[:, 0], stats[:, 1])
        elif units == "lon_lat_elev":
            stn_lon = stats[:, 0]
            stn_lat = stats[:, 1]
        elif units == "lat_lon_elev":
            stn_lon = stats[:, 1]
            stn_lat = stats[:, 0]

        stn_data["Longitude"] = stn_lon
        stn_data["Latitude"] = stn_lat
        stn_data["Elevation"] = stats[:, 2]
        stn_data["Name"] = stats[:, 3]

        self.station_data = stn_data

    def station_xyz(self, station=None):
        if station is None:
            stn = self.station_data
        else:
            station = self._select_station(station)
            stn = self.station_data[station]
        x, y = self.xy2lonlat(stn["Longitude"], stn["Latitude"], inverse=True)
        coord = np.c_[x, y, stn["Elevation"]]
        return coord

    def station_offset(self, station=None):
        coord = self.station_xyz(station)
        return coord - self.grid_centre

    @property
    def maps(self):
        """Get and set the traveltime tables"""
        return self._maps

    @maps.setter
    def maps(self, value):
        self._maps = value

    def _select_station(self, station_data):
        if self.station_data is None:
            return station_data

        nstn = len(self.station_data["Name"])
        flag = np.array(np.zeros(nstn, dtype=np.bool))
        for i, stn in enumerate(self.station_data["Name"]):
            if stn in station_data:
                flag[i] = True

        return flag

    def decimate(self, ds, inplace=False):
        """
        Up- or down-sample the travel-time tables by some factor

        Parameters
        ----------
        ds :

        inplace : bool
            Performs the operation to the travel-time table directly

        TO-DO
        -----
        I"m not sure that the inplace operation is doing the right thing? - CB


        """

        if not inplace:
            self = copy(self)
            self.maps = copy(self.maps)
        else:
            self = self

        ds = np.array(ds, dtype=np.int)
        cell_count = 1 + (self.cell_count - 1) // ds
        c1 = (self.cell_count - ds * (cell_count - 1) - 1) // 2
        cn = c1 + ds * (cell_count - 1) + 1
        centre_cell = (c1 + cn - 1) / 2
        centre = self.xyz2index(centre_cell, inverse=True)
        self.cell_count = cell_count
        self.cell_size = self.cell_size * ds
        self.centre = centre

        maps = self.maps
        if maps is not None:
            for id_, map_ in maps.items():
                maps[id_] = np.ascontiguousarray(map_[c1[0]::ds[0],
                                                      c1[1]::ds[1],
                                                      c1[2]::ds[2], :])
        if not inplace:
            return self

    def decimate_array(self, data, ds):
        self = self
        ds = np.array(ds, dtype=np.int)
        cell_count = 1 + (self.cell_count - 1) // ds
        c1 = (self.cell_count - ds * (cell_count - 1) - 1) // 2
        cn = c1 + ds * (cell_count - 1) + 1
        centre_cell = (c1 + cn - 1) / 2
        centre = self.xyz2index(centre_cell, inverse=True)
        self.cell_count = cell_count
        self.cell_size = self.cell_size * ds
        self.centre = centre

        array = np.ascontiguousarray(data[c1[0]::ds[0],
                                          c1[1]::ds[1],
                                          c1[2]::ds[2]])
        return array

    def get_values_at(self, loc, station=None):
        val = {}
        for map_ in self.maps.keys():
            val[map_] = self.get_value_at(map_, loc, station)
        return val

    def get_value_at(self, map_, loc, station=None):
        return self.interpolate(map_, loc, station)

    def value_at(self, map_, xyz, station=None):
        loc = self.xyz2index(xyz)
        return self.interpolate(map_, loc, station)

    def values_at(self, xyz, station=None):
        loc = self.xyz2index(xyz)
        return self.get_values_at(loc, station)

    def interpolator(self, map_, station=None):
        maps = self.fetch_map(map_, station)
        nc = self.cell_count
        cc = (np.arange(nc[0]), np.arange(nc[1]), np.arange(nc[2]))
        return RegularGridInterpolator(cc, maps, bounds_error=False)

    def interpolate(self, map_, loc, station=None):
        interp_fcn = self.interpolator(map_, station)
        return interp_fcn(loc)

    def fetch_map(self, map_, station=None):
        if station is None:
            return self.maps[map_]
        else:
            station = self._select_station(station)
            return self.maps[map_][..., station]

    def fetch_index(self, map_, sampling_rate, station=None):
        maps = self.fetch_map(map_, station)
        return np.rint(sampling_rate * maps).astype(np.int32)

    def compute_homogeneous_vmodel(self, vp, vs):
        """
        Calculate the travel-time tables for each station in a uniform velocity
        model

        Parameters
        ----------
        vp : float
            P-wave velocity (units: km / s)
        vs : float
            S-wave velocity (units: km / s)

        """

        rloc = self.station_xyz()
        nstn = rloc.shape[0]
        gx, gy, gz = self.grid_xyz
        ncell = self.cell_count
        p_map = np.zeros(np.r_[ncell, nstn])
        s_map = np.zeros(np.r_[ncell, nstn])
        for stn in range(nstn):
            dx = gx - float(rloc[stn, 0])
            dy = gy - float(rloc[stn, 1])
            dz = gz - float(rloc[stn, 2])
            dist = np.sqrt(dx**2 + dy**2 + dz**2)
            p_map[..., stn] = (dist / vp)
            s_map[..., stn] = (dist / vs)
        self.maps = {"TIME_P": p_map,
                     "TIME_S": s_map}

    def compute_1d_vmodel(self, p0, p1, gridspec, vmodel, nlloc_dx=0.1,
                          nlloc_path="", block_model=False):
        """
        Calculate 3D travel time lookup-tables from a 1D velocity model.

        NonLinLoc Grid2Time is used to generate a 2D lookup-table which is then
        swept across a 3D distance from station grid to populate a 3D travel
        time grid. The location of the stations should already have been added
        to the LUT using the function set_station().

        Parameters
        ----------
        p0 : dict
            Coordinate projection information
        p1 : dict
            Grid projection information
        gridspec : array-like
            Contains lon/lat of lower-left corner, lon/lat of upper-right
            corner, min/max grid depth and grid spacing (units: m)
        vmodel : pandas DataFrame
            Contains columns with names "depth", "vp" and "vs"
        nlloc_dx : float, optional
            NLLoc 2D grid spacing (default: 0.1 km)
        nlloc_path : str
            Path to NonLinLoc binaries
        block_model : bool
            Interpret velocity model with constant velocity blocks
        """

        from subprocess import call, check_output, STDOUT
        self.velocity_model = vmodel
        p0_x0, p0_y0, p0_z0 = gridspec[0]
        p0_x1, p0_y1, p0_z1 = gridspec[1]
        dx, dy, dz = gridspec[2]

        # Define the projection
        self._coord_proj = p0
        self._grid_proj = p1

        # Define traveltime grid for QuakeMigrate
        p1_x0, p1_y0, p1_z0 = _coord_transform_np(p0, p1,
                                                  np.asarray([p0_x0,
                                                              p0_y0,
                                                              p0_z0]))
        p1_x1, p1_y1, p1_z1 = _coord_transform_np(p0, p1,
                                                  np.asarray([p0_x1,
                                                              p0_y1,
                                                              p0_z1]))

        # extract the number of nodes
        d_x = (p1_x1 - p1_x0)
        nx = int(np.ceil(d_x / dx)) + 1
        d_y = (p1_y1 - p1_y0)
        ny = int(np.ceil(d_y / dy)) + 1
        d_z = (p1_z1 - p1_z0)
        nz = int(np.ceil(d_z / dz)) + 1
        print(nx, ny, nz)

        xvec = p1_x0 + (np.linspace(0, nx - 1, nx) * dx)
        yvec = p1_y0 + (np.linspace(0, ny - 1, ny) * dy)
        zvec = p1_z0 + (np.linspace(0, nz - 1, nz) * dz)
        print(zvec)

        X, Y, Z = np.meshgrid(xvec, yvec, zvec, indexing="ij")

        # make a folder structure to run nonlinloc in
        os.makedirs("time", exist_ok=True)
        os.makedirs("model", exist_ok=True)

        nstation = len(self.station_data["Name"])

        p_travel_times = np.empty((nx, ny, nz, nstation))
        s_travel_times = np.empty_like(p_travel_times)
        i = 0
        while i < nstation:
            p0_st_y = self.station_data["Latitude"][i]
            p0_st_x = self.station_data["Longitude"][i]
            p0_st_z = -self.station_data["Elevation"][i]
            name = self.station_data["Name"][i]

            print("Calculating Travel Time for station", name)

            # get the maximum distance from station to corner of grid
            p1_st_loc = _coord_transform_np(p0, p1,
                                            np.asarray([p0_st_x,
                                                        p0_st_y,
                                                        p0_st_z]))
            p1_st_x, p1_st_y, p1_st_z = p1_st_loc

            # for nonlinloc the distances must be in km
            distance_grid = np.sqrt(np.square(X - p1_st_x) +
                                    np.square(Y - p1_st_y))
            max_dist = np.max(distance_grid)

            # NLLOC needs the station to lie within the 2D section,
            # therefore we pick the depth extent of the 2D grid from
            # the maximum possible extent of the station and the grid
            min_z = np.min([p1_z0, p1_st_z])
            max_z = np.max([p1_z1, p1_st_z])
            depth_extent = np.asarray([min_z, max_z])

            for phase in ["P", "S"]:
                # Allow 2 nodes on depth extent as a computational buffer
                write_control_file(p1_st_x, p1_st_y,
                                   p1_st_z, name,
                                   max_dist, self.velocity_model,
                                   depth_extent,
                                   phase=phase, dx=nlloc_dx,
                                   block_model=block_model)

                print("\tRunning NonLinLoc phase =", phase)
                out = check_output([os.path.join(nlloc_path, "Vel2Grid"),
                                   "control.in"], stderr=STDOUT)
                if b"ERROR" in out:
                    raise Exception("Vel2Grid Error", out)

                out = check_output([os.path.join(nlloc_path, "Grid2Time"),
                                    "control.in"], stderr=STDOUT)
                if b"ERROR" in out:
                    raise Exception("Grid2Time Error", out)

                to_read = "./time/layer.{}.{}.time".format(phase, name)
                data, _, _, nll_gridspec = read_2d_nlloc(to_read)

                distance = distance_grid.flatten()
                depth = Z.flatten()
                travel_time = bilinear_interp(np.vstack((distance, depth)).T,
                                              [nll_gridspec[0][1:],
                                               nll_gridspec[1][1:],
                                               nll_gridspec[2][1:]],
                                              data[0, :, :])

                travel_time = np.reshape(travel_time, (nx, ny, nz))
                if phase == "P":
                    p_travel_times[..., i] = travel_time
                elif phase == "S":
                    s_travel_times[..., i] = travel_time
                else:
                    raise Exception("HELP")

            i += 1

        # Define rest of the LUT parameters
        self.cell_count = np.asarray([nx, ny, nz])
        self.cell_size = np.asarray([dx, dy, dz])
        self.grid_origin = np.asarray([p1_x0, p1_y0, p1_z0])
        self.longitude = p0_x0
        self.latitude = p0_y0
        self.elevation = -p0_z0
        self.azimuth = 0.0
        self.dip = 0.0

        self.maps = {"TIME_P": p_travel_times, "TIME_S": s_travel_times}

        # call(["rm", "-rf", "control.in", "time", "model"])

    def compute_1d_vmodel_skfmm(self, path, delimiter=","):
        """
        Calculate the travel-time tables for each station in a velocity model
        that varies with depth

        Parameters
        ----------
        z : array-like
            Depth of each layer in model (units: km)
        vp : array-like
            P-wave velocity for each layer in model (units: km / s)
        vs : array-like
            S-wave velocity for each layer in model (units: km / s)

        """

        import pandas as pd

        vmod = pd.read_csv(path, delimiter=delimiter).values
        z, vp, vs = vmod[:, 0], vmod[:, 1] * 1000, vmod[:, 2] * 1000

        rloc = self.station_xyz()
        nstn = rloc.shape[0]
        ix, iy, iz = self.grid_xyz
        p_map = np.zeros(ix.shape + (rloc.shape[0],))
        s_map = np.zeros(ix.shape + (rloc.shape[0],))

        z = np.insert(np.append(z, -np.inf), 0, np.inf)
        vp = np.insert(np.append(vp, vp[-1]), 0, vp[0])
        vs = np.insert(np.append(vs, vs[-1]), 0, vs[0])

        f = interp1d(z, vp)
        gvp = f(iz)
        f = interp1d(z, vs)
        gvs = f(iz)

        for stn in range(nstn):
            msg = "Generating 1D Travel-Time Table - {} of {}"
            msg = msg.format(stn + 1, nstn)
            print(msg)

            p_map[..., stn] = eikonal(ix, iy, iz,
                                      self.cell_size[0],
                                      self.cell_size[1],
                                      self.cell_size[2],
                                      gvp, rloc[stn][np.newaxis, :])
            s_map[..., stn] = eikonal(ix, iy, iz,
                                      self.cell_size[0],
                                      self.cell_size[1],
                                      self.cell_size[2],
                                      gvs, rloc[stn][np.newaxis, :])

        self.maps = {"TIME_P": p_map,
                     "TIME_S": s_map}

    def compute_3d_vmodel(self, path):
        """

        """
        raise NotImplementedError

    # def read_3d_nlloc_lut(self, path, regrid=False, decimate=[1, 1, 1]):
    #     """
    #     Calculate the travel-time tables for each station in a velocity model
    #     that varies over all dimensions.

    #     This velocity model comes from a NonLinLoc velocity model file.

    #     Parameters
    #     ----------
    #     path : str
    #         Location of .buf and .hdr files

    #     Raises
    #     ------
    #     MemoryError
    #         If travel-time grids size exceeds available memory

    #     """

    #     nstn = len(self.station_data["Name"])
    #     for st in range(nstn):
    #         name = self.station_data["Name"][st]
    #         msg = "Loading P- and S- traveltime maps for {}"
    #         msg = msg.format(name)
    #         print(msg)

    #         # Reading in P-wave
    #         self.nlloc_load_file("{}.P.{}.time".format(path, name))
    #         if not regrid:
    #             self.nlloc_project_grid()
    #         else:
    #             self.nlloc_regrid(decimate)

    #         if ("p_map" not in locals()) and ("s_map" not in locals()):
    #             ncell = self.NLLoc_data.shape
    #             try:
    #                 p_map = np.zeros(np.r_[ncell, nstn])
    #                 s_map = np.zeros(np.r_[ncell, nstn])
    #             except MemoryError:
    #                 msg = "P- and S-traveltime maps exceed available memory."
    #                 raise MemoryError(msg)

    #         p_map[..., st] = self.NLLoc_data

    #         self.nlloc_load_file("{}.S.{}.time".format(path, name))
    #         if not regrid:
    #             self.nlloc_project_grid()
    #         else:
    #             self.nlloc_regrid(decimate)

    #         s_map[..., st] = self.NLLoc_data

    #     self.maps = {"TIME_P": p_map,
    #                  "TIME_S": s_map}

    def write_grids(self, froot=None, fpath=None):
        '''
        Writes the travel time look-up tables as binary files.
        File format is similar, but not identical to the NLLoc
        format.

        froot specifies a root file naming format, i.e.
                /path/to/my/directory/layer.{phase:s}.{name:s}.time.buf
            this is primarily used for outputting NonLinLoc compatible files
            The format specifier can include a phase and/or a name field

        fpath specifies the path to where you want the files stored. i.e.
                /path/to/my/directory
            The individual grid files will be named according to their name and 
            phase autmoatically
        '''

        import struct

        if not froot and not fpath:
            raise ValueError('Must define one of froot or fpath')
        
        if fpath:
            fname = '.'.join(["{name:s}", "{phase:s}", 'buf'])  
            froot = os.path.join(fpath, fname)


        nstation = len(self.station_data['Name'])
        for phase in ['P', 'S']:
            m = self.fetch_map('_'.join(['TIME', phase]))

            for i in range(nstation):
                station = self.station_data['Name'][i]

                fname = froot.format(phase=phase, name=station)                

                time = m[..., i].flatten(order='C')
                npts = len(time)
                byte_string = struct.pack('f' * npts, *time)
                with open(fname, 'wb') as fid:
                    fid.write(byte_string)

    def save(self, fpath):
        """
        Writes the header information of the look-up table as a
        JSON file and saves the grids as binary floats
        """

        import json

        fname = 'lut.hdr.json'
        save_dic = {'cell_count' : self.cell_count.tolist(),
                    'cell_size' : self.cell_size.tolist(),
                    'coord_proj' : self._coord_proj.srs,
                    'grid_proj' : self._grid_proj.srs,
                    'grid_origin' : self.grid_origin.tolist(),
                    'azimuth' : self.azimuth,
                    'dip' : self.dip,
                    'longitude' : self.longitude, 
                    'latitude' : self.latitude,
                    'elevation' : self.elevation,
                    'stations' : dict(tuple([(key, self.station_data[key].tolist()) for key in self.station_data.keys()])),
                    'vmodel' : self.velocity_model.values.tolist()}
        with open(os.path.join(fpath, fname), 'w') as fid:
            json.dump(save_dic, fid)
        
        self.write_grids(fpath=fpath)

    def load(self, fpath, header_only=False):
        """
        The default way of loading a saved look-up table
        """

        import json
        import struct
        import pandas as pds
        # read in the JSON file

        fname = 'lut.hdr.json'
        with open(os.path.join(fpath, fname), 'r') as fid:
            save_dic = json.load(fid)
        
        self.cell_count = save_dic['cell_count']
        self.cell_size = save_dic['cell_size']
        self._coord_proj = pyproj.Proj(save_dic['coord_proj'], preserve_units=True)
        self._grid_proj = pyproj.Proj(save_dic['grid_proj'], preserve_units=True)
        self.grid_origin = save_dic['grid_origin']
        self.azimuth = save_dic['azimuth']
        self.dip = save_dic['dip']
        self._longitude = save_dic['longitude']
        self._latitude = save_dic['latitude']
        self._elevation = save_dic['elevation']
        self.station_data = dict(tuple([(key, np.asarray(save_dic['stations'][key])) for key in save_dic['stations'].keys()]))
        self.velocity_model = pds.DataFrame(save_dic['vmodel'], columns=['depth', 'vp', 'vs'])

        if not header_only:
            stations = self.station_data['Name']
            nstations = len(stations)
            npts = np.prod(self.cell_count)

            p_time = np.empty((self.cell_count[0], 
                              self.cell_count[1],
                              self.cell_count[2],
                              nstations))
            s_time = np.empty_like(p_time)
            self.maps = {"TIME_P": p_time, "TIME_S": s_time}

            for phase in ['P', 'S']:
                m = self.maps['TIME_' + phase]
                for i, station in enumerate(stations):
                    fname = '.'.join([station, phase, 'buf'])  
                    froot = os.path.join(fpath, fname)

                    with open(froot, 'rb') as fid:
                        data = fid.read(npts * 4)
                    data = np.asarray(struct.unpack('f' * npts, data))
                    m[..., i] = data.reshape(self.cell_count, order='C')


    def save_nonlinloc(self, froot):
        """
        Save the look up table as nonlinloc compatible grids

        Here we use the grid coordinate system
        """

        # this is the format of the .hdr file
        out_string = "{0:d} {1:d} {2:d} "
        out_string += "{3:f} {4:f} {5:f} "
        out_string += "{6:f} {7:f} {8:f} "
        out_string += "TIME\n"

        out_string += "{9:s} {10:f} {11:f} {12:f}"

        # nonlinloc saves the grids with respect to the lower left corner
        corners = self.grid_corners
        LL = corners[0, :]

        # station positions
        stat_xyz = self.station_xyz()

        # adds the NLL format string to the froot argument
        trans_fname = froot + '_trans.in'
        froot += '.{phase:s}.{name:s}.time'

        for station in self.station_data['Name']:
            xyz = stat_xyz[0, :]
            for phase in ['P', 'S']:
                fname = froot.format(name=station, phase=phase)
                with open(fname + '.hdr', 'w') as fid:
                    fid.write(out_string.format(self.cell_count[0], self.cell_count[1], self.cell_count[2],
                                                LL[0], LL[1], LL[2],
                                                self.cell_size[0], self.cell_size[1], self.cell_size[2],
                                                station, xyz[0], xyz[1], xyz[2]))
        
        # cartesian projection information is not stored in the NonLinLoc grid files
        # here we decide to write a NLL control file TRANS statement in a separate 
        # file called froot + "trans.in"

        p_string = self._grid_proj.srs.replace('+', '').split()
        p_dict = dict([(s.split('=')[0], s.split('=')[1]) for s in p_string])
        if p_dict['ellps'] == 'WGS84':
            p_dict['ellps'] = 'WGS-84'

        if self._coord_proj == self._grid_proj:
            out_string = 'TRANS NONE'
        elif 'lcc' in self._grid_proj.srs:
            out_string = "TRANS LAMBERT {0:s} {1:f} {2:f} {3:f} {4:f} {5:f}" 
            out_string = out_string.format(p_dict['ellps'], float(p_dict['lat_0']), 
                                          float(p_dict['lon_0']), float(p_dict['lat_1']), 
                                          float(p_dict['lat2']), self.azimuth)
        elif 'tmerc' in self._grid_proj:
            out_string = "TRANS TRANS_MERC {0:s} {1:f} {2:f} {3:f}" 
            out_string = out_string.format(p_dict['ellps'], float(p_dict['lat_0']), 
                                          float(p_dict['lon_0']),  self.azimuth)
        else:
            raise ValueError('Projection is not valid with NonLinLoc')
            
        with open(trans_fname, 'w') as fid:
            fid.write(out_string)

        self.write_grids(froot=froot + '.buf')

    def load_nonlinloc(self, froot, trans_file=None):
        
        """
        if trans file is not provided then a file is assumed to exist called
        froot + '_trans.in" as would have been created using the save_nonlinloc
        function
        """
        from glob import glob


        if not trans_file:
            trans_file = froot + '_trans.in'
        
        if not os.path.exists(trans_file):
            raise IOError(trans_file, 'does not exist')
        
        with open(trans_file, 'r') as fid:
            line = fid.readline().split()
        
        self.dip = 0.0
        if line[1] == 'NONE':
            raise NotImplementedError
        elif line[1] == 'LAMBERT':
            lat_0 = float(line[3])
            lon_0 = float(line[4])
            lat_1 = float(line[5])
            lat_2 = float(line[6])
            az = float(line[7])

            self.azimuth = az
            self._coord_proj = _proj(projection='WGS84')
            self._grid_proj = _proj(projection='LCC', lon0=lon_0,
                                    lat0=lat_0, parallel_1=lat_1,
                                    parallel_2=lat_2, units='km')
        
        elif line[1] == 'TRANS_MERC':
            lat_0 = float(line[3])
            lon_0 = float(line[4])
            az = float(line[5])

            self.azimuth = az
            self._coord_proj = _proj(projection='WGS84')
            self._grid_proj = _proj(projection='TM', lon0=lon_0,
                                    lat0=lat_0, units='km')
        
        else:
            raise AttributeError('Well, this should not happen')

        
        hdr_files = glob(froot +'*.hdr')
        names, lats, lons, heights = [], [], [], []
        for hdr_file in hdr_files:

            phase = hdr_file.split('/')[-1].split('.')[1]

            if phase == 'S':
                continue

            with open(hdr_file, 'r') as fid:
                grid_line = fid.readline().split()
                station_line = fid.readline().split()
            
            # first define the calculation grid
            nx = int(grid_line[0])
            ny = int(grid_line[1])
            nz = int(grid_line[2])
            x0 = float(grid_line[3])
            y0 = float(grid_line[4])
            z0 = float(grid_line[5])
            dx = float(grid_line[6])
            dy = float(grid_line[7])
            dz = float(grid_line[8])

            assert grid_line[9] == 'TIME'

            self.cell_count = np.asarray([nx, ny, nz])
            self.cell_size = np.asarray([dx, dy, dz])

            self.grid_origin = np.array([x0, y0, z0])
            lon, lat = self.xy2lonlat(x0, y0)
            self._longitude = lon
            self._latitude = lat
            self.elevation = z0

            # now define the station location
            name = station_line[0]
            x = float(station_line[1])
            y = float(station_line[2])
            z = float(station_line[3])
            heights.append(z)
            names.append(name)

            lon, lat = self.xy2lonlat(x, y)
            lons.append(lon)
            lats.append(lat)

            stations = self.station_data['Name']
            nstations = len(stations)
            npts = np.prod(self.cell_count)

        self.station_data = {}
        self.station_data['Name'] = names
        self.station_data['Latitude'] = lats
        self.station_data['Longitude'] = lons
        self.station_data["Elevation"] = heights


        nstations = len(self.station_data['Name'])
        p_time = np.empty((self.cell_count[0], 
                            self.cell_count[1],
                            self.cell_count[2],
                            nstations))
        s_time = np.empty_like(p_time)
        self.maps = {"TIME_P": p_time, "TIME_S": s_time}

        p_i, s_i = 0, 0
        for hdr_file in hdr_files:
            buf_file = hdr_file[:-3] + 'buf'
            phase = hdr_file.split('/')[-1].split('.')[1]
            if phase == 'P':
                i = p_i
            elif phase == 'S':
                i = s_i
            else:
                raise ValueError(phase, 'is an invalid phase')
            m = self.maps['TIME_' + phase]
            with open(buf_file, 'rb') as fid:
                data = fid.read(npts * 4)
            data = np.asarray(struct.unpack('f' * npts, data))
            m[..., i] = data.reshape(self.cell_count, order='C')
            if phase == 'P':
                p_i += 1
            else:
                s_i += 1

    def save_pickle(self, filename):
        """
        Create a pickle file containing the look-up table

        Parameters
        ----------
        filename : str
            Path to location to save pickle file

        """

        file = open(filename, "wb")
        pickle.dump(self.__dict__, file, 2)
        file.close()

    def load_pickle(self, filename):
        """
        Read the contents of a pickle file to __dict__

        Parameters
        ----------
        filename : str
            Path to pickle file to load

        """

        file = open(filename, "rb")
        tmp_dict = pickle.load(file)
        self.__dict__.update(tmp_dict)

    def plot_station(self):
        """
        Produce a 2D map view of station locations

        """

        plt.scatter(self.station_data["Longitude"],
                    self.station_data["Latitude"])
        plt.show()

    def plot_3d(self, map_, station, output_file=None):
        """
        Creates a 3-dimensional representation of the station locations with
        optional velocity model if specified.

        Parameters
        ----------
        map_ : str
            Specifies which velocity model to plot
        station : str

        output_file : str, optional
            Location to save file to
        """
        raise NotImplementedError
