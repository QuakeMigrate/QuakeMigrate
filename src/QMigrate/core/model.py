############################################################################
############## Scripts for Generation of Travel-Time LUT ###################
############################################################################
#   Adaptations from IntraSeis LUT generation and saving.
#
# ##########################################################################
# ---- Import Packages -----
import math
import warnings
from copy import copy


import numpy as np
import pyproj
from scipy.interpolate import RectBivariateSpline, RegularGridInterpolator, griddata, interp1d
import matplotlib
import subprocess
import os
import pandas as pd
import pickle
import struct
import skfmm

try:
    os.environ['DISPLAY']
    matplotlib.use('Qt4Agg')
except KeyError:
    matplotlib.use('Agg')
import matplotlib.pylab as plt

# ---- Coordinate transformations ----

def _cart2sph_np_array(xyz):
    # theta_phi_r = _cart2sph_np_array(xyz)
    tpr = np.zeros(xyz.shape)
    xy = xyz[:, 0] ** 2 + xyz[:, 1] ** 2
    tpr[:, 0] = np.arctan2(xyz[:, 1], xyz[:, 0])
    tpr[:, 1] = np.arctan2(xyz[:, 2], np.sqrt(xy))  # for elevation angle defined from XY-plane up
    tpr[:, 2] = np.sqrt(xy + xyz[:, 2] ** 2)
    return tpr


def _cart2sph_np(xyz):
    # theta_phi_r = _cart2sph_np(xyz)
    if xyz.ndim == 1:
        tpr = np.zeros(3)
        xy = xyz[0] ** 2 + xyz[1] ** 2
        tpr[0] = np.arctan2(xyz[1], xyz[0])
        tpr[1] = np.arctan2(xyz[2], np.sqrt(xy))  # for elevation angle defined from XY-plane up
        tpr[2] = np.sqrt(xy + xyz[2] ** 2)
    else:
        tpr = np.zeros(xyz.shape)
        xy = xyz[:, 0] ** 2 + xyz[:, 1] ** 2
        tpr[:, 0] = np.arctan2(xyz[:, 1], xyz[:, 0])
        tpr[:, 1] = np.arctan2(xyz[:, 2], np.sqrt(xy))  # for elevation angle defined from XY-plane up
        tpr[:, 2] = np.sqrt(xy + xyz[:, 2] ** 2)
    return tpr


def _sph2cart_np(tpr):
    # xyz = _sph2cart_np(theta_phi_r)
    if tpr.ndim == 1:
        xyz = np.zeros(3)
        xyz[0] = tpr[2] * np.cos(tpr[1]) * np.cos(tpr[0])
        xyz[1] = tpr[2] * np.cos(tpr[1]) * np.sin(tpr[0])
        xyz[2] = tpr[2] * np.sin(tpr[1])
    else:
        xyz = np.zeros(tpr.shape)
        xyz[:, 0] = tpr[:, 2] * np.cos(tpr[:, 1]) * np.cos(tpr[:, 0])
        xyz[:, 1] = tpr[:, 2] * np.cos(tpr[:, 1]) * np.sin(tpr[:, 0])
        xyz[:, 2] = tpr[:, 2] * np.sin(tpr[:, 1])
    return xyz


def _coord_transform_np(p1, p2, loc):
    xyz = np.zeros(loc.shape)
    if loc.ndim == 1:
        xyz[0], xyz[1], xyz[2] = pyproj.transform(p1, p2, loc[0], loc[1], loc[2])
    else:
        xyz[:, 0], xyz[:, 1], xyz[:, 2] = pyproj.transform(p1, p2, loc[:, 0], loc[:, 1], loc[:, 2])
    return xyz

def _proj_wgs84():
    return pyproj.Proj("+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs")  # "+init=EPSG:4326"


def _proj_nad27():
    return pyproj.Proj("+proj=longlat +ellps=clrk66 +datum=NAD27 +no_defs")  # "+init=EPSG:4267"

def _utm_zone(longitude):
    return (int(1 + math.fmod((longitude + 180.0) / 6.0, 60)))

def _proj_wgs84_utm(longitude):
    zone = _utm_zone(longitude)
    return pyproj.Proj("+proj=utm +zone={0:d} +datum=WGS84 +units=m +no_defs".format(zone))

def _proj_wgs84_lambertcc(lon_Org,lat_Org,lat_1pl,lat_2pl):
    return pyproj.Proj("+proj=lcc +lon_0={} +lat_0={} +lat_1={} +lat_2={} +datum=WGS84 +units=m +no_defs".format(float(lon_Org),float(lat_Org),float(lat_1pl),float(lat_2pl)))

def _proj_wgs84_tm(lon_Org,lat_Org):
    return pyproj.Proj("+proj=tmerc +lon_0={} +lat_0={} +datum=WGS84 +units=m +no_defs".format(float(lon_Org),float(lat_Org)))


# def _proj_nlloc_simple(latOrg,lonOrg,rotAngle):
#     x = (long - longOrig) * 111.111 * cos(lat_radians)
#     y = (lat - latOrig) * 111.111
#     lat = latOrig + y / 111.111
#     long = longOrig + x / (111.111 * cos(lat_radians))
#     x=(lon)


def eikonal(ix,iy,iz,dxi,dyi,dzi,V,S):
    '''
        Travel-Time formulation using a simple eikonal method.
        Requires the skifmm python package.

        Inputs:
            x - np.array of points in X-direction
            y - np.array of points in Y-direction
            z - np.array of points in Z-direction
            V - np.array of velocity in Z,Y,X for P- and S-wave
            S - Definition of the station location in grid

        Outputs:
            t - Travel-time numpy array

    '''
    phi = -np.ones(ix.shape)
    indx = np.argmin(abs((ix - S[:,0])) + abs((iy - S[:,1])) + abs((iz - S[:,2])))
    phi[np.unravel_index(indx,ix.shape)] = 1.0
    t = skfmm.travel_time(phi,V,dx=[dxi,dyi,dzi])
    return t




# ------- Class definition of the structure and manipulation of grid -------------
class Grid3D:
    def __init__(self, center=np.array([10000.0, 10000.0, -5000.0]), cell_count=np.array([51, 51, 31]),
                 cell_size=np.array([30.0, 30.0, 30.0]),
                 azimuth=0.0, dip=0.0, sort_order='C'):
        self._latitude = 51.4826
        self._longitude = 0.0077
        self._coord_proj = None
        self._grid_proj = None
        self._grid_center = None
        self._cell_count = None
        self._cell_size = None
        self.grid_center = center
        self.cell_count = cell_count
        self.cell_size = cell_size
        self.grid_azimuth = azimuth
        self.grid_dip = dip
        self.sort_order = sort_order
        self.UTM_zones_different = False
        self.lcc_standard_parallels=(0.0,0.0)

    @property
    def grid_center(self):
        return self._grid_center

    @grid_center.setter
    def grid_center(self, value):
        value = np.array(value, dtype='float64')
        assert (value.shape == (3,)), 'Grid center must be [x, y, z] array.'
        self._grid_center = value
        self._update_coord()

    @property
    def grid_proj(self):
        return self._grid_proj

    @grid_proj.setter
    def grid_proj(self, value):
        self._grid_proj = value
        self._update_grid_center()

    @property
    def coord_proj(self):
        return self._coord_proj

    @coord_proj.setter
    def coord_proj(self, value):
        self._coord_proj = value
        self._update_coord()

    @property
    def cell_count(self):
        return self._cell_count

    @cell_count.setter
    def cell_count(self, value):
        value = np.array(value, dtype='int32')
        if value.size == 1:
            value = np.repeat(value, 3)
        else:
            assert (value.shape == (3,)), 'Cell count must be [nx, ny, nz] array.'
        assert (np.all(value > 0)), 'Cell count must be greater than [0]'
        self._cell_count = value

    @property
    def cell_size(self):
        return self._cell_size

    @cell_size.setter
    def cell_size(self, value):
        value = np.array(value, dtype='float64')
        if value.size == 1:
            value = np.repeat(value, 3)
        else:
            assert (value.shape == (3,)), 'Cell size must be [dx, dy, dz] array.'
        assert (np.all(value > 0)), 'Cell size must be greater than [0]'
        self._cell_size = value

    @property
    def elevation(self):
        return self._grid_center[2]

    @elevation.setter
    def elevation(self, value):
        self._grid_center[2] = value

    @property
    def latitude(self):
        return self._latitude

    @property
    def longitude(self):
        return self._longitude

    def set_proj(self, coord_proj=None, grid_proj=None):
        if coord_proj:
            self._coord_proj = coord_proj
        if grid_proj:
            self._grid_proj = grid_proj
        self._update_coord()

    def _nlloc_grid_proj(self):
        if self.NLLoc_proj:
            if self.NLLoc_proj == 'SIMPLE':
                return "ERROR -- simple not yet supported"
            elif self.NLLoc_proj == 'LAMBERT':
                return _proj_wgs84_lambertcc(self.NLLoc_MapOrg[0],self.NLLoc_MapOrg[1],self.NLLoc_MapOrg[4],self.NLLoc_MapOrg[5])
            elif self.NLLoc_proj == 'TRANS_MERC':
                return _proj_wgs84_tm(self.NLLoc_MapOrg[0],self.NLLoc_MapOrg[1])

    def get_grid_proj(self):
        if self._grid_proj is None:
            warnings.warn("Grid Projection has not been set: Assuming WGS84")
            return _proj_wgs84_utm(self.longitude)
        else:
            return self._grid_proj

    def get_coord_proj(self):
        if self._coord_proj is None:
            warnings.warn("Coordinte Projection has not been set: Assuming WGS84")
            return _proj_wgs84()
        else:
            return self._coord_proj

    def _update_grid_center(self):
        if self._coord_proj and self._grid_proj and self._latitude and self._longitude:
            x, y = pyproj.transform(self._coord_proj, self._grid_proj, self._longitude, self._latitude)
            self._grid_center[0] = x
            self._grid_center[1] = y
            return True
        else:
            return False

    def _update_coord(self):
        if self._coord_proj and self._grid_proj:
            center = self._grid_center
            lat, lon = pyproj.transform(self._grid_proj, self._coord_proj, center[0], center[1])
            self._latitude = lat
            self._longitude = lon
            return True
        else:
            return False

    def get_NLLOC_gridcenter(self,NLLOCorg_lon,NLLOCorg_lat):
        self._longitude = NLLOCorg_lon
        self._coord_proj = _proj_wgs84()
        if self.NLLoc_proj is not 'NONE':
            self._grid_proj = self._nlloc_grid_proj()
        self.grid_origin_xy=self.lonlat2xy(NLLOCorg_lon,NLLOCorg_lat)
        self._grid_center[0],self._grid_center[1]=(self.grid_origin_xy[0]+self.center[0],self.grid_origin_xy[1]+self.center[1])
        self._longitude,self._latitude=self.xy2lonlat(self._grid_center[0],self._grid_center[1])
        # if _utm_zone(self.longitude) != _utm_zone(NLLOCorg_lon):
        #     self.UTM_zones_different=True
        #     self._coord_proj = _proj_wgs84()
        #     self._grid_proj = _proj_wgs84_utm(self.longitude)
        #     self.grid_origin_xy=self.lonlat2xy(NLLOCorg_lon,NLLOCorg_lat)
        #     self._grid_center[0],self._grid_center[1]=(self.grid_origin_xy[0]+self.center[0],self.grid_origin_xy[1]+self.center[1])
        #     self._longitude,self._latitude=self.xy2lonlat(self._grid_center[0],self._grid_center[1])
        self._update_grid_center()

    def set_lonlat(self, longitude=None, latitude=None, coord_proj=None, grid_proj=None):
        if coord_proj:
            self._coord_proj = coord_proj
        if grid_proj:
            self._grid_proj = grid_proj
        if latitude:
            self._latitude = latitude
        if longitude:
            self._longitude = longitude
        self._update_grid_center()

    def setproj_wgs84(self,proj):
        self._coord_proj = _proj_wgs84()
        if proj == 'UTM':
            self._grid_proj = _proj_wgs84_utm(self.longitude)
        elif proj == 'LCC':
            self._grid_proj = _proj_wgs84_lambertcc(self.longitude,self.latitude,self.lcc_standard_parallels[0],self.lcc_standard_parallels[1])
        elif proj == 'TM':
            self._grid_proj = _proj_wgs84_tm(self.longitude,self.latitude)
        else:
            raise Exception('Projection type must be specified! CMS currently supports UTM, LCC (Lambert Conical Conformic) or TM (Transverse Mercator)')
        if not self._update_grid_center():
            self._update_coord()

    def xy2lonlat(self, x, y):
        return pyproj.transform(self.get_grid_proj(), self.get_coord_proj(), np.array(x), np.array(y))

    def lonlat2xy(self, lon, lat):
        return pyproj.transform(self.get_coord_proj(), self.get_grid_proj(), np.array(lon), np.array(lat))

    def local2global(self, loc):
        tpr = _cart2sph_np(loc - self._grid_center)
        tpr += [self.grid_azimuth, self.grid_dip, 0.0]
        return (_sph2cart_np(tpr) + self._grid_center)

    def global2local(self, loc):
        tpr = _cart2sph_np(loc - self._grid_center)
        tpr -= [self.grid_azimuth, self.grid_dip, 0.0]
        return (_sph2cart_np(tpr) + self._grid_center)

    def loc2xyz(self, loc):
        return self.local2global(self._grid_center + self._cell_size * (loc - (self._cell_count - 1) / 2))

    def xyz2loc(self, cord):
        return ((self.global2local(cord) - self._grid_center) / self._cell_size) + (self._cell_count - 1) / 2

    def loc2index(self, loc):
        return np.ravel_multi_index(loc, self._cell_count, mode='clip', order=self.sort_order)

    def index2loc(self, index):
        loc = np.vstack(np.unravel_index(index, self._cell_count, order=self.sort_order)).transpose()
        return loc

    def index2xyz(self, index):
        return self.loc2xyz(self.index2loc(index))

    def xyz2index(self, cord):
        return self.loc2index(self.xyz2loc(cord))

    def xyz2coord(self, loc):
        lon, lat = self.xy2lonlat(loc[:,0], loc[:,1])
        return np.array([lon, lat, loc[:,2]]).transpose()

    def loc2coord(self,loc):
        return self.xyz2coord(self.loc2xyz(loc))

    def coord2loc(self,loc):
        return self.xyz2loc(self.coord2xyz(loc))


    def coord2xyz(self, loc):
        X, Y = self.lonlat2xy(loc[:,0], loc[:,1])
        Z = loc[:,2]

        Bounds = self.get_grid_xyz()
        Xmin,Ymin,Zmin = np.min(Bounds,axis=0)
        Xmax,Ymax,Zmax = np.max(Bounds,axis=0)

        if X < Xmin:
            X = np.array([Xmin + self._cell_size[0]/2])
        if X > Xmax:
            X = np.array([Xmax - self._cell_size[0]/2])
        if Y < Ymin:
            Y = np.array([Ymin + self._cell_size[1]/2])
        if Y >  Ymax:
            Y = np.array([Ymax - self._cell_size[1]/2])
        if Z < Zmin:
            Z = np.array([Zmin + self._cell_size[2]/2])
        if Z > Zmax:
            Z = np.array([Zmax - self._cell_size[2]/2])

        return np.array([X,Y,Z]).transpose()

    def coord2index(self,coord):
        return self.loc2index(self.coord2loc(coord))


    def grid_origin(self):
        grid_size = (self._cell_count) * self._cell_size # Defining origin as on surface
        return self.local2global(self._grid_center - grid_size / 2)

    def get_grid_xyz(self, cells='corner'):
        if cells == 'corner':
            lc = self._cell_count - 1
            ly, lx, lz = np.meshgrid([0, lc[1]], [0, lc[0]], [0, lc[2]])
            loc = np.c_[lx.flatten(), ly.flatten(), lz.flatten()]
            return self.loc2xyz(loc)
        else:
            lc = self._cell_count
            ly, lx, lz = np.meshgrid(np.arange(lc[1]), np.arange(lc[0]), np.arange(lc[2]))
            loc = np.c_[lx.flatten(), ly.flatten(), lz.flatten()]
            coord = self.loc2xyz(loc)
            lx = coord[:, 0].reshape(lc)
            ly = coord[:, 1].reshape(lc)
            lz = coord[:, 2].reshape(lc)
            return lx, ly, lz



class NonLinLoc:
    '''
        NonLinLoc - Class

        Reading and manipulating NLLoc Grids in a 2D or 3D format


        # Additional Features coming soon;
            - Loading of 2D Travel-Times

    '''

    def __init__(self):
        self.NLLoc_n       = np.array([0,0,0])
        self.NLLoc_org     = np.array([0,0,0])
        self.NLLoc_siz     = np.array([0,0,0])
        self.NLLoc_type    = 'TIME'
        self.NLLoc_proj    = 'NONE'
        self.NLLoc_MapOrg  =   [0.0,0.0,0.0,'SIMPLE',0.0,0.0] # Defining Long,Lat,Rotation,ReferenceEllipsoid,FirstSTD,SecondSTD

        self.NLLoc_data    = None



    def NLLOC_LoadFile(self,FileName):

        # read header file
        fp = open('{}.hdr'.format(FileName, 'r'))


        # Defining the grid dimensions
        params = fp.readline().split()
        self.NLLoc_n    = np.array([int(params[0]),int(params[1]),int(params[2])])
        self.NLLoc_org = np.array([float(params [3]),float(params [4]),float(params [5])])
        self.NLLoc_siz = np.array([float(params[6]),float(params[7]),float(params[8])])
        self.NLLoc_type = params[9]


        # Defining the station information
        stations = fp.readline().split()

        # Defining the Transform information
        trans    = fp.readline().split()
        if trans[1] == 'NONE':
            self.NLLoc_proj = 'NONE'
        if trans[1] == 'SIMPLE':
            self.NLLoc_proj    = 'SIMPLE'
            self.NLLoc_MapOrg  =  [trans[5],trans[3],trans[7],'Simple','0.0','0.0']
        if trans[1] == 'LAMBERT':
            self.NLLoc_proj    = 'LAMBERT'
            self.NLLoc_MapOrg  =  [trans[7],trans[5],trans[13],trans[3],trans[9],trans[11]]
        if trans[1] == 'TRANS_MERC':
            self.NLLoc_proj    = 'TRANS_MERC'
            self.NLLoc_MapOrg  =  [trans[7],trans[5],trans[9],trans[3],'0.0','0.0']



        # Reading the buf file
        fid = open('{}.buf'.format(FileName),'rb')
        data = struct.unpack('{}f'.format(self.NLLoc_n[0]*self.NLLoc_n[1]*self.NLLoc_n[2]),fid.read(self.NLLoc_n[0]*self.NLLoc_n[1]*self.NLLoc_n[2]*4))
        self.NLLoc_data = np.array(data).reshape(self.NLLoc_n[0],self.NLLoc_n[1],self.NLLoc_n[2])


    def NLLOC_ProjectGrid(self):
        '''
            Projecting the grid to the new coordinate system. This function also determines the 3D grid from the 2D
            grids from NonLinLoc
        '''

        # Generating the correct NonLinLoc Formatted Grid
        if (self.NLLoc_proj == 'NONE'):
            GRID_NLLOC = Grid3D(center=(self.NLLoc_org + self.NLLoc_siz*self.NLLoc_n), cell_count=self.NLLoc_n,cell_size=self.NLLoc_siz,azimuth=0.0, dip=0.0, sort_order='C')

        if (self.NLLoc_proj == 'SIMPLE'):
            GRID_NLLOC = Grid3D(center=(self.NLLoc_org + self.NLLoc_siz*self.NLLoc_n), cell_count=self.NLLoc_n,cell_size=self.NLLoc_siz,azimuth=self.NLLoc_MapOrg[2], dip=0.0, sort_order='C')
            GRID_NLLOC.set_lonlat(self.NLLoc_MapOrg[0],self.NLLoc_MapOrg[1])

        if (self.NLLoc_proj == 'LAMBERT'):
            GRID_NLLOC = Grid3D(center=(self.NLLoc_org + self.NLLoc_siz*self.NLLoc_n), cell_count=self.NLLoc_n,cell_size=self.NLLoc_siz,azimuth=self.NLLoc_MapOrg[2], dip=0.0, sort_order='C')
            GRID_NLLOC.set_lonlat(self.NLLoc_MapOrg[0],self.NLLoc_MapOrg[1])
            GRID_NLLOC.set_proj(self.NLLoc_MapOrg[3])

        if (self.NLLoc_proj == 'TRANS_MERC'):
            GRID_NLLOC = Grid3D(center=(self.NLLoc_org + self.NLLoc_siz*self.NLLoc_n), cell_count=self.NLLoc_n,cell_size=self.NLLoc_siz,azimuth=self.NLLoc_MapOrg[2], dip=0.0, sort_order='C')
            GRID_NLLOC.set_lonlat(self.NLLoc_MapOrg[0],self.NLLoc_MapOrg[1])
            GRID_NLLOC.set_proj(self.NLLoc_MapOrg[3])


        OrgX,OrgY,OrgZ = GRID_NLLOC.get_grid_xyz(cells='full')
        NewX,NewY,NewZ = self.get_grid_xyz(cells='full')

        self.NLLoc_data = griddata((OrgX.flatten(),OrgY.flatten(),OrgZ.flatten()),self.NLLoc_data.flatten(),(NewX,NewY,NewZ),method='nearest')



    def NLLOC_RedefineGrid(self,Decimate):
        '''
            Redefining coordinate system to the file loaded
        '''

        # Decimating the grid by the factor defined


        self.center     = (self.NLLoc_org + self.NLLoc_siz*(self.NLLoc_n-1)/2)*[1000,1000,-1000]
        self.cell_count = self.NLLoc_n
        self.cell_size  = self.NLLoc_siz*1000
        self.dip        = 0.0

        if (self.NLLoc_proj == 'NONE'):
            self.azimuth    = 0.0
            self.grid_center = self.center

        if (self.NLLoc_proj == 'SIMPLE'):
            self.azimuth = self.NLLoc_MapOrg[2]
            self.get_NLLOC_gridcenter(float(self.NLLoc_MapOrg[0]),float(self.NLLoc_MapOrg[1]))
            self.grid_center[2] = self.center[2]

        if (self.NLLoc_proj == 'LAMBERT'):
            self.azimuth = float(self.NLLoc_MapOrg[2])
            self.get_NLLOC_gridcenter(float(self.NLLoc_MapOrg[0]),float(self.NLLoc_MapOrg[1]))
            self.grid_center[2] = self.center[2]

        if (self.NLLoc_proj == 'TRANS_MERC'):
            self.azimuth = float(self.NLLoc_MapOrg[2])
            self.get_NLLOC_gridcenter(float(self.NLLoc_MapOrg[0]),float(self.NLLoc_MapOrg[1]))
            self.grid_center[2] = self.center[2]

        self.NLLoc_data = self.decimate_array(self.NLLoc_data,np.array(Decimate))[:,:,::-1]

# ------------ LUT Generation for the 3D LUT -------------

class LUT(Grid3D,NonLinLoc):
    '''
        Generating and Altering the Travel-Time LUT for


        maps            - Used later to apply Coalescence 4D data.
        _select_station - Selecting the stations to be used in the LUT
        decimate        - Downsample the intitial velocity model tables that are loaded before processing.
        get_station_xyz - Getting the stations relative x,y,z positions to the origin
        set_station     - Defining the station locations to be used

        ADDITON - Currently 'maps' stored in RAM. Need to use JSON or HDF5

    '''

    #   Additions to be made to the program:
    #       - Weighting of the stations with distance, allow the user to define their own tables
    #         or define a fixed weighting for the problem.
    #
    #       -
    #
    #

    def __init__(self, center=np.array([10000.0, 10000.0, -5000.0]), cell_count=np.array([51, 51, 31]),
                 cell_size=np.array([30.0, 30.0, 30.0]), azimuth=0.0, dip=0.0):


        Grid3D.__init__(self, center, cell_count, cell_size, azimuth, dip)
        NonLinLoc.__init__(self)

        self.velocity_model = None
        self.station_data = None
        self._maps = dict()
        self.data = None

    @property
    def maps(self):
        return self._maps

    @maps.setter
    def maps(self, maps):
        self._maps = maps

    def _select_station(self, station_data):
        if self.station_data is None:
            return station_data

        nstn = len(self.station_data)
        flag = np.array(np.zeros(nstn, dtype=np.bool))
        for i, stn in enumerate(self.station_data['Name']):
            if stn in station_data:
                flag[i] = True

    def decimate(self, ds, inplace=False):
        '''
            Function used to decimate the travel-time tables either supplied by NonLinLoc or through
            the inbuilt functions:


        '''
        if not inplace:
            self = copy(self)
            self.maps = copy(self.maps)
        else:
            self = self

        ds = np.array(ds, dtype=np.int)
        cell_count = 1 + (self.cell_count - 1) // ds
        c1 = (self.cell_count - ds * (cell_count - 1) - 1) // 2
        cn = c1 + ds * (cell_count - 1) + 1
        center_cell = (c1 + cn - 1) / 2
        center = self.loc2xyz(center_cell)
        self.cell_count = cell_count
        self.cell_size = self.cell_size * ds
        self.center = center

        maps = self.maps
        if maps is not None:
            for id, map in maps.items():
                maps[id] = np.ascontiguousarray(map[c1[0]::ds[0], c1[1]::ds[1], c1[2]::ds[2], :])
        if not inplace:
            return self


    def decimate_array(self,DATA,ds):
        self = self
        ds = np.array(ds, dtype=np.int)
        cell_count = 1 + (self.cell_count - 1) // ds
        c1 = (self.cell_count - ds * (cell_count - 1) - 1) // 2
        cn = c1 + ds * (cell_count - 1) + 1
        center_cell = (c1 + cn - 1) / 2
        center = self.loc2xyz(center_cell)
        self.cell_count = cell_count
        self.cell_size = self.cell_size * ds
        self.center = center

        ARRAY = np.ascontiguousarray(DATA[c1[0]::ds[0], c1[1]::ds[1], c1[2]::ds[2]])
        return ARRAY


    def get_station_xyz(self, station=None):
        if station is not None:
            station = self._select_station(station)
            stn = self.station_data[station]
        else:
            stn = self.station_data
        x, y = self.lonlat2xy(stn['Longitude'], stn['Latitude'])
        coord = np.c_[x, y, stn['Elevation']]
        return coord

    def get_station_offset(self, station=None):
        coord = self.get_station_xyz(station)
        return coord - self.grid_center

    def get_values_at(self, loc, station=None):
        val = dict()
        for map in self.maps.keys():
            val[map] = self.get_value_at(map, loc, station)
        return val

    def get_value_at(self, map, loc, station=None):
        return self.interpolate(map, loc, station)

    def value_at(self, map, coord, station=None):
        loc = self.xyz2loc(coord)
        return self.interpolate(map, loc, station)

    def values_at(self, coord, station=None):
        loc = self.xyz2loc(coord)
        return self.get_values_at(loc, station)

    def get_interpolator(self, map, station=None):
        maps = self.fetch_map(map, station)
        nc = self._cell_count
        cc = (np.arange(nc[0]), np.arange(nc[1]), np.arange(nc[2]))
        return RegularGridInterpolator(cc, maps, bounds_error=False)

    def interpolate(self, map, loc, station=None):
        interp_fcn = self.get_interpolator(map, station)
        return interp_fcn(loc)

    def fetch_map(self, map, station=None):
        if station is None:
            return self.maps[map]
        else:
            station = self._select_station(station)
            return self.maps[map][..., station]

    def fetch_index(self, map, srate, station=None):
        maps = self.fetch_map(map, station)
        return np.rint(srate * maps).astype(np.int32)

    def set_station(self,loc,units):
        # Changing Pandas to Numpy Array
        nstn = loc.shape[0]
        stn_data={}
        if units == 'offset':
            stn_lon, stn_lat = self.xy2lonlat(loc[:, 0].astype('float') + self.grid_center[0], loc[:, 1].astype('float') + self.grid_center[1])
            stn_data['Longitude'] = stn_lon
            stn_data['Latitude'] = stn_lat
            stn_data['Elevation'] = loc[:, 2]
            stn_data['Name'] = loc[:,3]
        elif units == 'xyz':
            stn_lon, stn_lat = self.xy2lonlat(loc[:, 0], loc[:, 1])
            stn_data['Longitude'] = stn_lon
            stn_data['Latitude'] = stn_lat
            stn_data['Elevation'] = loc[:, 2]
            stn_data['Name'] = loc[:,3]
        elif units == 'lon_lat_elev':
            stn_data['Longitude'] = loc[:, 0]
            stn_data['Latitude'] = loc[:, 1]
            stn_data['Elevation'] = loc[:, 2]
            stn_data['Name'] = loc[:,3]
        elif units == 'lat_lon_elev':
            stn_data['Longitude'] = loc[:, 1]
            stn_data['Latitude'] = loc[:, 0]
            stn_data['Elevation'] = loc[:, 2]
            stn_data['Name'] = loc[:,3]
        self.station_data = stn_data



    def compute_Homogeous(self,VP,VS):
        '''
            Function used to compute Travel-time tables in a homogeous
            velocity model

            Input:
                VP - P-wave velocity (km/s, float)
                VS - S-wave velocity (km/s, float)
        '''
        rloc = self.get_station_xyz()
        gx, gy, gz = self.get_grid_xyz(cells='all')
        nstn = rloc.shape[0]
        ncell = self.cell_count
        map_p1 = np.zeros(np.r_[ncell, nstn])
        map_s1 = np.zeros(np.r_[ncell, nstn])
        for stn in range(nstn):
            dx = gx - float(rloc[stn, 0])
            dy = gy - float(rloc[stn, 1])
            dz = gz - float(rloc[stn, 2])
            dist = np.sqrt(dx * dx + dy * dy + dz * dz)
            map_p1[..., stn] = (dist / VP)
            map_s1[..., stn] = (dist / VS)
        self.maps = {'TIME_P': map_p1, 'TIME_S': map_s1}


    def compute_1DVelocity(self,Z,VP,VS):
        '''
            Function used to compute Travel-time tables in a 1D Velocity model
            defined using the input VP and VS arrays

            INPUTS:
                Z  - Depth of each point in the velocity mode e.g. [0,1,2,3]
                VP - P-Wave velocity 1D array
                VS - S-Wave velocity 1D array

        '''
        # Constructing the velocity model
        #      Interpolating the velocity model to each point in the 3D grid. Defined Smoothing parameter based by




        stn = self.get_station_xyz()
        coord = self.get_grid_xyz()
        ix, iy, iz = self.get_grid_xyz(cells='all')
        ttp = np.zeros(ix.shape + (stn.shape[0],))
        tts = np.zeros(ix.shape + (stn.shape[0],))

        Z  = np.insert(np.append(Z,-np.inf),0,np.inf)
#        print(Z)
        VP = np.insert(np.append(VP,VP[-1]),0,VP[0])
        VS = np.insert(np.append(VS,VS[-1]),0,VS[0])

        f = interp1d(Z,VP)
        gvp = f(iz)
        f = interp1d(Z,VS)
        gvs = f(iz)


        for s in range(stn.shape[0]):
            print("Generating 1D Travel-Time Table - {} of {}".format(s+1,stn.shape[0]))

            x = np.arange(min(coord[:,0]),max(coord[:,0]),self.cell_size[0])
            y = -np.arange(min(coord[:,1]),max(coord[:,1]),self.cell_size[1])
            z = np.arange(min(coord[:,2]),max(coord[:,2]),self.cell_size[2])

            #print(eikonal(x,y,z,gvp,np.array([s])))

            ttp[..., s] = eikonal(ix,iy,iz,self.cell_size[0],self.cell_size[1],self.cell_size[2],gvp,stn[s][np.newaxis,:])
            tts[..., s] = eikonal(ix,iy,iz,self.cell_size[0],self.cell_size[1],self.cell_size[2],gvs,stn[s][np.newaxis,:])

        self.maps = {'TIME_P': ttp, 'TIME_S': tts}

#    def compute_3DVelocity(self,INPUT_FILE):
#        '''
#            Function used to compute Travel-time tables in a 1D Velocity model
#            defined using the input VP and VS arrays

#            INPUTS:
#                INPUT_FILE - File containg comma seperated X,Y,Z,VP,VS


#        '''
#        # Constructing the velocity model
#        #      Interpolating the velocity model to each point in the 3D grid. Defined Smoothing parameter based by

#        VEL = pd.read_csv(INPUT_FILE,names=['X','Y','Z','VP','VS'])

#        stn = self.get_station_xyz()
#        coord = self.get_grid_xyz()
#        ix, iy, iz = self.get_grid_xyz(cells='all')
#        ttp = np.zeros(ix.shape + (nstn,))
#        tts = np.zeros(ix.shape + (nstn,))

#        gvp = scipy.interpolate.griddata(VEL[['X','Y','Z']], VEL['VP'], (ix,iy,iz), 'linear')
#        gvs = scipy.interpolate.griddata(VEL[['X','Y','Z']], VEL['VP'], (ix,iy,iz), 'linear')


#        for s in range(stn.shape[0]):
#            print("Generating 1D Travel-Time Table - {}".format(i))

#            x = np.arange(min(coord[:,0]),max(coord[:,0]),self.cell_size[0])
#            y = np.arange(min(coord[:,1]),max(coord[:,1]),self.cell_size[1])
#            Z = np.arange(min(coord[:,2]),max(coord[:,2]),self.cell_size[2])

#            ttp[..., p] = eikonal(x,y,z,gvp,stn[s][np.newaxis,:])[0]
#            tts[..., s] = eikonal(x,y,z,gvs,stn[s][np.newaxis,:])[0]

#        self.maps = {'TIME_P': ttp1, 'TIME_S': tts}




    def compute_3DNLLoc(self,PATH,RedefineCoord=False,Decimate=[1,1,1]):

        '''
            Function to read in NonLinLoc Tables to be used for the Travel-Time
            tables.

            INPUTS:
                PATH - Full path to where the .buf and .hdr files can be found from
                        the NonLinLoc output files



        '''
        nstn = len(self.station_data['Name'])
        for st in range(nstn):
            name = self.station_data['Name'][st]
            print('Loading TTp and TTs for {}'.format(name))

            # Reading in P-wave
            self.NLLOC_LoadFile('{}.P.{}.time'.format(PATH,name))

            if (RedefineCoord == False):
                self.NLLOC_ProjectGrid()
            else:
                self.NLLOC_RedefineGrid(Decimate)

            if ('map_p1' not in locals()) and ('map_s1' not in locals()):
                ncell = self.NLLoc_data.shape
                try:
                    map_p1 = np.zeros(np.r_[ncell, nstn])
                    map_s1 = np.zeros(np.r_[ncell, nstn])
                except MemoryError:
                    raise MemoryError('P- and S-Wave Travel-Time Grids too large for memory !')

            map_p1[...,st] = self.NLLoc_data


            self.NLLOC_LoadFile('{}.S.{}.time'.format(PATH,name))
            if (RedefineCoord == False):
                self.NLLOC_ProjectGrid()
            else:
                self.NLLOC_RedefineGrid(Decimate)

            map_s1[...,st] = self.NLLoc_data



        self.maps = {'TIME_P':map_p1, 'TIME_S':map_s1}


    def save(self,FILENAME):
        '''
            Saving the LUT format for future use.
        '''
        file = open('{}'.format(FILENAME),'wb')
        pickle.dump(self.__dict__,file,2)
        file.close()


    def load(self,FILENAME):
        '''
            Saving the LUT format for future use.
        '''
        file = open('{}'.format(FILENAME),'rb')
        tmp_dict = pickle.load(file)
        self.__dict__.update(tmp_dict)



    def plot_station(self):
        '''
            Function to plot a 2D representation of the station locations

        '''

        plt.scatter(self.station_data['Longitude'],self.station_data['Latitude'])
        plt.show()


    # def plot3D(self,TYPE,STAION,save_filename=None):

    #     '''
    #         Function to plot a 3D representation of the station locations
    #         with optional velocity model if specified

    #         CURRENTLY ONLY WORKS IF IN LAT/LONG


    #         OPTIONAL-INPUT:
    #             save_filename = Save plot to the defined filename
    #             TravelTimeSlice=

    #     '''



    #     # ---- Plotting the Station Locations ----

    #     # Plotting X-Y

    #     fig = plt.figure()
    #     XYSlice  =  plt.subplot2grid((3, 3), (0, 0), colspan=2,rowspan=2)
    #     YZSlice  =  plt.subplot2grid((3, 3), (2, 0),colspan=2)
    #     XZSlice  =  plt.subplot2grid((3, 3), (0, 2),rowspan=2)

    #     EXTENT=lut.loc2coord(lut.get_grid_xyz())



    #     # Plotting the MAP
    #     gridXY_X,gridXY_Y = np.mgrid[ min(EXTENT[:,0]):max(EXTENT[:,0]):(max(EXTENT[:,0])-min(EXTENT[:,0]))/lut.cell_count[0],[min(EXTENT[:,1]):max(EXTENT[:,1]):(max(EXTENT[:,1])-min(EXTENT[:,1]))/lut.cell_count[1]]]
    #     gridXZ_X,gridXZ_Z = np.mgrid[min(EXTENT[:,0]):max(EXTENT[:,0]):(max(EXTENT[:,0])-min(EXTENT[:,0]))/lut.cell_count[0],[min(EXTENT[:,2]):max(EXTENT[:,2]):(max(EXTENT[:,2])-min(EXTENT[:,2]))/lut.cell_count[2]]]
    #     gridYZ_X,gridYZ_Z = np.mgrid[min(EXTENT[:,1]):max(EXTENT[:,1]):(max(EXTENT[:,1])-min(EXTENT[:,1]))/lut.cell_count[1],[min(EXTENT[:,2]):max(EXTENT[:,2]):(max(EXTENT[:,2])-min(EXTENT[:,2]))/lut.cell_count[2]]]
    #     XYSlice.pcolormesh(gridXY_X,gridXY_Y,lut.fetch_map(TYPE)[:,:,100,1])


    #     # Plotting the Station Location
    #     XYSlice.scatter(lut.station_data['Longitude'],lut.station_data['Latitude'])
    #     XZSlice.scatter(lut.station_data['Elevation'],lut.station_data['Longitude'])
    #     YZSlice.scatter(lut.station_data['Latitude'],lut.station_data['Elevation'])




    #     # # ---- Plotting the Velocity Model Slices
    #     # if VelSlice is not None:
    #     #     try:
    #     #         StationIndex = np.where(self.station_data['Name'] == TravelTimeSlice)[0][0]
    #     #         StationInfo = np.array([self.station_data['Longitude'][StationIndex],self.station_data['Latitude'][StationIndex],self.station_data['Elevation'][StationIndex]])

    #     #         loc2index(coord2loc(StationInfo))



    #     #     except:
    #     #         print('Please give a defined station name!')

    #     try:
    #         StationIndex = np.where(lut.station_data['Name'] == STATION)[0][0]


    #     except:
    #         print(' Please specify ')




    #     if save_filename is not None:
    #         plt.savefig(save_filename)
    #     else:
    #         plt.show()


