"""
intra.core.project - Project Module
"""

import os
import pyproj
import pandas
import math
import numpy as np
from uuid import uuid4 as new_uid


def _proj_wgs84():
    # "+init=EPSG:4326"
    return pyproj.Proj("+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs")


def _proj_nad27():
    # "+init=EPSG:4267"
    return pyproj.Proj("+proj=longlat +ellps=clrk66 +datum=NAD27 +no_defs")


def _proj_wgs84_utm(longitude):
    zone = (int(1+math.fmod((longitude+180.0)/6.0, 60)))
    return pyproj.Proj("+proj=utm +zone={0:d} +datum=WGS84 +units=m +no_defs".format(zone))


def _get_projection(data, longitude=None, latitude=None):
    if isinstance(data, dict):
        if data['datum'] == 'WGS84' and data['proj'] == 'utm':
            if data['zone'] is None:
                return _proj_wgs84_utm(longitude)
            else:
                return pyproj.Proj("+proj=utm +zone={0:d} +datum=WGS84 +units=m +no_defs".format(data['zone']))
        else:
            raise ValueError('Projection not understood / supported.')
    elif isinstance(data, pyproj.Proj):
        return data
    else:
        raise ValueError('Projection not understood / supported.')


class SeisProject(object):

    latitude = 0.0
    longitude = 51.4
    timezone = 0
    units = 'Metric'

    def __init__(self):
        super().__init__()
        self._coord_proj = _proj_wgs84()
        self._grid_proj = {'datum': 'WGS84', 'proj': 'utm', 'zone': None}
        self._items = {}
        self._trees = {}

    @property
    def easting(self):
        """
        The project "Easting" calculated from the project coordinates and grid projection.

        :rtype: float
        :return: easting
        """
        x, y = self.lonlat2xy(self.longitude, self.latitude)
        return x

    @property
    def northing(self):
        """
        The project "Northing" calculated from the project coordinates and grid projection.

        :rtype: float
        :return: northing
        """
        x, y = self.lonlat2xy(self.longitude, self.latitude)
        return y

    def set_coord(self, longitude, latitude):
        """
        Project coordinates (longitude, latitude). These correspond to the approximate coordinates and are
        used for example to set UTM zone.

        :type longitude: float
        :param longitude: Project longitude
        :type latitude: float
        :param latitude: Project latitude

        """
        self.longitude = longitude
        self.latitude = latitude
        self._grid_proj = _get_projection(self._grid_proj, self.longitude, self.latitude)

    def set_location(self, easting, northing):
        lon, lat = self.xy2lonlat(easting, northing)
        self.set_coord(lon, lat)

    @property
    def coord_proj(self):
        """
        Coordinate Projection as defined by "pyproj" Module. This is usually either "WGS84" or "NAD27". Alternatively a
        custom defined projection can be passed either as a string or as a "pyproj.Proj" type.

        :rtype: pyproj.Proj
        :return: Coordinate projection definition for pyproj.
        """
        return self._coord_proj

    @coord_proj.setter
    def coord_proj(self, value):
        if isinstance(value, str):
            if value == 'WGS84':
                self._coord_proj = _proj_wgs84()
            elif value == 'NAD27':
                self._coord_proj = _proj_nad27()
            else:
                self._coord_proj = pyproj.Proj(str)
        elif isinstance(value, pyproj.Proj):
            self._coord_proj = value
        else:
            raise ValueError('Datum must be one of "NAD27" or "WGS84" or pyprojProj.')

    @property
    def grid_proj(self):
        """
        Local projection as defined by "pyproj" Module. This could be UTM or some local grid. The IntraSeis
        library is primarily for local rather than global analysis hence data processed on a local grid (Eastings and
        Northings) then optionally exported as global coordinates (latitude / longitude).

        :rtype: pyproj.Proj
        :return: Grid projection definition for pyproj.
        """
        return _get_projection(self._grid_proj, self.longitude, self.latitude)

    @grid_proj.setter
    def grid_proj(self, value):
        self._grid_proj = value

    def xy2lonlat(self, x, y):
        lon, lat = pyproj.transform(self.grid_proj, self.coord_proj, np.array(x), np.array(y))
        return lon, lat

    def lonlat2xy(self, lon, lat):
        x, y = pyproj.transform(self.coord_proj, self.grid_proj, np.array(lon), np.array(lat))
        return x, y

    def find(self, type_name):
        pass

    def __setitem__(self, key, item):
        item._uuid = key
        item._project = self
        cls = type(item).__name__
        self._items.__setitem__(key, item)
        self._bytype[cls] = item

    def setitem(self, key, item):
        self[key] = item

    def additem(self, item):
        key = item._uuid if hasattr(item, '_uuid') else None
        if key is None:
            key = new_uid().hex
        self[key] = item
        return key

    def __getitem__(self, key):
        return self._items.__getitem__(key)

    def getitem(self, key):
        return self._items.__getitem__(key)

    def items(self):
        return self._items.keys()

    def __contains__(self, key):
        return key in self._items

    def __delitem__(self, key):
        return self._items.__delitem__(self, key)


class DefaultProject(SeisProject):
    pass


class LocalProject(SeisProject):
    """ Local project, data stored in files.\n
        Root = root directory (path) for the project
    """

    Root = ''

    def __init__(self, root):
        """
        Init method
        :param root: Root directory (Path) for the project.
        :return:
        """
        super().__init__()
        if os.path.exists(root):
            self.Root = root
        else:
            raise FileNotFoundError(root)

    def file(self, path, file=None):
        if file is None:
            return os.path.join(self.Root, path)
        else:
            return os.path.join(self.Root, path, file)

    def fetch(self, path, constructor=None):
        """
        :param path:
        :param constructor:
        :return:
        """
        if constructor is None:
            filename = os.path.join(self.Root, path)
            with open(filename, 'rb') as fp:
                data = fp.read()
                return data
        else:
            raise Exception("Constructor interface not yet supported.")

    def read_csv(self, path, file=None):
        return pandas.read_csv(self.file(path, file))



