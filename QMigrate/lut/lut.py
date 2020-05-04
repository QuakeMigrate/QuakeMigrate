# -*- coding: utf-8 -*-
"""
Module to produce travel-time lookup tables defined on a Cartesian grid.

"""

import copy
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pyproj
from scipy.interpolate import RegularGridInterpolator


class Grid3D(object):
    """
    A grid object represents a collection of points in a 3-D Cartesian space
    that can be used to produce regularised travel-time lookup tables that
    sample the continuous travel-time space for each station in a seismic
    network.

    This class also provides the series of transformations required to move
    between the input projection, the grid projection and the grid index
    coordinate spaces.

    The size and shape specifications of the grid are defined by providing the
    (input projection) coordinates for the lower-left and upper-right corners,
    a cell size and the projections (defined using pyproj) of the input and
    grid spaces.

    Attributes
    ----------
    ll_corner : array-like, [float, float, float]
        Location of the lower-left corner of the grid in the grid
        projection. Should also contain the minimum depth in the grid.
    ur_corner : array-like ,[float, float, float]
        Location of the upper-right corner of the grid in the grid
        projection. Should also contain the maximum depth in the grid.
    cell_size : array-like, [float, float, float]
        Size of a cell in each dimension of the grid.
    grid_proj : pyproj Proj object
        Grid space projection.
    coord_proj : pyproj Proj object
        Input coordinate space projection.
    cell_count : array-like, [int, int, int]
        Number of cells in each dimension of the grid. This is calculated by
        finding the number of cells with cell_size will fit between the
        lower-left and upper-right corners. This value is rounded up if the
        number of cells returned is non-integer, to ensure the requested area
        is included in the grid.
    maps : dict
        A dictionary containing the travel-time lookup tables. The structure of
        this dictionary is:
            maps
                - "<Station1-ID>"
                    - "<PHASE>"
                    - "<PHASE>"
                - "<Station2-ID"
                    - "<PHASE>"
                    - "<PHASE>"
                etc
    velocity_model : pandas DataFrame object
        Contains the input velocity model specification.
        Columns: "Z" "Vs" "Vp"
    grid_corners : array-like, shape (8, 3)
        Positions of the corners of the grid in the grid coordinate space.
    grid_xyz : array-like, shape (3,)
        Positions of the grid nodes in the grid coordinate space. The shape of
        each element of the list is defined by the number of cells in each
        dimension.
    stations_xyz : array-like, shape (n, 3)
        Positions of the stations in the grid coordinate space.

    Methods
    -------
    decimate(df, inplace=False)
        Downsamples the travel-time lookup tables by some decimation factor.
    index2grid(value, inverse=False, unravel=False)
        Provides a transformation between grid indices (can be a flattened
        index or an [i, j, k] position) and the grid coordinate space.
    coord2grid(value, inverse=False, clip=False)
        Provides a transformation between the input projection and grid
        coordinate spaces.
    index2coord(value, inverse=False, unravel=False, clip=False)
        Provides a transformation between grid dindices (can be a flattened
        index or an [i, j, k] position) and the input projection coordinate
        space.

    """

    def __init__(self, ll_corner=None, ur_corner=None, cell_size=None,
                 grid_proj=None, coord_proj=None, lut_file=None):
        """
        Class initialisation.

        Parameters
        ----------
        ll_corner : array-like, [float, float, float]
            Location of the lower-left corner of the grid in the input
            projection. Should also contain the minimum depth in the grid.
        ur_corner : array-like ,[float, float, float]
            Location of the upper-right corner of the grid in the input
            projection. Should also contain the maximum depth in the grid.
        cell_size : array-like, [float, float, float]
            Size of a cell in each dimension of the grid.
        grid_proj : pyproj Proj object
            Grid space projection.
        coord_proj : pyproj Proj object
            Input coordinate space projection.

        """

        if lut_file is not None:
            self.load(lut_file)
        else:
            self.grid_proj = grid_proj
            self.coord_proj = coord_proj

            # Transform the geographical grid corners into grid coordinates
            self.ll_corner = self.coord2grid(ll_corner)[0]
            self.ur_corner = self.coord2grid(ur_corner)[0]

            # Calculate the grid dimensions and the number of cells required
            grid_dims = self.ur_corner - self.ll_corner
            self.cell_size = cell_size
            self.cell_count = np.ceil(grid_dims / self.cell_size) + 1

            self.maps = {}
            self.velocity_model = ""

    def decimate(self, df, inplace=False):
        """
        Resample the travel-time lookup tables by decimation by some factor.

        Parameters
        ----------
        df : array-like [int, int, int]
            Decimation factor in each dimension.
        inplace : bool, optional
            Perform the operation on the lookup table object or a copy.

        Returns
        -------
        grid : Grid3D object (optional)
            Returns a Grid3D object with decimated travel-time lookup tables.

        """

        df = np.array(df, dtype=np.int)

        new_cell_count = 1 + (self.cell_count - 1) // df
        c1 = (self.cell_count - df * (new_cell_count - 1) - 1) // 2

        if inplace:
            grid = self
        else:
            grid = copy.deepcopy(self)

        grid.cell_count = new_cell_count
        grid.cell_size = self.cell_size * df

        for station, map_ in grid.maps.items():
            for phase, ttimes in map_.items():
                grid[station][phase] = ttimes[c1[0]::df[0],
                                              c1[1]::df[1],
                                              c1[2]::df[2]]

        if not inplace:
            return grid

    def index2grid(self, value, inverse=False, unravel=False):
        """
        Convert between grid indices and grid coordinate space.

        Parameters
        ----------
        value : array-like
            Array (of arrays) containing the grid indices (grid coordinates)
            to be transformed. Can be an array of flattened indices.
        inverse : bool, optionale
            Reverses the direction of the transform.
            Default indices -> grid coordinates.
        unravel : bool, optional
            Convert a flat index or array of flat indices into a tuple of
            coordinate arrays.

        Returns
        -------
        out : array-like
            Returns an array of arrays of the transformed values.

        """

        if unravel:
            value = np.column_stack(np.unravel_index(value, self.cell_count))
        else:
            value = np.array(value)

        if inverse:
            out = np.rint((value - self.ll_corner) / self.cell_size)
            out = np.vstack(out.astype(int))
        else:
            out = np.vstack(self.ll_corner + (value * self.cell_size))

        # Handle cases where only a single ijk index is requested
        if out.shape[1] == 1:
            out = out.T

        return out

    def coord2grid(self, value, inverse=False, clip=False):
        """
        Convert between input coordinate space and grid coordinate space.

        Parameters
        ----------
        value : array-like
            Array (of arrays) containing the coordinate locations to be
            transformed. Each sub-array should describe a single point in the
            3-D input space.
        inverse : bool, optional
            Reverses the direction of the transform.
            Default input coordinates -> grid coordinates
        clip : bool, optional

        Returns
        -------
        out : array-like
            Returns an array of arrays of the transformed values.

        """

        v1, v2, v3 = np.array(value).T

        if inverse:
            inproj, outproj = self.grid_proj, self.coord_proj
        else:
            inproj, outproj = self.coord_proj, self.grid_proj

        return np.column_stack(pyproj.transform(inproj, outproj, v1, v2, v3))

    def index2coord(self, value, inverse=False, unravel=False, clip=False):
        """
        Convert between grid indices and input coordinate space.

        This is a utility function that wraps the other two defined transforms.

        Parameters
        ----------
        value : array-like
            Array (of arrays) containing the grid indices (grid coordinates)
            to be transformed. Can be an array of flattened indices.
        inverse : bool, optional
            Reverses the direction of the transform.
            Default indices -> input projection coordinates.
        unravel : bool, optional
            Convert a flat index or array of flat indices into a tuple of
            coordinate arrays.
        clip : bool, optional

        Returns
        -------
        out : array-like
            Returns an array of arrays of the transformed values.

        """

        if inverse:
            value = self.coord2grid(value, clip=clip)
            out = self.index2grid(value, inverse=True)
        else:
            value = self.index2grid(value, unravel=unravel)
            out = self.coord2grid(value, inverse=True, clip=clip)

        return out

    def xyz2lonlatdep(self, x, y, z, inverse=False, clip=False):
        """
        Convert between grid (x/y/z) and geographical (lon/lat/dep)
        coordinates.

        Parameters
        ----------
        x : array-like
            Grid x or longitudinal coordinates to convert
        y : array-like
            Grid y or latitudinal coordinates to convert
        z : array-like
            Grid z or depth coordinates to convert
        inverse : bool, optional
            Reverses the direction of the transform. Default xyz -> lonlatdep
        clip : bool, optional
            Collapse all values outside the grid onto the edge of the grid.
            CB: I don't think this will behave, need to find where it is used.

        Returns
        -------
        x' (x_p) : array-like
            Converted grid x or longitudinal coordinates
        y' (y_p) : array-like
            Converted grid y or latitudinal coordinates
        z' (z_p) : array-like
            Converted grid z or depth coordinates (no change from input)

        """

        x, y = np.asarray(x), np.asarray(y)
        z_p = np.asarray(z)

        if inverse:
            x_p, y_p = pyproj.transform(self.coord_proj, self.grid_proj, x, y)

            if clip:
                corners = self.grid_corners

                xmin, ymin, zmin = np.min(corners, axis=0)
                xmax, ymax, zmax = np.max(corners, axis=0)

                # Replace all this with np.clip
                x_p[x_p < xmin] = xmin + self.cell_size[0] / 2
                y_p[y_p < ymin] = ymin + self.cell_size[1] / 2
                z_p[z_p < zmin] = zmin + self.cell_size[2] / 2
                x_p[x_p > xmax] = xmax - self.cell_size[0] / 2
                y_p[y_p > ymax] = ymax - self.cell_size[1] / 2
                z_p[z_p > zmax] = zmax - self.cell_size[2] / 2

        else:
            x_p, y_p = pyproj.transform(self.grid_proj, self.coord_proj, x, y)

        return x_p, y_p, z_p

    @property
    def grid_corners(self):
        """Get the xyz positions of the cells on the edge of the grid."""

        c = self.cell_count - 1
        i, j, k = np.meshgrid([0, c[0]], [0, c[1]], [0, c[2]], indexing="ij")

        return self.index2grid(np.c_[i.flatten(), j.flatten(), k.flatten()])

    @property
    def grid_xyz(self):
        """Get the xyz positions of all of the cells in the grid."""

        cc = self.cell_count
        i, j, k = np.meshgrid(np.arange(cc[0]), np.arange(cc[1]),
                              np.arange(cc[2]), indexing="ij")
        xyz = self.index2grid(np.c_[i.flatten(), j.flatten(), k.flatten()])
        x, y, z = [xyz[:, dim].reshape(cc) for dim in range(3)]

        return x, y, z

    @property
    def stations_xyz(self):
        """Get station locations in the grid space [X, Y, Z]."""

        return self.coord2grid(self.station_data[["Longitude",
                                                  "Latitude",
                                                  "Elevation"]].values)

    @property
    def cell_count(self):
        """Get and set the number of cells in each dimension of the grid."""

        return self._cell_count

    @cell_count.setter
    def cell_count(self, value):
        value = np.array(value, dtype="int32")
        assert (np.all(value > 0)), "Cell count must be greater than [0]"
        self._cell_count = value

    @property
    def cell_size(self):
        """Get and set the size of a cell in each dimension of the grid."""

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


class LUT(Grid3D):
    """
    A lookup table object is a simple data structure that is used to store a
    series of regularised tables that, for each seismic station in a network,
    store the travel times to every point in the 3-D volume. These lookup
    tables are pre-computed to reduce the computational cost of the
    back-projection method.

    This class provides utility functions that can be used to serve up or query
    these pre-computed lookup tables.

    This object is-a Grid3D.

    Attributes
    ----------
    max_ttime : float
        Returns the max travel time in the lookup table for each station.

    Methods
    -------
    ttimes(sampling_rate)
        Serve up the travel-time lookup tables.
    traveltime_to(phase, ijk)
        Query travel times to a grid location (in terms of indices) for a
        particular phase.
    save(filename)
        Dumps the current state of the lookup table object to a pickle file.
    load(filename)
        Restore the state of the saved LUT object from a pickle file.
    plot(fig, gs, slices=None, hypocentre=None, station_clr="k")
        Plot cross-sections of the LUT with station locations. Optionally plot
        slices through a coalescence volume.

    """

    def __str__(self):
        """Return short summary string of the lookup table object."""

        ll, *_, ur = self.coord2grid(self.grid_corners, inverse=True)
        cc = self.cell_count
        cs = self.cell_size

        out = ("QuakeMigrate traveltime lookup table\nGrid parameters"
               "\n\tLower-left corner  : {lat1:10.5f}\u00b0N "
               "{lon1:10.5f}\u00b0E {dep1:10.3f} m"
               "\n\tUpper-right corner : {lat2:10.5f}\u00b0N "
               "{lon2:10.5f}\u00b0E {dep2:10.3f} m"
               f"\n\tNumber of cells    : {cc}"
               f"\n\tCell dimensions    : {cs} m\n\n")

        out = out.format(lat1=ll[0], lon1=ll[1], dep1=ll[2],
                         lat2=ur[0], lon2=ur[1], dep2=ur[2])

        out += ("\tVelocity model:\n"
                "\t{}".format(str(self.velocity_model).replace("\n", "\n\t")))

        return out

    def ttimes(self, sampling_rate):
        """
        Serve up the travel-time lookup tables.

        The travel times are multiplied by the scan sampling rate and converted
        to integers.

        Parameters
        ----------
        sampling_rate : int
            Samples per second used in the scan run.

        Returns
        -------
        ttimes : array-like
            Stacked travel-time lookup tables for P and S phases, concatenated
            along the station axis.

        """

        pttimes = np.zeros(tuple(self.cell_count) + (len(self.station_data),))
        sttimes = np.zeros(tuple(self.cell_count) + (len(self.station_data),))

        for i, station in self.station_data.iterrows():
            pttimes[..., i] = self[station["Name"]]["TIME_P"]
            sttimes[..., i] = self[station["Name"]]["TIME_S"]

        pttimes = np.rint(pttimes * sampling_rate).astype(np.int32)
        sttimes = np.rint(sttimes * sampling_rate).astype(np.int32)

        return np.c_[pttimes, sttimes]

    def traveltime_to(self, phase, ijk):
        """
        Serve up the travel times to a grid location for a particular phase.

        Parameters
        ----------
        phase : str
            The seismic phase to lookup.
        ijk : array-like
            Grid indices for which to serve travel time.

        Returns
        -------
        travel_times : array-like
            Array of interpolated travel times to the requested grid position.

        """

        grid = tuple([np.arange(cc) for cc in self.cell_count])

        maps = np.zeros(tuple(self.cell_count) + (len(self.station_data),))
        for i, station in self.station_data.iterrows():
            maps[..., i] = self[station["Name"]]["TIME_{}".format(phase)]

        interpolator = RegularGridInterpolator(grid, maps, bounds_error=False,
                                               fill_value=None)
        return interpolator(ijk)[0]

    @property
    def max_ttime(self):
        """Get the maximum travel time from any station across the grid."""

        # Get all S maps
        ttimes = np.zeros(tuple(self.cell_count) + (len(self.station_data),))
        for i, station in self.station_data.iterrows():
            ttimes[..., i] = self[station["Name"]]["TIME_S"]

        return np.max(ttimes)

    def save(self, filename):
        """
        Dump the current state of the lookup table object to a pickle file.

        Parameters
        ----------
        filename : str
            Path to location to save pickle file.

        """

        with open(filename, "wb") as f:
            pickle.dump(self.__dict__, f, 4)

    def load(self, filename):
        """
        Read the contents of a pickle file and restore state of the lookup
        table object.

        Parameters
        ----------
        filename : str
            Path to pickle file to load.

        """

        with open(filename, "rb") as f:
            tmp_dict = pickle.load(f)

        self.__dict__.update(tmp_dict)

    def plot(self, fig, gs, slices=None, hypocentre=None, station_clr="k"):
        """
        Plot the lookup table for a particular station.

        Parameters
        ----------
        fig : `matplotlib.Figure` object
            Canvas on which LUT is plotted.
        gs : tuple(int, int)
            Grid specification for the plot.
        slices : array of arrays, optional
            Slices through a coalescence volume to plot.
        hypocentre : array of floats
            Event hypocentre - will add cross-hair to plot.
        station_clr : str, optional
            Plot the stations with a particular colour.

        """

        xy = plt.subplot2grid(gs, (2, 0), colspan=5, rowspan=5, fig=fig)
        xz = plt.subplot2grid(gs, (7, 0), colspan=5, rowspan=2, fig=fig)
        yz = plt.subplot2grid(gs, (2, 5), colspan=2, rowspan=5, fig=fig)

        xz.get_shared_x_axes().join(xy, xz)
        yz.get_shared_y_axes().join(xy, yz)

        # --- Set bounds ---
        corners = self.coord2grid(self.grid_corners, inverse=True)
        mins = [np.min(dim) for dim in corners.T]
        maxs = [np.max(dim) for dim in corners.T]
        sizes = (np.array(maxs) - np.array(mins)) / self.cell_count
        stack = np.c_[mins, maxs, sizes]

        for idx1, idx2, ax in [(0, 1, xy), (0, 2, xz), (2, 1, yz)]:
            min1, max1, size1 = stack[idx1]
            min2, max2, size2 = stack[idx2]

            ax.set_xlim([min1, max1])
            ax.set_ylim([min2, max2])

            # --- Plot slices through coalescence volume ---
            if slices is not None:
                idx = (idx1 + idx2) - 1
                slice_ = slices[idx]
                grid1, grid2 = np.mgrid[min1:max1 + size1:size1,
                                        min2:max2 + size2:size2]
                grid1 = grid1[:slice_.shape[0]+1, :slice_.shape[1]+1]
                grid2 = grid2[:slice_.shape[0]+1, :slice_.shape[1]+1]
                sc = ax.pcolormesh(grid1, grid2, slice_, cmap="viridis",
                                   edgecolors="face")

                if idx == 0:
                    # --- Add colourbar ---
                    cax = plt.subplot2grid(gs, (2, 7), colspan=1, rowspan=5,
                                           fig=fig)
                    cax.set_axis_off()
                    cbar = fig.colorbar(sc, ax=cax, orientation="vertical",
                                        fraction=0.4)
                    cbar.ax.set_ylabel("Coalescence value", rotation=90,
                                       fontsize=14)

            # --- Plot crosshair for event hypocentre ---
            if hypocentre is not None:
                ax.axvline(x=hypocentre[idx1], ls="--", lw=1.5, c="white")
                ax.axhline(y=hypocentre[idx2], ls="--", lw=1.5, c="white")

        # --- Plot stations ---
        xy.scatter(self.station_data.Longitude, self.station_data.Latitude,
                   s=15, marker="v", zorder=20, c=station_clr)
        xz.scatter(self.station_data.Longitude, self.station_data.Elevation,
                   s=15, marker="v", zorder=20, c=station_clr)
        yz.scatter(self.station_data.Elevation, self.station_data.Latitude,
                   s=15, marker=">", zorder=20, c=station_clr)
        for i, row in self.station_data.iterrows():
            xy.annotate(row["Name"], [row.Longitude, row.Latitude], zorder=20,
                        c=station_clr)

        # --- Axes labelling ---
        xy.xaxis.tick_top()

        xz.yaxis.tick_right()
        xz.invert_yaxis()
        xz.set_xlabel("Longitude (deg)", fontsize=14)
        xz.set_ylabel("Depth (m)", fontsize=14)
        xz.yaxis.set_label_position("right")

        yz.yaxis.tick_right()
        yz.set_xlabel("Depth (m)", fontsize=14)
        yz.set_ylabel("Latitude (deg)", fontsize=14)
        yz.yaxis.set_label_position("right")

    def __add__(self, other):
        """
        Define behaviour for the rich addition operator, "+".

        Two lookup tables which have identical grid definitions (as per "==")
        can be combined by adding the travel-time lookup tables from other.maps
        for which the station key is not already in self.maps.

        Parameters
        ----------
        other : QuakeMigrate LUT object
            LUT with travel-time lookup tables to add to self.

        """

        if not isinstance(other, LUT):
            print("Addition not defined for non-LUT object.")
            return self
        else:
            if self == other:
                for key, ttime in other.maps.items():
                    if key not in self.maps.keys():
                        self.maps[key] = ttime
                return self
            else:
                print("Grid definitions do not match - cannot combine.")

    def __eq__(self, other):
        """
        Define behaviour for the rich equality operator, "==".

        Two lookup tables are defined to be equal if their grid definitions are
        identical - corners, cell size, projections.

        Parameters
        ----------
        other : QuakeMigrate LUT object
            LUT with which to test equality with self.

        """

        # Test if other isinstance of LUT
        if not isinstance(other, LUT):
            print("Equality of LUT with non-LUT object is undefined.")
            return False
        else:
            # Test equality of grid corners
            eq_corners = (self.grid_corners == other.grid_corners).all()

            # Test equality of cell sizes
            eq_sizes = (self.cell_size == other.cell_size).all()

            # Test equality of projections
            eq_projections = (self.grid_proj == other.grid_proj
                              and self.coord_proj == other.coord_proj)

            return eq_corners and eq_sizes and eq_projections

    def __getitem__(self, key):
        """
        Provide a method to directly access travel-time maps by station key
        without having to go through the maps dictionary.

        Parameters
        ----------
        key : str
            Station ID for which to search.

        Returns
        -------
        ttimes : array-like
            Travel-time lookup table for key, if key exists.

        """

        try:
            return self.maps[key]
        except KeyError:
            print(f"No travel-time lookup table available for '{key}'.")
