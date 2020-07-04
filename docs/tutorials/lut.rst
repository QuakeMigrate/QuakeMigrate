The traveltime lookup table
===========================
This tutorial will cover the basic ideas and definitions underpinning the traveltime lookup table, as well as showing how they can be created.

In order to reduce computational costs during runtime, we pre-compute traveltime
lookup tables (LUTs). These LUTs contain P- and S-phase traveltimes for each station in the network to every point in a 3-D grid. This grid spans the volume of interest, herein termed the coalescence volume, within which QuakeMigrate will search for events.

Defining the underlying 3-D grid
--------------------------------
Before we can create our traveltime lookup table, we have to define the underlying 3-D grid which spans the volume of interest.

Coordinate projections
######################
First, we choose a pair of projections to represent the input coordinate space (``cproj``) and the Cartesian grid space (``gproj``). We do this using the Python interface with the PROJ library, pyproj. It is important to think about which projection is best suited to your particular study region. 

We use here the WGS84 reference ellipsoid (used as standard by the Global Positioning System) as our input space and the Lambert Conformal Conic projection to form our Cartesian space. The units of the Cartesian space are specified as metres. The values used in the LCC projection are for a study region in northern Borneo.

::

	from pyproj import Proj

	cproj = Proj(proj="longlat", ellps="WGS84", datum"=WGS84", no_defs=True)
	gproj = Proj(proj="lcc", lon_0=116.75, lat_0=6.25, lat_1=4.0, lat_2=7.5,
	         datum="WGS84", ellps="WGS84", units="m", no_defs=True)

Geographical location and spatial extent
########################################
In order to geographically situate our lookup table, we define two points, herein called the lower-left and upper-right corners (``ll_corner`` and ``ur_corner``, respectively). By default, we work in a depth-positive frame (i.e. positive down or left-handed coordinate system) and use metres. It is in theory possible to run QuakeMigrate with distances measured in kilometres, as long as the user specifies this requirement when defining the grid projection. In order to avoid any unexpected problems, however, we recommend using metres.

This schematic shows the relative positioning of the two corners:

.. image:: img/LUT_definition.png

The final piece of information required to fully define the grid on which we will calculate traveltimes is the size (in each dimension, `x`, `y`, `z`) of a cell (``cell_size``). The LUT class will automatically find the number of cells required in each dimension to span the specified geographical region. If a cell dimension doesn't fit into the corresponding grid dimension an integer number of times, the location of the upper-right corner is shifted to accommodate an additional cell.

::

	ll_corner = [116.075, 5.573, -1750]
	ur_corner = [117.426, 6.925, 27750]
	cell_size = [500., 500., 500.]

Creating an instance of the LUT class
-------------------------------------
We are now ready to create an instance of the :class:`LUT` class, which we can then populate with traveltimes. We import the :mod:`QMigrate.lut` module, which contains two submodules: :mod:`lut.py`, which contains the :class:`LUT` class; and :mod:`create_lut.py`, which contains a suite of utility functions to compute traveltimes.

::

	from QMigrate.lut import compute, LUT, read_nlloc

	# --- Create a new LUT ---
	lut = LUT(ll_corner=ll_corner, ur_corner=ur_corner, cell_size=cell_size,
	          grid_proj=gproj, coord_proj=cproj)

Computing traveltimes
---------------------
We have bundled a few methods of computing traveltimes into QuakeMigrate.

In all cases we will make use of the :mod:`QMigrate.io` module, so let's import that first and read in our station file:

::

    from QMigrate.io import read_stations, read_vmodel

    stations = read_stations("/path/to/station_file")

Homogeneous velocity model
##########################
Simply calculates the straight line traveltimes between stations and points in the grid.

::

	compute(lut, stations, method="homogeneous", vp=5000., vs=3000.)

Fast-marching method
####################
The fast-marching method implicitly tracks the evolution of the wavefront. See Rawlinson & Sambridge (2005) for more details.

::

	vmod = read_vmodel("/path/to/vmodel_file")
	compute(lut, stations, method="1dfmm", vmod=vmod)

NonLinLoc style 2-D sweep
#########################
Uses the Eikonal solver from NonLinLoc under the hood to generate a traveltime grid for the 2-D slice that passes through the station and the point in the grid furthest away from that station. This slice is then "swept" using a bilinear interpolation scheme to produce a 3-D traveltime grid. This has the benefit of being able to include stations outside of the volume of interest, without having to increase the size of the grid.

::

	vmod = read_vmodel("/path/to/vmodel_file")
	compute(lut, stations, method="1dsweep", vmod=vmod, block_model=True)

Other formats
#############
It is also easy to import traveltime lookup tables generated by other means. We have provided a parser for lookup tables in the NonLinLoc format (:func:`read_nlloc()`). It is straightforward to adapt this code to read any other traveltime lookup table, so long as it is stored as an array. Create an instance of the LUT class with the correct grid dimensions, then add the traveltime arrays (in C-order) to the ``LUT.maps`` dictionary.

Saving your LUT
---------------
Finally, you will need to save the lookup table to file. The default approach is to pickle the entire object.

::

	lut.save("/path/to/output/lut")

Reading in a saved LUT
----------------------
When running the main stages of QuakeMigrate (`detect`, `trigger`, and `locate`)
it is necessary to read in the saved LUT, which can be done as:

::

    from QMigrate.io import read_lut
    lut = read_lut(lut_file="/path/to/lut_file")