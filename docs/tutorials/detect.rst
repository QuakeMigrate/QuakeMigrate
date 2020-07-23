The Detect Stage
===========================
This tutorial will cover the basic ideas and definitions underpinning the initial stage of a QuakeMigrate run, the detect stage.

During this stage, the waveform data is continuously migrated through your travel time lookup tables (LUTs) to generate a coalescence function through time. This function records the x, y, z position of maximum coalescence in your volume for each timestep. Peaks in this function are then triggered during the `trigger` stage.

The migration of the data is performed for each node and timestep in a 4D sense and can be very computationally intense. For this reason, it is typical to decimate the LUT to reduce the computation time. Muti-core machines or HPC clusters can also be used to split the time period and perform the computation in parallel.

Before you start
-----------------

You will need a semi-continuous waveform archive organised in a known way (see `Archive` tutorial), a travel time LUT (as generated in the previous tutorial) and your station file (as used to generate the LUT). You will also need to choose a location to store your results and a name for your run. QuakeMigrate will automatically generate an output structure to store all your results and place this in a folder in your chosen location named as the run name. You may well run QuakeMigrate many times before you reach the final set of parameter values which produce the best results. It is therefore important to choose a clear and documented run naming scheme.

.. note:: Your run name and directory does not have to be the same for the three QuakeMigrate stages (`detect`, `trigger` and `locate`).

We proceed by defining these parameters as variables.

::
    from quakemigrate.io import read_stations

    archive_path = "/path/to/archived/data"
    lut_file = "/path/to/lut_file"
    station_file = "/path/to/station_file"

    run_path = "/path/to/output"
    run_name = "name_of_run"
    
    stations = read_stations(station_file)

Detect runs on continuous data between two defined timestamps. Internally, QuakeMigrate uses UTCDateTime (from obspy) to parse the timestamps so you can input your dates in any way that is understandable. However, using an isoformat like datetime string (as below) is recommended.

::
    starttime = "2018-001T00:00:00.0"
    endtime = "2018-002T00:00:00.0"

The waveform achive is defined using an `Archive` object (see `Archive` tutorial) and the LUT can be importing using the `read_lut` function.

::
    from quakemigrate.io import Archive, read_lut
    
    archive = Archive(archive_path=archive_path, stations=stations,
                  archive_format="YEAR/JD/STATION")

Decimation of the LUT
-----------------

To reduce computation time the decimation functionality of the LUT can be used. This reduces the number of nodes at which the coalesence is calculated resulting in a spatially low-pass filtered coalesence function. This is reasonable when we use the data for earthquake detection, but should be carefully considered if the locations output from a decimated grid are to be relied upon. 

Typically, you the final node spacing of your decimated grid should be similar to the expected uncertainty on your earthquake location. For the example below, the raw LUT has a node spacing of 0.5 km. After decimation the node spacing is 2.5 km in the horizontal directions and 2 km in the vertical direction. These are good starting values for a local seismic network with an aperature of 20 - 80 km.

::
    lut = lut.decimate([5, 5, 4])


XXX NOTES HERE ABOUT PARALLEL XXXX












In order to reduce computational costs during runtime, we pre-compute traveltime
lookup tables (LUTs) for each seismic phase and each station in the network to every node in a regularised 3-D grid. This grid spans the volume of interest, herein termed the coalescence volume, within which QuakeMigrate will search for events.

Defining the underlying 3-D grid
--------------------------------
Before we can create our traveltime lookup table, we have to define the underlying 3-D grid which spans the volume of interest.

Coordinate projections
######################
First, we choose a pair of coordinate reference systems to represent the input coordinate space (``cproj``) and the Cartesian grid space (``gproj``). We do this using `pyproj`, which provides the Python bindings for the PROJ library. It is important to think about which projection is best suited to your particular study region. More information can be found [in their documentation](https://pyproj4.github.io/pyproj/stable/).

.. warning:: The default units of :class:`Proj` are `metres`! It is strongly advised that you explicitly state which units you wish to use.

We use here the WGS84 reference ellipsoid (used as standard by the Global Positioning System) as our input space and the Lambert Conformal Conic projection to form our Cartesian space. The units of the Cartesian space are specified as kilometres. The values used in the LCC projection are for a study region in Sabah, Borneo.