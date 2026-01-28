Overview of interfaces
======================
Our aim with QuakeMigrate is to provide a tool that can be used by any researcher/student, regardless of their experience with computer programming. For the experienced user, the script-based interface will likely be the most appropriate. For those with little-to-no Python experience, we have provided a command line + config file interface, which sacrifices a little flexibility for ease-of-use. This has the added benefit of making suite of config files the definitive record of a run, which are lightweight and easy to share. In the future, we would like to provide a browser-based interface to provide a complete project management environment that is abstracted from the underlying code.

Script-based interface
----------------------
The script-based approach to using QuakeMigrate is the original and most flexible option. It requires a little experience with the Python programming language, but is fairly straightforward. All example applications provided in the examples directory currently use this interface—for further information, please refer to them.

Command-line interface
----------------------
In an effort to offer a minimal-coding-experience-required interface to QuakeMigrate, we have added a basic command-line interface (accessed via ``qmctl`` on the command line, or alternatively via the alias ``quakemigrate``) and the option to configure every stage of QuakeMigrate using human-readable config files. These files use the TOML (Tom's Obvious Markup Language) file format, which is used elsewhere in the Python project, with each stage being configured by a standalone file.

This command-line interface currently has the following top-level commands:

- ``init``: this command will initialise a new QuakeMigrate project. Default config files are initialised that must be completed by the user. These default config files come bundled with the QuakeMigrate installation. Optionally, the user can pre-link a station file and/or a velocity model file.

- ``new``: this command has two sub-commands for instantiating a new traveltime lookup table or a new run. Config files are copied from the project config templates, so it is advised that the user populates these with the basic information first. The sub-commands are:
    
    * ``lut``: this is used to add a new project-wide traveltime lookup table. Expects a LUT name.
    * ``run``: this is used to add a new run configuration. Expects a run name.

- ``build-lut``: used to build a traveltime lookup table. Expects a run name.
- ``detect``: used to run the detect stage of QuakeMigrate. Expects a run name.
- ``trigger``: used to run the trigger stage of QuakeMigrate. Expects a run name.
- ``locate``: used to run the locate stage of QuakeMigrate. Expects a run name.

FOr now, the user must be in root directory of a QuakeMigrate project for these commands to run (tracked by a ``.`` file).

An example usage of the command-line interface follows:

::

    qmctl init --name test-project --station-file <path/to/station_file> --velocity-model <path/to/velocity_model_file>
    cd test-project

The user must now edit the config files for each stage. Note: it is important to _only_ edit the values of each entry, leaving the rest of the file unchanged.

::

    # Create a new LUT config file
    qmctl new lut example-lut

::

    # Create a new suite of run stage config files and setup directories
    qmctl new run example-run

Once the LUT and run configuration files have been set up, each stage can be executed as:

::

    qmctl build-lut example-run
    qmctl run detect example-run
    qmctl run trigger example-run
    qmctl run locate example-run

Of course, it remains possible to use the original script-based interface to QuakeMigrate. Future efforts may expand on the command-line interface options (facilitating, for example, easier duplication of basic projects for batched processing etc.).
