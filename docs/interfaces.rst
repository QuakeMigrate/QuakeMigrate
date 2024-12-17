Overview of interfaces
======================
Our aim with QuakeMigrate is to provide a tool that can be used by any researcher/student, regardless of their experience with computer programming. For the experienced user, the script-based interface will likely be the most appropriate. For those with little-to-no Python experience, we have provided a command line + config file interface, which sacrifices a little flexibility for ease-of-use. In the future, we would like to provide a browser-based interface to provide a complete project management environment that is abstracted from the underlying code.

Script-based interface
----------------------
The script-based approach to using QuakeMigrate is the original and most flexible option. It requires a little experience with the Python programming language, but is fairly straightforward. All example applications provided in the examples directory currently use this interface—for further information, please refer to them.

Command-line interface
----------------------
In an effort to offer a minimal-coding-experience-required interface to QuakeMigrate, we have added a basic command-line interface (accessed via ``qmctl`` on the command line, or alternatively via the alias ``quakemigrate``) and the option to configure every stage of QuakeMigrate using human-readable config files. These files use the TOML (Tom's Obvious Markup Language) file format, which is used elsewhere in the Python project, with each stage being configured by a standalone file.

There are currently two basic commands for the command-line interface:

- ``init``: this command will initialise a new QuakeMigrate project. Default config files are initialised that must be completed by the user. These default config files come bundled with the QuakeMigrate installation. Optionally, the user can pre-link a station file and/or a velocity model file.

- ``run``: this command is used to specify which stage is to be run. The user must be in root directory of a QuakeMigrate project for this to run (tracked by a ``.`` file), as well as have a ``.toml`` file for the requested stage. There are four stages that can be run using this command:

    * ``lut``: used to build a traveltime lookup table;
    * ``detect``: used to run the detect stage of QuakeMigrate;
    * ``trigger``: used to run the trigger stage of QuakeMigrate;
    * ``locate``: used to tun the locate stage of QuakeMigrate.

An example usage of the command-line interface follows:

::

    qmctl init --name test-project --station-file <path/to/station_file> --velocity-model <path/to/velocity_model_file>
    cd test-project

The user must now edit the config files for each stage. Note: it is important to _only_ edit the values of each entry, leaving the rest of the file unchanged.

::

    qmctl run -s lut
    qmctl run -s detect
    qmctl run -s trigger
    qmctl run -s locate

Of course, it remains possible to use the original script-based interface to QuakeMigrate. Future efforts may expand on the command-line interface options (facilitating, for example, easier duplication of basic projects for batched processing etc.).
