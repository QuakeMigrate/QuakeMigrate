1.0.3
=====
Make some minor updates to tooling used by the package and address some issues that had arisen from underlying dependencies. Further, we have added a backdoor that allows us to debug issues with our CI/CD testing.

- Switched to Ruff autoformatter (9580ccf)
- Addressed a Matplotlib bug arising from the deprecation of the method of sharing/joining axes we were previously using (a4fe6f8)
- Resolved issue with test workflow failure (a11677b/1363a7a) and added tmate backdoor for debugging (9b433d3)
- Updated an issue with the data downloading in the Iceland volcano-tectonic example. This had the added benefit of being faster, too! (b11a6b8)

1.0.2
=====
This patch release introduces a few code management/styling tools to help maintain consistent styling across contributors, as well as re-establishing routine testing.

- Introduced a pre-commit hook for developers that performs a number of checks that must pass before a commit can be made. Principle among these tests is the use of Black ("The uncompromising Python code formatter"). This ensures consistent styling across all commits.
- A GitHub workflow now manages test running and coverage analysis. The workflow is triggered by a number of scenarios, such as a new PR being opened etc. For full details, see 24bdb2e. Importantly, the workflow is also scheduled to run on a regular basis. Once a week, all tests are run—any breaking changes introduced by our dependencies should be caught this way.
- The Contributing guidelines were updated to reflect these changes to the developer workflow.

1.0.1
=====
Add the ability to install QuakeMigrate from the Python Package Index. This involves building Python wheels for different operating systems / architectures / Python versions. This is done using a GitHub Action and build machines provided by GitHub.

- Overhaul the build system from the old, quite messy, setup.py file to a pyproject.toml file (specifying mostly metadata) and a setup.py file that specifically handles the C extension module.
- Make the codebase compatible with most recent versions of dependencies (and specify some minimum version requirements).
- Handled a number of deprecation warnings that had started to crop up.
- Reviewed the installation instructions and provided significant additional detail for installing compilers across different operating systems. While no longer necessary (`pip install quakemigrate` will install QuakeMigrate complete with its precompiled C library), it may still be useful for anyone wanting to build from the source code themselves.
- Updated base README.md file to reflect these updates.
- Drop TravisCI—not currently working and will migrate to testing and coverage uploading via GitHub Actions in the future.

1.0.0
=====

This provides a limited summary of the major changes to the code since the
previous release. For full details please see the linked pull requests, and
commit messages therein.

- Change to top-level package name - `QMigrate` → `quakemigrate` See #85
- `quakemigrate.core`
  * C libraries now build automatically when installing the package.
  * C functions have been cleaned up and documented properly. See #21.
- `quakemigrate.export` – new!
  * Add some export utilities to a new module (`export`).
  * Currently includes functions to take the outputs of QuakeMigrate to: ObsPy
    catalogue; ObsPy catalogue to NLLoc OBS file; ObsPy catalogue to MFAST SAC
    input; QuakeMigrate to pyrocko Snuffler input.
- `quakemigrate.io`
  * Refactor the I/O system entirely. This was in response to Issue #66. See
    68c13f7 for full details.
- The re-write of the `quakemigrate.io.data` sub-module includes fixing bugs
  and making breaking changes related to the detrending and
  upsampling/decimation of waveform data. See c1ff447 and #103.
  * Add support for reading response information (via a light wrapper of the
    obspy function) and functions for response removal & Wood-Anderson
    seismogram simulation, for use with the new
    `quakemigrate.signal.local_mag` module.
  * Add a new, more comprehensive, transparent & flexible
    `check_availability()` function, to check what data is available given a
    set of provided quality criteria.
  * Added functions for removing response and simulating WA response
    here, allowing for this functionality to be accessed from anywhere across
    the package. This includes the ability to output real and/or WA cut
    waveforms (as velocity or displacement seismograms), for example for
    spectral analysis. Response removal parameters are specified (along
    with the instrument response inventory) when creating the `Archive`
    object.
  * The format of triggered event files, event files, pick files and
    station-availabiity files have been heavily overhauled. See #76.
  * Fix to allow for numerical station names.
- `quakemigate.lut` – significant rewrite (see #54, #65)
  * Significant changes to API – see examples.
  * Users from the 0.x series will need to update their look-up table files
    (using an included utility function) or re-compute them.
  * Fully-documented, including a tutorial in the documentation.
  * Handling of a possible bug/ambiguity in the scipy RegularGridInterpolator
    API. The default for the fill_value parameter does not appear to be
    consistent with documentation.
  * The traveltime lookup tables are now stored in a dictionary structure
    `maps[station][phase]` to enable migration of a flexible combination of
    seismic phases, and to make it possible to migrate onsets for a subset of
    the phases and stations contained in the LUT. See #75, #103.
  * The 3-D grid on which the lookup tables are defined is now more intuitive
    to build. The user simply chooses the positions of a pair of opposite
    corners (the lower-left and upper-right) in geographic coordinates, the
    geographic and cartesian projections (using pyproj), and a node spacing
    along each axis.
    The number of grid nodes will be calculated automatically to span the
    volume of interest.
  * User-specified units: the user must specify when making an LUT whether to
    use units of metres or kilometres; this will then be used consistently
    throughout the package. See #79.
- `quakemigrate.plot` – new! See #83.
  * Extracted all base plotting methods to individual modules within the
    `quakemigrate.plot` module. No longer using a class to pass the information
    around.
  * Revamp all of the figures produced by QuakeMigrate to include more useful
    information and to make better use of the available space. See 72b1c47 for
    details.
- `quakemigrate.signal`
  * Refactor to be more flexible with the input data.
    * QuakeMigrate now allows for single-component data to be used, or
      stacking to be performed on just one phase (e.g. just P or just S). The
      required changes reached quite deep into the package and have changed how
      Onset objects are created, but is ultimately very straightforward to use.
      See #103.
    * Channel names can now be specified by the user, by default they are “Z”
      for vertical component, and “[N,1]” and [“E,2”] for horizontal
      components.
  * Internally use `obspy.Trace` objects to store data up to the point of
    passing it to the C functions; this adds greater flexibility and more
    built-in methods for quality checking, filtering, re-sampling etc. than the
    previous framework using arrays.
  * `quakemigate.signal.onsets` – significant re-write
    * Extracted the embedded onset function generation from the core QuakeScan
      class to a new submodule, `quakemigrate.signal.onsets`.
    * Various changes to parameter names – see examples.
    * Created an Abstract Base Class – Onset. This class can be used as a base
      class for a class implementing an alternative onset function algorithm,
      while ensuring compliance with the embedded code in QuakeScan.
    * Created a new class OnsetData to store the pre-processed waveforms used
      for onset calculation, the onset functions themselves, and all associated
      parameters and attributes.
    * Significant expansion to the options available for data quality-checking;
      now exposed to the user, giving the flexibility to select which to use.
    * The STA/LTA onset function remains the default.
  * `quakemigate.signal.pickers` – significant re-write
    * Extracted the embedded picking functions from the core QuakeScan class to
      a new module, `quakemigrate.signal.pickers`.
    * General improvements to picking through a mix of fixes and features.
      Major improvement to clarity/style.
    * Created an Abstract Base Class – PhasePicker. This class can be used as a
      base class for a class implementing an alternative phase picking
      algorithm, while ensuring compliance with the embedded code in
      QuakeMigrate.
    * The user can provide a different onset function for phase picking using
      GaussianPicker than they used for migration in locate().
    * Bug fixes related to calculating the pick threshold; new option to use
      the median absolute deviation of the noise, rather than a percentile.
      See #116.
    * Fitting a 1-D Gaussian to the phase onset function remains the default
      method of phase picking.
  * ` quakemigate.signal.trigger`
    * Add dynamic trigger threshold method based on the median absolute
      deviation, a robust statistical estimator that is insensitive to extreme
      outliers. See #59.
    * Added the ability to trigger events restricted to a specific region of
      the grid (specified as geographic coordinates, and illustrated on the
      trigger summary plot).
    * Numerous bug-fixes related to the handling of overlapping triggers.
    * Fix an indexing bug in trigger that caused the last event to be missed if
      it was a single sample in length.
  * `quakemigrate.signal.local_mag` – new! See #71
    * A comprehensive suite of codes for local earthquake magnitude calculation
      by measuring displacement amplitudes from Wood-Anderson simulated
      seismograms.
    * This can optionally be used as part of a locate run to automatically
      output ML estimates for each located event.
- General changes
  * Update to various class attributes. Deprecation warnings + internal
    attribute re-mappings have been included to ease the transition.
  * Add additional examples (Rutford icequakes + Iceland dike intrusion). This
    includes a script that performs data download from IRIS through an FDSN
    client. See #105.
  * Option to locate events from a user-provided triggered event file, as well
    as the existing functionality to automatically read triggers between two
    timestamps from the default output directory structure. See #53.
  * Re-write `setup.py` to automatically build the C extension library and link
    it to the rest of the package. Should be more robust on different
    platforms. Functions to read .dll or .so files (depending on operating
    system) have been added to automatically load the correct linked library.
    See #84.
  * Add a system of module imports through `__init__.py` files to reduce the
    verbosity of input statements.
  * New, intuitive directory structure for outputs, allowing straightforward
    batch processing of data archives. See #68.
  * Option to run trigger() multiple times from the same detect() output by
    using different run sub-names. Also incorporated into the functions used by
    locate() to read triggered event files.
  * Add readthedocs documentation with sphinx. Here we can host the
    documentation for the source code, instructions on installation, and
    tutorials on how to use the package.
  * The internal reference frame has Z being positive down (the depth frame).
    Station elevations in the station input file are in the positive up
    (elevation/natural frame) and converted internally.
  * More information on intermediate results is retained in the final output
    files. For example, the coalescence and normalised coalescence values for a
    triggered event, along with which value it was triggered against.
  * Added tests based on the "Iceland Icequake" and "Volcanotectonic_Iceland"
    example use-cases to veryify correct operation ofter installation.
  * Added Continuous Integration (Travis CI) allow us to catch breaking
    issues before new features/fixes are merged and, in particular, should
    help us keep on top of changes related to upstream changes to
    dependencies.
  * License: QuakeMigrate v1.0.0 is released under the GPLv3 license.
- Optimisations
  * Optimise compute by changing the flags passed to the the compiler. Bumped
    more of the compute() method to the C library to maximise efficiency.
    See a9ffdb6
  * Moved to logging with the native Python logging module, which allows for
    writing logs to file as well as stdout in a more concise manner. See #81
