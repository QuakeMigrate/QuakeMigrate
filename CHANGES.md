
- Examples:
  * Added a new example use-case from Askja volcano, Iceland, which will be featured in the forthcoming manuscript. This example showcases the capability of QuakeMigrate to detect and locate a wide variety of seismic events with different source types, and in this case simultaneously with the same set of parameters. febabcd
  * Add an example that demonstrates the functioning of QuakeMigrate using a synthetic dataset. This dataset is one of the examples presented in the manuscript. The example is also accompanied by documentation covering the example, which is a great plus to our online documentation!
- quakemigrate.io.marginal_coalescence:
  * Introduce support for saving the marginalised 3D coalescence map generated within `locate()`, and also a utility function to read it for e.g. plotting purposes, or further interrogation of the topology of the coalescence peak. ebcae96
- switched off pick plotting for the VT_Iceland example. 7bab86d
- quakemigrate.signal.trigger:
  * Introduce the `median_ratio` method for determining a dynamic trigger threshold, by taking a multiplier of the median value of the coalescence trace in a user-defined window. 6d16f18
  * Fix a bug in `trigger.chunks2trace()` - still using deprecated `numpy.product()` 6d16f18
  * Rename the existing dynamic trigger threshold method (based on a multiple of the median absolute devation of the coalescence trace) from `dynamic` -> `mad` 9ef138a
  * Introduce "trigger smoothing" functionality; the option to smooth the coalescence trace by convolving a gaussian kernel of user-defined sigma and width before determining and applying the trigger threshold to identify candidate events. 1ea643c
  * Revert to always using the `COA` timeseries for determining the trigger peak index (i.e. candidate event origin time) to ensure better correspondence with origin times determined within locate() 836521a
- tests.test_trigger:
  * Add unit tests for trigger dynamic threshold and smoothing functions 78f3e73

1.1.1
=====
Address a bug in amplitude plotting introduced in v1.1.0, and modify the Volcanotectonic_Iceland example such that this issue is now covered by tests going forwards.

- Explicitly use `pandas.Series` for input to plot.amplitudes.label_stations(), and fix a minor plotting bug e63851f
- Add filters for magnitude calculation in the Volcanotectonic_Iceland example, as well as making some additional improvements to the amp_params and mag_params configuration 7e5768a

1.1.0
=====
Move the onset function computation into a compiled C library. The ability to use the original Python backend is retained. Settings for the icequake examples were updated to improve the detections/locations, reflecting improvements to the core code that allows for higher resolutions to be computed without incurring prohibitive computational costs. We also added a feature to allow users to select the transformation applied to thea waveform data before onset functions are computed.

Importantly, we addressed a bug in the ordering of when the onset functions for the horizontal (S) components were combined relative to when the log of the onset function was taken. This changes the values for the example benchmarks, which have been updated accordingly. This motivated the bump from version 1.0.x to 1.1.0.

- Add C onset function library dd3b52d
- Add Python backend for onset functions 907873a
- Move the logging of the onset functions to the correct position (after combining the two horizontal components for the S onset) 1600f11
- Update Rutford icequake example 5d7510f and 9e2c363
- Update Iceland icequake example 8f61a64 and 9e2c363
- Update minimum and maximum supported Python versions to 3.9 and 3.12, respectively 28d2537
- Move onset log (breaking change, fixes a bug) 1600f11
- Add the ability to select the type of signal transform applied to the raw data before computing the onset functions 1600f11
  - Energy (waveform squared)
  - Absolute value
  - Envelope (absolute value of the analytic signal, i.e., the Hilbert transform)
  - Envelope squared

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
