<p align="center">
  <!-- DOI -->
  <a href="https://doi.org/10.5281/zenodo.4442749">
    <img src="https://zenodo.org/badge/DOI/10.5281/zenodo.4442749.svg" alt="DOI" />
  </a>
  <!-- ReadTheDocs -->
  <a href="https://quakemigrate.readthedocs.io/en/latest">
    <img src="https://readthedocs.org/projects/quakemigrate/badge/?version=latest" />
  </a>
  <!-- Build Action -->
  <a href="https://github.com/QuakeMigrate/QuakeMigrate/actions">
    <img src="https://github.com/QuakeMigrate/QuakeMigrate/actions/workflows/build_wheels.yml/badge.svg" />
  </a>
  <!-- PyPI -->
  <a href="https://pypi.org/project/quakemigrate/">
    <img src="https://img.shields.io/pypi/v/quakemigrate" />
  </a>
  <!-- Coverage -->
  <a href="https://codecov.io/gh/QuakeMigrate/QuakeMigrate">
    <img src="https://codecov.io/gh/QuakeMigrate/QuakeMigrate/branch/master/graph/badge.svg">
  </a>
  <!-- MyBinder Example -->
  <a href="https://mybinder.org/v2/gh/QuakeMigrate/QuakeMigrate/master">
    <img src="https://mybinder.org/badge_logo.svg" />
  </a>
  <!-- Python version-->
  <a href="https://www.python.org/downloads/release/python-380/">
    <img src="https://img.shields.io/badge/python-3.8+-blue.svg" />
  </a>
  <!-- License -->
  <a href="https://www.gnu.org/licenses/gpl-3.0">
    <img src="https://img.shields.io/badge/License-GPLv3-blue.svg" />
  </a>
</p>

<p align="center">
  <a href="https://quakemigrate.readthedocs.io/en/latest">QuakeMigrate</a> is a Python package for automatic earthquake detection and location using waveform migration and stacking.</a>
</p>

<p align="center">
<img src="https://github.com/QuakeMigrate/QuakeMigrate/raw/master/docs/img/QMlogoBig.png", width="80%">
</p>

Key Features
------------
QuakeMigrate uses a waveform migration and stacking algorithm to search for coherent seismic phase arrivals across a network of instruments. It produces—from raw data—catalogues of earthquakes with locations, origin times, phase arrival picks, and local magnitude estimates, as well as rigorous estimates of the associated uncertainties.

The package has been built with a modular architecture, providing the potential for extension and adaptation at numerous entry points. This includes, but is not limited to:
* the calculation or import of traveltime grids
* the choice of algorithm used to identify phase arrivals (for example by kurtosis, cross-covariance analysis between multiple components, machine learning techniques and more)
* the stacking function used to combine onset functions
* the algorithm used to perform phase picking

Documentation
-------------
Documentation for QuakeMigrate is hosted [here](https://quakemigrate.readthedocs.io/en/latest/index.html).

Installation
------------
Detailed installation instructions can be found [here](https://quakemigrate.readthedocs.io/en/latest/installation.html).

If you're comfortable with virtual environments and just want to get started, QuakeMigrate is available via the Python Package Index, and can be installed via pip:

```console
pip install quakemigrate
```

Usage
-----
We are working on tutorials covering how each individual aspect of the package works, as well as example use cases where we provide substantive reasoning for the parameter choices used. These examples include applications to cryoseismicity and volcano seismology.

This is a work in progress - [see our documentation for full details](https://quakemigrate.readthedocs.io/en/latest/tutorials.html).

### Examples you can run in your browser
To quickly get a taste for how the software works, try out the two icequake examples hosted on Binder:
* Icequakes at the Rutford Ice Stream, Antarctica  [![badge](https://img.shields.io/badge/launch-Icequake%20Rutford%20notebook-579ACA.svg)](https://mybinder.org/v2/gh/QuakeMigrate/QuakeMigrate/master?filepath=examples%2FIcequake_Rutford%2Ficequakes_rutford.ipynb)
* Icequakes at the Skeiðarárjökull outlet glacier, Iceland [![badge](https://img.shields.io/badge/launch-Icequake%20Iceland%20notebook-E66581.svg)](https://mybinder.org/v2/gh/QuakeMigrate/QuakeMigrate/master?filepath=examples%2FIcequake_Iceland%2Ficequakes_iceland.ipynb)

And for a more comprehensive demonstration of the options available, see the [template scripts](examples/template_scripts).

Citation
--------
If you use this package in your work, please cite the following conference presentation:

```console
Winder, T., Bacon, C.A., Smith, J.D., Hudson, T., Greenfield, T. and White, R.S., 2020. QuakeMigrate: a Modular, Open-Source Python Package for Automatic Earthquake Detection and Location. In AGU Fall Meeting 2020. AGU.
```

as well as the relevant version of the source code on [Zenodo](https://doi.org/10.5281/zenodo.4442749).

We hope to have a publication coming out soon:

```console
Winder, T., Bacon, C.A., Smith, J.D., Hudson, T.S., Drew, J., and White, R.S. QuakeMigrate: a Python Package for Automatic Earthquake Detection and Location Using Waveform Migration and Stacking. (to be submitted to Seismica).
```

Contributing to QuakeMigrate
----------------------------
Contributions to QuakeMigrate are welcomed. Whether you have identified a bug or would like to request a new feature, your first stop should be to reach out, either directly or—preferably—via the GitHub Issues panel, to discuss the proposed changes. Once we have had a chance to scope out the proposed changes you can proceed with making your contribution following the instructions in our [contributions guidelines](https://github.com/QuakeMigrate/QuakeMigrate/blob/master/CONTRIBUTING.md).

Bug reports, suggestions for new features and enhancements, and even links to projects that have made use of QuakeMigrate are most welcome.

Contact
-------
You can contact us directly at: quakemigrate.developers@gmail.com

Any additional comments/questions can be directed to:
* **Tom Winder** - tom.winder@esc.cam.ac.uk
* **Conor Bacon** - conor.bacon@cantab.net

License
-------
This package is written and maintained by the QuakeMigrate developers, Copyright QuakeMigrate developers 2020–2023. It is distributed under the GPLv3 License. Please see the [LICENSE](LICENSE) file for a complete description of the rights and freedoms that this provides the user.
