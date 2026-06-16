<p align="center">
  <!-- DOI -->
  <a href="https://doi.org/10.5281/zenodo.4442748">
    <img src="https://zenodo.org/badge/DOI/10.5281/zenodo.4442748.svg" alt="DOI" />
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
  <!-- Python version-->
  <a href="https://www.python.org/downloads/release/python-3100/">
    <img src="https://img.shields.io/badge/python-3.10+-blue.svg" />
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

For a demonstration of the options available, and a starting point to write scripts for your own use-case, see the [template scripts](examples/template_scripts).

Citation
--------
If you use QuakeMigrate in your work, please cite the software and the accompanying paper.

### Paper

**Winder, T., Bacon, C.A., Smith, J.D., Hudson, T.S., and White, R.S. (2026).**
QuakeMigrate: a Python Package for Automatic Earthquake Detection and Location Using Waveform Migration and Stacking.
*Seismica*, 5(1).

<p align="center">
  <!-- DOI -->
  <a href="https://doi.org/10.26443/seismica.v5i1.1854">
    <img src="https://img.shields.io/badge/DOI-10.26443/seismica.v5i1.1854-blue.svg" alt="DOI" />
  </a>
</p>

```bibtex
@article{winder2026quakemigrate,
  title = {QuakeMigrate: a Python Package for Automatic Earthquake Detection and Location Using Waveform Migration and Stacking},
  author = {Winder, Tom and Bacon, Conor Andrew and Smith, Jonathan D. and Hudson, Thomas Samuel and White, Robert S.},
  journal = {Seismica},
  year = {2026},
  volume = {5},
  number = {1},
  doi = {10.26443/seismica.v5i1.1854},
  url = {https://doi.org/10.26443/seismica.v5i1.1854}
}
```

### Software

Please also cite the relevant QuakeMigrate software release on Zenodo:

<p align="center">
  <!-- DOI -->
  <a href="https://doi.org/10.5281/zenodo.4442748">
    <img src="https://zenodo.org/badge/DOI/10.5281/zenodo.4442748.svg" alt="DOI" />
  </a>
</p>

Full citation metadata is provided in [`CITATION.cff`](CITATION.cff).

Contributing to QuakeMigrate
----------------------------
Contributions to QuakeMigrate are welcomed. Whether you have identified a bug or would like to request a new feature, your first stop should be to reach out, either directly or—preferably—via the GitHub Issues panel, to discuss the proposed changes. Once we have had a chance to scope out the proposed changes you can proceed with making your contribution following the instructions in our [contribution guidelines](https://github.com/QuakeMigrate/QuakeMigrate/blob/master/CONTRIBUTING.md).

Bug reports, suggestions for new features and enhancements, and even links to projects that have made use of QuakeMigrate are most welcome.

Contact
-------
You can contact us directly at: quakemigrate.developers@gmail.com

Any additional comments/questions can be directed to:
* **Tom Winder** - tom.winder@esc.cam.ac.uk
* **Conor Bacon** - conor.bacon@norsar.no

License
-------
This package is written and maintained by the QuakeMigrate developers, Copyright QuakeMigrate developers 2020–2026. It is distributed under the GPLv3 License. Please see the [LICENSE](LICENSE) file for a complete description of the rights and freedoms that this provides the user.
