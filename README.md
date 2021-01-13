<p align="center">
<img src="./docs/img/QMlogoBig.png" width="80%">
</p>

[![Build Status](https://travis-ci.org/QuakeMigrate/QuakeMigrate.svg?branch=master)](https://travis-ci.org/QuakeMigrate/QuakeMigrate)
[![codecov](https://codecov.io/gh/QuakeMigrate/QuakeMigrate/branch/master/graph/badge.svg)](https://codecov.io/gh/QuakeMigrate/QuakeMigrate)
[![Documentation Status](https://readthedocs.org/projects/quakemigrate/badge/?version=latest)](https://quakemigrate.readthedocs.io/en/latest/?badge=latest)

QuakeMigrate is a Python package for automatic earthquake detection and location using waveform migration and stacking. It can be used to produce catalogues of earthquakes, including hypocentres, origin times, phase arrival picks, and local magnitude estimates, as well as rigorous estimates of the associated uncertainties.

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
Installation instructions can be found [here](https://quakemigrate.readthedocs.io/en/latest/installation.html).

Usage
-----
We are working on tutorials covering how each individual aspect of the package works, as well as example use cases where we provide substantive reasoning for the parameter choices used. These examples include applications to cryoseismicity and volcano seismology.

This is a work in progress - [see our documentation for full details](https://quakemigrate.readthedocs.io/en/latest/tutorials.html).

To quickly get a taste for how the software works, try out the two icequake examples hosted on Binder:
* Icequakes at the Rutford Ice Stream, Antarctica [here](https://mybinder.org/v2/gh/QuakeMigrate/QuakeMigrate/3f6f9ab030109e32c1c68d267bc456bbf79d82c9?filepath=examples%2FIcequake_Rutford%2Ficequakes_rutford.ipynb)
* Icequakes at the Skeiðarárjökull outlet glacier, Iceland [here](https://mybinder.org/v2/gh/QuakeMigrate/QuakeMigrate/AGU_2020_binder?filepath=examples%2FIcequake_Iceland%2Ficequakes_iceland.ipynb)

And for a more comprehensive demonstration of the options available, see the [template scripts](examples/template_scripts).

Citation
--------
If you use this package in your work, please cite the following conference presentation:

Winder, T., Bacon, C.A., Smith, J.D., Hudson, T., Greenfield, T. and White, R.S., 2020. QuakeMigrate: a Modular, Open-Source Python Package for Automatic Earthquake Detection and Location. In AGU Fall Meeting 2020. AGU.

or, if this is not possible, please cite the following journal article:

Smith, J.D., White, R.S., Avouac, JP, and S. Bourne (2020), Probabilistic earthquake locations of induced seismicity in the Groningen region, Netherlands, Geophysical Journal International.

We hope to have a publication coming out soon:

Winder, T., Bacon, C.A., Smith, J.D., Bacon, C.A., Hudson, T.S., Drew, J., and White, R.S. QuakeMigrate: a Python Package for Automatic Earthquake Detection and Location Using Waveform Migration and Stacking. (to be submitted to Seismological Research Letters).

Contact
-------
You can contact us directly at: quakemigrate.developers@gmail.com

Any additional comments/questions can be directed to:
* **Tom Winder** - tom.winder@esc.cam.ac.uk
* **Conor Bacon** - conor.bacon@esc.cam.ac.uk

License
-------
This package is written and maintained by the QuakeMigrate developers, Copyright QuakeMigrate developers 2020-2021. It is distributed under the GPLv3 License. Please see the [LICENSE](LICENSE) file for a complete description of the rights and freedoms that this provides the user.
