.. figure:: img/QMlogoBig.png
   :figwidth: 70 %
   :width: 90%
   :align: center
   :alt: QuakeMigrate: a Python package for automatic earthquake detection and location using waveform migration and stacking.

QuakeMigrate
============

:mod:`QuakeMigrate` is a Python package for automatic earthquake detection and location using waveform migration and stacking.

QuakeMigrate uses a waveform migration and stacking algorithm to search for coherent seismic phase arrivals across a network of instruments. It produces, from raw data, a catalogue of earthquakes with locations, origin times, phase arrival picks, and local magnitude estimates, as well as well as rigorous estimates of the associated uncertainties.

The package has been built with a modular architecture, providing the potential for extension and adaptation at numerous entry points. This includes, but is not limited to:

* the calculation or import of traveltime grids
* the choice of algorithm used to identify phase arrivals (for example by kurtosis, cross-covariance analysis between multiple components, machine learning techniques and more)
* the stacking function used to combine onset functions
* the algorithm used to perform phase picking

The source code for the project is hosted on |github|.

This package is written by the QuakeMigrate developers, and is distributed under
the GPLv3 License, Copyright QuakeMigrate developers 2023.


.. |github| raw:: html

    <a href="https://github.com/QuakeMigrate/QuakeMigrate" target="_blank">GitHub</a>

Citation
--------
If you use this package in your work, please cite the following conference presentation:

Winder, T., Bacon, C.A., Smith, J.D., Hudson, T., Greenfield, T. and White, R.S., 2020. QuakeMigrate: a Modular, Open-Source Python Package for Automatic Earthquake Detection and Location. AGUFM, 2020. pp.S38-0013.

as well as the relevant version of the source code on `Zenodo <https://doi.org/10.5281/zenodo.4442749>`_.

We hope to have a publication coming out soon:

Winder, T., Bacon, C.A., Smith, J.D., Hudson, T.S., Drew, J., and White, R.S. QuakeMigrate: a Python Package for Automatic Earthquake Detection and Location Using Waveform Migration and Stacking. (to be submitted to *Seismica*).

Contact
-------
You can contact us directly at quakemigrate.developers@gmail.com

Any additional comments/questions can be directed to:

* **Tom Winder** - tom.winder@esc.cam.ac.uk
* **Conor Bacon** - conor.bacon@cantab.net

License
-------
This package is written and maintained by the QuakeMigrate developers, Copyright QuakeMigrate developers 2020-2021. It is distributed under the GPLv3 License. Please see the `LICENSE <https://www.gnu.org/licenses/gpl-3.0.html>`_ for a complete description of the rights and freedoms that this provides the user.

Contents:
---------

.. toctree::
   :numbered:
   :maxdepth: 1

   installation
   tutorials
   sourcecode
