.. figure:: img/QMlogoBig.png
   :figwidth: 70 %
   :width: 90%
   :align: center
   :alt: QuakeMigrate: a Python package for automatic earthquake detection and location using waveform migration and stacking.

QuakeMigrate
============

:mod:`QuakeMigrate` is a Python package for automatic earthquake detection and location using waveform migration and stacking.

QuakeMigrate uses a waveform migration and stacking algorithm to search for coherent seismic phase arrivals across a network of instruments. It produces, from raw data, a catalogue of earthquakes with locations, origin times, phase arrival picks, and local magnitude estimates, as well as well as rigorous estimates of the associated uncertainties.

The source code for the project is hosted on |github|.

This package is written by the QuakeMigrate developers, and is distributed under
the GPLv3 License, Copyright QuakeMigrate developers 2020-2022.


.. |github| raw:: html

    <a href="https://github.com/QuakeMigrate/QuakeMigrate" target="_blank">github</a>

Supported operating systems
---------------------------
QuakeMigrate was developed and tested on Ubuntu 16.04/18.04, with the intention of being "platform agnostic". As of June 2022, the package has been successfully built and run on:

- Ubuntu 16.04/18.04/20.04
- Red Hat Enterprise Linux
- Debian
- Windows 10
- macOS High Sierra 10.13, Catalina 10.15

Citation
--------
If you use this package in your work, please cite the following conference presentation:

Winder, T., Bacon, C.A., Smith, J.D., Hudson, T., Greenfield, T. and White, R.S., 2020. QuakeMigrate: a Modular, Open-Source Python Package for Automatic Earthquake Detection and Location. AGUFM, 2020. pp.S38-0013.

or, if this is not possible, please cite the following journal article:

Smith, J.D., White, R.S., Avouac, JP, and S. Bourne (2020), Probabilistic earthquake locations of induced seismicity in the Groningen region, Netherlands, Geophysical Journal International.

We hope to have a publication coming out soon:

Winder, T., Bacon, C.A., Smith, J.D., Hudson, T.S., Drew, J., and White, R.S. QuakeMigrate: a Python Package for Automatic Earthquake Detection and Location Using Waveform Migration and Stacking. (to be submitted to Seismological Research Letters).

Contact
-------
You can contact us directly at quakemigrate.developers@gmail.com

Any additional comments/questions can be directed to:

* **Tom Winder** - tom.winder@esc.cam.ac.uk
* **Conor Bacon** - conor.bacon@esc.cam.ac.uk

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
