QuakeMigrate
============

QuakeMigrate is a Python package for the detection and location of earthquakes.

QuakeMigrate uses an efficient, parallelised waveform stacking algorithm, implemented in C, to search for coherent seismic phase arrivals across a network of instruments. It produces, from raw data, a catalogue of earthquakes with locations, origin times and phase arrival picks, as well as estimates of the uncertainties associated with these measurements.

The package has been designed to be simple to use, even for those with no prior experience with Python. For more details, including installation, tutorials and documentation, please visit readthedocs.io/qmigrate.

---

Citing
======




QuakeMigrate is currently in preparation for publication. However, an outline to the technique is outline 

  `Smith, J.D., White, R.S., Avouac, JP, and S. Bourne (2020), Probabilistic earthquake locations of induced seismicity in the Groningen region, Netherlands, Geophysical Journal International.`
  
We hope that the publication will be submitted shortly at: 

  `Smith, J.D., Winder, T., Hudson, T.S., Bacon, C., Greenfield, T., Drew, J. and R.S. White, QuakeMigrate: a Python package for earthquake detection and location using waveform stacking. Seismological Research Letters.`

---

Requirements
============

* [pandas](https://pandas.pydata.org/) - Easy-to-use data structures and data analysis tools.
* [ObsPy](https://github.com/obspy/obspy/wiki) - Python framework for processing seismological data
* [scikit-fmm](https://pythonhosted.org/scikit-fmm/) - Python extension module which implements the fast marching method
* [NumPy](http://www.numpy.org/) - NumPy is the fundamental package for scientific computing with Python
* [SciPy](https://www.scipy.org/) - Python-based ecosystem of open-source software for mathematics, science, and engineering

---

Collaboration
=============

* **Jonathan Smith** - *Project Leader & Developer* - *Induced Seismicity, Geomechanics and Mitigating Geohazard* [Link](https://www.gps.caltech.edu/people/jonathan-d-smith)
* **Tom Winder** - *Project Leader & Developer* - *Volcano Seismology and Earthquake Triggering* [Link](https://www.esc.cam.ac.uk/directory/tom-winder)
* **Tom Hudson**  - *Developer* - *Glacioseismology and Ice dynamics* [Link](https://www.esc.cam.ac.uk/directory/tom-s-hudson)
* **Conor Bacon** - *Developer* - *Volcano Seismology and Seismic Anisotropy* [Link](https://www.esc.cam.ac.uk/directory/conor-bacon)
* **Tim Greenfield** - *Developer* - *Volcano Seismology* [Link](https://www.esc.cam.ac.uk/directory/tim-greenfield)

---

This project is licensed under the MIT License so can be used freely for academic use and non-comercial gain. Pleasesee the [LICENSE.md](LICENSE.md) file for more details.
