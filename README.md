---
---
# <span style="color:blue">some *QuakeMigrate: Earthquake Onset Detection and Location text</span>
---
---

QuakeMigrate is a Python and C software package that detects and locates seismic events from raw data, using a migration and coalescence based technique. The ability to fine tune the input parameters and analyse the results using a suite of visualisation functions give it an edge over the alternatives. Simple scripts (see the provided examples) allow the user to rapidly deploy the software in any seismological setting.

---

# References
If you use QuakeMigrate, please reference: 
   (PUBLICATION COMING SOON)


---
# Getting Started
These instructions will get you a copy of QuakeMigrate up and running on your local machine.  

## Prerequisites
We recommend that you use Python environments to isolate the specific dependencies of QuakeMigrate from any other projects you may be working on (see Setting Up an Environment).

## Setting Up an Environment
If you are using conda to manage your Python environments, you can use our QuakeMigrate.yml file to create a minimally complete environment:

`conda env create -f QuakeMigrate.yml`

## Installing
There are two methods of getting a copy of QuakeMigrate:

### From GitHub
- Fork this repository (see [here](https://help.github.com/en/articles/fork-a-repo) for more details)
- Clone the repository to your local machine using `git clone <repo link> <target_dir>` (see [here](https://help.github.com/en/articles/cloning-a-repository) for more details)
- Make sure you have activated your environment (see Setting Up an Environment)
- Move into the new directory and run `python setup.py install`

### pip install
Feature coming soon...

- Make sure you have activated your environment (see Setting up an Environment)
- Run `pip install QuakeMigrate`

# Examples and Testing
To ensure your version is working as expected, we have provided a set of examples that demonstrate the various features of QuakeMigrate:

#### Icequake
This example uses data recorded at a high sampling rate on ice. The velocity model is homogeneous.

#### Iceland 
This example features data from a dense local seismic network over an area of ~ 150 x 150 km in Central Iceland. It features a 1D velocity model, and extremely high earthquake rates during the 2014 Bardarbunga-Holuhraun dike intrusion.

#### Kinabalu
This example demonstrates QuakeMigrate's ability to detect earthquakes using a limited number of seismometers recording at only 20Hz.

For each example, you will need to make sure you generate the traveltime lookup table first, as these files are quite large and not stored on GitHub.
---
# Software Manual
Coming soon...

# Built With
* [pandas](https://pandas.pydata.org/) - Easy-to-use data structures and data analysis tools.
* [ObsPy](https://github.com/obspy/obspy/wiki) - Python framework for processing seismological data
* [scikit-fmm](https://pythonhosted.org/scikit-fmm/) - Python extension module which implements the fast marching method
* [NumPy](http://www.numpy.org/) - NumPy is the fundamental package for scientific computing with Python
* [SciPy](https://www.scipy.org/) - Python-based ecosystem of open-source software for mathematics, science, and engineering

---
# Collaborating

Please contact lead authors for corresponding collaboration.

---
# Authors

* **Jonathan Smith** - *Project Leader & Developer* - *Induced Seismicity, Geomechanics and Mitigating Geohazard* [Link](https://www.esc.cam.ac.uk/directory/jonathan-smith)
* **Tom Winder** - *Project Leader & Developer* - *Volcano Seismology and Earthquake Triggering* [Link](https://www.esc.cam.ac.uk/directory/tom-winder)
* **Tom Hudson**  - *Developer* - *Glacioseismology and Ice dynamics* [Link](https://www.esc.cam.ac.uk/directory/tom-s-hudson)
* **Conor Bacon** - *Developer* - *Volcano Seismology: Seismic Anisotropy* [Link](https://www.esc.cam.ac.uk/directory/conor-bacon)
* **Tim Greenfield** - *Developer* - *Volcano Seismology* [Link](https://www.esc.cam.ac.uk/directory/tim-greenfield)

Future contributors will be added to this list.

---
# License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

