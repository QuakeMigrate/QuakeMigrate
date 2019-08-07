
![](./QMigrate/Title.png)

> QuakeMigrate is a Python and C software package that detects and locates seismic events from raw data, using a migration and coalescence back projection technique. The ability to fine tune the input parameters and analyse the results using a suite of visualisation functions give it an edge over the alternatives. Simple scripts (see the provided examples) allow the user to rapidly deploy the software in any seismological setting. The open source and module nature allows for the continious development with the inclusion of cutting edge techniques. 

---
<img src="./QMigrate/References.png" alt="drawing" height="50"/>

QuakeMigrate is currently in prperation for publication. Once accepted we hope that the publication will have the reference: 
  
  `Smith, J.D., Winder, T., Hudson, T.S., Bacon, C., Greenfield, T., Drew, J. and R.S. White (2019), QuakeMigrate: Earthquake Onset Detection adn Back-Migration Location Method, Seismological Research Letters.`


---
<img src="./QMigrate/Installation.png" alt="drawing" height="50"/>

Our installation process can be directly from the GitHub repository. However, we recommend that the user creates a Anaconda environment and installs the required packages using:

`conda env create -f QuakeMigrate.yml`

Once the environment is installed the user can install the software using
`git clone https://github.com/Ulvetanna`
into the required host directory. 

More information on the required version of packages can be found under the setup.py file or in the QuakeMigrate.yml file.

<img src="./QMigrate/ExamplesTesting.png" alt="drawing" height="75"/>
To ensure your version is working as expected, we have provided a set of examples that demonstrate the various features of QuakeMigrate. Outlined below are the three examples given with the software release. Additional examples will be gradulally uploaded to corresponding GitHub wiki pages.

To get started with the software we recomend that the user runs the 'Icequake' example, as this provides an insite into the entire processing procedure. 

### Antarctic Basal Icequakes
This example uses data recorded at a high sampling rate on ice. The velocity model is homogeneous.

### Bárðabunga-Holoraun Dyke Propagation, Iceland
This example features data from a dense local seismic network over an area of ~ 150 x 150 km in Central Iceland. It features a 1D velocity model, and extremely high earthquake rates during the 2014 Bardarbunga-Holuhraun dike intrusion.

### Kinabalu
This example demonstrates QuakeMigrate's ability to detect earthquakes using a limited number of seismometers recording at only 20Hz.

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


---
# Authors, Collaboration & Licensing

* **Jonathan Smith** - *Project Leader & Developer* - *Induced Seismicity, Geomechanics and Mitigating Geohazard* [Link](https://www.esc.cam.ac.uk/directory/jonathan-smith)
* **Tom Winder** - *Project Leader & Developer* - *Volcano Seismology and Earthquake Triggering* [Link](https://www.esc.cam.ac.uk/directory/tom-winder)
* **Tom Hudson**  - *Developer* - *Glacioseismology and Ice dynamics* [Link](https://www.esc.cam.ac.uk/directory/tom-s-hudson)
* **Conor Bacon** - *Developer* - *Volcano Seismology: Seismic Anisotropy* [Link](https://www.esc.cam.ac.uk/directory/conor-bacon)
* **Tim Greenfield** - *Developer* - *Volcano Seismology* [Link](https://www.esc.cam.ac.uk/directory/tim-greenfield)

Future contributors will be added to this list.

Please contact lead authors for corresponding collaboration.
   **Jonathan Smith** - jdsmith@caltech.edu
   **Tom Winder** - tebw2@cam.ac.uk

This project is licensed under the MIT License so can be used freely for academic use and non-comercial gain. Pleasesee the [LICENSE.md](LICENSE.md) file for more details.

