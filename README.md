# QuakeMigrate: Earthquake Onset Detection and Location Method

A python and C software package for detection and location of seismic events using a coherency and coalescence based technique. The advantages over other software packages is the generation of several visual outputs and information. The simple initiation scripts and examples help the user to define parameters for any seismological situation. The modular and open-source nature of the software package allows for the continued development from many institutions and research groups. 

## Install the software
Installing all the requirments for QuakeMigrate you can create a new conda environment using the command:
```
conda create -n qmigrate python=3.6 pandas=0.19 numpy obspy scikit-fmm
```
and then activating the environment by running
```
source activate qmigrate
```

The user then needs to clone the git repository and then installing:
```
git clone https://github.com/Ulvetanna/QuakeMigrate.git
python setup.py install
```

## Software Manual
We are currently developing a usage manual for the software. Outlined below are the main processing stages from our publication currently in prep.

QuakeMigrate uses a modular process for the detection and location of seismic events. An outline of this procedure is displayed in Figure ???. QuakeMigrate can be broadly separated into five modules: Travel Time - Determination or Loading of travel-time look-up tables for each seismic station within a predefined velocity structure; Seismic Data - Definition of the continuous seismic data structure to load; Detect - Determination of the maximum coalescence value for a reduced size model; Trigger - Determination of time periods which exceed a given threshold value; and Location - re-analysis of triggered events to refine event location. The main processing stages are the Detect and Location stages, interfacing with a series of C modules for the efficient calculation of energy coalescence over a given time and space domain. The module structure of the software and the extensive output information allows the user to provide different onset function, travel-time grids and picking methods to allow for continious development of the software package and help improve the user experience.


## Built With
* [pandas](https://pandas.pydata.org/) - Easy-to-use data structures and data analysis tools.
* [ObsPy](https://github.com/obspy/obspy/wiki) - Python framework for processing seismological data
* [scikit-fmm](https://pythonhosted.org/scikit-fmm/) - Python extension module which implements the fast marching method
* [NumPy](http://www.numpy.org/) - NumPy is the fundamental package for scientific computing with Python
* [SciPy](https://www.scipy.org/) - Python-based ecosystem of open-source software for mathematics, science, and engineering

## Collaborating

Please contact jds70@alumni.esc.cam.ac.uk or more information on collaborating on the project

## Authors

* **Jonathan Smith** - *Project Leader and Developer* - *Induced Seismicity, Geomechanics and Mitigating Geohazard* [Link](https://www.esc.cam.ac.uk/directory/jonathan-smith)
* **Tom Hudson** - *Developer* - *Glacioseismology and Icedynamics* [Link](https://www.esc.cam.ac.uk/directory/tom-s-hudson)
* **Tom Winder** - *Developer* - *Volcanoseismology and Earthquake Triggering* [Link](https://www.esc.cam.ac.uk/directory/tom-winder)
* **Conor Bacon** - *Developer* - *Volcanoseismology and Earthquake Triggering* [Link](https://www.esc.cam.ac.uk/directory/conor-bacon)
* **Tim Greenfield** - *Developer* - *Volcanoseismology and Earthquake Triggering* [Link](https://www.esc.cam.ac.uk/directory/tim-greenfield)

Future contributors will be added to this list.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.
