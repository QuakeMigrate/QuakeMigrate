Installation
============
These instructions will get the latest version of QuakeMigrate up and running.

Prerequisites
-------------
We recommend using Python environments to isolate the specific dependencies of QuakeMigrate.

Setting up an environment
*************************
If you are using conda to manage your Python environments, you can use our QuakeMigrate.yml file to create and activate a minimally complete environment:

.. code-block:: bash

    conda env create -f QuakeMigrate.yml
    conda activate QuakeMigrate

Compiling C extensions
**********************

You will need a C compiler to build the C extensions. We recommend doing this `through conda <https://docs.conda.io/projects/conda-build/en/latest/resources/compiler-tools.html>`_, though the install will fall back on your system compiler (if available).

Linux
#####

Install the gcc compiler using:

.. code-block:: bash
    
    conda install gcc_linux-64

MacOS
#####

Coming soon.

Windows
#######

Coming soon.

Installing
----------
There are a few ways to get a copy of QuakeMigrate:

From source
***********
`Clone the repository <https://help.github.com/en/github/creating-cloning-and-archiving-repositories/cloning-a-repository>`_ from |github|, navigate to the new QuakeMigrate directory and run (ensuring your environment is activated):

.. code-block:: bash

   python setup.py install

.. _cloning 

.. |github| raw:: html

    <a href="https://github.com/QuakeMigrate/QuakeMigrate" target="_blank">GitHub</a>

pip install
***********
We will be linking the package to PyPi soon, after which you can use the following command to install the package:

.. code-block:: bash

    pip install quakemigrate

conda install
*************
We hope to link the package with the conda forge soon.

Testing your installation
-------------------------
In order to test your installation, you will need to have cloned the GitHub repository. This will ensure you have all of the required benchmarked data (which is not included in pip/conda installs). Then, navigate to `QuakeMigrate/examples/Icequake_Iceland` and run the example scripts in the following order:

.. code-block:: bash

    python iceland_lut.py
    python iceland_detect.py
    python iceland_trigger.py
    python iceland_locate.py

Once these have all run successfully, navigate to `QuakeMigrate/tests` and run:

.. code-block:: bash

    python test_benchmarks.py

This should execute with no failed tests.
