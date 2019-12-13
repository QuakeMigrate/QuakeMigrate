Installation
============
QuakeMigrate is a Python package with C extensions. These instructions will get you the latest version of the software up and running on your local machine.

Prerequisites
-------------
We recommend using Python environments to isolate the specific dependencies of QuakeMigrate from any other projects on which you may be working. You will also need a working C compiler, such as the GNU project 

Setting up an environment
-------------------------
If you are using conda to manage your Python environments, you can use our QuakeMigrate.yml file to create and activate a minimally complete environment:

.. code-block:: bash

    conda env create -f QuakeMigrate.yml
    conda activate QuakeMigrate

Installing
----------
There are a few ways to get a copy of QuakeMigrate:

From source
***********
Download the package (either by `cloning the repository <https://help.github.com/en/github/creating-cloning-and-archiving-repositories/cloning-a-repository>`_ or downloading the source code) from |github|, navigate to the QuakeMigrate directory and run (ensuring your environment is activated):

.. code-block:: bash

   python setup.py install

.. _cloning 

.. |github| raw:: html

    <a href="https://github.com/Ulvetanna/QuakeMigrate" target="_blank">github</a>

pip install
***********
We will be linking the package to PyPi soon, after which you can use the following command to install the package:

.. code-block:: bash

    pip install QMigrate

conda install
*************
We will be linking the package with the conda forge soon, which will allow the user to bypass the need to have a suitable C compiler.