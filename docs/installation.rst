Installation
============
:mod:`QuakeMigrate` is a predominantly Python package with some routines written and optimised in C. These are built and linked to QuakeMigrate at installation, which means you will need to ensure that there is a suitable compiler available (see :ref:`C compilers`).

Supported operating systems
---------------------------
QuakeMigrate was developed and tested on Ubuntu 16.04/18.04, with the intention of being "platform agnostic". As of January 2021, the package has been successfully built and run on:

- Ubuntu 16.04/18.04/20.04
- Red Hat Enterprise Linux
- Debian
- Windows 10
- macOS High Sierra 10.13, Catalina 10.15

Prerequisites
-------------
QuakeMigrate supports Python 3.6 or newer (3.7/3.8). We recommend using Anaconda as a package manager and environment management system to isolate and install the specific dependencies of QuakeMigrate. Instructions for downloading and installing Anaconda can be found `here <https://docs.anaconda.com/anaconda/install/>`_. If drive space is limited, consider using Miniconda instead, which ships with a minimal collection of useful packages.

Setting up an environment
*************************
Using conda, you can use our ``quakemigrate.yml`` file to create and activate a minimally complete environment:

.. code-block:: bash
    
    conda env create -f quakemigrate.yml
    conda activate quakemigrate

This will install the explicit dependencies of QuakeMigrate (as well as some additional sub-dependencies/useful packages). The full list of dependencies (and versions, where relevant) is:

- matplotlib < 3.3
- numpy
- obspy >= 1.2
- pandas >= 1 and < 1.1
- pyproj >= 2.6
- scipy

.. note:: These version pins are subject to change. We defer to ObsPy to select suitable versions for NumPy/SciPy.

.. warning:: Some changes to datetime handling were introduced in matplotlib 3.3, which caused some conflicts with pandas versions > 1.0.5. A patch was applied, but for the time being we have pinned these two packages until we find time to fully resolve the issues arising from these changes.

In addition, we use `NonLinLoc <http://alomax.free.fr/nlloc/>`_ and `scikit fmm <https://pythonhosted.org/scikit-fmm/>`_ as backends for producing 1-D traveltime lookup tables.

NonLinLoc
#########
To download, unpack, and compile NonLinLoc, you can use:

.. code-block:: bash
    
    curl http://alomax.free.fr/nlloc/soft7.00/tar/NLL7.00_src.tgz -o NLL7.00_src.tgz
    tar -xzvf NLL7.00_src.tgz
    cd src
    mkdir bin; export MYBIN=./bin
    make -R all

.. note:: In order to install NonLinLoc, you will need an accessible C compiler, such as gcc (see :ref:`C compilers`). Our testing has suggested that there may be issues using gcc-10 (the latest version) to compile NonLinLoc, at least on macOS, so we recommend installing gcc-9.

Once the source code has been compiled, we recommend you add the bin to your system path. For Unix systems, this can be done by adding the following to your `.bashrc` file - or `.bash_profile` file for macOS - (typically found in your home directory, ``~/``):

.. code-block:: bash
    
    export PATH=/path/to/nonlinloc/bin:$PATH

replacing the ``/path/to/nonlinloc`` with the path to where you downloaded/installed NonLinLoc. Save your `.bashrc` / `.bash_profile` and open a new terminal window to activate the change. This will allow your shell to access the ``Vel2Grid`` and ``Grid2Time`` programs anywhere.

scikit-fmm
##########
scikit-fmm is a 3rd-party package which implements the fast-marching method. We specify the version ``2019.1.30`` as previous versions did not catch a potential numerical instability which may lead to unphysical traveltimes. It can be installed using:

.. code-block:: bash
    
    pip install scikit-fmm==2019.1.30

It can also be installed along with the rest of package (see :ref:`Installing`).

.. note:: In order to install scikit-fmm, you will need an accessible C++ compiler, such as gxx (see :ref:`C compilers`).

C compilers
***********
In order to install and use QuakeMigrate, you will need a C compiler that will build the migration extension library.

If you already have a suitable compiler (e.g. gcc, MSVC) at the OS level, then you can proceed to the Installing section.

If you do not, or to be sure, we recommend installing a compiler using conda. Instructions for doing this on :ref:`Linux` and :ref:`macOS` operating systems are given below.

.. note:: In order to build the (optional) dependency scikit-fmm you will need a C++ compiler (e.g. gxx, MSVC). This can also be done either at the OS level, or using conda (see guidance on the conda compiler tools page, linked below).

Linux
#####
We recommend installing the GNU compiler collection (GCC, which previously stood for the GNU C Compiler) `through conda <https://docs.conda.io/projects/conda-build/en/latest/resources/compiler-tools.html>`_.

.. code-block:: bash
    
    conda install gcc_linux-64

It is generally useful to install compilers at the OS level, including a C++ compiler (e.g. gxx), which is required to build the scikit-fmm package.

Once installed, you can proceed with the QuakeMigrate :ref:`installation <Installing>`.

macOS
#####
As with Linux, we recommend installing GCC through conda.

.. code-block:: bash
    
    conda install gcc

.. note:: We have not yet tested compiling and/or running QuakeMigrate against the Clang compiler.

Alternatively, installation of compilers at an OS level can be done using ``Homebrew``, `a package manager for macOS <https://brew.sh/>`_. It is then as simple as:

.. code-block:: bash
    
    brew install gcc

.. note:: To install gcc-9, replace ``gcc`` with ``gcc@9``

Once installed, you can proceed with the QuakeMigrate :ref:`installation <Installing>`.

Windows
#######
Compilation and linking of the C extensions has been successful using the Microsoft Visual C++ (MSVC) build tools. We strongly recommend that you download and install these tools in order to use QuakeMigrate. You can either install Visual Studio in its entirety, or just the Build Tools - `available here <https://visualstudio.microsoft.com/downloads/>`_. You will need to restart your computer once the installation process has completed. We recommend using the anaconda command line interface (unix shell-like) to install QuakeMigrate over command prompt.

.. warning:: QuakeMigrate has been tested and validated on Windows, but there may yet remain some unknown issues. If you encounter an issue (and/or resolve it), please let us know!

Once installed, you can proceed with the QuakeMigrate :ref:`installation <Installing>`.

Installing
----------
There are several ways to get a copy of QuakeMigrate:

From source
***********
`Clone the repository <https://help.github.com/en/github/creating-cloning-and-archiving-repositories/cloning-a-repository>`_ from our `GitHub <https://github.com/QuakeMigrate/quakemigrate>`_ (note: you will need ``git`` installed on your system), or alternatively download the source code directly through the GitHub web interface. Once you have a local copy, navigate to the new QuakeMigrate directory and run (ensuring your environment is activated):

.. code-block:: bash
    
    pip install .

You can optionally pass a ``-e`` argument to install the package in 'editable' mode.

If you wish to use :mod:`scikit-fmm`, you can install it here as an optional package using:

.. code-block:: bash
    
    pip install .[fmm]

You should now be able to import quakemigrate within a Python session:

.. code-block:: bash
    
    python
    >>> import quakemigrate

pip install
***********
We will be linking the package to PyPI (the Python Package Index) soon, after which you will be able to use the following command to install the package:

.. code-block:: bash
    
    pip install quakemigrate

conda install
*************
We hope to link the package with the conda forge soon, after which you will be able to use the following command to install the package:

.. code-block:: bash
    
    conda install -c conda-forge quakemigrate

Testing your installation
-------------------------
In order to test your installation, you will need to have cloned the GitHub repository. This will ensure you have all of the required benchmarked data (which is not included in pip/conda installs).

Iceland icequake test
*********************
Navigate to ``QuakeMigrate/examples/Icequake_Iceland`` and run the example scripts in the following order:

.. code-block:: bash
    
    python iceland_lut.py
    python iceland_detect.py
    python iceland_trigger.py
    python iceland_locate.py

Once these have all run successfully, navigate to ``QuakeMigrate/tests`` and run:

.. code-block:: bash
    
    python test_benchmarks.py

This should execute with no failed tests.

Iceland dike intrusion test
***************************
If you have installed the optional ``NonLinLoc`` dependency, you can also run the Iceland dike intrusion example. Navigate to ``QuakeMigrate/examples/Volcanotectonic_Iceland`` and run the example scripts in the following order:

.. code-block:: bash

    python get_dike_intrusion_data.py
    python dike_intrusion_lut.py
    python dike_intrusion_detect.py
    python dike_intrusion_trigger.py
    python dike_intrusion_locate.py

Once these have all run successfully (note this test locates more than 20 events, so takes longer to execute than the icequake example!) run the ``test_benchmarks.py`` script as described :ref:`above <Iceland icequake test>`. This should again execute with no failed tests.

.. note:: We hope to work this into a more complete suite of tests that can be run in a more automated sense - coming soon!

Notes
-----
There is a known issue with PROJ version 6.2.0 which causes vertical coordinates to be incorrectly transformed when using units other than metres (the PROJ default). If you encounter this issue (you will get an ``ImportError`` when trying to use the ``lut`` subpackage), you should update :mod:`pyproj`. Using conda will install an up-to-date PROJ backend, but you may need to clear your cache of downloaded packages. This can be done using:

.. code-block:: bash
    
    conda clean --all

Then reinstall :mod:`pyproj`:

.. code-block:: bash
    
    conda uninstall pyproj
    conda install pyproj
