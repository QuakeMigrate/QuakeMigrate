Installation
============
:mod:`QuakeMigrate` is a predominantly Python package with some routines written and optimised in C. These are built and linked to QuakeMigrate at installation - if installing from source you will need to ensure that there is a suitable compiler available (see :ref:`C compilers`).

However, most users can bypass this step by installing QuakeMigrate using ``pip``.

Supported operating systems
---------------------------
QuakeMigrate was developed and tested on Ubuntu 16.04/18.04, with the intention of being "platform agnostic". As of March 2023, the package has been successfully built and run on:

- Ubuntu 16.04/18.04/20.04/22.04
- Red Hat Enterprise Linux
- Debian
- Windows 10
- macOS High Sierra 10.13, Mojave 10.14, Catalina 10.15, Big Sur 11, Monterey 12 (including M1)

Prerequisites
-------------
QuakeMigrate supports Python 3.8 or newer (3.8/3.9/3.10/3.11). We recommend using Anaconda as a package manager and environment management system to isolate and install the specific dependencies of QuakeMigrate.

Instructions for downloading and installing Anaconda can be found `here <https://docs.anaconda.com/anaconda/install/>`_. If drive space is limited, consider using Miniconda instead, which ships with a minimal collection of useful packages.

Installation via `pip`
----------------------
The simplest way to get a working copy of QuakeMigrate is to install it from the Python Package Index (PyPI) using ``pip`` (the Python package installer).

To do this you first need to set up an enivironment. We recommend creating a minimal environment initially:

.. code-block:: bash
    
    conda create --name quakemigrate python=3.9
    conda activate quakemigrate

All other dependencies will be handled during the installation of QuakeMigrate. After activating your environment, type the following command into terminal:

.. code-block:: bash
    
    pip install quakemigrate

This will install QuakeMigrate **and** its explicit dependencies!

.. note:: Installing the package this way will not provide you with the examples. These can be retrieved directly from the GitHub repository (see :ref:`testing your installation`).

The full list of dependencies is:

- matplotlib
- numpy
- obspy >= 1.2
- pandas
- pyproj >= 2.5
- scipy

.. note:: We are currently not pinning the version of any dependencies. We aim to keep on top of any new syntax changes etc. as new versions of these packages are released - but please submit an issue if you come across something!

If you want to explore the example notebooks, you will also need to install `ipython` and `jupyter`. This can be done with conda (making sure your environment is still activated) as:

.. code-block:: bash

    conda install ipython jupyter

Finally, if you wish to apply QuakeMigrate in situations where the velocity is not uniform for each phase (including for the provided ``Volcanotectonic_Iceland`` usage example), you will need to :ref:`install a traveltime solver <Installing a traveltime solver>`.

Installing a traveltime solver
------------------------------
In addition to the explicit dependencies, QuakeMigrate includes wrapper functions that use `NonLinLoc <http://alomax.free.fr/nlloc/>`_ and `scikit-fmm <https://pythonhosted.org/scikit-fmm/>`_ as backends for producing 1-D traveltime lookup tables (see :doc:`The traveltime lookup table </tutorials/lut>`).

Users can choose to install one or both of these software packages, which will enable them to use the corresponding wrapper function. (If you already have NonLinLoc installed, you may skip this step!)

NonLinLoc
*********
Obtaining binaries
++++++++++++++++++
To download, unpack, and compile NonLinLoc, **Linux** and *most* **macOS** users can use:

.. note:: In order to install NonLinLoc, you will need an accessible C compiler, such as `gcc` or `clang` (see :ref:`C compilers`).

.. warning:: The NLLoc MakeFile specifies the compiler as ``gcc``. For **macOS users** this means that the system (XCode) clang compiler will be used even if you activate the relevant environment for an alternative. To change this, edit the MakeFile to specify ``clang`` instead of ``gcc``.

.. code-block:: bash
    
    curl http://alomax.free.fr/nlloc/soft7.00/tar/NLL7.00_src.tgz -o NLL7.00_src.tgz
    tar -xzvf NLL7.00_src.tgz
    cd src
    mkdir bin; export MYBIN=./bin
    make -R all

If this is not successful, **macOS** users (at least those using systems with an Intel CPU) can instead download the binaries directly:

.. code-block:: bash

    curl http://alomax.free.fr/nlloc/soft7.00/tar/NLL7.00_bin_x86_64-apple-darwin14.tgz -o NLL7.00_bin_x86_64-apple-darwin14.tgz
    tar -xvzf NLL7.00_bin_x86_64-apple-darwin14.tgz

Alternatively, for newer versions of NonLinLoc (and instructions for installation using CMake) see the instructions on the `NonLinLoc GitHub page <https://github.com/alomax/NonLinLoc>`_.

Adding to the system path
+++++++++++++++++++++++++

Once you have successfully obtained the binary executables, we recommend you add the newly created ``bin`` directory to your system path. For Unix systems, this can be done by **adding the following** to your ``.bashrc`` file (for Linux users), or either ``.zshrc`` or ``.bash_profile`` file (for macOS - use ``echo $SHELL`` to check your default login shell, and therefore the appropriate file to use). This file is typically found in your home directory, ``~/``):

.. code-block:: bash
    
    export PATH=/path/to/nonlinloc/bin:$PATH

replacing the ``/path/to/nonlinloc`` with the path to where you downloaded or installed NonLinLoc. Save the changes to your ``.bashrc``, ``.zshrc`` or ``.bash_profile`` file, and open a new terminal window to activate the change. This will allow your shell to access the ``Vel2Grid`` and ``Grid2Time`` programs from anywhere. To test this has worked, type:

.. code-block:: bash
    
    which Grid2Time

This should return ``/path/to/nonlinloc/bin/Grid2Time``, as described above.

Alternatively, if you do not wish to add the NonLinLoc executables to your system path, you can explicitly specify the ``nlloc_path`` variable when using NonLinLoc to generate a QuakeMigrate lookup table (see :ref:`The traveltime lookup table <1-D NonLinLoc Grid2Time Eikonal solver>`).

scikit-fmm
**********
.. note:: In order to install scikit-fmm, you will need an accessible C++ compiler, such as `gxx` or `clangxx` (see :ref:`C compilers`).

scikit-fmm is a 3rd-party Python package which implements the fast-marching method. It can be installed using:

.. code-block:: bash
    
    pip install scikit-fmm

It can also be installed along with the rest of package if installing from source (see :ref:`Other installation methods`).

Other installation methods
--------------------------
From source
***********

.. note:: In order to install from source, you will need an accessible C compiler, such as `gcc` or `clang` (see :ref:`C compilers`).

`Clone the repository <https://help.github.com/en/github/creating-cloning-and-archiving-repositories/cloning-a-repository>`_ from our `GitHub <https://github.com/QuakeMigrate/quakemigrate>`_ (note: you will need ``git`` installed on your system), or alternatively download the source code directly through the GitHub web interface. Once you have a local copy, navigate to the new ``QuakeMigrate`` directory.

You can build a complete environment using the ``environment.yml`` file which can be found in the top level of the cloned repository.

.. code-block:: bash

    conda env create -f environment.yml
    conda activate quakemigrate

Finally, you can install the package (making sure your environment is activated) by running:

.. code-block:: bash
    
    pip install .

You can optionally pass a ``-e`` argument to install the package in 'editable' mode.

If you wish to use :mod:`scikit-fmm`, you can install it here as an optional package using:

.. code-block:: bash
    
    pip install .[fmm]
    # or for zsh users:
    pip install .\[fmm]

You should now be able to import :mod:`quakemigrate` within a Python session:

.. warning:: You should try this import in any directory that is *not* the root of the git repository (i.e. ``QuakeMigrate/``. Here, the local ``quakemigrate`` directory will override the version of QuakeMigrate installed in your environment site-packages!

.. code-block:: bash
    
    cd  # Moving out of QuakeMigrate directory - see warning above!
    python
    >>> import quakemigrate
    >>> quakemigrate.__version__

If successful, this should return '|Version|'.

.. note:: If you wish to use NonLinLoc as a traveltime solver, you will need to install that as detailed :ref:`above <NonLinLoc>`.

conda install
*************
We hope to link the package with the conda forge soon, after which you will be able to use the following command to install the package:

.. code-block:: bash
    
    conda install -c conda-forge quakemigrate

Testing your installation
-------------------------
In order to test your installation, you will need to have cloned the GitHub repository (see :ref:`installation from source <From Source>`). This will ensure you have all of the required benchmarked data (which is not included in pip/conda installs). It is also recommended that you :ref:`install NonLinLoc <NonLinLoc>`, which is required for the ``Volcanotectonic_Iceland`` example.

To run the tests, navigate to ``QuakeMigrate/tests`` and run the test scripts. First, test all packages have correctly installed and you can import QuakeMigrate:

.. code-block:: bash

    python test_import.py

This may output some warning messages about deprecations - so long as the final output line says "OK" and not "FAILED", these aren't an issue.

.. note:: Check if there is a message about matplotlib backends - there ought to be a suitable backend (e.g. macOSX, Qt, or Tk), but there is a chance you might not have any. If this warning is present, see :ref:`matplotlib backends`.
    
Next, run the examples.

.. note:: This requires NonLinLoc to be installed. If you have not installed (or can not install) NonLinLoc, you may edit the ``run_test_examples.py`` script to only run the ``Icequake_Iceland`` example by commenting out the relevant section.

.. code-block:: bash

    python run_test_examples.py

This script collates and runs the scripts for each stage in the ``Icequake_Iceland`` and ``Volcanotectonic_Iceland`` examples. This process will take a number of minutes. Once this has completed successfully, run:

.. code-block:: bash
    
    python test_benchmarks.py

.. note:: If you edited the ``run_test_examples.py`` script to only run the ``Icequake_Iceland`` example, you will also need to edit the ``test_benchmarks.py`` script to reflect this, otherwise the test will report as failed!

If your installation is working as intended, this should execute with no failures.


C compilers
-----------
In order to install and use QuakeMigrate and/or NonLinLoc & scikit-fmm from source, you will need a C compiler.

If you already have a suitable compiler (e.g. `gcc`, `MSVC`, `clang`) at the OS level, then you can proceed with installation. If this fails, then read on for tips to overcome common issues!

Checking for a C compiler
*************************
On Linux or macOS, to check if you have a C compiler, open a terminal and type:

.. code-block:: bash
    
    which gcc
    gcc --version

If a compiler is present, the first command will return ``/usr/bin/gcc``. However, this does not guarantee it is present! The second command will confirm this.

On **Linux** the second command should output something like:

.. code-block:: console

    gcc (Ubuntu 11.3.0-1ubuntu1~22.04) 11.3.0
    Copyright (C) 2021 Free Software Foundation, Inc.
    This is free software; see the source for copying conditions. There is NO
    warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

As long as the version is relatively recent (version 9 or later), you should be good to go. To additionally confirm that you have a C++ compiler installed, type:

.. code-block:: bash
    
    which g++
    g++ --version

For which you should obtain a similar result.

On **macOS** it will be obvious if the compiler is not actually installed -- you will be faced with a prompt to install the Xcode Command Line Tools. You can go ahead and install this (press ``Install`` and wait for the process to complete). If these are already installed, the second command should output something like:

.. code-block:: console

    Configured with: --prefix=/Library/Developer/CommandLineTools/usr --with-gxx-include-dir=/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/include/c++/4.2.1
    Apple clang version 12.0.5 (clang-1205.0.22.11)
    Target: x86_64-apple-darwin20.5.0
    Thread model: posix
    InstalledDir: /Library/Developer/CommandLineTools/usr/bin

.. warning:: Even if `clang` is installed, it may not have all necessary libraries included. See :ref:`OpenMP on macOS`.

Note that this indicates that the system compiler is ``clang``, and that the accompanying C++ compiler is also installed. These are all supplied as part of the Xcode Command Line Tools (see e.g. `here <https://mac.install.guide/commandlinetools/index.html>`_ for a rundown).

If you do not have a compiler, or to be sure, we provide a simple guide for :ref:`Linux`, :ref:`macOS` and :ref:`Windows` operating systems below.

.. note:: In order to build the (optional) dependency scikit-fmm you will need a C++ compiler (e.g. `gxx`, `MSVC`, `clangxx`). This can also be done either at the OS level, or using conda (see guidance on the conda compiler tools page, linked below).

Linux
*****
If you have root access, the simplest route is to install `gcc` and `gxx` at system-level. You should search for the correct way to do this for your Linux Distribution. For example, on Ubuntu you would type:

.. code-block:: console

    sudo apt-get install build-essential

This includes `gcc`, `g++` as well as `make`. The commands will differ on other distros (CentOS, Red Hat, etc.).

Alternatively, you can install `gcc` and `g++` `through conda <https://docs.conda.io/projects/conda-build/en/latest/resources/compiler-tools.html>`_. Make sure you have activated your environment, then type:

.. code-block:: bash
    
    conda install gcc_linux-64 gxx_linux-64

You can test this was successful with the same procedure detailed :ref:`above<Checking for a C compiler>`. Once installed, you can proceed with the QuakeMigrate :ref:`installation from source <Other installation methods>`.

macOS
*****
By default, there is no C compiler included with macOS. If you have previously installed the Xcode Command Line Tools (via the web or the App Store), the `clang` compiler will be installed. However, this may not include all necessary libraries to install QuakeMigrate (see :ref:`OpenMP on macOS`).

Whether you already have Xcode installed or not, there are two options to install what is required: the user can either install all dependencies :ref:`through conda <conda>` - noting that they will only be available in that specific environment - or using `HomeBrew <https://brew.sh/>`_. We generally recommend using conda, unless the user is already familiar with brew (in which case, see :ref:`brew`).

OpenMP on macOS
+++++++++++++++
The default C compiler on macOS does not include support for OpenMP. This will result in the following error during installation from source:

.. code-block:: console

    ld: library not found for -lomp
    clang: error: linker command failed with exit code 1 (use -v to see invocation)
    error: command '/usr/bin/clang' failed with exit code 1

As above, this can either be solved with :ref:`conda` or :ref:`brew`.

conda
+++++
First create and/or activate your environment:

.. code-block:: bash

    conda create -n quakemigrate python=3.9  # if not already created
    conda activate quakemigrate  # replace with alternative environment name if desired

Then use conda to install the compiler (along with the OpenMP libraries). **Note the syntax is different if your machine is running on an Apple Silicon (M1, M2, etc.) chip**:

.. code-block:: bash

    conda install openmp clang_osx-64 clangxx_osx-64  # Intel chip
    conda install openmp clang_osx-arm64 clangxx_osx-arm64  # Apple Silicon chip (M1, M2 etc.)

.. note:: If you did not already have Xcode Command Line Tools installed, you will be prompted to install them now. Click ``Install`` and wait for installation to complete.

You should now open a fresh terminal, and activate your environment. To test the installation was successful, type:

.. code-block:: bash

    echo $CC
    $CC --version

This should return something like:

.. code-block:: console

    echo $CC
    x86_64-apple-darwin13.4.0-clang
    $CC --version
    clang version 14.0.6
    Target: x86_64-apple-darwin13.4.0
    Thread model: posix
    InstalledDir: /Users/user/miniconda3/envs/quakemigrate/bin

You can proceed with the QuakeMigrate :ref:`installation from source <Other installation methods>`.

brew
++++
If brew is not already installed (check with ``which brew``), follow the instructions on the `HomeBrew frontpage <https://brew.sh/>`_. This will offer to install the Xcode Command Line Tools if they are not already present (press 'ENTER' or 'Y' to accept this suggestion).

You can then proceed to install the OpenMP libraries with brew:

.. code-block:: bash
    
    brew install libomp

You can safely ignore the warning about explicitly adding the relevant LDFLAGS etc. - this is already handled in the QuakeMigrate ``setup.py`` script.

You can proceed with the QuakeMigrate :ref:`installation from source <Other installation methods>`.

*Legacy*: brew gcc
++++++++++++++++++
Alternatively, you can use the `gcc` compiler to install QuakeMigrate (and NonLinLoc). As with `clang`, we recommend installing GCC through ``Homebrew``. First, check if you already have `gcc` installed, with:

.. code-block:: bash

    which gcc

If this doesn't return anything, continue to installing `gcc`. If this returns the path to a gcc executable (e.g. `/usr/bin/gcc`), then you should check the version, with:

.. code-block:: bash

    gcc --version

If the version string includes `Apple clang`, or is a version number lower than 9, you should proceed to install with ``Homebrew``:

.. code-block:: bash
    
    brew install gcc
    brew link gcc

Note that the ``brew link`` command should add ``gcc`` to your path, but might not succeed if a previous ``gcc`` install was present. To test this, type:

.. code-block:: bash

    which gcc
    gcc --version

If the linking was successful, this should point to a new gcc executable, and the version string should contain ``gcc (Homebrew GCC 9.4.0) 9.4.0`` or similar. If not, you will need to manually link the new ``gcc`` executable. To do this, find the path to your new ``gcc``` installation with:

.. code-block:: bash

    brew --prefix gcc

Then create a symlink to this executable:

.. code-block:: bash

    ln -s /usr/local/bin/gcc /path/to/brew/gcc

Where ``/path/to/brew/gcc`` is the path returned by the ``brew --prefix`` command.

Finally, test this has worked by repeating the check from above:

.. code-block:: bash

    which gcc
    gcc --version

This should now return the ``Homebrew`` ``gcc`` version string. If not, please get in touch and we will try to help if we can...


Windows
*******
Compilation and linking of the C extensions has been successful using the Microsoft Visual C++ (MSVC) build tools. 

We strongly recommend that you download and install these tools in order to use QuakeMigrate. You can either install Visual Studio in its entirety, or just the Build Tools - `available here <https://visualstudio.microsoft.com/downloads/>`_.

You will need to restart your computer once the installation process has completed. We recommend using the anaconda command line interface (unix shell-like) to install QuakeMigrate over command prompt.

.. warning:: QuakeMigrate has been tested and validated on Windows, but there may yet remain some unknown issues. If you encounter an issue (and/or resolve it), please submit a GitHub issue (or send an email) to let us know!

Once installed, you can proceed with the QuakeMigrate :ref:`installation from source <Other installation methods>`.

Notes
-----

PROJ
****
There is a known issue with PROJ version 6.2.0 which causes vertical coordinates to be incorrectly transformed when using units other than metres (the PROJ default). If you encounter this issue (you will get an ``ImportError`` when trying to use the ``lut`` subpackage), you should update :mod:`pyproj`. Using conda will install an up-to-date PROJ backend, but you may need to clear your cache of downloaded packages. This can be done using:

.. code-block:: bash
    
    conda clean --all

Then reinstall :mod:`pyproj`:

.. code-block:: bash
    
    conda uninstall pyproj
    conda install pyproj

matplotlib backends
*******************
If you receive the warning about only the ``'Agg'`` backend being available, you should first verify this manually. Open a Python session, and type the following commands to attempt to open an interactive plotting window:

.. code-block:: bash

    python
    >>> import matplotlib.pyplot as plt
    >>> plt.plot([1, 2], [1, 2])
    >>> plt.show()

If an interactive plot window appears, then this was a false alarm, and you can proceed. Else, double-verify with:

.. code-block:: bash

    >>> import matplotlib
    >>> matplotlib.get_backend()

If this returns ``'Agg'``, then you definitely need to install a backend capable of drawing interactive plots. You can do this with conda (making sure your environment is activated):

.. code-block:: bash

    conda intall pyqt

Then re-do the steps above to verify that this was successful.
