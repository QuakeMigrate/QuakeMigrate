# -*- coding: utf-8 -*-
try:
    # Use setuptools if we can
    from setuptools import setup, Extension
    from setuptools.command.build_ext import build_ext
    using_setuptools = True
except ImportError:
    from distutils.core import setup, Extension
    from distutils.command.build_ext import build_ext
    using_setuptools = False

import os
import pathlib
from pkg_resources import get_build_platform
import re
import shutil
import sys


# Check if we are on RTD and don't build extensions if we are.
READ_THE_DOCS = os.environ.get("READTHEDOCS", None) == "True"

if READ_THE_DOCS:
    try:
        environ = os.environb
    except AttributeError:
        environ = os.environ

    environ[b"CC"] = b"x86_64-linux-gnu-gcc"
    environ[b"LD"] = b"x86_64-linux-gnu-ld"
    environ[b"AR"] = b"x86_64-linux-gnu-ar"

# Directory of the current file
SETUP_DIRECTORY = pathlib.Path.cwd()


long_description = """A Python package for the detection and location of
                      seismicity, based on waveform backprojection and
                      stacking"""


def read(*parts):
    """
    Build an absolute path from *parts* and and return the contents of the
    resulting file.
    """
    p = SETUP_DIRECTORY
    for part in parts:
        p /= part
    with p.open("r") as f:
        return f.read()


META_FILE = read("quakemigrate", "__init__.py")


def find_meta(meta):
    """
    Extract __*meta*__ from META_FILE.
    """
    meta_match = re.search(
        r"^__{meta}__ = ['\"]([^'\"]*)['\"]".format(meta=meta),
        META_FILE, re.M
    )
    if meta_match:
        return meta_match.group(1)
    raise RuntimeError("Unable to find __{meta}__ string.".format(meta=meta))


def get_package_data():
    package_data = {}
    if not READ_THE_DOCS:
        if get_build_platform() in ("win32", "win-amd64"):
            package_data["quakemigrate.core"] = ["quakemigrate/core/src/*.dll"]

    return package_data


def get_package_dir():
    package_dir = {}
    if get_build_platform() in ("win32", "win-amd64"):
        package_dir["quakemigrate.core"] = str(
            pathlib.Path("quakemigrate") / "core")

    return package_dir


def get_extras_require():
    if READ_THE_DOCS:
        return {"docs": ['Sphinx >= 1.8.1', 'docutils']}
    else:
        return {}


def get_include_dirs():
    import numpy

    include_dirs = [str(pathlib.Path.cwd() / "quakemigrate" / "core" / "src"),
                    numpy.get_include(),
                    str(pathlib.Path(sys.prefix) / "include")]

    if get_build_platform().startswith("freebsd"):
        include_dirs.append("/usr/local/include")

    return include_dirs


def get_library_dirs():
    library_dirs = []
    if get_build_platform() in ("win32", "win-amd64"):
        library_dirs.append(str(pathlib.Path.cwd() / "quakemigrate" / "core"))
        library_dirs.append(str(pathlib.Path(sys.prefix) / "bin"))

    library_dirs.append(str(pathlib.Path(sys.prefix) / "lib"))
    if get_build_platform().startswith("freebsd"):
        library_dirs.append("/usr/local/lib")

    return library_dirs


def export_symbols(*parts):
    """
    Required for Windows systems - functions defined in qmlib.def.
    """
    p = SETUP_DIRECTORY
    for part in parts:
        p /= part
    with p.open("r") as f:
        lines = f.readlines()[2:]
    return [s.strip() for s in lines if s.strip() != ""]


def get_extensions():
    if READ_THE_DOCS:
        return []

    common_extension_args = {
        "include_dirs": get_include_dirs(),
        "library_dirs": get_library_dirs()}

    sources = [str(pathlib.Path("quakemigrate")
               / "core" / "src" / "quakemigrate.c")]
    exp_symbols = export_symbols("quakemigrate/core/src/qmlib.def")

    if get_build_platform() not in ("win32", "win-amd64"):
        if get_build_platform().startswith("freebsd"):
            # Clang uses libomp not libgomp
            extra_link_args = ["-lm", "-lomp"]
        else:
            extra_link_args = ["-lm", "-lgomp"]
        extra_compile_args = ["-fopenmp", "-fPIC", "-Ofast"]
    else:
        extra_link_args = []
        extra_compile_args = ["/openmp", "/TP", "/PIC", "/Ofast"]

    common_extension_args["extra_link_args"] = extra_link_args
    common_extension_args["extra_compile_args"] = extra_compile_args
    common_extension_args["export_symbols"] = exp_symbols

    ext_modules = [Extension("quakemigrate.core.src.qmlib", sources=sources,
                   **common_extension_args)]

    return ext_modules


def setup_package():
    """Setup package"""

    if not READ_THE_DOCS:
        install_requires = ["matplotlib", "numpy", "obspy>=1.2", "pandas>=1",
                            "pyproj>=2.5", "scikit-fmm==2019.1.30", "scipy"]
    else:
        install_requires = ["matplotlib", "mock", "numpy", "obspy>=1.2",
                            "pandas>=1", "pyproj>=2.5",
                            "scikit-fmm==2019.1.30", "scipy"]

    setup_args = {
        "name": "quakemigrate",
        "version": find_meta("version"),
        "description": find_meta("description"),
        "long_description": long_description,
        "url": "https://github.com/QuakeMigrate/QuakeMigrate",
        "author": find_meta("author"),
        "author_email": find_meta("email"),
        "license": find_meta("license"),
        "classifiers": [
            "Development Status :: Beta",
            "Intended Audience :: Science/Research",
            "Topic :: Scientific/Engineering",
            "Natural Language :: English",
            "License :: OSI Approved :: MIT License",
            "Operating System :: OS Independent",
            "Programming Language :: Python :: 3.6",
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3.8",
        ],
        "keywords": "seismic waveform event detection location",
        "install_requires": install_requires,
        "extras_require": get_extras_require(),
        "zip_safe": False,
        "packages": ["quakemigrate",
                     "quakemigrate.core",
                     "quakemigrate.export",
                     "quakemigrate.io",
                     "quakemigrate.lut",
                     "quakemigrate.plot",
                     "quakemigrate.signal",
                     "quakemigrate.signal.local_mag",
                     "quakemigrate.signal.onsets",
                     "quakemigrate.signal.picker"],
        "ext_modules": get_extensions(),
        "package_data": get_package_data(),
        "package_dir": get_package_dir()
                  }

    shutil.rmtree(str(SETUP_DIRECTORY / "build"), ignore_errors=True)

    setup(**setup_args)


if __name__ == "__main__":
    # clean --all does not remove extensions automatically
    if "clean" in sys.argv and "--all" in sys.argv:
        # Delete complete build directory
        path = SETUP_DIRECTORY / "build"
        shutil.rmtree(str(path), ignore_errors=True)

        # Delete all shared libs from clib directory
        path = SETUP_DIRECTORY / "quakemigrate" / "core" / "src"
        for filename in path.glob("*.pyd"):
            filename.unlink(missing_ok=True)
        for filename in path.glob("*.so"):
            filename.unlink(missing_ok=True)
        for filename in path.glob("*.dll"):
            filename.unlink(missing_ok=True)
    else:
        setup_package()
