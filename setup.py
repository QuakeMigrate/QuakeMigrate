# -*- coding: utf-8 -*-
"""
QuakeMigrate: A Python package for automatic earthquake detection and location using
waveform migration and stacking.

:copyright:
    2020–2023, QuakeMigrate developers.
:license:
    GNU General Public License, Version 3
    (https://www.gnu.org/licenses/gpl-3.0.html)

"""

import os
import pathlib
import platform
import shutil
import sys

from distutils.ccompiler import get_default_compiler
from setuptools import Extension, setup


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

# Check for MSVC (Windows)
if platform.system() == "Windows" and (
    "msvc" in sys.argv or "-c" not in sys.argv and get_default_compiler() == "msvc"
):
    IS_MSVC = True
else:
    IS_MSVC = False

# Monkey patch for MS Visual Studio
if IS_MSVC:
    # Remove 'init' entry in exported symbols
    def _get_export_symbols(self, ext):
        return ext.export_symbols

    from setuptools.command.build_ext import build_ext

    build_ext.get_export_symbols = _get_export_symbols


def export_symbols(path):
    """
    Required for Windows systems - functions defined in qmlib.def.
    """
    with (SETUP_DIRECTORY / path).open("r") as f:
        lines = f.readlines()[2:]
    return [s.strip() for s in lines if s.strip() != ""]


def get_extensions():
    """
    Config function used to compile C code into a Python extension.
    """
    import numpy

    extensions = []

    if READ_THE_DOCS:
        return []

    extension_args = {
        "include_dirs": [
            str(pathlib.Path.cwd() / "quakemigrate" / "core" / "src"),
            str(pathlib.Path(sys.prefix) / "include"),
            numpy.get_include(),
        ],
        "library_dirs": [str(pathlib.Path(sys.prefix) / "lib")],
    }
    if platform.system() == "Darwin":
        extension_args["include_dirs"].extend(
            ["/usr/local/include", "/usr/local/opt/llvm/include"]
        )
        extension_args["library_dirs"].extend(
            ["/usr/local/lib", "/usr/local/opt/llvm/lib", "/usr/local/opt/libomp/lib"]
        )

    sources = [str(pathlib.Path("quakemigrate") / "core/src/quakemigrate.c")]

    extra_link_args = []
    if IS_MSVC:
        extra_compile_args = ["/openmp", "/TP", "/O2"]
        extension_args["export_symbols"] = export_symbols(
            "quakemigrate/core/src/qmlib.def"
        )
        extension_args["library_dirs"].extend(
            [
                str(pathlib.Path.cwd() / "quakemigrate" / "core"),
                str(pathlib.Path(sys.prefix) / "bin"),
            ]
        )
    else:
        extra_compile_args = []
        extra_link_args.extend(["-lm"])
        if platform.system() == "Darwin":
            extra_link_args.extend(["-lomp"])
            extra_compile_args.extend(["-Xpreprocessor"])
        else:
            extra_link_args.extend(["-lgomp"])
        extra_compile_args.extend(["-fopenmp", "-fPIC", "-Ofast"])

    extension_args["extra_link_args"] = extra_link_args
    extension_args["extra_compile_args"] = extra_compile_args

    extensions.extend(
        [Extension("quakemigrate.core.src.qmlib", sources=sources, **extension_args)]
    )

    return extensions


def setup_package():
    """Setup package"""

    package_dir, package_data = {}, {}
    if IS_MSVC:
        package_dir["quakemigrate.core"] = str(pathlib.Path("quakemigrate") / "core")
        package_data["quakemigrate.core"] = ["quakemigrate/core/src/*.dll"]

    setup_args = {
        "ext_modules": get_extensions(),
        "package_data": package_data,
        "package_dir": package_dir,
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
