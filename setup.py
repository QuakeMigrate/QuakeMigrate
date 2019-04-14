import codecs
import os
import re
import time
import sys
import numpy.distutils.misc_util

from setuptools import setup, find_packages, Extension
from setuptools.dist import Distribution

#
# https://hynek.me/articles/sharing-your-labor-of-love-pypi-quick-and-dirty/
#
###################################################################

#
# Specifying files to Distribute
# https://docs.python.org/2/distutils/sourcedist.html#specifying-the-files-to-distribute
#

NAME = "QuakeMigrate"
PACKAGES = find_packages(where="src")
#EXT_MODULE=[Extension('intra.seis.scan01',['ext/scan01.c','ext/scan01module.c'],extra_compile_args=['/openmp'],
#                           extra_link_args=['-lgomp'])]
INCLUDE_DIRS=numpy.distutils.misc_util.get_numpy_include_dirs()

META_PATH = os.path.join("src", "QMigrate", "__init__.py")
KEYWORDS = ["Seismic", "Location", "Analysis"]
CLASSIFIERS = [
    "Development Status :: Beta",
    "Intended Audience :: Geophysicist",
    "Natural Language :: English",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
    "Programming Language :: Python :: 3.6",
    "Topic :: Scientific/Engineering",
]
# hjson is not in Conda Repository
INSTALL_REQUIRES = ['numpy', 'pandas', 'scipy','scikit-fmm', 'pyproj', 'matplotlib', 'vispy', 'pyzmq', 'msgpack-python']


# ADD A COMPILE STAGE THAT GENERATES THE REQUIRED C-Compile of code. Get working for several operating systems.
os.system('gcc -shared -fPIC -std=gnu99 ./src/QMigrate/lib/src/onset.c ./src/QMigrate/lib/src/QMigrate.c ./src/QMigrate/lib/src/levinson.c -fopenmp -O0 -o ./src/QMigrate/lib/QMigrate.so')


###################################################################

HERE = os.path.abspath(os.path.dirname(__file__))

def read(*parts):
    """
    Build an absolute path from *parts* and and return the contents of the
    resulting file.  Assume UTF-8 encoding.
    """
    with codecs.open(os.path.join(HERE, *parts), "rb", "utf-8") as f:
        return f.read()


META_FILE = read(META_PATH)

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


# class BinaryDistribution(Distribution):
#     def is_pure(self):
#         return False

if __name__ == "__main__":
    setup(
        name=NAME,
        description=find_meta("description"),
        version=find_meta("version") + time.strftime(".%y.%m.%d"),
        author=find_meta("author"),
        author_email=find_meta("email"),
        maintainer=find_meta("author"),
        maintainer_email=find_meta("email"),
        keywords=KEYWORDS,
        long_description=read("README.rst"),
        packages=PACKAGES,
        include_package_data=True,
        include_dirs=INCLUDE_DIRS,
        package_dir={"": "src"},
        package_data={"QMigrate": ["lib/*.so"]},
        zip_safe=False,
        classifiers=CLASSIFIERS,
        install_requires=INSTALL_REQUIRES,
    )
