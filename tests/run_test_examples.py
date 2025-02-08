"""
Script that will run the examples required to run the QuakeMigrate benchmark
tests.

:copyright:
    2020–2025, QuakeMigrate developers.
:license:
    GNU General Public License, Version 3
    (https://www.gnu.org/licenses/gpl-3.0.html)

"""

import os
import pathlib

base_path = pathlib.Path(__file__).resolve().parent.parent / "examples"

# Run Iceland icequake example scripts
os.chdir(base_path / "Icequake_Iceland")
exec(open("iceland_lut.py").read())
exec(open("iceland_detect.py").read())
exec(open("iceland_trigger.py").read())
exec(open("iceland_locate.py").read())

# Run volcano-tectonic example scripts
os.chdir(base_path / "Volcanotectonic_Iceland")
exec(open("get_dike_intrusion_data.py").read())
exec(open("dike_intrusion_lut.py").read())
exec(open("dike_intrusion_detect.py").read())
exec(open("dike_intrusion_trigger.py").read())
exec(open("dike_intrusion_locate.py").read())
