# -*- coding:utf-8 -*-
"""
Generate NLLoc OBS phase pick files from QuakeMigrate output.

Author: Conor Bacon
Date: 31/10/2019
"""

import pathlib

import quakemigrate.export as qexport
import quakemigrate.io.obspy_catalog as obs

run_dir = ""

output_dir = pathlib.Path(run_dir / "nonlinloc")
output_dir.mkdir(parents=True, exist_ok=True)

cat = obs.read_quakemigrate(run_dir)

for event in cat:
    filename = "{}.nonlinloc".format(event.resource_id)
    qexport.nlloc_obs(event, str(output_dir / filename))
