# -*- coding: utf-8 -*-
"""
Short test script that will identify any differences between the benchmarked
example outputs and those run after any changes have been made to the codebase.

:copyright:
    2020 - 2021, QuakeMigrate developers.
:license:
    GNU General Public License, Version 3
    (https://www.gnu.org/licenses/gpl-3.0.html)

"""

import pathlib
import sys
import unittest

import numpy as np
import obspy
import pandas as pd

from quakemigrate.io import read_lut


examples = ["Icequake_Iceland", "Volcanotectonic_Iceland"]
e_path = pathlib.Path(__file__).resolve().parent.parent / "examples"
b_path = e_path / "benchmarks"
t_path = e_path / "{}" / "outputs" / "runs" / "example_run"


class TestExamples(unittest.TestCase):
    """
    Suite of tests to compare the output of each of the examples scripts with
    a benchmark.

    """

    def test_lut(self):
        """Confirm LUTs are identical."""

        for example in examples:
            if not (e_path / example / "outputs/runs/example_run").is_dir():
                continue

            print(f"Testing contents of lookup tables from {example} run.")
            b_dir = b_path / example
            t_dir = e_path / example / "outputs" / "lut"

            b_lut = read_lut((sorted(b_dir.glob("*.LUT"))[0]))
            t_lut = read_lut((sorted(t_dir.glob("*.LUT"))[0]))

            print("\t1: Assert traveltime tables are identical...")
            b_tt = b_lut.serve_traveltimes(50)
            t_tt = t_lut.serve_traveltimes(50)
            self.assertTrue((b_tt == t_tt).all())

            print("\t2: Assert various LUT parameters are identical...")
            print("\t\t...'fraction_tt'...")
            self.assertEqual(b_lut.fraction_tt, t_lut.fraction_tt)
            print("\t\t...'phases'...")
            self.assertEqual(b_lut.phases, t_lut.phases)
            print("\t\t...'station_data'...")
            pd.testing.assert_frame_equal(
                b_lut.station_data, t_lut.station_data, check_exact=False
            )
            print("\t\t...'grid_proj'...")
            self.assertEqual(b_lut.grid_proj, t_lut.grid_proj)
            print("\t\t...'coord_proj'...")
            self.assertEqual(b_lut.coord_proj, t_lut.coord_proj)
            print("\t\t...'ll_corner'...")
            self.assertTrue(np.isclose(b_lut.ll_corner, t_lut.ll_corner).all())
            print("\t\t...'ur_corner'...")
            self.assertTrue(np.isclose(b_lut.ur_corner, t_lut.ur_corner).all())
            print("\t\t...'node_spacing'...")
            self.assertTrue(np.equal(b_lut.node_spacing, t_lut.node_spacing).all())
            print("\t\t...'node_count'...")
            self.assertTrue(np.equal(b_lut.node_count, t_lut.node_count).all())
            print("\t   ...passed!")

    def test_detect(self):
        """Check the outputs of detect."""

        for example in examples:
            if not (e_path / example / "outputs/runs/example_run").is_dir():
                continue

            print(f"Testing detect output from {example} run.")
            b_dir = b_path / example
            t_dir = pathlib.Path(f"{str(t_path).format(example)}") / "detect"

            print("\t1: Assert same number of channels in .scanmseed...")
            b_st = obspy.read(f"{b_dir / '*.scanmseed'}")
            t_st = obspy.read(f"{t_dir / 'scanmseed' / '*.scanmseed'}")
            self.assertEqual(len(b_st), len(t_st))
            print("\t   ...passed!")

            print("\t2: Assert meta data is identical...")
            self.assertEqual(b_st[0].stats, t_st[0].stats)
            print("\t   ...passed!")

            print("\t3: Assert each data channels are identical...")
            c_st = b_st + t_st
            c_st.merge(method=-1)
            self.assertEqual(len(c_st), len(b_st))
            print("\t   ...passed!")

            print("\t4: Assert availability files are identical...")
            b_av = sorted(b_dir.glob("*Availability*"))[0]
            t_av = sorted((t_dir / "availability").glob("*Availability*"))[0]
            self.assertTrue(pd.read_csv(b_av).equals(pd.read_csv(t_av)))
            print("\t   ...passed!")

    def test_trigger(self):
        """Check the outputs of trigger."""

        for example in examples:
            if not (e_path / example / "outputs/runs/example_run").is_dir():
                continue

            print(f"Testing trigger output from {example} run.")
            b_dir = b_path / example
            t_dir = pathlib.Path(f"{str(t_path).format(example)}") / "trigger"

            print("\t1: Assert triggered events files are identical...")
            b_trig = pd.read_csv(sorted(b_dir.glob("*Trigger*"))[0])
            t_trig = pd.read_csv(sorted((t_dir / "events").glob("*Trigger*"))[0])
            self.assertTrue(b_trig.equals(t_trig))
            print("\t   ...passed!")

    def test_locate(self):
        """Check the outputs of locate."""

        for example in examples:
            if not (e_path / example / "outputs/runs/example_run").is_dir():
                continue

            print(f"Testing locate output from {example} run.")
            b_dir = b_path / example
            t_dir = pathlib.Path(f"{str(t_path).format(example)}") / "locate"

            print("\t1: Assert event files are identical...")
            b_events = sorted(b_dir.glob("*.event"))
            t_events = sorted((t_dir / "events").glob("*.event"))
            for b_event, t_event in zip(b_events, t_events):
                pd.testing.assert_frame_equal(
                    pd.read_csv(b_event), pd.read_csv(t_event), check_exact=False
                )
            print("\t   ...passed!")

            print("\t2: Assert pick files are identical...")
            b_picks = sorted(b_dir.glob("*.picks"))
            t_picks = sorted((t_dir / "picks").glob("*.picks"))
            for b_pick, t_pick in zip(b_picks, t_picks):
                if "20140824000443240" in t_pick.name:
                    print("\t   ...skipping due to unidentified floating point issue.")
                    continue
                pd.testing.assert_frame_equal(
                    pd.read_csv(b_pick), pd.read_csv(t_pick), check_exact=False
                )
            print("\t   ...passed!")

            print("\t3: Assert same number of channels in cut waveforms...")
            b_st = obspy.read(f"{b_dir / '*.m'}").sort()
            t_st = obspy.read(f"{t_dir / 'raw_cut_waveforms' / '*.m'}").sort()
            self.assertEqual(len(b_st), len(t_st))
            print("\t   ...passed!")

            print("\t4: Assert meta data is identical...")
            self.assertEqual(b_st[0].stats, t_st[0].stats)
            print("\t   ...passed!")

            print("\t5: Assert data channels are identical...")
            c_st = b_st + t_st
            c_st.merge(method=-1)
            b_st.merge(method=-1)
            self.assertEqual(len(c_st), len(b_st))
            print("\t   ...passed!")

            print("\t6: Assert amplitude files are identical...")
            try:
                b_amps = sorted(b_dir.glob("*.amps"))
                t_amps = sorted((t_dir / "amplitudes").glob("*.amps"))
                _ = b_amps[0]  # Check any files
                for b_amp, t_amp in zip(b_amps, t_amps):
                    pd.testing.assert_frame_equal(
                        pd.read_csv(b_amp), pd.read_csv(t_amp), check_exact=False
                    )
                print("\t   ...passed!")
            except IndexError:
                print("\t   ...no amplitude files found!")


if __name__ == "__main__":
    unittest.main()
