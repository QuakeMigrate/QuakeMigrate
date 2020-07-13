# -*- coding: utf-8 -*-
"""
Short test script that will identify any differences between the benchmarked
example outputs and those run after any changes have been made to the codebase.

"""

import pathlib
import unittest

import obspy
import pandas as pd

from quakemigrate.io import read_lut


examples = ["Icequake_Iceland"]
e_path = pathlib.Path().cwd().parent / "examples"
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
            print(f"Testing contents of lookup tables from {example} run.")
            b_dir = b_path / example
            t_dir = e_path / example / "outputs" / "lut"

            b_lut = read_lut((list(b_dir.glob("*.LUT"))[0]))
            t_lut = read_lut((list(t_dir.glob("*.LUT"))[0]))

            # b_lut = LUT(lut_file=(list(b_dir.glob("*.LUT"))[0]))
            # t_lut = LUT(lut_file=(list(t_dir.glob("*.LUT"))[0]))

            print("\t1: Assert grid specifications identical...")
            self.assertEqual(b_lut, t_lut)
            print("\t   ...passed!")

            print("\t2: Assert traveltime tables are identical...")
            b_tt = b_lut.serve_traveltimes(50)
            t_tt = t_lut.serve_traveltimes(50)
            self.assertTrue((b_tt == t_tt).all())
            print("\t   ...passed!")

    def test_detect(self):
        """Check the outputs of detect."""

        for example in examples:
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
            b_av = list(b_dir.glob("*Availability*"))[0]
            t_av = list((t_dir / "availability").glob("*Availability*"))[0]
            self.assertTrue(pd.read_csv(b_av).equals(pd.read_csv(t_av)))
            print("\t   ...passed!")

    def test_trigger(self):
        """Check the outputs of trigger."""

        for example in examples:
            print(f"Testing trigger output from {example} run.")
            b_dir = b_path / example
            t_dir = pathlib.Path(f"{str(t_path).format(example)}") / "trigger"

            print("\t1: Assert triggered events files are identical...")
            b_trig = pd.read_csv(list(b_dir.glob("*Trigger*"))[0])
            t_trig = pd.read_csv(list((t_dir / "events").glob("*Trigger*"))[0])
            self.assertTrue(b_trig.equals(t_trig))
            print("\t   ...passed!")

    def test_locate(self):
        """Check the outputs of locate."""

        for example in examples:
            print(f"Testing locate output from {example} run.")
            b_dir = b_path / example
            t_dir = pathlib.Path(f"{str(t_path).format(example)}") / "locate"

            print("\t1: Assert event files are identical...")
            b_event = pd.read_csv(list(b_dir.glob("*.event"))[0])
            t_event = pd.read_csv(list((t_dir / "events").glob("*.event"))[0])
            self.assertTrue(b_event.equals(t_event))
            print("\t   ...passed!")

            print("\t2: Assert pick files are identical...")
            b_picks = pd.read_csv(list(b_dir.glob("*.picks"))[0])
            t_picks = pd.read_csv(list((t_dir / "picks").glob("*.picks"))[0])
            self.assertTrue(b_picks.equals(t_picks))
            print("\t   ...passed!")

            print("\t3: Assert same number of channels in cut waveforms...")
            b_st = obspy.read(f"{b_dir / '*.m'}")
            t_st = obspy.read(f"{t_dir / 'cut_waveforms' / '*.m'}")
            self.assertEqual(len(b_st), len(t_st))
            print("\t   ...passed!")

            print("\t4: Assert meta data is identical...")
            self.assertEqual(b_st[0].stats, t_st[0].stats)
            print("\t   ...passed!")

            print("\t5: Assert data channels are identical...")
            c_st = b_st + t_st
            c_st.merge(method=-1)
            self.assertEqual(len(c_st), len(b_st))
            print("\t   ...passed!")


if __name__ == "__main__":
    unittest.main()
