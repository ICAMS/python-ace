#!/usr/bin/env python
import argparse
import numpy as np

from pyace.utils.timing import run_timing_test

parser = argparse.ArgumentParser(prog="pace_timing",
                                 description="Utility to run the single-CPU timing test for PACE (.yaml) potential ")

parser.add_argument("potential_file", help="B-basis file name (.yaml)", type=str, nargs='+', default=[])
parser.add_argument("-n", "--nstructures", help="number of structures", type=int, default=10)

args_parse = parser.parse_args()
potential_files = args_parse.potential_file
nstructures = args_parse.nstructures

for potential_file in potential_files:
    np.random.seed(42)
    print()
    print("**************************************************")
    print("Using potential file: ", potential_file)
    print("**************************************************")
    run_timing_test(potential_file, n_struct=nstructures)
