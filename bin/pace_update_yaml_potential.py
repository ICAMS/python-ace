import argparse

from pyace import ACEBBasisSet

parser = argparse.ArgumentParser(prog="pace_update_yaml_potential",
                                 description="Update utility for B-basis (.yaml file)")
parser.add_argument("input", help="input B-basis file name (.yaml)", type=str)
parser.add_argument("-o", "--output", help="output B-basis file name", type=str, default="")

args_parse = parser.parse_args()
input_yaml_filename = args_parse.input
output_ace_filename = args_parse.output

if output_ace_filename == "":
    output_ace_filename = input_yaml_filename

print("Loading B-basis from '{}'".format(input_yaml_filename))
bbasis = ACEBBasisSet(input_yaml_filename)
print("Saving B-basis to '{}'".format(output_ace_filename))
bbasis.save(output_ace_filename)
