import argparse

from pyace import ACECTildeBasisSet

parser = argparse.ArgumentParser(prog="pace_update_ace",
                                 description="Conversion utility from old-style Ctilde-basis for single species (.ace file) to new Ctilde-basis YAML (.yace file)")
parser.add_argument("input", help="input old-style Ctilde-basis for single species (.ace file)", type=str)
parser.add_argument("-o", "--output", help="output new Ctilde-basis file name (.yace)", type=str, default="")

args_parse = parser.parse_args()
input_yaml_filename = args_parse.input
output_ace_filename = args_parse.output

if output_ace_filename == "":
    if input_yaml_filename.endswith("ace"):
        output_ace_filename = input_yaml_filename.replace("ace", "yace")
    else:
        output_ace_filename = input_yaml_filename + ".yace"

print("Loading old-style C-tilde basis from '{}'".format(input_yaml_filename))
cbasis = ACECTildeBasisSet(input_yaml_filename)
print("Saving new-style Ctilde-basis to '{}'".format(output_ace_filename))
cbasis.save_yaml(output_ace_filename)
