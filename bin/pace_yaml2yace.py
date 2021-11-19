import argparse

from pyace import ACEBBasisSet

parser = argparse.ArgumentParser(prog="pace_yaml2yace",
                                 description="Conversion utility from B-basis (.yaml file) to new-style Ctilde-basis (.yace file)")
parser.add_argument("input", help="input B-basis file name (.yaml)", type=str,  nargs='+', default=[])
parser.add_argument("-o", "--output", help="output Ctilde-basis file name (.yace)", type=str, default="")

args_parse = parser.parse_args()
input_yaml_filenames = args_parse.input
output_ace_filename = args_parse.output


for input_yaml_filename in input_yaml_filenames:
    if output_ace_filename == "":
        if input_yaml_filename.endswith("yaml"):
            actual_output_ace_filename = input_yaml_filename.replace("yaml", "yace")
        elif input_yaml_filename.endswith("yml"):
            actual_output_ace_filename = input_yaml_filename.replace("yml", "yace")
        else:
            actual_output_ace_filename = input_yaml_filename + ".yace"
    else:
        actual_output_ace_filename = output_ace_filename

    print("Loading B-basis from '{}'".format(input_yaml_filename))
    bbasis = ACEBBasisSet(input_yaml_filename)
    print("Converting to Ctilde-basis")
    cbasis = bbasis.to_ACECTildeBasisSet()
    print("Saving Ctilde-basis to '{}'".format(actual_output_ace_filename))
    cbasis.save_yaml(actual_output_ace_filename)
