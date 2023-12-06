# `pacemaker` command line interface

`pacemaker` is a utility for fitting the atomic cluster expansion potential. Usage:

```
usage: pacemaker [-h] [-c] [-o OUTPUT] [-p POTENTIAL] [-ip INITIAL_POTENTIAL] [-d DATA] [-l LOG] [-dr] [-t] [-v] [--no-fit] [--no-predict] [--verbose-tf] [input]

Fitting utility for atomic cluster expansion potentials.

positional arguments:
  input                 input YAML file, default: input.yaml

optional arguments:
  -h, --help            show this help message and exit
  -c, --clean           Remove all generated data
  -o OUTPUT, --output OUTPUT
                        output B-basis YAML file name, default: output_potential.yaml
  -p POTENTIAL, --potential POTENTIAL
                        input potential YAML file name, will override input file 'potential' section
  -ip INITIAL_POTENTIAL, --initial-potential INITIAL_POTENTIAL
                        initial potential YAML file name, will override input file 'potential::initial_potential' section
  -d DATA, --data DATA  data file, will override section 'YAML:fit:filename' from input file
  -l LOG, --log LOG     log filename, default: log.txt
  -dr, --dry-run        Dry run: performs all preprocessing and analysis, but do not do the fitting
  -t, --template        Generate a template 'input.yaml' file by dialog
  -v, --version         Show version info
  --no-fit              Do not fit the potential
  --no-predict          Do not compute and save the predictions
  --verbose-tf          Make tensorflow more verbose (off by defeault)

``` 