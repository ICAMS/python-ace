import os
from pyace.basis import ACE_B_to_CTildeBasisSet

def convert_yaml_to_ace(infile, outfile):
    """
    Utility to convert yaml format to ace format

    Parameters
    ----------
    infile: string
        input file name

    outfile: string
        output file name
    """
    if not os.path.exists(infile):
        raise FileNotFoundError("input file does not exist")

    basis = ACE_B_to_CTildeBasisSet()
    basis.load_yaml(infile)
    basis.save(outfile)
    
