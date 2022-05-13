#!/usr/bin/env python

import glob
import sys
import os
import numpy as np
import pandas as pd
import argparse

from ase import Atoms
from ase.io import read

from collections import defaultdict, Counter
import logging



logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

      
def walk_file_or_dir(root_directory):
    
    if root_directory is None:
        raise ValueError("Root directory not provided")
        
    if os.path.isfile(root_directory):
        dirname, basename = os.path.split(root_directory)
        yield dirname, [], [basename]
    else:
        for path, dirnames, filenames in os.walk(root_directory):
            yield path, dirnames, filenames
            

def get_ase_atoms_energy_forces(configuration):

    cell = configuration.get_cell()
    positions = configuration.get_positions()
    symbols = configuration.get_chemical_symbols()
    pbc = configuration.get_pbc()

    ase_atoms = Atoms(symbols=symbols, positions=positions, cell=cell, pbc=pbc)
    
    free_energy = configuration.get_potential_energy(force_consistent=True)
    forces = configuration.get_forces()
    
    return ase_atoms, free_energy, forces    
    

def read_vasp_output(root_directory, vasprun_file, free_atom_energy_dict):

    vasprun = os.path.join(root_directory, vasprun_file)

    if not os.path.getsize(vasprun) > 0:
        pass

    vasp_output = read(vasprun, index=':')
    
    vasp_output_dict = {"name": [], "energy": [], "forces": [], "energy_corrected": [], "ase_atoms": []}
    for index, configuration in enumerate(vasp_output):
        
        vasp_output_dict["name"].append('{}##{}'.format(root_directory,index))
        
        ase_atoms, free_energy, forces = get_ase_atoms_energy_forces(configuration)
        vasp_output_dict["ase_atoms"].append(ase_atoms)
        vasp_output_dict["energy"].append(free_energy)
        vasp_output_dict["forces"].append(forces)
        
        chemical_symbols = ase_atoms.get_chemical_symbols()
        counter_elements = Counter(chemical_symbols)
        reference_energy = 0.
        for symbol, counter in counter_elements.items():
            reference_energy += free_atom_energy_dict[symbol]*counter
        
        vasp_output_dict["energy_corrected"].append(free_energy - reference_energy)
            
    return vasp_output_dict


class ParseKwargs(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, dict())
        for value in values:
            key, value = value.split(':')[0], float(value.split(':')[1]) 
            getattr(namespace, self.dest)[key] = value


def main(args):
    
    ##############################################################################################
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-wd", "--working-dir", help="top directory where keep calculations",
                        type=str, default='.', dest="working_dir")
        
    parser.add_argument("--output-dataset-filename", help="pickle filename, default is collected.pckl.gzip",
                        type=str, default="collected.pckl.gzip", dest="output_dataset_filename")
    
    parser.add_argument('--free-atom-energy', help="dictionary of reference energies (i.e. Al:-0.123 Cu:-0.456 Zn:-0.789)",
                        nargs='*', dest="free_atom_energy", default=defaultdict(lambda:0), action=ParseKwargs)
    
    args_parse = parser.parse_args(args)
    working_dir = os.path.abspath(args_parse.working_dir)
    free_atom_energy_dict = args_parse.free_atom_energy
    output_dataset_filename = args_parse.output_dataset_filename
    
    ##############################################################################################
    logger.info('Following atomic reference energies will be subtracted : {}'.format(', '.join([str(k)+':'+str(v) for k,v in free_atom_energy_dict.items()])))
   
    vasprun_file = 'vasprun.xml'
    outcar_file = 'OUTCAR'
    data = {"name": [], "energy": [], "forces": [], "energy_corrected":[], "ase_atoms": []}

    for root, _, filenames in walk_file_or_dir(working_dir):

        try:
            if filenames and vasprun_file in filenames:                
                vasp_output_dict = read_vasp_output(root, vasprun_file, free_atom_energy_dict)
                for key, value in vasp_output_dict.items():
                    data[key] += value
                logger.info('Data collected successfully from {} with entries {}'.format(os.path.join(root,vasprun_file),len(vasp_output_dict["name"])))
            else:
                vasp_output_dict = read_vasp_output(root, outcar_file, free_atom_energy_dict)
                for key, value in vasp_output_dict.items():
                    data[key] += value
                logger.info('Data collected successfully from {} with entries {}'.format(os.path.join(root,outcar_file),len(vasp_output_dict["name"])))
        except Exception as e:
            logger.error('Filename could not be read: {}'.format(str(e)))

            
    df = pd.DataFrame(data)
    df.to_pickle('{}'.format(output_dataset_filename), compression='gzip', protocol=4)
    logger.info('Store dataset into {}'.format(output_dataset_filename))
    
    df['NUMBER_OF_ATOMS'] = df['ase_atoms'].map(len)
    df['energy_corrected_per_atom'] = df['energy_corrected']/df['NUMBER_OF_ATOMS']
    df['absolute_energy_collected_per_atom'] = df['energy_corrected_per_atom'].abs()

    logger.info('Total number of structures: {}'.format(len(df)))
    number_atoms = df['ase_atoms'].map(len).sum()
    logger.info('Total number of atoms: {}'.format(number_atoms))
    logger.info('Mean number of atoms per structure: {}'.format(number_atoms/len(df)))
    
    logger.info('Mean of energy per atom: {:.3f} eV/atom'.format(df['energy_corrected_per_atom'].mean()))
    logger.info('Std of energy per atom: {:.3f} eV/atom'.format(df['energy_corrected_per_atom'].std()))
    logger.info('Minimum/maximum energy per atom: {:.3f}/{:.3f} eV/atom'.format(df['energy_corrected_per_atom'].min(),df['energy_corrected_per_atom'].max()))
    logger.info('Minimum/maximum abs energy per atom: {:.3f}/{:.3f} eV/atom'.format(df['absolute_energy_collected_per_atom'].min(),df['absolute_energy_collected_per_atom'].max()))
    
    df['magnitude_forces'] = df['forces'].map(np.linalg.norm)
    logger.info('Mean force magnitude: {:.3f} eV/A'.format(df['magnitude_forces'].mean()))
    logger.info('Std of force magnitude: {:.3f} eV/A'.format(df['magnitude_forces'].std()))
    logger.info('Minimum/maximum force magnitude: {:.3f}/{:.3f} eV/A'.format(df['magnitude_forces'].min(),df['magnitude_forces'].max()))
    
    

if __name__ == "__main__":
    main(sys.argv[1:])
