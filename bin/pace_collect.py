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

LOG_FMT = '%(asctime)s %(levelname).1s - %(message)s'
logging.basicConfig(level=logging.INFO, format=LOG_FMT, datefmt="%Y/%m/%d %H:%M:%S")
logger = logging.getLogger()


class InvalidArgumentError(ValueError):
    pass


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


def generate_selected_indices(selection, number_of_structures):
    """
    select_steps:
    :param selection:
    "all", "first", "last", "first_and_last", "0:10:-1"
    :param select_steps:
    :param number_of_structure:
    :return: list of selected indices
    """
    selected_index = list(range(number_of_structures))
    if selection == "last":
        selected_index = [number_of_structures - 1]
    elif selection == "first":
        selected_index = [0]
    elif selection == "all":
        selected_index = range(number_of_structures)
    elif selection == "first_and_last":
        if number_of_structures > 1:
            selected_index = [0, number_of_structures - 1]
        else:
            selected_index = [0]
    else:
        if ":" in selection:
            i = []
            for s in selection.split(':'):
                if s == '':
                    i.append(None)
                else:
                    i.append(int(s))
            i += (3 - len(i)) * [None]
            slc = slice(*i)
            selected_index = selected_index[slc]
        else:
            raise InvalidArgumentError(
                'select_steps option `{}` is not valid. Should be one of the  "all", "first", "last", "first_and_last"'.format(
                    selection))
    return selected_index


def read_vasp_output(root_directory, vasp_output_file_name, selection="last"):
    """

    :param root_directory:
    :type root_directory:
    :param vasp_output_file_name:
    :type vasp_output_file_name:
    :param selection: "all", "first", "last", "first_and_last"

    :return:

    """
    vasp_output_file_name = os.path.join(root_directory, vasp_output_file_name)

    if not os.path.getsize(vasp_output_file_name) > 0:
        pass

    vasp_output = read(vasp_output_file_name, index=':')
    number_of_structures = len(vasp_output)
    selected_index = generate_selected_indices(selection, number_of_structures)

    vasp_output_dict = {"name": [], "energy": [], "forces": [], "ase_atoms": []}
    for index, configuration in enumerate(vasp_output):
        if index not in selected_index:
            continue

        vasp_output_dict["name"].append('{}##{}'.format(root_directory, index))

        ase_atoms, free_energy, forces = get_ase_atoms_energy_forces(configuration)
        vasp_output_dict["ase_atoms"].append(ase_atoms)
        vasp_output_dict["energy"].append(free_energy)
        vasp_output_dict["forces"].append(forces)

    return vasp_output_dict


def get_safe_volume(at):
    if np.all(at.get_pbc()):
        return at.get_volume()
    else:
        return 0


class ParseKwargs(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, dict())
        for value in values:
            if value == "auto":
                key = 'auto'
                val = True
            else:
                splt = value.split(':')
                key = splt[0]
                try:
                    val = float(splt[1])
                except ValueError:
                    val = splt[1]
            getattr(namespace, self.dest)[key] = val


def get_free_atom_energy(df, el):
    mask = (df["n_" + el] == 1) & (df['NUMBER_OF_ATOMS'] == 1) & (df['volume_per_atom'] > 500)
    sel_df = df.loc[mask].sort_values('volume_per_atom', ascending=False)
    if len(sel_df) > 0:
        row = sel_df.iloc[0]
        ref_epa = row["energy"]
        ref_vpa = row["volume_per_atom"]
        name = row["name"]
        logger.info(
            "Auto-identify free-atom reference energy for {}: E={:.3f} eV/atom, V={:.3f} A^3/atom (from {})".format(el,
                                                                                                                    ref_epa,
                                                                                                                    ref_vpa,
                                                                                                                    name))
    else:
        logger.warning('No reference free atom energy found for {}, set to E=0'.format(el))
        ref_epa = 0
    return ref_epa


def main(args):
    ##############################################################################################
    parser = argparse.ArgumentParser()

    parser.add_argument("-wd", "--working-dir", help="top directory where keep calculations",
                        type=str, default='.', dest="working_dir")

    parser.add_argument("--output-dataset-filename", help="pickle filename, default is collected.pkl.gz",
                        type=str, default="collected.pkl.gz", dest="output_dataset_filename")

    parser.add_argument('--free-atom-energy',
                        help="dictionary of reference energies (auto for extraction from dataset), i.e. `Al:-0.123 Cu:-0.456 Zn:auto`,"
                             " default is zero. If option is `auto`, then it will be extracted from dataset",
                        nargs='*', dest="free_atom_energy", default=defaultdict(lambda: 0), action=ParseKwargs)

    parser.add_argument('--selection', type=str, default='last', dest='selection',
                        help='Option to select from multiple configurations of single VASP calculation:'
                             ' first, last, all, first_and_last (default: last)')

    args_parse = parser.parse_args(args)
    working_dir = os.path.abspath(args_parse.working_dir)
    free_atom_energy_dict = args_parse.free_atom_energy
    output_dataset_filename = args_parse.output_dataset_filename
    selection = args_parse.selection

    ##############################################################################################

    logger.info('Selection from multiple configurations of single calculation: {}'.format(selection))
    vasprun_file = 'vasprun.xml'
    outcar_file = 'OUTCAR'
    data = {"name": [], "energy": [], "forces": [], "ase_atoms": []}

    for root, _, filenames in walk_file_or_dir(working_dir):

        try:
            if filenames and vasprun_file in filenames:
                vasp_output_dict = read_vasp_output(root, vasprun_file, selection)
                for key, value in vasp_output_dict.items():
                    data[key] += value
                logger.info(
                    'Data collected successfully from {} with entries {}'.format(os.path.join(root, vasprun_file),
                                                                                 len(vasp_output_dict["name"])))
            elif filenames and outcar_file in filenames:
                vasp_output_dict = read_vasp_output(root, outcar_file, selection)
                for key, value in vasp_output_dict.items():
                    data[key] += value
                logger.info(
                    'Data collected successfully from {} with entries {}'.format(os.path.join(root, outcar_file),
                                                                                 len(vasp_output_dict["name"])))
        except InvalidArgumentError as e:
            logger.error('Invalid argument: {}'.format(str(e)))
            raise e
        except Exception as e:
            logger.error('Filename could not be read: {}'.format(str(e)))

    df = pd.DataFrame(data)
    logger.info('Free atomic energy options: {}'.format(
        ', '.join([str(k) + ':' + str(v) for k, v in free_atom_energy_dict.items()])))

    df['NUMBER_OF_ATOMS'] = df['ase_atoms'].map(len)
    df["comp_dict"] = df["ase_atoms"].map(lambda at: Counter(at.get_chemical_symbols()))
    df['volume'] = df["ase_atoms"].map(get_safe_volume)
    df['volume_per_atom'] = df['volume'] / df['NUMBER_OF_ATOMS']

    elements = set()
    for cd in df["comp_dict"]:
        elements.update(cd.keys())
    elements = sorted(elements)
    logger.info("List of elements: {}".format(elements))

    for el in elements:
        df["n_" + el] = df["comp_dict"].map(lambda d: d.get(el, 0))

    # free_atom_energy_dict
    auto_determined_free_atom_energy = {}
    for el, val in free_atom_energy_dict.items():
        if val == "auto":
            ref_epa = get_free_atom_energy(df, el)
            auto_determined_free_atom_energy[el] = ref_epa

    # missing elements:
    if "auto" in free_atom_energy_dict:
        for el in elements:
            if el not in free_atom_energy_dict.keys():
                logger.info("Element {} is found but free-atom reference energy is missing, try to extract".format(el))
                ref_epa = get_free_atom_energy(df, el)
                auto_determined_free_atom_energy[el] = ref_epa

    free_atom_energy_dict.update(auto_determined_free_atom_energy)

    free_atom_energy_dict = {el: free_atom_energy_dict[el] for el in elements}
    logger.info('Following atomic reference energies will be subtracted: {}'.format(
        ', '.join([str(k) + ':' + str(v) for k, v in free_atom_energy_dict.items()])))

    n_el_cols = ["n_" + el for el in elements]

    free_atom_arr = np.array([free_atom_energy_dict[e] for e in elements])
    df["energy_corrected"] = df["energy"] - (df[n_el_cols] * free_atom_arr).sum(axis=1)
    df['energy_corrected_per_atom'] = df['energy_corrected'] / df['NUMBER_OF_ATOMS']

    #######
    df.drop(columns=n_el_cols + ['comp_dict', 'volume', 'volume_per_atom', 'NUMBER_OF_ATOMS'], inplace=True)
    df.to_pickle('{}'.format(output_dataset_filename), protocol=4)
    logger.info('Store dataset into {}'.format(output_dataset_filename))
    ######
    df['absolute_energy_collected_per_atom'] = df['energy_corrected_per_atom'].abs()

    logger.info('Total number of structures: {}'.format(len(df)))
    number_atoms = df['ase_atoms'].map(len).sum()
    logger.info('Total number of atoms: {}'.format(number_atoms))
    logger.info('Mean number of atoms per structure: {:.3f}'.format(number_atoms / len(df)))

    logger.info('Mean of energy per atom: {:.3f} eV/atom'.format(df['energy_corrected_per_atom'].mean()))
    logger.info('Std of energy per atom: {:.3f} eV/atom'.format(df['energy_corrected_per_atom'].std()))
    logger.info('Minimum/maximum energy per atom: {:.3f}/{:.3f} eV/atom'.format(df['energy_corrected_per_atom'].min(),
                                                                                df['energy_corrected_per_atom'].max()))
    logger.info('Minimum/maximum abs energy per atom: {:.3f}/{:.3f} eV/atom'.format(
        df['absolute_energy_collected_per_atom'].min(), df['absolute_energy_collected_per_atom'].max()))

    df['magnitude_forces'] = df['forces'].map(np.linalg.norm)
    logger.info('Mean force magnitude: {:.3f} eV/A'.format(df['magnitude_forces'].mean()))
    logger.info('Std of force magnitude: {:.3f} eV/A'.format(df['magnitude_forces'].std()))
    logger.info('Minimum/maximum force magnitude: {:.3f}/{:.3f} eV/A'.format(df['magnitude_forces'].min(),
                                                                             df['magnitude_forces'].max()))


if __name__ == "__main__":
    main(sys.argv[1:])
