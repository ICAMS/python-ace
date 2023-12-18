import logging
import numpy as np
import pickle
import pkg_resources
import re

from collections import defaultdict
from copy import deepcopy
from itertools import combinations, permutations, combinations_with_replacement, product
from typing import Dict, List, Union, Tuple

from pyace import BBasisConfiguration, BBasisFunctionSpecification, BBasisFunctionsSpecificationBlock, ACEBBasisSet
from pyace.basisextension import *
from pyace.const import *

element_patt = re.compile("([A-Z][a-z]?)")

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

ALL = "ALL"
UNARY = "UNARY"
BINARY = "BINARY"
TERNARY = "TERNARY"
QUATERNARY = "QUATERNARY"
QUINARY = "QUINARY"
KEYWORDS = [ALL, UNARY, BINARY, TERNARY, QUATERNARY, QUINARY, 'number_of_functions_per_element']

NARY_MAP = {UNARY: 1, BINARY: 2, TERNARY: 3, QUATERNARY: 4, QUINARY: 5}
PERIODIC_ELEMENTS = chemical_symbols = [
    'H', 'He',
    'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
    'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar',
    'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
    'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr',
    'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
    'In', 'Sn', 'Sb', 'Te', 'I', 'Xe',
    'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy',
    'Ho', 'Er', 'Tm', 'Yb', 'Lu',
    'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi',
    'Po', 'At', 'Rn',
    'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk',
    'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr',
    'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc',
    'Lv', 'Ts', 'Og']

default_mus_ns_uni_to_rawlsLS_np_rank_filename = pkg_resources.resource_filename('pyace.data',
                                                                                 'mus_ns_uni_to_rawlsLS_np_rank.pckl')

def clean_bbasisconfig(initial_bbasisconfig):
    for block in initial_bbasisconfig.funcspecs_blocks:
        block.lmaxi = 0
        block.nradmaxi = 0
        block.nradbaseij = 0
        block.radcoefficients = []
        block.funcspecs = []


def reset_bbasisconfig(bconf):
    """set crad=delta_nk, func.coeffs=[0...]"""
    for block in bconf.funcspecs_blocks:
        block.set_func_coeffs(np.zeros_like(block.get_func_coeffs()))
        radcoefficients = np.array(block.radcoefficients)
        if len(radcoefficients.shape) == 3:
            # C_ nlk = delta_nk
            radcoefficients[:, :, :] = 0.0
            for nk in range(min(radcoefficients.shape[0], radcoefficients.shape[2])):
                radcoefficients[nk, :, nk] = 1.0
            block.set_radial_coeffs(radcoefficients.flatten())



def unify_to_minimized_indices(seq, shift=0):
    """
    Unify to minimized ordered sequence of indices
    """
    seq_map = {e: i + shift for i, e in enumerate(sorted(set(seq)))}
    return tuple([seq_map[e] for e in seq])


def unify_by_ordering(mus_comb, ns_comb):
    """
    Unify mus_comb and ns_comb to minimized-indices sequence, combine pairwise and  sort
    """
    return tuple(sorted(zip(unify_to_minimized_indices(mus_comb), unify_to_minimized_indices(ns_comb))))


def unify_absolute_by_ordering(mus_comb, ns_comb):
    """
    Unify mus_comb and ns_comb by combining pairwise and sort
    """
    return tuple(sorted(zip(mus_comb, ns_comb)))


def unify_mus_ns_comb(mus_comb, ns_comb):
    """
    Unify mus_comb, ns_comb by unifying to min-inds, combining, sorting and
    unifying the pair one more time to minimized-indices sequence
    """
    unif_comb = unify_by_ordering(mus_comb, ns_comb)
    return unify_to_minimized_indices(unif_comb)


def unify_absolute_mus_ns_comb(mus_comb, ns_comb):
    """
    Unify mus_comb, ns_comb by combining, sorting
    """
    unif_comb = unify_absolute_by_ordering(mus_comb, ns_comb)
    return unif_comb


def create_species_block_without_funcs(elements_vec: List[str], block_spec: Dict) -> BBasisFunctionsSpecificationBlock:
    """
    Create a BBasisFunctionsSpecificationBlock
    :param elements_vec: block's elements, i.e. (ele1, ele2, ...)
    :param block_spec: block specification dictionary
    :param crad_initializer: "delta" or "random"
    :return: BBasisFunctionsSpecificationBlock
    """
    block = BBasisFunctionsSpecificationBlock()

    block.block_name = " ".join(elements_vec)
    block.elements_vec = elements_vec

    # embedding
    if "fs_parameters" in block_spec:
        block.fs_parameters = block_spec["fs_parameters"]

    if "ndensity" in block_spec:
        block.ndensityi = block_spec["ndensity"]

    if "npot" in block_spec:
        block.npoti = block_spec["npot"]

    if "drho_core_cut" in block_spec:
        block.drho_cut = block_spec["drho_core_cut"]
    if "rho_core_cut" in block_spec:
        block.rho_cut = block_spec["rho_core_cut"]

    # bonds
    if "NameOfCutoffFunction" in block_spec:
        block.NameOfCutoffFunctionij = block_spec["NameOfCutoffFunction"]

    if "core-repulsion" in block_spec:
        block.core_rep_parameters = block_spec["core-repulsion"]

    if "rcut" in block_spec:
        block.rcutij = block_spec["rcut"]
    if "dcut" in block_spec:
        block.dcutij = block_spec["dcut"]
    if "radbase" in block_spec:
        block.radbase = block_spec["radbase"]
    if "radparameters" in block_spec:
        block.radparameters = block_spec["radparameters"]

    if "nradbase" in block_spec:
        block.nradbaseij = block_spec["nradbase"]
    if "nradmax" in block_spec:
        block.nradmaxi = block_spec["nradmax"]
    if "lmax" in block_spec:
        block.lmaxi = block_spec["lmax"]

    if "r_in" in block_spec:
        block.r_in = block_spec["r_in"]
        block.inner_cutoff_type = "distance"
    if "delta_in" in block_spec:
        block.delta_in = block_spec["delta_in"]
        block.inner_cutoff_type = "distance"
    if "inner_cutoff_type" in block_spec:
        block.inner_cutoff_type = block_spec["inner_cutoff_type"]
    else:
        block.inner_cutoff_type = "distance"

    # if crad_initializer == "delta":
    crad = np.zeros((block.nradmaxi, block.lmaxi + 1, block.nradbaseij))

    for n in range(0, min(block.nradmaxi, block.nradbaseij)):  # +1 is excluded, because ns=1...
        crad[n, :, n] = 1.0
    # elif crad_initializer == "random":
    #     crad = np.random.randn(block.nradmaxi, block.lmaxi + 1, block.nradbaseij)
    # else:
    #     raise ValueError("Unknown crad_initializer={}. Could be only 'delta' or 'random'".format(crad_initializer))

    block.radcoefficients = crad

    return block


def generate_species_keys(elements, r):
    """
    Generate all ordered permutations of the elements if size `r`

    :param elements: list of elements
    :param r: permutations size
    :return: list of speices blocks names (permutation) of size `r`
    """
    keys = set()
    for el in elements:
        rest_elements = [e for e in elements if e != el]

        for rst in product(rest_elements, repeat=r - 1):
            rst = list(dict.fromkeys(sorted(rst)))
            key = tuple([el] + rst)
            if len(key) == r:
                keys.add(key)
    return sorted(keys)


def generate_all_species_keys(elements):
    """
    Generate all ordered in [1:...] slice permutations of elements,
    that would be the species blocks names

    :param elements: list of elements (str)
    :return: list of all generated block names
    """
    keys = []
    nelements = len(elements)
    for r in range(1, nelements + 1):
        for key in generate_species_keys(elements, r):
            keys.append(key)
    return keys


def species_key_to_bonds(key):
    """
    Unify the tuple `key` to a list of bond pairs:

        A     -> [A,A]
        A:B   -> [A,B] [B,A],
        A:BC  -> [A,B] [B,A], [A,C], [C,A]
        A:BCD -> [A,B] [B,A], [A,C], [C,A], [A,D], [D,A]

    :param key: tuple of elements
    :return: list of bond pairs
    """
    if len(key) == 1:
        bonds = [(key[0], key[0])]
    else:
        k0 = key[0]
        rkeys = key[1:]
        bonds = []
        for rk in rkeys:
            bonds.append((k0, rk))
            bonds.append((rk, k0))
    return bonds


def create_multispecies_basis_config(potential_config: Dict,
                                     unif_mus_ns_to_lsLScomb_dict: Dict = None,
                                     func_coefs_initializer="zero",
                                     initial_basisconfig: BBasisConfiguration = None,
                                     overwrite_blocks_from_initial_bbasis=False
                                     ) -> BBasisConfiguration:
    """
    Creates a BBasisConfiguration using potential_config dict

    Possible keywords: ALL, UNARY, BINARY, TERNARY, QUATERNARY, QUINARY

    Example:

    potential_config = {
        'deltaSplineBins': 0.001,
        'elements': ['Al', 'H'],

        'embeddings': {'ALL': {'drho_core_cut': 250,
                               'fs_parameters': [1, 1],
                               'ndensity': 1,
                               'npot': 'FinnisSinclair',
                               'rho_core_cut': 200000},
                       'Al': {'drho_core_cut': 250,
                              'fs_parameters': [1, 1, 1, 0.5],
                              'ndensity': 2,
                              'npot': 'FinnisSinclairShiftedScaled',
                              'rho_core_cut': 200000}

                       },

        'bonds': {'ALL': {'NameOfCutoffFunction': 'cos',
                          'core-repulsion': [10000.0, 5.0],
                          'dcut': 0.01,
                          'radbase': 'ChebPow',
                          'radparameters': [2.0],
                          'rcut': 3.9},

                  ('Al', 'H'): {'NameOfCutoffFunction': 'cos',
                                'core-repulsion': [10000.0, 5.0],
                                'dcut': 0.01,
                                'radbase': 'ChebExpCos',
                                'radparameters': [2.0],
                                'rcut': 3.5},
                  },

        'functions': {
            'UNARY': {
                'nradmax_by_orders': [5, 2, 2],
                'lmax_by_orders': [0, 1, 1]
            },

            'BINARY': {
                'nradmax_by_orders': [5, 1, 1],
                'lmax_by_orders': [0, 1, 1],
            },
        }
    }



    :param potential_config: potential configuration dictionary, see above
    :param unif_mus_ns_to_lsLScomb_dict: "whitelist" (dictionary) of the { unify_mus_ns_comb(mus_comb, ns_comb): list of (ls,LS) }
    :param func_coefs_initializer: "zero" or "random"
    :param initial_basisconfig:
    :param overwrite_blocks_from_initial_bbasis: (default False)
    :return: BBasisConfiguration
    """

    overwrite_blocks_from_initial_bbasis = potential_config.get('overwrite_blocks_from_initial_bbasis',
                                                                overwrite_blocks_from_initial_bbasis)
    element_ndensity_dict = None
    if initial_basisconfig is not None:
        # extract embeddings from initial_basisconfig
        bas = ACEBBasisSet(initial_basisconfig)
        initial_embeddings = {}
        element_ndensity_dict = {}
        for el_ind, emb in bas.map_embedding_specifications.items():
            emb_dict = {}
            emb_dict["npot"] = emb.npoti
            emb_dict["ndensity"] = emb.ndensity
            emb_dict["fs_parameters"] = emb.FS_parameters
            emb_dict["rho_core_cut"] = emb.rho_core_cutoff
            emb_dict["drho_core_cut"] = emb.drho_core_cutoff
            initial_embeddings[bas.elements_name[el_ind]] = emb_dict
            element_ndensity_dict[bas.elements_name[el_ind]] = emb.ndensity

        if "embeddings" not in potential_config:
            potential_config["embeddings"] = {}

        embeddings = potential_config["embeddings"]
        for elm, emb_spec in initial_embeddings.items():
            if elm not in embeddings:
                embeddings[elm] = emb_spec
        potential_config["embeddings"] = embeddings

    blocks_list = create_multispecies_basisblocks_list(potential_config,
                                                       element_ndensity_dict=element_ndensity_dict,
                                                       func_coefs_initializer=func_coefs_initializer,
                                                       unif_mus_ns_to_lsLScomb_dict=unif_mus_ns_to_lsLScomb_dict,
                                                       )
    # compare with initial_basisconfig, if some blocks are missing in generated config - add them:
    if initial_basisconfig is not None:
        new_block_dict = {bl.block_name: bl for bl in blocks_list}
        if overwrite_blocks_from_initial_bbasis: # overwrite new blocks with old-ones
            for initial_block in initial_basisconfig.funcspecs_blocks:
                new_block_dict[initial_block.block_name] = initial_block
                log.info("Block {} is overwritten from initial potential".format(initial_block.block_name))
                if initial_block.block_name not in new_block_dict:
                    blocks_list.append(initial_block)
            blocks_list = [bl for bl in new_block_dict.values()]
        else: # only add missing blocks
            for initial_block in initial_basisconfig.funcspecs_blocks:
                if initial_block.block_name not in new_block_dict:
                    blocks_list.append(initial_block)
                    log.info("New block {} is added from initial potential".format(initial_block.block_name))

    new_basis_conf = BBasisConfiguration()
    new_basis_conf.deltaSplineBins = potential_config.get("deltaSplineBins", 0.001)
    new_basis_conf.funcspecs_blocks = blocks_list
    validate_bonds_nradmax_lmax_nradbase(new_basis_conf)
    new_basis_conf.validate(raise_exception=True)

    if ("functions" in potential_config and
            "number_of_functions_per_element" in potential_config['functions']):
        num_block = len(new_basis_conf.funcspecs_blocks)
        number_of_functions_per_element = potential_config["functions"]["number_of_functions_per_element"]
        target_bbasis = ACEBBasisSet(new_basis_conf)
        nelements = target_bbasis.nelements
        ladder_step = number_of_functions_per_element * nelements // num_block

        initial_basisconfig = new_basis_conf.copy()
        clean_bbasisconfig(initial_basisconfig)
        current_bbasisconfig = extend_multispecies_basis(initial_basisconfig, new_basis_conf,
                                                         "power_order", ladder_step)
        new_basis_conf = current_bbasisconfig

    return new_basis_conf


def get_element_ndensity_dict(block_spec_dict):
    element_ndensity_dict = {}
    for el, spec_val in block_spec_dict.items():
        if len(el) == 1:
            element_ndensity_dict[el[0]] = spec_val['ndensity']
    return element_ndensity_dict


def generate_blocks_specifications_dict(potential_config: Dict) -> Dict:
    """
   Creates a blocks_specifications_dict using potential_config dict

   Possible keywords: ALL, UNARY, BINARY, TERNARY, QUATERNARY, QUINARY

   Example:

   potential_config = {
       'deltaSplineBins': 0.001,
       'elements': ['Al', 'H'],

       'embeddings': {'ALL': {'drho_core_cut': 250,
                              'fs_parameters': [1, 1],
                              'ndensity': 1,
                              'npot': 'FinnisSinclair',
                              'rho_core_cut': 200000},
                      'Al': {'drho_core_cut': 250,
                             'fs_parameters': [1, 1, 1, 0.5],
                             'ndensity': 2,
                             'npot': 'FinnisSinclairShiftedScaled',
                             'rho_core_cut': 200000}

                      },

       'bonds': {'ALL': {'NameOfCutoffFunction': 'cos',
                         'core-repulsion': [10000.0, 5.0],
                         'dcut': 0.01,
                         'radbase': 'ChebPow',
                         'radparameters': [2.0],
                         'rcut': 3.9},

                 ('Al', 'H'): {'NameOfCutoffFunction': 'cos',
                               'core-repulsion': [10000.0, 5.0],
                               'dcut': 0.01,
                               'radbase': 'ChebExpCos',
                               'radparameters': [2.0],
                               'rcut': 3.5},
                 },

       'functions': {
           'UNARY': {
               'nradmax_by_orders': [5, 2, 2],
               'lmax_by_orders': [0, 1, 1]
           },

           'BINARY': {
               'nradmax_by_orders': [5, 1, 1],
               'lmax_by_orders': [0, 1, 1],
           },
       }
   }

   :param potential_config: potential configuration dictionary, see above
   :return: blocks_specifications_dict
   """

    ### Embeddings
    ### possible keywords: ALL, UNARY  + el
    if "embeddings" in potential_config:
        embeddings_ext = generate_embeddings_ext(potential_config)
    else:
        embeddings_ext = {}
    ### Bonds
    ### possible keywords: ALL, UNARY, BINARY + (el,el)
    if "bonds" in potential_config:
        bonds_ext = generate_bonds_ext(potential_config)
    else:
        bonds_ext = {}
    ### Functions
    ### possible keywords: ALL, UNARY, BINARY, TERNARY, QUATERNARY, QUINARY + (el,el,...)
    if "functions" in potential_config:
        functions_ext = generate_functions_ext(potential_config)
    else:
        functions_ext = {}
    ### Update bonds specifications according to maximum observable nmax, lmax, nradbasemax in functions specifications
    bonds_ext = update_bonds_ext(bonds_ext, functions_ext)
    ### Combine together to have block_spec specs
    block_spec_dict = deepcopy(functions_ext)
    # update with embedding info
    for key, emb_ext_val in embeddings_ext.items():
        if key in block_spec_dict:
            block_spec_dict[key].update(emb_ext_val)
    # update with bond info
    for key, bonds_ext_val in bonds_ext.items():
        if len(set(key)) == 1:
            key = (key[0],)
        if key in block_spec_dict:
            block_spec_dict[key].update(bonds_ext_val)
    return block_spec_dict


def generate_functions_ext(potential_config):
    elements = potential_config["elements"]
    elements = sorted(elements)

    functions = potential_config["functions"].copy()
    functions_ext = defaultdict(dict)

    if ALL in functions:
        all_species_keys = generate_all_species_keys(elements)
        for key in all_species_keys:
            functions_ext[key].update(functions[ALL])
    for nary_key, nary_val in NARY_MAP.items():
        if nary_key in functions:
            for key in generate_species_keys(elements, r=nary_val):
                functions_ext[key].update(functions[nary_key])
    for k in functions:
        if k not in KEYWORDS:
            if isinstance(k, str):  # single species string
                key = tuple(element_patt.findall(k))
            else:
                key = tuple(k)
            # TODO extend permutations
            functions_ext[key].update(functions[k])

    # drop all keys, that has no specifications
    functions_ext = {k: v for k, v in functions_ext.items() if len(v) > 0}

    return functions_ext


def generate_bonds_ext(potential_config):
    elements = potential_config["elements"]
    elements = sorted(elements)

    bonds = potential_config["bonds"].copy()
    bonds_ext = {pair: {} for pair in product(elements, repeat=2)}
    if ALL in bonds:
        for pair in bonds_ext:
            bonds_ext[pair].update(bonds[ALL])
    if UNARY in bonds:
        for el in elements:
            bonds_ext[(el, el)].update(bonds[UNARY])
    if BINARY in bonds:
        for pair in permutations(elements, 2):
            bonds_ext[pair].update(bonds[BINARY])
    for pair in bonds:
        if pair not in KEYWORDS:  # assume that pair is valid (el1, el2)
            if isinstance(pair, str):
                # dpair= (pair, pair)
                # use regex to
                dpair = tuple(element_patt.findall(pair))
                if len(dpair) == 1:
                    dpair = (dpair[0], dpair[0])
            else:
                dpair = pair
            bonds_ext[dpair].update(bonds[pair])
            r_pair = tuple(reversed(dpair))
            bonds_ext[r_pair].update(bonds[pair])
    # drop all keys, that has no specifications
    bonds_ext = {k: v for k, v in bonds_ext.items() if len(v) > 0}
    return bonds_ext


def generate_embeddings_ext(potential_config):
    elements = potential_config["elements"]
    elements = sorted(elements)

    embeddings = potential_config["embeddings"].copy()
    embeddings_ext = {(el,): {} for el in elements}
    # ALL and UNARY behave identically
    if ALL in embeddings:
        for el in elements:
            embeddings_ext[(el,)].update(embeddings[ALL])
    if UNARY in embeddings:
        for el in elements:
            embeddings_ext[(el,)].update(embeddings[UNARY])
    for el, val in embeddings.items():
        if el in elements:
            embeddings_ext[(el,)].update(val)
        elif el not in [ALL, UNARY]:
            raise ValueError(f"{el} is not in specified elements: {elements}")

    # drop all keys, that has no specifications
    embeddings_ext = {k: v for k, v in embeddings_ext.items() if len(v) > 0}
    return embeddings_ext


def update_bonds_ext(bonds_ext, functions_ext):
    # if bonds_ext is empty - return it as it is
    if not bonds_ext:
        return bonds_ext

    bonds_ext_updated = deepcopy(bonds_ext)
    # run through functions specifications and update/validate bond's nradbase, nradmax, lmax
    for key, funcs_spec in functions_ext.items():
        nradbasemax = max(funcs_spec[ORDERS_NRADMAX_KW][:1])
        if len(funcs_spec[ORDERS_NRADMAX_KW][1:]) > 0:
            nradmax = max(funcs_spec[ORDERS_NRADMAX_KW][1:])
        else:
            nradmax = 0
        lmax = max(funcs_spec[ORDERS_LMAX_KW])

        for bkey in species_key_to_bonds(key):

            bond = bonds_ext[bkey]

            if POTENTIAL_NRADBASE_KW not in bond:
                if bonds_ext_updated[bkey].get(POTENTIAL_NRADBASE_KW, 0) < nradbasemax:
                    bonds_ext_updated[bkey][POTENTIAL_NRADBASE_KW] = nradbasemax
            else:
                if bond[POTENTIAL_NRADBASE_KW] < nradbasemax:
                    raise ValueError(f"Given `{POTENTIAL_NRADBASE_KW}`={bond[POTENTIAL_NRADBASE_KW]} for bond {bkey} " + \
                                     f"is less than nradbasemax={nradbasemax} from {key}")

            if POTENTIAL_NRADMAX_KW not in bond:
                if bonds_ext_updated[bkey].get(POTENTIAL_NRADMAX_KW, 0) < nradmax:
                    bonds_ext_updated[bkey][POTENTIAL_NRADMAX_KW] = nradmax
            else:
                if bond[POTENTIAL_NRADMAX_KW] < nradmax:
                    raise ValueError(f"Given `{POTENTIAL_NRADMAX_KW}`={bond[POTENTIAL_NRADMAX_KW]} for bond {bkey} " + \
                                     f"is less than nradmax={nradmax} from {key}")

            if POTENTIAL_LMAX_KW not in bond:
                if bonds_ext_updated[bkey].get(POTENTIAL_LMAX_KW, 0) < lmax:
                    bonds_ext_updated[bkey][POTENTIAL_LMAX_KW] = lmax
            else:
                if bond[POTENTIAL_LMAX_KW] < lmax:
                    raise ValueError(f"Given `{POTENTIAL_LMAX_KW}`={bond[POTENTIAL_LMAX_KW]} for bond {bkey} " + \
                                     f"is less than nradmax={lmax} from {key}")
    return bonds_ext_updated


def create_multispecies_basisblocks_list(potential_config: Dict,
                                         element_ndensity_dict: Dict = None,
                                         func_coefs_initializer="zero",
                                         unif_mus_ns_to_lsLScomb_dict=None,
                                         verbose=False) -> List[BBasisFunctionsSpecificationBlock]:
    blocks_specifications_dict = generate_blocks_specifications_dict(potential_config)

    if unif_mus_ns_to_lsLScomb_dict is None:
        with open(default_mus_ns_uni_to_rawlsLS_np_rank_filename, "rb") as f:
            unif_mus_ns_to_lsLScomb_dict = pickle.load(f)

    element_ndensity_dict =  element_ndensity_dict or {}
    constr_element_ndensity_dict = get_element_ndensity_dict(blocks_specifications_dict)
    for k, v in constr_element_ndensity_dict.items():
        if k not in element_ndensity_dict:
            element_ndensity_dict[k] = v
    if not element_ndensity_dict:
        raise ValueError("`element_ndensity_dict` neither provided nor constructed")

    blocks_list = []
    for elements_vec, block_spec_dict in blocks_specifications_dict.items():
        if verbose:
            print("Block elements:", elements_vec)

        ndensity = element_ndensity_dict[elements_vec[0]]
        spec_block = create_species_block(elements_vec, block_spec_dict, ndensity,
                                          func_coefs_initializer, unif_mus_ns_to_lsLScomb_dict)
        if verbose:
            print(len(spec_block.funcspecs), " functions added")
        blocks_list.append(spec_block)
    return blocks_list


def create_species_block(elements_vec: List, block_spec_dict: Dict,
                         ndensity: int,
                         func_coefs_initializer="zero",
                         unif_mus_ns_to_lsLScomb_dict=None) -> BBasisFunctionsSpecificationBlock:
    central_atom = elements_vec[0]

    elms = tuple(sorted(set(elements_vec)))
    nary = len(elms)
    spec_block = create_species_block_without_funcs(elements_vec, block_spec_dict)
    current_block_func_spec_list = []
    if "nradmax_by_orders" in block_spec_dict and "lmax_by_orders" in block_spec_dict:
        max_rank = len(block_spec_dict["nradmax_by_orders"])
        unif_abs_combs_set = set()
        for rank, nmax, lmax in zip(range(1, max_rank + 1),
                                    block_spec_dict["nradmax_by_orders"],
                                    block_spec_dict["lmax_by_orders"]):

            ns_range = range(1, nmax + 1)

            for mus_comb in combinations_with_replacement(elms, rank):
                mus_comb_ext = tuple([central_atom] + list(mus_comb))  # central atom + ordered tail
                current_nary = len(set(mus_comb_ext))
                if current_nary != nary:
                    continue

                for ns_comb in product(ns_range, repeat=rank):  # exhaustive list
                    unif_abs_comb = unify_absolute_mus_ns_comb(mus_comb, ns_comb)
                    if unif_abs_comb in unif_abs_combs_set:
                        continue
                    unif_abs_combs_set.add(unif_abs_comb)
                    unif_comb = unify_mus_ns_comb(mus_comb, ns_comb)
                    if unif_comb not in unif_mus_ns_to_lsLScomb_dict:
                        raise ValueError(
                            "Specified potential shape is too big " + \
                            "and goes beyond the precomputed BBasisFunc white-list" + \
                            "for unified combination {}".format(unif_comb))

                    mus_ns_white_list = unif_mus_ns_to_lsLScomb_dict[unif_comb]  # only ls, LS are important
                    for (pre_ls, pre_LS) in mus_ns_white_list:
                        if max(pre_ls) <= lmax:
                            if "coefs_init" in block_spec_dict:
                                func_coefs_initializer = block_spec_dict["coefs_init"]

                            if func_coefs_initializer == "zero":
                                coefs = [0] * ndensity
                            elif func_coefs_initializer == "random":
                                coefs = np.random.randn(ndensity) * 1e-4
                            else:
                                raise ValueError(
                                    "Unknown func_coefs_initializer={}. Could be only 'zero' or 'random'".format(
                                        func_coefs_initializer))

                            new_spec = BBasisFunctionSpecification(elements=mus_comb_ext,
                                                                   ns=ns_comb,
                                                                   ls=pre_ls,
                                                                   LS=pre_LS,
                                                                   coeffs=coefs
                                                                   )

                            current_block_func_spec_list.append(new_spec)
        spec_block.funcspecs = current_block_func_spec_list
    return spec_block


def single_to_multispecies_converter(potential_config):
    new_multi_species_potential_config = {}

    if "deltaSplineBins" in potential_config:
        new_multi_species_potential_config["deltaSplineBins"] = potential_config["deltaSplineBins"]
    element = potential_config["element"]
    new_multi_species_potential_config["elements"] = [element]

    embeddings = {}
    embeddings_kw_list = ["npot", "fs_parameters", "ndensity", "rho_core_cut", "drho_core_cut"]

    for kw in embeddings_kw_list:
        if kw in potential_config:
            embeddings[kw] = potential_config[kw]

    new_multi_species_potential_config["embeddings"] = {element: embeddings}

    bonds = {}
    bonds_kw_list = ["NameOfCutoffFunction",
                     "core-repulsion",
                     "dcut",
                     "rcut",
                     "radbase",
                     "radparameters"]
    for kw in bonds_kw_list:
        if kw in potential_config:
            bonds[kw] = potential_config[kw]

    new_multi_species_potential_config["bonds"] = {element: bonds}

    functions = {}
    functions_kw_list = ["nradmax_by_orders", "lmax_by_orders", ]
    for kw in functions_kw_list:
        if kw in potential_config:
            functions[kw] = potential_config[kw]
    if "func_coefs_init" in potential_config:
        functions["coefs_init"] = potential_config["func_coefs_init"]

    new_multi_species_potential_config["functions"] = {element: functions}

    return new_multi_species_potential_config


def tail_sort(combs):
    return tuple(list(combs[:1]) + sorted(combs[1:]))


def species_tail_sorted_permutation(elements, r):
    combs = set()
    for comb in list(permutations(elements, r)):
        comb = tail_sort(comb)
        combs.add(comb)
    return tuple(sorted(combs))


def expand_trainable_parameters(elements: list, trainable_parameters: Union[str, list, dict] = None) -> dict:
    if trainable_parameters is None:
        trainable_parameters = []

    if isinstance(trainable_parameters, str):
        if trainable_parameters in ["func", "radial"]:
            trainable_parameters = {"ALL": trainable_parameters}
        else:
            trainable_parameters = [trainable_parameters]

    if len(trainable_parameters) == 0:
        trainable_parameters = [ALL]

    is_dict_format = isinstance(trainable_parameters, dict)

    nelements = len(elements)
    DEFAULT_PARAMS = ["func", "radial"]

    # if trainable_parameters is list -> options=["radial","func"]
    new_trainable_parameters_dict = {}

    # check ALL keyword
    if ALL in trainable_parameters:
        params = trainable_parameters[ALL] if is_dict_format else DEFAULT_PARAMS

        # wrap pure str into list
        if isinstance(params, str):
            params = [params]

        if params == ["all"]:
            params = DEFAULT_PARAMS

        for r in range(1, nelements + 1):
            for comb in species_tail_sorted_permutation(elements, r):
                new_trainable_parameters_dict[comb] = params

    # check NARY's keywords
    for kw, r in NARY_MAP.items():
        if kw in trainable_parameters:
            params = trainable_parameters[kw] if is_dict_format else DEFAULT_PARAMS
            # wrap pure str nto list
            if isinstance(params, str):
                params = [params]
            if params == ["all"]:
                params = DEFAULT_PARAMS
            for comb in species_tail_sorted_permutation(elements, r):
                new_trainable_parameters_dict[comb] = params

    # check exact combinations
    for comb in trainable_parameters:
        if comb not in KEYWORDS:
            # translate str -> to element tuple using regex
            if isinstance(comb, str):
                ext_comb = tuple(element_patt.findall(comb))
            else:
                ext_comb = comb

            params = trainable_parameters[comb] if is_dict_format else DEFAULT_PARAMS
            # wrap pure str nto list
            if isinstance(params, str):
                params = [params]
            if params == ["all"]:
                params = DEFAULT_PARAMS
            new_trainable_parameters_dict[ext_comb] = params

    # clear "radial" from r>2
    new_trainable_parameters_dict_cleared = {}

    for comb, params in new_trainable_parameters_dict.items():
        params = params.copy()
        r = len(comb)
        if r > 2 and "radial" in params:
            params.remove("radial")
        elif r == 2:  # check the bonds symmetry
            inv_comb = tuple(reversed(comb))
            if ("radial" in comb) != ("radial" in inv_comb):
                raise ValueError(
                    "Inconsisteny setup of 'radial' parameters trainability for {} and {}:".format(comb, inv_comb) +
                    "This option should be identical"
                )

        new_trainable_parameters_dict_cleared[comb] = params

    return new_trainable_parameters_dict_cleared


def compute_bbasisset_train_mask(bbasisconf: Union[BBasisConfiguration, ACEBBasisSet],
                                 extended_trainable_parameters_dict: dict):
    if isinstance(bbasisconf, BBasisConfiguration):
        bbasis_set = ACEBBasisSet(bbasisconf)
    elif isinstance(bbasisconf, ACEBBasisSet):
        bbasis_set = bbasisconf

    elements_to_ind_map = bbasis_set.elements_to_index_map

    ind_trainable_parameters_dict = {tuple(elements_to_ind_map[el] for el in k): v for k, v in
                                     extended_trainable_parameters_dict.items()}
    crad_train_mask = np.zeros(len(bbasis_set.crad_coeffs_mask), dtype=bool)
    basis_train_mask = np.zeros(len(bbasis_set.basis_coeffs_mask), dtype=bool)

    crad_coeffs_mask = [tuple(c) for c in bbasis_set.crad_coeffs_mask]
    basis_coeffs_mask = [tuple(c) for c in bbasis_set.basis_coeffs_mask]

    for ind_comb, params in ind_trainable_parameters_dict.items():
        if "radial" in params:
            crad_train_mask = np.logical_or(crad_train_mask, [c == ind_comb for c in crad_coeffs_mask])
        if "func" in params:
            basis_train_mask = np.logical_or(basis_train_mask, [c == ind_comb for c in basis_coeffs_mask])

    total_train_mask = np.concatenate((crad_train_mask, basis_train_mask))
    return total_train_mask


def is_mult_basisfunc_equivalent(func1: BBasisFunctionSpecification, func2: BBasisFunctionSpecification) -> bool:
    return (func1.elements == func2.elements) and \
           (func1.ns == func2.ns) and \
           (func1.ls == func2.ls) and \
           (func1.LS == func2.LS)


class BlockBasisFunctionsList:

    def __init__(self, block: BBasisFunctionsSpecificationBlock):
        self.funcs = block.funcspecs

    def find_existing(self, func: BBasisFunctionSpecification) -> bool:
        for other_func in self.funcs:
            if is_mult_basisfunc_equivalent(func, other_func):
                return other_func
        return None


def extend_basis_block(init_block: BBasisFunctionsSpecificationBlock,
                       final_block: BBasisFunctionsSpecificationBlock,
                       num_funcs=None,
                       ladder_type="body_order") -> Tuple[BBasisFunctionsSpecificationBlock, bool]:
    # check that block name is identical
    assert init_block.block_name == final_block.block_name, ValueError("Could not extend block '' to new block ''". \
                                                                       format(init_block.block_name,
                                                                              final_block.block_name
                                                                              ))

    nelements = init_block.number_of_species

    init_block_list = BlockBasisFunctionsList(init_block)
    final_basis_funcs = sort_funcspecs_list(final_block.funcspecs, ladder_type)

    # existing_funcs_list = []
    existing_funcs_list = init_block.funcspecs

    new_funcs_list = []
    for new_func in final_basis_funcs:
        existing_func = init_block_list.find_existing(new_func)
        if existing_func is None:
            #             print("Func ", new_func, " added")
            new_funcs_list.append(new_func)

    if num_funcs is not None and len(new_funcs_list) > num_funcs:
        new_funcs_list = new_funcs_list[:num_funcs]
    else:
        new_funcs_list = new_funcs_list  # use all new funcs

    extended_block = init_block.copy()

    # if no new functions to add, return init_block
    if len(new_funcs_list) == 0:
        return extended_block, False

    extended_func_list = sort_funcspecs_list(existing_funcs_list + new_funcs_list, "body_order")

    # Update crad only for nelements<=2
    if nelements <= 2:
        validate_radial_shape_from_funcs(extended_block, extended_func_list)
        initialize_block_crad(extended_block)

        extended_radcoeffs = np.array(extended_block.radcoefficients)
        init_radcoeffs = np.array(init_block.radcoefficients)
        merge_crad_matrix(extended_radcoeffs, init_radcoeffs)

        extended_block.radcoefficients = extended_radcoeffs

    extended_block.funcspecs = extended_func_list

    # core-repulsion translating from final_basis

    if nelements <= 2:
        extended_block.core_rep_parameters = final_block.core_rep_parameters

    if nelements == 1:
        extended_block.rho_cut = final_block.rho_cut
        extended_block.drho_cut = final_block.drho_cut

    return extended_block, True


def merge_crad_matrix(extended_radcoeffs, init_radcoeffs):
    init_radcoeffs = np.array(init_radcoeffs)
    if len(init_radcoeffs.shape) == 3:
        common_shape = [min(s1, s2) for s1, s2 in zip(np.shape(init_radcoeffs), np.shape(extended_radcoeffs))]
        if len(common_shape) == 3:
            extended_radcoeffs[:common_shape[0], :common_shape[1], :common_shape[2]] = \
                init_radcoeffs[:common_shape[0], :common_shape[1], :common_shape[2]]


def initialize_block_crad(extended_block, crad_init="delta"):
    init_radcoeffs = np.array(extended_block.radcoefficients)
    if len(init_radcoeffs.shape) == 3:
        new_nradmax = max(extended_block.nradmaxi, init_radcoeffs.shape[0])
        new_lmax = max(extended_block.lmaxi, init_radcoeffs.shape[1] - 1)
        new_nradbase = max(extended_block.nradbaseij, init_radcoeffs.shape[2])
    else:
        new_nradbase = extended_block.nradbaseij
        new_lmax = extended_block.lmaxi
        new_nradmax = extended_block.nradmaxi

    if crad_init == "delta":
        extended_radcoeffs = np.zeros((new_nradmax, new_lmax + 1, new_nradbase))
        for n in range(min(new_nradmax, new_nradbase)):
            extended_radcoeffs[n, :, n] = 1.
    elif crad_init == "zero":
        extended_radcoeffs = np.zeros((new_nradmax, new_lmax + 1, new_nradbase))
    elif crad_init == "random":
        extended_radcoeffs = np.random.randn(*(new_nradmax, new_lmax + 1, new_nradbase))
    else:
        raise ValueError("Unknown value for crad_init ({}). Use delta, zero or random".format(crad_init))

    merge_crad_matrix(extended_radcoeffs, init_radcoeffs)

    extended_block.nradmaxi = new_nradmax
    extended_block.lmaxi = new_lmax
    extended_block.nradbaseij = new_nradbase

    extended_block.radcoefficients = extended_radcoeffs


def validate_radial_shape_from_funcs(extended_block, func_list=None):
    new_nradmax = 0
    new_nradbase = 0
    new_lmax = 0
    if func_list is None:
        func_list = extended_block.funcspecs
    for func in func_list:
        rank = len(func.ns)
        if rank == 1:
            new_nradbase = max(max(func.ns), new_nradbase)
        else:
            new_nradmax = max(max(func.ns), new_nradmax)
        new_lmax = max(max(func.ls), new_lmax)
    extended_block.nradbaseij = new_nradbase
    extended_block.lmaxi = new_lmax
    extended_block.nradmaxi = new_nradmax


def validate_bonds_nradmax_lmax_nradbase(ext_basis: BBasisConfiguration):
    """
    Check the nradbase, lmax and nradmax over all bonds in all species blocks
    """
    ext_blocks_dict = {block.block_name: block for block in ext_basis.funcspecs_blocks}

    max_nlk_dict = defaultdict(lambda: defaultdict(int))

    for block_name, block in ext_blocks_dict.items():
        for f in block.funcspecs:
            rank = len(f.ns)
            mu0 = f.elements[0]
            mus = f.elements[1:]
            ns = f.ns
            ls = f.ls

            for mu, n, l in zip(mus, ns, ls):
                bond = (mu0, mu)

                if rank == 1:
                    max_nlk_dict[bond]["nradbase"] = max(max_nlk_dict[bond]["nradbase"], n)
                else:
                    max_nlk_dict[bond]["nradmax"] = max(max_nlk_dict[bond]["nradmax"], n)

                max_nlk_dict[bond]["lmax"] = max(max_nlk_dict[bond]["lmax"], l)

    # loop over max_nlk_dict and symmetrize pair bonds
    for bond_pair, dct in max_nlk_dict.items():
        if len(bond_pair) == 2:
            sym_bond_pair = (bond_pair[1], bond_pair[0])
            sym_dct = max_nlk_dict[sym_bond_pair]
            max_nradbase = max(dct["nradbase"], sym_dct["nradbase"])
            max_lmax = max(dct["lmax"], sym_dct["lmax"])
            max_nradmax = max(dct["nradmax"], sym_dct["nradmax"])

            max_nlk_dict[bond_pair]["nradbase"] = max_nlk_dict[sym_bond_pair]["nradbase"] = max_nradbase
            max_nlk_dict[bond_pair]["lmax"] = max_nlk_dict[sym_bond_pair]["lmax"] = max_lmax
            max_nlk_dict[bond_pair]["nradmax"] = max_nlk_dict[sym_bond_pair]["nradmax"] = max_nradmax

    bonds_dict = {}
    for k, v in max_nlk_dict.items():
        if k[0] == k[1]:
            bonds_dict[k[0]] = v
        else:
            bonds_dict[" ".join(k)] = v

    for block in ext_basis.funcspecs_blocks:
        k = block.block_name

        # skip more ternary and higher blocks, because bond specification are defined only in unary/binary blocks
        if len(k.split()) > 2:
            block.radcoefficients = []
            block.nradbaseij = 0
            block.lmaxi = 0
            block.nradmaxi = 0
            continue

        if block.nradbaseij < bonds_dict[k]["nradbase"]:
            block.nradbaseij = bonds_dict[k]["nradbase"]

        if block.nradmaxi < bonds_dict[k]["nradmax"]:
            block.nradmaxi = bonds_dict[k]["nradmax"]

        if block.lmaxi < bonds_dict[k]["lmax"]:
            block.lmaxi = bonds_dict[k]["lmax"]

        initialize_block_crad(block)


def extend_multispecies_basis(initial_basis: BBasisConfiguration,
                              final_basis: BBasisConfiguration,
                              ladder_type="body_order",
                              num_funcs=None,
                              return_is_extended=False
                              ) -> Tuple[BBasisConfiguration, bool]:
    if num_funcs == 0:
        if return_is_extended:
            return initial_basis, False
        else:
            return initial_basis

    # create a dict of block_name -> block for initial and final configs
    initial_blocks_dict = {block.block_name: block for block in initial_basis.funcspecs_blocks}
    final_blocks_dict = {block.block_name: block for block in final_basis.funcspecs_blocks}

    # initialize accumulator lists
    is_extended_list = []
    extended_block_list = []

    # 1. loop over final expected blocks
    for f_block_name, f_block in final_blocks_dict.items():
        # if such block exists in initial blocks
        if f_block_name in initial_blocks_dict:
            # take it
            i_block = initial_blocks_dict[f_block_name]
        else:
            # otherwise need to create empty block as for f_block
            i_block = f_block.copy()
            # remove funcs and crad
            i_block.funcspecs = []
            i_block.radcoefficients = []
            # set nradbase, lmax, nradmax to zero
            i_block.lmaxi = 0
            i_block.nradbaseij = 0
            i_block.nradmaxi = 0

        # growth from i_block to f_block
        ext_block, is_extended = extend_basis_block(i_block, f_block, num_funcs, ladder_type)
        #         print(ext_block.block_name,":",is_extended)
        is_extended_list.append(is_extended)
        extended_block_list.append(ext_block)

    # check that all blocks in initial basis are copied into extended basis
    for i_block_name, i_block in initial_blocks_dict.items():
        if i_block_name not in final_blocks_dict:
            extended_block_list.append(i_block)

    is_basis_extended = np.any(is_extended_list)
    extended_basis = final_basis.copy()
    extended_basis.funcspecs_blocks = extended_block_list
    validate_bonds_nradmax_lmax_nradbase(extended_basis)
    extended_basis.validate(True)
    if return_is_extended:
        return extended_basis, is_basis_extended
    else:
        return extended_basis
