import logging
from collections import defaultdict

import numpy as np
import pandas as pd
from typing import List, Dict, Union

from pyace.basis import ACEBBasisSet, BBasisConfiguration, BBasisFunctionsSpecificationBlock, \
    BBasisFunctionSpecification
from pyace.const import *

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


def get_actual_ladder_step(ladder_step_param: Union[int, float, List],
                           current_number_of_funcs: int,
                           final_number_of_funcs: int) -> int:
    ladder_discrete_step: int = 0
    ladder_frac: float = 0.0

    if isinstance(ladder_step_param, int) and ladder_step_param >= 1:
        ladder_discrete_step = int(ladder_step_param)
    elif isinstance(ladder_step_param, float) and 1. > ladder_step_param > 0:
        ladder_frac = float(ladder_step_param)
    elif isinstance(ladder_step_param, (list, tuple)):
        if len(ladder_step_param) > 2:
            raise val_exc
        for p in ladder_step_param:
            if p >= 1:
                ladder_discrete_step = int(p)
            elif 0 < p < 1:
                ladder_frac = float(p)
            else:
                raise ValueError(
                    "Invalid ladder step parameter: {}. Should be integer >= 1 or  0<float<1 or list of both [int, float]".format(
                        ladder_step_param))
    elif ladder_step_param is None:
        ladder_discrete_step = final_number_of_funcs - current_number_of_funcs
        log.info('Ladder step parameter is None - all functions will be added')
    else:
        raise ValueError(
            "Invalid ladder step parameter: {}. Should be integer >= 1 or  0<float<1 or list of both [int, float]".format(
                ladder_step_param))

    ladder_frac_step = int(round(ladder_frac * current_number_of_funcs))
    ladder_step = max(ladder_discrete_step, ladder_frac_step, 1)
    log.info(
        'Possible ladder steps: discrete - {}, fraction - {}. Selected maximum - {}'.format(ladder_discrete_step,
                                                                                            ladder_frac_step,
                                                                                            ladder_step))

    if current_number_of_funcs + ladder_step > final_number_of_funcs:
        ladder_step = final_number_of_funcs - current_number_of_funcs
        log.info("Ladder step is too large and adjusted to {}".format(ladder_step))

    return ladder_step


def construct_bbasisconfiguration(potential_config: Dict,
                                  initial_basisconfig: BBasisConfiguration = None,
                                  overwrite_blocks_from_initial_bbasis=False) -> BBasisConfiguration:
    from pyace.multispecies_basisextension import single_to_multispecies_converter, create_multispecies_basis_config
    # for backward compatibility with pacemaker 1.0 potential_dict format
    check_backward_compatible_parameters(potential_config)

    # if old-single specie format
    if not ("elements" in potential_config and  # "embeddings" in potential_config and
            "bonds" in potential_config and "functions" in potential_config):
        # convert to new general multispecies format
        potential_config = single_to_multispecies_converter(potential_config)

    return create_multispecies_basis_config(potential_config, initial_basisconfig=initial_basisconfig,
                                            overwrite_blocks_from_initial_bbasis=overwrite_blocks_from_initial_bbasis)


def sort_funcspecs_list(lst: List[BBasisFunctionSpecification], ladder_type: str) -> List[BBasisFunctionSpecification]:
    if ladder_type == 'power_order':
        return list(sorted(lst, key=lambda func: len(func.ns) + sum(func.ns) + sum(func.ls)))
    elif ladder_type == 'body_order':
        return list(
            sorted(lst, key=lambda func: (
                len(tuple(func.ns)), tuple(func.ns), tuple(func.ls), tuple(func.LS), tuple(func.elements))))
    else:
        raise ValueError('Specified Ladder type ({}) is not implemented'.format(ladder_type))


def extend_basis(initial_basis: BBasisConfiguration, final_basis: BBasisConfiguration,
                 ladder_type: str, func_step: int = None, return_is_extended=False) -> BBasisConfiguration:
    # TODO: move from here, optimize import
    from pyace.multispecies_basisextension import extend_multispecies_basis
    return extend_multispecies_basis(initial_basis, final_basis, ladder_type, func_step,
                                     return_is_extended=return_is_extended)


def check_backward_compatible_parameters(potential_config: Dict):
    # "nradmax" -> "nradmax_by_orders"
    # "lmax" -> "lmax_by_orders"

    if POTENTIAL_NRADMAX_KW in potential_config:
        log.warn("potential_config:'{}' is deprecated parameter, please use '{}'".format(POTENTIAL_NRADMAX_KW,
                                                                                         ORDERS_NRADMAX_KW))
        potential_config[ORDERS_NRADMAX_KW] = potential_config[POTENTIAL_NRADMAX_KW]

    if POTENTIAL_LMAX_KW in potential_config:
        log.warn(
            "potential_config:'{}' is deprecated parameter, please use '{}'".format(POTENTIAL_LMAX_KW, ORDERS_LMAX_KW))
        potential_config[ORDERS_LMAX_KW] = potential_config[POTENTIAL_LMAX_KW]
