import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    # datefmt='%Y-%m-%d %H:%M:%S.%f'
                    )
logging.getLogger().setLevel(logging.INFO)

from pyace.asecalc import PyACECalculator, PyACEEnsembleCalculator

from pyace.atomicenvironment import ACEAtomicEnvironment, create_cube, create_linear_chain, \
    aseatoms_to_atomicenvironment

from pyace.basis import BBasisFunctionSpecification, BBasisConfiguration, BBasisFunctionsSpecificationBlock, \
    ACEBBasisFunction, ACECTildeBasisFunction, ACERadialFunctions, ACECTildeBasisSet, ACEBBasisSet, Fexp
from pyace.calculator import ACECalculator
from pyace.coupling import ACECouplingTree, generate_ms_cg_list, validate_ls_LS, is_valid_ls_LS, expand_ls_LS
from pyace.evaluator import ACECTildeEvaluator, ACEBEvaluator,get_ace_evaluator_version
from pyace.pyacefit import PyACEFit
from pyace.preparedata import *
from pyace.radial import RadialFunctionsValues, RadialFunctionsVisualization, RadialFunctionSmoothness

from pyace.multispecies_basisextension import create_multispecies_basis_config

__all__ = ["ACEAtomicEnvironment", "create_cube", "create_linear_chain", "aseatoms_to_atomicenvironment",
           "BBasisFunctionSpecification", "BBasisConfiguration", "BBasisFunctionsSpecificationBlock",
           "ACEBBasisFunction",
           "ACECTildeBasisFunction", "ACERadialFunctions", "ACECTildeBasisSet", "ACEBBasisSet",
           "ACECalculator",
           "ACECouplingTree", "generate_ms_cg_list", "validate_ls_LS", "is_valid_ls_LS", 'expand_ls_LS',
           "ACECTildeEvaluator", "ACEBEvaluator",
           "PyACEFit", "PyACECalculator", "PyACEEnsembleCalculator",
           "StructuresDatasetSpecification", "EnergyBasedWeightingPolicy", "Fexp",
           "get_ace_evaluator_version",
           "EnergyBasedWeightingPolicy", "UniformWeightingPolicy",
           "RadialFunctionsValues", "RadialFunctionsVisualization", "RadialFunctionSmoothness",
           "StructuresDatasetSpecification",

           "create_multispecies_basis_config"
           ]

from . import _version
__version__ = _version.get_versions()['version']
