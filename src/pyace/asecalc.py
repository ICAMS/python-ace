import numpy as np
import time

from ase.calculators.calculator import Calculator, all_changes
from ase.io import write

from pyace.atomicenvironment import aseatoms_to_atomicenvironment_old, aseatoms_to_atomicenvironment
from pyace.basis import ACEBBasisSet, ACECTildeBasisSet, BBasisConfiguration
from pyace.calculator import ACECalculator
from pyace.catomicenvironment import ACEAtomicEnvironment
from pyace.evaluator import ACEBEvaluator, ACECTildeEvaluator, ACERecursiveEvaluator

from pyace.grace_fs import GRACEFSBasisSet, GRACEFSBEvaluator, GRACEFSCalculator


class PyACECalculator(Calculator):
    """
    PyACE ASE calculator
    :param basis_set - specification of ACE potential, could be in following forms:
                      ".ace" potential filename
                      ".yaml" potential filename
                      ACEBBasisSet object
                      ACECTildeBasisSet object
                      BBasisConfiguration object
    """
    implemented_properties = ['energy', 'forces', 'stress', 'energies', 'free_energy']

    def __init__(self, basis_set, **kwargs):
        """
PyACE ASE calculator
:param basis_set - specification of ACE potential, could be in following forms:
                  ".ace" potential filename
                  ".yaml" potential filename
                  ACEBBasisSet object
                  ACECTildeBasisSet object
                  BBasisConfiguration object
:param recursive_evaluator (default: False)
:param recursive (default: False)
:param keep_extrapolative_structures (default: False)
:param dump_extrapolative_structures (default: False)
:param gamma_lower_bound (default: 1.1)
:param gamma_upper_bound (default: 5.0)
:param stop_at_large_extrapolation (default: False)
:param verbose (default: 0).   0 - no extrapolation output,
                               1 - output for upper-bound extrapolation
                               2 - output for upper- and lower-bound extrapolation
                               3 - output for all interpolation/extrapolation calculations
"""
        if "recursive_evaluator" not in kwargs:
            kwargs["recursive_evaluator"] = False
        if "recursive" not in kwargs:
            kwargs["recursive"] = False
        if "fast_nl" not in kwargs:
            kwargs["fast_nl"] = True
        if "gamma_lower_bound" not in kwargs:
            kwargs["gamma_lower_bound"] = 1.1
        if "gamma_upper_bound" not in kwargs:
            kwargs["gamma_upper_bound"] = 5.0
        if "stop_at_large_extrapolation" not in kwargs:
            kwargs["stop_at_large_extrapolation"] = False
        if "dump_extrapolative_structures" not in kwargs:
            kwargs["dump_extrapolative_structures"] = False
        if "keep_extrapolative_structures" not in kwargs:
            kwargs["keep_extrapolative_structures"] = False

        if "verbose" not in kwargs:
            kwargs["verbose"] = 0

        Calculator.__init__(self, basis_set=basis_set, **kwargs)
        self.nl = None
        self.skin = 0.
        # self.reset_nl = True  # Set to False for MD simulations
        self.ae = ACEAtomicEnvironment()

        self._create_evaluator()

        self.cutoff = self.basis.cutoffmax  # self.parameters.basis_config.funcspecs_blocks[0].rcutij

        self.energy = None
        self.energies = None
        self.forces = None
        self.virial = None
        self.stress = None
        self.projections = None
        self.current_extrapolation_structure_index = 0
        self.is_active_set_configured = False
        self.extrapolative_structures_list = []
        self.extrapolative_structures_gamma = []
        self.ace = ACECalculator()
        self.ace.set_evaluator(self.evaluator)
        self.compute_projections = True

    def _create_evaluator(self):

        basis_set = self.parameters.basis_set
        if isinstance(basis_set, BBasisConfiguration):
            self.basis = ACEBBasisSet(self.parameters.basis_set)
        elif isinstance(basis_set, (ACEBBasisSet, ACECTildeBasisSet)):
            self.basis = basis_set
        elif isinstance(basis_set, str):
            if basis_set.endswith(".yaml"):
                self.basis = ACEBBasisSet(basis_set)
            elif basis_set.endswith(".ace") or basis_set.endswith(".yace"):
                self.basis = ACECTildeBasisSet(basis_set)
            else:
                raise ValueError("Unrecognized file format: " + basis_set)
        else:
            raise ValueError("Unrecognized basis set specification")

        self.elements_name = np.array(self.basis.elements_name).astype(dtype="S2")
        self.elements_mapper_dict = {el: i for i, el in enumerate(self.elements_name)}

        if isinstance(self.basis, ACEBBasisSet):
            self.evaluator = ACEBEvaluator(self.basis)
        elif isinstance(self.basis, ACECTildeBasisSet) and self.parameters.recursive_evaluator:
            self.evaluator = ACERecursiveEvaluator(self.basis)
            self.evaluator.set_recursive(self.parameters.recursive)
        elif isinstance(self.basis, ACECTildeBasisSet):
            self.evaluator = ACECTildeEvaluator(self.basis)

    def get_atomic_env(self, atoms):
        try:
            if self.parameters.fast_nl:
                self.ae = aseatoms_to_atomicenvironment(atoms, cutoff=self.cutoff,
                                                        elements_mapper_dict=self.elements_mapper_dict)
            else:
                self.ae = aseatoms_to_atomicenvironment_old(atoms, cutoff=self.cutoff,
                                                            skin=self.skin,
                                                            elements_mapper_dict=self.elements_mapper_dict)
        except KeyError as e:
            raise ValueError("Unsupported species type: " + str(e) + ". Supported elements: " + str(self.elements_name))
        return self.ae

    def set_active_set(self, filename_or_list_of_active_set_inv):
        if isinstance(self.evaluator, ACEBEvaluator):
            if isinstance(filename_or_list_of_active_set_inv, str):
                self.evaluator.load_active_set(filename_or_list_of_active_set_inv)
                self.is_active_set_configured = True
            elif isinstance(filename_or_list_of_active_set_inv, list):
                self.evaluator.set_active_set(filename_or_list_of_active_set_inv)
                self.is_active_set_configured = True
            else:
                raise ValueError("Unsopported type for `filename_or_list_of_active_set_inv`: {}".format(
                    type(filename_or_list_of_active_set_inv)))
        else:
            raise ValueError("PyACECalculator.set_active_set works only with ACEBEvaluator/B-basis potential(.yaml)")

    def calculate(self, atoms=None, properties=('energy', 'forces', 'stress', 'energies'),
                  system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)

        self.energy = 0.0
        self.energies = np.zeros(len(atoms))
        self.forces = np.empty((len(atoms), 3))

        self.get_atomic_env(atoms)
        self.ace.compute(self.ae, compute_projections=self.compute_projections)

        self.energy, self.forces = np.array(self.ace.energy), np.array(self.ace.forces)
        nat = len(atoms)
        try:
            self.projections = np.reshape(self.ace.projections, (nat, -1))
        except ValueError:
            # if projections has different shapes
            self.projections = self.ace.projections

        self.energies = np.array(self.ace.energies)

        self.results = {
            'energy': np.float64(self.energy.reshape(-1, )),
            'free_energy': np.float64(self.energy.reshape(-1, )),
            'forces': self.forces.astype(np.float64),
            'energies': self.energies.astype(np.float64),
            'gamma': np.array(self.ace.gamma_grade, dtype=np.float64)
        }
        if self.atoms.cell.rank == 3:
            self.volume = atoms.get_volume()
            self.virial = np.array(self.ace.virial)  # order is: xx, yy, zz, xy, xz, yz
            # swap order of the virials to fullfill ASE Voigt stresses order:  (xx, yy, zz, yz, xz, xy)
            self.stress = self.virial[[0, 1, 2, 5, 4, 3]] / self.volume
            self.results["stress"] = self.stress

        if (self.parameters.dump_extrapolative_structures or self.parameters.stop_at_large_extrapolation or
            self.parameters.keep_extrapolative_structures) and not self.is_active_set_configured:
            raise RuntimeError("Active set is not configured, please do it with `set_active_set` method")

        if self.is_active_set_configured:
            max_gamma = max(self.ace.gamma_grade)
            # collect or dump calculations if extrapolation larger than lower bound
            if max_gamma >= self.parameters.gamma_lower_bound:
                # collect extrapolative structures if needed
                if self.parameters.keep_extrapolative_structures:
                    self.extrapolative_structures_list.append(atoms.copy())
                    self.extrapolative_structures_gamma.append(max_gamma)
                # dump extrapolative structures if needed
                if self.parameters.dump_extrapolative_structures:
                    self.dump_current_configuration(atoms, max_gamma)

            if max_gamma >= self.parameters.gamma_upper_bound:
                msg = "Upper extrapolation threshold exceeded: max(gamma) = {}".format(max_gamma)
                if self.parameters.verbose >= 1:
                    print(msg)
                # stop if extrapolation larger than upper bound if needed
                if self.parameters.stop_at_large_extrapolation:
                    raise RuntimeError(msg)
            elif max_gamma >= self.parameters.gamma_lower_bound and self.parameters.verbose >= 2:
                print("Lower extrapolation threshold exceeded: max(gamma) = {}".format(max_gamma))
            elif self.parameters.verbose >= 3:
                print("Extrapolation: max(gamma) = {}".format(max_gamma))

    def dump_current_configuration(self, atoms, max_gamma):

        fname = "extrapolation_{ind}_gamma={gamma}.cfg".format(ind=self.current_extrapolation_structure_index,
                                                               gamma=max_gamma)
        write(fname, atoms, format="cfg")
        self.current_extrapolation_structure_index += 1


class PyACEEnsembleCalculator(Calculator):
    """
    PyACE ASE ensemble calculator
    :param basis_set - list of specification of ACE potential, could be in following forms:
                      ".ace" potential filename
                      ".yaml" potential filename
                      ACEBBasisSet object
                      ACECTildeBasisSet object
                      BBasisConfiguration object
    """
    implemented_properties = ['energy', 'forces', 'stress', 'energies', 'free_energy',
                              'energy_std', 'forces_std', 'stress_std', 'energies_std', 'free_energy_std',
                              'energy_dev', 'forces_dev', 'stress_dev', 'energies_dev'
                              ]

    def __init__(self, basis_set, **kwargs):
        """
PyACE ASE ensemble calculator
:param basis_set - specification of ACE potential, could be in following forms:
                  ".ace" potential filename
                  ".yaml" potential filename
                  ACEBBasisSet object
                  ACECTildeBasisSet object
                  BBasisConfiguration object
"""
        Calculator.__init__(self, basis_set=basis_set, **kwargs)

        self.calcs = [PyACECalculator(pot) for pot in basis_set]
        self.ensemble_size = len(self.calcs)

        self.energy = None
        self.energies = None
        self.forces = None
        self.stress = None

        self.energy_std = None
        self.energies_std = None
        self.forces_std = None
        self.stress_std = None

        self.energy_dev = None
        self.energies_dev = None
        self.forces_dev = None
        self.stress_dev = None

    def calculate(self, atoms=None, properties=('energy', 'forces', 'stress', 'energies',
                                                'energy_std', 'forces_std', 'stress_std', 'energies_std',
                                                ),
                  system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)

        energy_lst = []
        energies_lst = []
        forces_lst = []

        stress_lst = []

        # loop over calculators
        for calc in self.calcs:
            cur_atoms = atoms.copy()
            cur_atoms.set_calculator(calc)
            cur_energy = cur_atoms.get_potential_energy()

            energy_lst.append(cur_energy)
            energies_lst.append(cur_atoms.get_potential_energies())
            forces_lst.append(cur_atoms.get_forces())

            if self.atoms.number_of_lattice_vectors == 3:
                cur_stress = cur_atoms.get_stress()
                stress_lst.append(cur_stress)

        energy_lst = np.array(energy_lst)
        energies_lst = np.array(energies_lst)
        forces_lst = np.array(forces_lst)

        # compute mean of energies and forces
        self.energy = np.mean(energy_lst, axis=0)
        self.energies = np.mean(energies_lst, axis=0)
        self.forces = np.mean(forces_lst, axis=0)

        # compute std of energies and forces
        self.energy_std = np.std(energy_lst, axis=0)
        self.energies_std = np.std(energies_lst, axis=0)
        self.forces_std = np.std(forces_lst, axis=0)

        # compute maximum deviation of energies and forces
        self.energy_dev = np.max(np.abs(energy_lst - self.energy), axis=0)
        self.energies_dev = np.max(np.abs(energies_lst - self.energies), axis=0)
        self.forces_dev = np.max(np.linalg.norm(forces_lst - self.forces, axis=2), axis=0)

        self.results = {
            # mean
            'energy': np.float64(self.energy.reshape(-1, )),
            'free_energy': np.float64(self.energy.reshape(-1, )),
            'forces': self.forces.astype(np.float64),
            'energies': self.energies.astype(np.float64),

            # std
            'energy_std': np.float64(self.energy_std.reshape(-1, )),
            'free_energy_std': np.float64(self.energy_std.reshape(-1, )),
            'forces_std': self.forces_std.astype(np.float64),
            'energies_std': self.energies_std.astype(np.float64),

            # dev
            'energy_dev': np.float64(self.energy_dev),
            'energies_dev': np.float64(self.energies_dev),
            'forces_dev': np.float64(self.forces_dev)
        }

        if self.atoms.number_of_lattice_vectors == 3:
            self.stress = np.mean(stress_lst, axis=0)
            self.stress_std = np.std(stress_lst, axis=0)
            self.stress_dev = np.max(abs(stress_lst - self.stress), axis=0)
            self.results["stress"] = self.stress
            self.results["stress_std"] = self.stress_std
            self.results["stress_dev"] = self.stress_dev


# TODO: made this calculator part of PyACECalculator
class PyGRACEFSCalculator(Calculator):
    """
    Python ASE calculator wrapper for GRACE/FS C++ native implementation
    :param basis_set - ".yaml" potential filename

    """
    implemented_properties = ['energy', 'forces', 'stress', 'energies', 'free_energy']

    def __init__(self, basis_set, **kwargs):
        """
PyGRACEFSCalculator calculator
:param basis_set - specification of GRACE/FS potential, could be in following forms:
                  ".yaml" potential filename
"""
        if "fast_nl" not in kwargs:
            kwargs["fast_nl"] = True

        Calculator.__init__(self, basis_set=basis_set, **kwargs)
        self.nl = None
        self.skin = 0.
        # self.reset_nl = True  # Set to False for MD simulations
        self.ae = ACEAtomicEnvironment()

        self._create_evaluator()

        self.cutoff = self.basis.cutoffmax  # self.parameters.basis_config.funcspecs_blocks[0].rcutij

        self.energy = None
        self.energies = None
        self.forces = None
        self.virial = None
        self.stress = None
        self.projections = None
        self.current_extrapolation_structure_index = 0
        self.is_active_set_configured = False
        self.extrapolative_structures_list = []
        self.extrapolative_structures_gamma = []
        self.ace = GRACEFSCalculator(self.evaluator)
        # self.ace.set_evaluator()
        self.compute_projections = False

    def _create_evaluator(self):

        basis_set = self.parameters.basis_set
        if isinstance(basis_set, GRACEFSBasisSet):
            self.basis = basis_set
        elif isinstance(basis_set, str):
            if basis_set.endswith(".yaml"):
                self.basis = GRACEFSBasisSet(basis_set)
            else:
                raise ValueError("Unrecognized file format: " + basis_set)
        else:
            raise ValueError("Unrecognized basis set specification")

        self.elements_name = np.array(self.basis.elements_name).astype(dtype="S2")
        self.elements_mapper_dict = {el: i for i, el in enumerate(self.elements_name)}

        if isinstance(self.basis, GRACEFSBasisSet):
            self.evaluator = GRACEFSBEvaluator()
            self.evaluator.set_basis(self.basis)

    def get_atomic_env(self, atoms):
        try:
            if self.parameters.fast_nl:
                self.ae = aseatoms_to_atomicenvironment(atoms, cutoff=self.cutoff,
                                                        elements_mapper_dict=self.elements_mapper_dict)
            else:
                self.ae = aseatoms_to_atomicenvironment_old(atoms, cutoff=self.cutoff,
                                                            skin=self.skin,
                                                            elements_mapper_dict=self.elements_mapper_dict)
        except KeyError as e:
            raise ValueError("Unsupported species type: " + str(e) + ". Supported elements: " + str(self.elements_name))
        return self.ae

    def calculate(self, atoms=None, properties=('energy', 'forces', 'stress', 'energies'),
                  system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)

        self.energy = 0.0
        self.energies = np.zeros(len(atoms))
        self.forces = np.empty((len(atoms), 3))

        self.get_atomic_env(atoms)
        t0 = time.perf_counter()
        self.ace.compute(self.ae, compute_projections=self.compute_projections)
        self.perf = (time.perf_counter() - t0)

        nat = len(atoms)
        try:
            self.projections = np.reshape(self.ace.projections, (nat, -1))
        except ValueError:
            # if projections has different shapes
            self.projections = self.ace.projections

        self.energy, self.forces = np.array(self.ace.energy), np.array(self.ace.forces)

        self.energies = np.array(self.ace.energies)

        self.results = {
            'energy': np.float64(self.energy.reshape(-1, )),
            'free_energy': np.float64(self.energy.reshape(-1, )),
            'forces': self.forces.astype(np.float64),
            'energies': self.energies.astype(np.float64),
            'gamma': np.array(self.ace.gamma_grade, dtype=np.float64)
        }
        if self.atoms.cell.rank == 3:
            self.volume = atoms.get_volume()
            self.virial = np.array(self.ace.virial)  # order is: xx, yy, zz, xy, xz, yz
            # swap order of the virials to fullfill ASE Voigt stresses order:  (xx, yy, zz, yz, xz, xy)
            self.stress = self.virial[[0, 1, 2, 5, 4, 3]] / self.volume
            self.results["stress"] = self.stress

    def set_active_set(self, filename_or_list_of_active_set_inv):
        if isinstance(filename_or_list_of_active_set_inv, str):
            self.evaluator.load_active_set(filename_or_list_of_active_set_inv)
            self.is_active_set_configured = True
            self.compute_projections = True
        else:
            raise ValueError("Unsupported type for `filename_or_list_of_active_set_inv`: {}".format(
                type(filename_or_list_of_active_set_inv)))
