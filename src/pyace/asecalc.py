import numpy as np

from ase.calculators.calculator import Calculator, all_changes

from pyace.basis import ACEBBasisSet, ACECTildeBasisSet, BBasisConfiguration
from pyace.calculator import ACECalculator
from pyace.catomicenvironment import ACEAtomicEnvironment
from pyace.atomicenvironment import aseatoms_to_atomicenvironment_old, aseatoms_to_atomicenvironment
from pyace.evaluator import ACEBEvaluator, ACECTildeEvaluator, ACERecursiveEvaluator


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
"""
        if "recursive_evaluator" not in kwargs:
            kwargs["recursive_evaluator"] = False
        if "recursive" not in kwargs:
            kwargs["recursive"] = False
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

        self.ace = ACECalculator()
        self.ace.set_evaluator(self.evaluator)

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
            raise ValueError("Unsupported species type: " + str(e))
        return self.ae

    def calculate(self, atoms=None, properties=['energy', 'forces', 'stress', 'energies'],
                  system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)

        self.energy = 0.0
        self.energies = np.zeros(len(atoms))
        self.forces = np.empty((len(atoms), 3))

        self.get_atomic_env(atoms)
        self.ace.compute(self.ae)

        self.energy, self.forces = np.array(self.ace.energy), np.array(self.ace.forces)
        nat = len(atoms)
        proj1 = np.reshape(self.ace.basis_projections_rank1, (nat, -1))
        proj2 = np.reshape(self.ace.basis_projections, (nat, -1))
        self.projections = np.concatenate([proj1, proj2], axis=1)

        self.energies = np.array(self.ace.energies)

        self.results = {
            'energy': np.float64(self.energy.reshape(-1, )),
            'free_energy': np.float64(self.energy.reshape(-1, )),
            'forces': self.forces.astype(np.float64),
            'energies': self.energies.astype(np.float64)
        }
        if self.atoms.number_of_lattice_vectors == 3:
            self.volume = atoms.get_volume()
            self.virial = np.array(self.ace.virial)  # order is: xx, yy, zz, xy, xz, yz
            # swap order of the virials to fullfill ASE Voigt stresses order:  (xx, yy, zz, yz, xz, xy)
            self.stress = self.virial[[0, 1, 2, 5, 4, 3]] / self.volume
            self.results["stress"] = self.stress


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
                              'energy_std', 'forces_std', 'stress_std', 'energies_std', 'free_energy_std'
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

        # compute mean of energies and forces
        self.energy = np.mean(energy_lst, axis=0)
        self.energies = np.mean(energies_lst, axis=0)
        self.forces = np.mean(forces_lst, axis=0)

        # compute std of energies and forces
        self.energy_std = np.std(energy_lst, axis=0)
        self.energies_std = np.std(energies_lst, axis=0)
        self.forces_std = np.std(forces_lst, axis=0)

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
            'energies_std': self.energies_std.astype(np.float64)
        }

        if self.atoms.number_of_lattice_vectors == 3:
            self.stress = np.mean(stress_lst, axis=0)
            self.stress_std = np.std(stress_lst, axis=0)
            self.results["stress"] = self.stress
            self.results["stress_std"] = self.stress_std


if __name__ == '__main__':
    from ase.build import bulk
    from pyace.basis import BBasisConfiguration, BBasisFunctionSpecification, BBasisFunctionsSpecificationBlock

    block = BBasisFunctionsSpecificationBlock()

    block.block_name = "Al"
    block.nradmaxi = 1
    block.lmaxi = 0
    block.npoti = "FinnisSinclair"
    block.fs_parameters = [1, 1, 1, 0.5]
    block.rcutij = 8.7
    block.dcutij = 0.01
    block.NameOfCutoffFunctionij = "cos"
    block.nradbaseij = 1
    block.radbase = "ChebExpCos"
    block.radparameters = [3.0]
    block.radcoefficients = [1]

    block.funcspecs = [
        BBasisFunctionSpecification(["Al", "Al"], ns=[1], ls=[0], LS=[], coeffs=[1.]),
        # BBasisFunctionSpecification(["Al", "Al", "Al"], ns=[1, 1], ls=[0, 0], LS=[], coeffs=[2])
    ]

    basisConfiguration = BBasisConfiguration()
    basisConfiguration.deltaSplineBins = 0.001
    basisConfiguration.funcspecs_blocks = [block]

    a = bulk('Al', 'fcc', a=4, cubic=True)
    a.pbc = False
    print(a)
    calc = PyACECalculator(basis_set=basisConfiguration)
    a.set_calculator(calc)
    e1 = (a.get_potential_energy())
    f1 = a.get_forces()
    print(e1)
    print(f1)

    calc2 = PyACECalculator(basis_set=ACEBBasisSet(basisConfiguration))
    a2 = bulk('Al', 'fcc', a=4, cubic=True)
    a2.set_calculator(calc2)
    e2 = (a2.get_potential_energy())
    f2 = a2.get_forces()
    print(e2)
    print(f2)
