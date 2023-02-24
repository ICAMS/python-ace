import numpy as np
from ase.io import write
from pyace import PyACECalculator
from scipy.optimize import minimize

from pyace.preparedata import calc_min_distance, aseatoms_to_atomicenvironment
from pyace.basis import BBasisConfiguration, ACEBBasisSet
from pyace.activelearning import load_active_inverse_set, compute_active_set, compute_A_active_inverse, \
    compute_B_projections, save_active_inverse_set


class EarlyStoppingExtrapolation(Exception):
    pass


def move_atom(delta_pos, atom_ind, original_structure):
    """
    Move atom with index `atom_ind` from `original_structure` by `delta_pos` in absolute units
    Return new ASE atoms

    :param delta_pos:
    :type delta_pos:
    :param atom_ind:
    :type atom_ind:
    :param original_structure:
    :type original_structure:
    :param calc:
    :type calc:
    :return:
    :rtype:
    """
    delta_pos = np.array(delta_pos)
    newpos = original_structure.get_positions().copy()
    newpos[atom_ind] += delta_pos
    new_atoms = original_structure.copy()
    new_atoms.set_positions(newpos)
    return new_atoms


class ActiveExploration:

    def __init__(self, bconf, orig_asi_fname: str):
        self.gamma_tol = 1
        self.gamma_lo = 2
        self.gamma_hi = 15
        self.min_dist = 1.1
        self.min_cutoff = 3

        self.setup_bconf(bconf)
        self.setup_asi0(orig_asi_fname)

        self.orig_atoms_ = None

        self.current_dpos_ = 0, 0, 0
        self.current_gamma_ = 0
        self.extrapolative_structures = []
        self.current_snapshot_fname = "current_AE_snapshot.extxyz"
        self.current_asi_fname = "current_ASI.asi"
        self.extrapolative_structures_fname = "current_active_explored.extxyz"

    def setup_asi0(self, orig_asi_fname):
        if not isinstance(orig_asi_fname, str):
            raise ValueError("`asi_fname` should be string - filename")
        self.orig_asi_fname = orig_asi_fname
        self.asi0 = load_active_inverse_set(self.orig_asi_fname)

    def setup_bconf(self, bconf):
        if isinstance(bconf, str):
            bconf = BBasisConfiguration(bconf)
        elif not isinstance(bconf, BBasisConfiguration):
            raise ValueError("bconf should be filename.yaml or BBasisConfiguration, but not {}".format(type(bconf)))
        self.bconf = bconf
        self.bbasis = ACEBBasisSet(bconf)
        self.elements_name = self.bbasis.elements_name
        self.calc = PyACECalculator(self.bconf)

    def obj(self, dpos, atom_ind, orig_structure):
        symbols = self.orig_atoms_.get_chemical_symbols()
        print(" [obj] {}(#{}) dpos=({:>7.3f}, {:>7.3f}, {:>7.3f})".format(symbols[atom_ind], atom_ind, *dpos), end=" ")
        new_atoms = move_atom(dpos, atom_ind=atom_ind, original_structure=orig_structure)

        # element mapper doesn't matter, only for NN distance
        ae = aseatoms_to_atomicenvironment(new_atoms, cutoff=self.min_cutoff)
        min_dist = ae.get_minimal_nn_distance()
        print("min dist={:<7.3f}".format(min_dist), end=" ")
        if min_dist < self.min_dist:
            print("(too small)")
            return 1000 * (self.min_dist - min_dist)

        new_atoms.set_calculator(self.calc)
        self.calc.reset()
        new_atoms.get_potential_energy()
        max_gamma = max(self.calc.results["gamma"])
        print("max_gamma = {:<7.3f}".format(max_gamma))

        self.current_dpos_ = dpos
        self.current_gamma_ = max_gamma
        res = (max_gamma - self.gamma_hi) ** 2
        if res < self.gamma_tol ** 2:
            raise EarlyStoppingExtrapolation("Minimization goal achieved, max_gamma={:.3f}".format(max_gamma))
        return res

    def active_exploration(self, orig_atoms,
                           min_dist=1.1,
                           gamma_hi=15,
                           n_iter=None,
                           gamma_lo=1,
                           gamma_tol=1,
                           initial_moving_atom_index=None,
                           movable_atoms_indices=None,
                           n_atom_shake_max_attempts=20,
                           seed=42):
        """
        Perform Active Exploration (AE):
        Try to displace atom in `orig_atoms` by maximizing extrapolation grade gamma until
        it not exceeds upper limit `gamma_hi` but keeping minimal distance not smaller than `min_dist`

        Procedure repeats up-to `n_iter` iterations

        :param orig_atoms: original ASE atoms structure
        :param min_dist: minimal distance between atoms
        :param n_iter: number of total iterations (number of displaced atoms)
        :param gamma_hi: limit of extrapolation grade, above which atom stop moving
        :param gamma_lo: low limit of extrapolation grad
        :param initial_moving_atom_index: index of first moving atom
        :param n_atom_shake_max_attempts: number of random displacement attempts to restart optimization
        :param seed: random seed

        :return: list of ASE atoms, original structure  + one structure for each AE iteration

        """
        self.gamma_lo = gamma_lo
        self.gamma_hi = gamma_hi
        self.min_dist = min_dist
        self.gamma_tol = gamma_tol
        orig_atoms = orig_atoms.copy()
        self.orig_atoms_ = orig_atoms
        self.attach_gamma_array(orig_atoms)
        self.extrapolative_structures = [orig_atoms]
        self.current_dpos_ = 0, 0, 0
        self.current_gamma_ = 0
        self.max_dpos = 0, 0, 0

        # reset ASI to original
        self.calc.set_active_set(self.orig_asi_fname)
        orig_symbols = orig_atoms.get_chemical_symbols()

        # restore active set from ASI
        lin_AS = compute_A_active_inverse(self.asi0)

        # atom indices
        if movable_atoms_indices is None:
            movable_atoms_indices = np.arange(len(orig_atoms))
        movable_atoms_indices = np.array(movable_atoms_indices)
        # first moving atom
        cur_atoms = orig_atoms.copy()

        if seed:
            np.random.seed(seed)

        if n_iter is None:
            n_iter = len(movable_atoms_indices)

        # n_iter = min(n_iter, len(movable_atoms_indices))
        if initial_moving_atom_index is None:
            initial_moving_atom_index = np.random.choice(movable_atoms_indices, size=1)[0]

        for it in range(n_iter):
            print("=" * 80)
            print("Iteration: {}/{}".format(it + 1, n_iter))
            mov_symb = cur_atoms.get_chemical_symbols()[initial_moving_atom_index]
            print("Moving atom {}(#{})".format(mov_symb, initial_moving_atom_index))

            att = 0
            x0 = [0, 0, 0]
            while True:
                print("Attempt local exploration #{}/{}".format(att + 1, n_atom_shake_max_attempts))

                try:
                    # self.atom_ind_ = initial_moving_atom_index
                    # self.orig_structure_ = cur_atoms
                    res = minimize(self.obj, x0=x0, args=(initial_moving_atom_index, cur_atoms),
                                   method="Nelder-Mead"  # , callback = self.callback
                                   )
                    self.max_gamma = -res.fun
                    self.max_dpos = res.x
                except EarlyStoppingExtrapolation as e:
                    print(e)
                    self.max_gamma = self.current_gamma_
                    self.max_dpos = self.current_dpos_

                if (self.max_gamma - self.gamma_hi) ** 2 < self.gamma_tol ** 2:
                    break
                att += 1
                if att > n_atom_shake_max_attempts:
                    raise RuntimeError("Could not find extrapolative strucrtures, too many attempts")
                print("Attempting to shake x0 position")
                x0 = x0 + np.random.randn(3) * 0.1

            new_atoms = move_atom(self.max_dpos, initial_moving_atom_index, cur_atoms)

            self.attach_gamma_array(new_atoms)

            self.extrapolative_structures.append(new_atoms)

            write(self.extrapolative_structures_fname, self.extrapolative_structures)

            new_proj = compute_B_projections(self.bconf, self.extrapolative_structures[-1:])

            current_ext_lin_proj = {}
            for mu, active_set_mu in lin_AS.items():
                current_ext_lin_proj[mu] = np.vstack((active_set_mu, new_proj[mu]))

            print("Constructing new active set")
            new_as = compute_active_set(current_ext_lin_proj, tol=1.001, verbose=True)

            print("Inverting new active set")
            new_asi = compute_A_active_inverse(new_as)

            print("Saving new active set into " + self.current_asi_fname)
            save_active_inverse_set(self.current_asi_fname, new_asi, elements_name=self.elements_name)

            cur_atoms = new_atoms
            lin_AS = new_as

            ####
            # reset ASI to original
            self.calc.set_active_set(self.orig_asi_fname)
            self.calc.reset()

            # compute gamma wrt. original ASI
            gat = new_atoms.copy()
            gat.set_calculator(self.calc)
            gat.get_potential_energy()

            gamma = self.calc.results['gamma']
            gat.arrays["gamma"] = gamma
            print("Save current snapshot with gamma (wrt. original ASI) to ", self.current_snapshot_fname)
            write(self.current_snapshot_fname, gat, format='extxyz')

            print(
                "Gamma distribution (wrt. original ASI): min={:.3f}, median={:.3f}, mean={:.3f}, max={:.3f}, std={:.3f}".format(
                    np.min(gamma), np.median(gamma), np.mean(gamma), np.max(gamma), np.std(gamma)
                ))
            gamma_pools = gamma[movable_atoms_indices]
            if self.gamma_lo is not None:
                atoms_pools = movable_atoms_indices[gamma_pools < self.gamma_lo]
            else:
                atoms_pools = movable_atoms_indices

            print("Pool of movable atoms: {} atoms".format(len(atoms_pools)))
            if not len(atoms_pools):
                print("Too high extrapolation, no atoms with gamma < {}. Stopping".format(self.gamma_lo))
                break
            initial_moving_atom_index = np.random.choice(atoms_pools, size=1)[0]
            print("Next moving atom ind: #{}-{} ".format(initial_moving_atom_index,
                                                         orig_symbols[initial_moving_atom_index]))

            self.calc.set_active_set(self.current_asi_fname)
            self.calc.reset()

        return cur_atoms

    def attach_gamma_array(self, atoms):
        atoms.set_calculator(self.calc)
        self.calc.reset()
        atoms.get_potential_energy()
        atoms.arrays["gamma"] = self.calc.results['gamma']
