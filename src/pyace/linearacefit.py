from typing import Optional

import numpy as np
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from time import perf_counter

import pandas as pd
from pyace import ACEBEvaluator, ACECalculator, BBasisConfiguration, ACEBBasisSet
from pyace.atomicenvironment import aseatoms_to_atomicenvironment

global g_nfunc, g_func_ind_shift
global g_basis, g_beval, g_calc

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(*args, **kwargs):
        return args


def compute_b_grad_ae(ae):
    """
    Compute B-gradients for given atomic environment 'ae
    :param ae: Atomic environment
    :return: energy_list (sum of B_projection over all atoms), forces_list (B-grad for each atom and B-function)
    """
    global g_calc, g_func_ind_shift, g_nfunc
    g_calc.compute(ae, True)

    # energy projections
    projections = g_calc.projections

    energy_list = np.zeros(g_nfunc)
    for s, proj in zip(ae.species_type, projections):
        f_shift = g_func_ind_shift[s]
        n_func = len(proj)  # species type dependent
        energy_list[f_shift:f_shift + n_func] += proj

    return energy_list, g_calc.forces_bfuncs


def potential_initializer(basis):
    global g_basis, g_calc, g_beval, g_nfunc, g_func_ind_shift
    g_basis = basis
    g_beval = ACEBEvaluator(g_basis)
    g_calc = ACECalculator(g_beval)
    g_nfunc, g_func_ind_shift = compute_nfunc_func_ind_shift(basis)


def compute_nfunc_func_ind_shift(basis):
    func_ind_shift = []
    num_funcs = 0
    for br1, br in zip(basis.basis_rank1, basis.basis):
        func_ind_shift.append(num_funcs)
        num_funcs += len(br1) + len(br)
    return num_funcs, func_ind_shift


# Generate random data using multiple processes
def generate_data(data, basis, shared_design_matrix, structures_chunk, atoms_chunk, verbose=False, proc_id=-1):
    if verbose:
        print(f"[Proc-{proc_id}]structures_chunk={structures_chunk}, atoms_chunk={atoms_chunk} " +
              f" # struct={structures_chunk[1] - structures_chunk[0]}, #at = {atoms_chunk[1] - atoms_chunk[0]}",
              flush=True)

    t1_start = perf_counter()

    beval = ACEBEvaluator(basis)
    calc = ACECalculator(beval)

    total_number_of_structures = len(data["NUMBER_OF_ATOMS"])
    total_number_of_atoms = data["NUMBER_OF_ATOMS"].sum()

    nfunc, func_ind_shift = compute_nfunc_func_ind_shift(basis)

    shared_design_matrix_np = np.frombuffer(shared_design_matrix).reshape(
        total_number_of_structures + total_number_of_atoms * 3, nfunc)

    # Get the data subset
    data_subset = data.iloc[structures_chunk[0]: structures_chunk[1]]

    global_atom_ind_shift = total_number_of_structures + atoms_chunk[0] * 3
    loc_n_structures = len(data_subset)
    step = max(loc_n_structures // 25, 1)
    t_loop_start = perf_counter()
    for struct_ind, ae in enumerate(data_subset["atomic_env"]):

        # do calculation with compute gradients flag
        calc.compute(ae, True)

        # energy projections
        projections = calc.projections
        e_ind = structures_chunk[0] + struct_ind

        cur_num_atoms = ae.n_atoms_real
        # loop over atoms
        for i, (s, proj) in enumerate(zip(ae.species_type, projections)):
            f_shift = func_ind_shift[s]
            n_func = len(proj)
            shared_design_matrix_np[e_ind, f_shift:f_shift + n_func] += proj
        # divide by number of atoms to get average-per-atom
        shared_design_matrix_np[e_ind] /= cur_num_atoms

        # force projections
        forces_bfuncs = np.array(calc.forces_bfuncs)  # [natoms, nfunc, 3]
        nat = ae.n_atoms_real

        forces_bfuncs = forces_bfuncs.transpose((0, 2, 1)).reshape(nat * 3, nfunc)
        shared_design_matrix_np[global_atom_ind_shift: global_atom_ind_shift + nat * 3] = forces_bfuncs
        global_atom_ind_shift += nat * 3
        if verbose:
            if (struct_ind + 1) % step == 0:
                t_loop_cur = perf_counter()
                cur_elaps = t_loop_cur - t_loop_start
                est_tot = cur_elaps / (struct_ind + 1) * loc_n_structures
                est_remain = est_tot - cur_elaps  # in sec
                print(
                    f"[Proc-{proc_id}] {struct_ind + 1}/{loc_n_structures} structures: "
                    f"[{cur_elaps:.2f}/{est_tot:.2f}s, {est_remain:.2f} s remains]",
                    flush=True)
    t1_stop = perf_counter()

    if verbose:
        print(
            f"[Proc-{proc_id}] {struct_ind + 1}/{loc_n_structures} structures, "
            f"elapsed time = {t1_stop - t1_start:.3g} s", flush=True)


class LinearACEDataset:
    def __init__(self, bconf_or_bbasis, df):
        """
        Representation of a linear ACE dataset, build from a dataframe of atomic environments and a basis set
        :param bconf_or_bbasis: BBasisConfiguration or ACEBBasisSet
        :param df: pandas dataframe with ASE atoms "ase_atoms" and atomic environments "ae"
        """
        self.f_pred: np.array = None
        self.e_pred: np.array = None
        self.shared_design_matrix: np.array = None
        self.total_shared_design_matrix_size: int = 0
        self.target_vector: np.array = None
        self.design_matrix: np.array = None
        self.model = None

        if isinstance(bconf_or_bbasis, BBasisConfiguration):
            self.basis = ACEBBasisSet(bconf_or_bbasis)
        elif isinstance(bconf_or_bbasis, ACEBBasisSet):
            self.basis = bconf_or_bbasis
        else:
            raise ValueError(
                f"Unsupported type of bconf_or_bbasis: {type(bconf_or_bbasis)}. Must be BBasisConfiguration or "
                f"ACEBBasisSet")

        self.beval = ACEBEvaluator(self.basis)
        self.calc = ACECalculator(self.beval)
        self.nfunc, self.func_ind_shift = compute_nfunc_func_ind_shift(self.basis)

        self.total_number_of_atoms: int = 0
        self.total_number_of_structures: int = 0
        self.df: pd.DataFrame = df
        if self.df is not None:
            self.prepare_df()

    def estimate_memory(self):
        """
        :return: Memory for storing e and flist in bytes
        """
        return (self.total_number_of_atoms * 3 * self.nfunc +
                self.total_number_of_structures * self.nfunc) * 8

    def prepare_df(self):
        if "NUMBER_OF_ATOMS" not in self.df.columns:
            self.df["NUMBER_OF_ATOMS"] = self.df["ase_atoms"].map(len)
        if "atomic_env" not in self.df.columns:
            elements_mapper_dict = self.basis.elements_to_index_map
            cutoff = self.basis.cutoffmax
            build_ae = lambda at: aseatoms_to_atomicenvironment(at, cutoff=cutoff,
                                                                elements_mapper_dict=elements_mapper_dict)
            self.df["atomic_env"] = self.df["ase_atoms"].map(build_ae)

        self.total_number_of_atoms: int = self.df["NUMBER_OF_ATOMS"].sum()
        self.total_number_of_structures: int = len(self.df)

    def construct_design_matrix(self, df=None, max_workers=4, verbose=False):
        """
        Compute energies_list and forces_list for design matrix in a parallel manner

        :param df: dataset with at least "ase_atoms" column
                   "atomic_env" column is optional
        :param max_workers: number of parallel processes
        :param verbose: verbosity flag

        :return:
            energies_list: [n_structures, n_funcs]
            forces_list: [n_atoms, n_funcs, 3]
        """
        if df is not None:
            self.df = df
            self.prepare_df()

        self.total_shared_design_matrix_size = int(
            self.total_number_of_structures + self.total_number_of_atoms * 3) * self.nfunc

        self.shared_design_matrix = mp.Array('d', self.total_shared_design_matrix_size, lock=False)

        self.design_matrix = np.frombuffer(self.shared_design_matrix).reshape(
            self.total_number_of_structures + self.total_number_of_atoms * 3, self.nfunc)

        max_workers = min(max_workers, self.total_number_of_structures)
        # Determine the number of rows to process in each iteration
        splt = np.array_split(np.arange(self.total_number_of_structures), max_workers)

        # Split the data into chunks for each process
        structure_ind_chunks = [(s[0], s[-1] + 1) for s in splt]
        # structure_ind_chunks[-1] = structure_ind_chunks[-1][0], self.total_number_of_structures

        atoms_ind_chunks = []
        prev_at_ind = 0
        for chunks in structure_ind_chunks:
            chunk_num_at = self.df.iloc[chunks[0]:chunks[1]]['NUMBER_OF_ATOMS'].sum()
            atoms_ind_chunks.append((prev_at_ind, prev_at_ind + chunk_num_at))
            prev_at_ind += chunk_num_at
        assert atoms_ind_chunks[-1][-1] == self.total_number_of_atoms

        self.design_matrix[:] = 0.
        # Create the processes and start them
        processes = []
        for proc_ind, (s_chunk, a_chunk) in enumerate(zip(structure_ind_chunks, atoms_ind_chunks)):
            p = mp.Process(target=generate_data,
                           args=(self.df, self.basis,
                                 self.shared_design_matrix, s_chunk, a_chunk, verbose, proc_ind))
            processes.append(p)
            p.start()

        # Wait for all processes to finish
        for p in processes:
            p.join()

    def get_design_matrix(self):
        """ Get design matrix """
        if self.design_matrix is None:
            self.construct_design_matrix()
        return self.design_matrix

    def construct_target_vector(self):
        """ Construct target matrix from the dataset """
        if "energy_corrected" in self.df.columns:
            energies_ref = self.df["energy_corrected"].values
        elif "energy" in self.df.columns:
            energies_ref = self.df["energy"].values
        else:
            raise ValueError("No `energy` or `energy_corrected` columns in the dataset")
        energies_ref /= self.df["NUMBER_OF_ATOMS"].values
        forces_ref = np.vstack(self.df["forces"]).reshape(-1)
        self.target_vector = np.hstack([energies_ref, forces_ref])

    def get_target_vector(self):
        """ Get target matrix """
        if self.target_vector is None:
            self.construct_target_vector()
        return self.target_vector

    def get_energies_per_atom(self, vector=None):
        """Get energies-per-atom from target vector
        :param vector: target vector to extract energies from, if None - return TRUE values from  self.targetvector
        :return: energies array (n_structures)
         """
        if vector is None:
            vector = self.get_target_vector()
        return vector[:self.total_number_of_structures]

    def get_forces(self, vector=None, reshape_forces=False):
        """ Get forces from target vector
        :param vector: target vector to extract energies from, if None - return TRUE values from  self.targetvector
        :param reshape_forces: reshape forces to (n_atoms, 3) shape
        :return: forces array (3*n_atoms) or reshaped forces array (n_atoms, 3)
        """
        if vector is None:
            vector = self.get_target_vector()
        if reshape_forces:
            return vector[self.total_number_of_structures:].reshape(-1, 3)
        else:
            return vector[self.total_number_of_structures:]

    # compute errors: MAE, RMSE, MAE per atom, RMSE per atom
    # return as dictionary
    def compute_errors(self, epa_pred=None, f_pred=None, epa_ref=None, f_ref=None):
        epa_ref = epa_ref if epa_ref is not None else self.get_energies_per_atom()  # shape [nstruct]
        f_ref = f_ref if f_ref is not None else self.get_forces()  # shape [natom, 3]
        # e_pred = e_pred if e_pred is not None else self.e_pred
        # f_pred = f_pred if f_pred is not None else self.f_pred
        # number_of_atoms = self.df["NUMBER_OF_ATOMS"].values
        depa = (epa_ref - epa_pred)  # / number_of_atoms
        epa_mae = np.mean(np.abs(depa))
        epa_rmse = np.sqrt(np.mean(depa ** 2))

        f_comp_mae = np.mean(np.abs(f_ref - f_pred).flatten())
        f_comp_rmse = np.sqrt(np.mean(((f_ref - f_pred) ** 2).flatten()))

        return {'epa_mae': epa_mae, 'epa_rmse': epa_rmse,
                'f_comp_mae': f_comp_mae, 'f_comp_rmse': f_comp_rmse}

    def get_bbasis(self, model):
        self.basis.basis_coeffs = model.coef_
        return self.basis


# class that takes two LinearACEDatasets: train and test
# and has .fit and .predict methods

class LinearACEFit:
    def __init__(self, model=None, train_dataset=None):
        """ Linear ACE fit class
        :param model: linear model to fit
        :param train_dataset: LinearACEDataset object
        """
        self.model = None
        self.set_model(model)
        self.train_dataset = train_dataset

    def set_model(self, model=None):
        """ Set model to self.model attribute
        :param model: linear model to fit. If None - use Ridge with default parameters
        """
        if model is None:
            from sklearn.linear_model import Ridge
            self.model = Ridge(alpha=1e-5, fit_intercept=False, copy_X=False, random_state=42, solver="auto")
        else:
            self.model = model

    def fit(self, train_dataset: Optional[LinearACEDataset] = None, model=None, **kwargs):
        """ Fit model to train dataset
        :param train_dataset: LinearACEDataset object. If None - use self.train_dataset
        :param model: linear model to fit. If None - use self.model
        :param kwargs: keyword arguments for model.fit method
        :return: fitted model
        """
        train_dataset = train_dataset if train_dataset is not None else self.train_dataset
        if model is not None:
            self.set_model(model)

        return self.model.fit(train_dataset.get_design_matrix(),
                              train_dataset.get_target_vector(), **kwargs)

    def predict(self, dataset: Optional[LinearACEDataset] = None, reshape_forces=False):
        """ Predict energies and forces from dataset
        :param dataset: LinearACEDataset object. If None - use self.train_dataset
        :param model: linear model to fit. If None - use self.model
        :param reshape_forces: reshape forces to (n_atoms, 3) shape
        :return: epa_pred, f_pred
        """
        dataset = dataset if dataset is not None else self.train_dataset
        predicted_vector = self.model.predict(dataset.get_design_matrix())

        epa_pred = dataset.get_energies_per_atom(predicted_vector)
        f_pred = dataset.get_forces(predicted_vector, reshape_forces=reshape_forces)
        return epa_pred, f_pred

    def compute_errors(self, dataset: Optional[LinearACEDataset] = None):
        """ Compute errors for dataset
        :param dataset: LinearACEDataset object. If None - use self.train_dataset
        :return: dictionary with errors
        """
        dataset = dataset if dataset is not None else self.train_dataset
        epa_pred, f_pred = self.predict(dataset, reshape_forces=False)
        return dataset.compute_errors(epa_pred, f_pred)

    def get_bbasis(self):
        return self.train_dataset.get_bbasis(self.model)
