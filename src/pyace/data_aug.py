import numpy as np
import pandas as pd

try:
    from matplotlib import pylab as plt
except Exception:
    plt = None

from ase.calculators.calculator import Calculator, all_changes
from ase.neighborlist import neighbor_list
from ase.units import _eps0, _e

from scipy.optimize import curve_fit
from scipy.signal import find_peaks

# string consts
ZBL = "zbl"
EOS = "eos"
KINK = "kink"
EXTRAPOLATION = "extrapolation"

# column names
GAMMA_C = "gamma"
GAMMA_PER_ATOM_C = "gamma_per_atom"
VPA_C = "vpa"
EPA_C = "epa"
Z_C = "z"
ASE_ATOMS_C = "ase_atoms"
EPA_ZBL_C = "epa_zbl"
FORCES_ZBL_C = "forces_zbl"
EPA_EOS_C = "epa_eos"
FORCES_EOS_C = "forces_eos"
EPA_CORRECTED_C = "energy_corrected_per_atom"
FORCES_C = "forces"
E_CORRECTED_C = "energy_corrected"
NAME_C = "name"

# transformation coefficients to eV
K = _e ** 2 / (4 * np.pi * _eps0) / 1e-10 / _e

# coefficients of ZBL potential
phi_coefs = np.array([0.18175, 0.50986, 0.28022, 0.02817])
phi_exps = np.array([-3.19980, -0.94229, -0.40290, -0.20162])


def phi(x):
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    return np.sum(phi_coefs * np.exp(x.reshape(-1, 1) * phi_exps), axis=1)


def dphi(x):
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    return np.sum(phi_coefs * phi_exps * np.exp(x.reshape(-1, 1) * phi_exps), axis=1)


def d2phi(x):
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    return np.sum(
        phi_coefs * phi_exps * phi_exps * np.exp(x.reshape(-1, 1) * phi_exps), axis=1
    )


# common factor: K*Zi*Zj*
def fun_E_ij(nl_dist, a):
    return 1 / nl_dist * phi(nl_dist / a)


# common factor: K*Zi*Zj*
def fun_dE_ij(nl_dist, a):
    return (-1 / nl_dist ** 2) * phi(nl_dist / a) + 1 / nl_dist * dphi(nl_dist / a) / a


# common factor: K*Zi*Zj*
def fun_d2E_ij(nl_dist, a):
    return (
            (+2 / nl_dist ** 3) * phi(nl_dist / a)
            + 2 * (-1 / nl_dist ** 2) * dphi(nl_dist / a) / a
            + (1 / nl_dist) * d2phi(nl_dist / a) / (a ** 2)
    )


class ZBLCalculator(Calculator):
    """Python implementation of Ziegler-Biersack-Littmark (ZBL) potential as ASE calculator
    References
        https://docs.lammps.org/pair_zbl.html
        https://docs.lammps.org/pair_gromacs.html
        https://github.com/lammps/lammps/blob/develop/src/pair_zbl.cpp

    """

    implemented_properties = ["energy", "free_energy", "forces"]

    def __init__(self, cut_in=0, cutoff=3, **kwargs):
        Calculator.__init__(self, **kwargs)
        self.cut_in = cut_in
        self.cutoff = cutoff

        self.energy = None
        self.forces = None

    def calculate(
            self,
            atoms=None,
            properties=["energy", "forces", "free_energy"],
            system_changes=all_changes,
    ):
        Calculator.calculate(self, atoms, properties, system_changes)

        nl_i, nl_j, d, D = neighbor_list("ijdD", atoms, cutoff=self.cutoff)

        atomic_numbers = atoms.get_atomic_numbers()
        Zi = atomic_numbers[nl_i]
        Zj = atomic_numbers[nl_j]

        a = 0.46850 / (Zi ** 0.23 + Zj ** 0.23)

        E_ij = fun_E_ij(d, a)

        Ec = fun_E_ij(self.cutoff, a)
        dEc = fun_dE_ij(self.cutoff, a)
        d2Ec = fun_d2E_ij(self.cutoff, a)

        drcut = self.cutoff - self.cut_in

        A = (-3 * dEc + drcut * d2Ec) / drcut ** 2
        B = (2 * dEc - drcut * d2Ec) / drcut ** 3
        C = -Ec + 1 / 2 * drcut * dEc - 1 / 12 * drcut ** 2 * d2Ec

        S = A / 3 * (d - self.cut_in) ** 3 + B / 4 * (d - self.cut_in) ** 4 + C  # S(r)

        S[d < self.cut_in] = C[d < self.cut_in]
        S[d > self.cutoff] = 0
        self.energy = K / 2 * np.sum(Zi * Zj * (E_ij + S))

        # forces
        # if "forces" in properties:
        dEdr = fun_dE_ij(d, a)
        dS_dr = A * (d - self.cut_in) ** 2 + B * (d - self.cut_in) ** 3

        dS_dr[d < self.cut_in] = 0
        dS_dr[d > self.cutoff] = 0

        pair_forces = -(dEdr + dS_dr).reshape(-1, 1) * (D / d.reshape(-1, 1))
        pair_forces *= (Zi * Zj).reshape(-1, 1) * K / 2

        nat = len(atoms)

        forces_x = np.bincount(
            nl_j, weights=pair_forces[:, 0], minlength=nat
        ) - np.bincount(nl_i, weights=pair_forces[:, 0], minlength=nat)
        forces_y = np.bincount(
            nl_j, weights=pair_forces[:, 1], minlength=nat
        ) - np.bincount(nl_i, weights=pair_forces[:, 1], minlength=nat)
        forces_z = np.bincount(
            nl_j, weights=pair_forces[:, 2], minlength=nat
        ) - np.bincount(nl_i, weights=pair_forces[:, 2], minlength=nat)

        self.forces = np.vstack([forces_x, forces_y, forces_z]).T

        self.results["energy"] = self.energy
        self.results["free_energy"] = self.energy
        self.results["forces"] = self.forces


def E_ER_pars(V, pars):
    E0, V0, c3, lr = pars
    xrs = (V ** (1 / 3) - V0 ** (1 / 3)) / lr
    return E0 * (1 + xrs + c3 * xrs ** 3) * np.exp(-xrs)


def E_ER(V, *pars):
    return E_ER_pars(V, pars)


def get_min_nn_dist(atoms, cutoff=7):
    """Compute minimal nearest-neighbours (NN) distance in `atoms` (maximum up to `cutoff`)"""
    min_nn_dist = np.min(neighbor_list("d", atoms, cutoff=cutoff))
    return min_nn_dist


def generate_nndist_atoms(original_atoms, nn_distances, cutoff=7):
    min_nn_dist = get_min_nn_dist(original_atoms, cutoff=cutoff)

    atoms_list = []
    for z in nn_distances:
        at = original_atoms.copy()
        cell = at.get_cell()
        cell *= z / min_nn_dist
        at.set_cell(cell, scale_atoms=True)
        atoms_list.append(at)

    return atoms_list


def fit_eos(vpas, epas, n_fit_iter, e_best_rmse_threshold, random_state):
    if len(vpas) < 4:
        raise RuntimeError(
            f"Number of reliable data-points ({len(vpas)}) is less than minimal required for EOS (4)"
        )

    if random_state is not None:
        np.random.seed(random_state)

    p0 = np.array((-5, 20, 0.5, 1))
    # random shuffle for best params optimization
    e_best_rmse = np.inf
    best_parsER = None
    for it in range(n_fit_iter):
        try:
            dp0 = 0 if it == 0 else np.random.randn(4) * np.array([10, 20, 5, 5]) * 2

            parsER, _ = curve_fit(
                E_ER,
                vpas,
                epas,
                p0=p0 + dp0,
                maxfev=1000,
            )
            e_rmse = np.sqrt(np.mean((E_ER_pars(vpas, parsER) - epas) ** 2))
            if e_rmse < e_best_rmse:
                e_best_rmse = e_rmse
                best_parsER = parsER
                print("E_rmse:", e_rmse)
            if e_best_rmse < e_best_rmse_threshold:
                break
        except Exception as e:
            print("Exception:", e)

    return best_parsER, e_best_rmse


def plot_all(df, df_reliable=None, df_selected=None, plot_eos=True, plot_zbl=False):
    if not plt:
        return

    title = df.iloc[0]["ase_atoms"].get_chemical_formula()

    if plot_zbl:
        plt.plot(df["z"], df["epa_zbl"], "--", label="ZBL")

    if plot_eos:
        plt.plot(df["z"], df["epa_eos"], "-", label="EOS", color="gray")

    plt.plot(df["z"], df["epa"], "o-", color="red", label="ACE")

    if df_reliable is not None:
        plt.plot(
            df_reliable["z"],
            df_reliable["epa"],
            "o-",
            color="green",
            ls="--",
            label="ACE-reliable",
        )

    if df_selected is not None:
        plt.plot(
            df_selected["z"],
            df_selected["energy_corrected_per_atom"],
            "d-",
            color="blue",
            label="AUG data",
        )

    plt.legend()
    plt.yscale("symlog")  # , linear_width=1e-1)
    # plt.xlim(*nn_distance_range)
    plt.ylim(-10, None)
    plt.title(title)
    plt.xlabel("z, A")
    plt.ylabel("E, eV/at")
    plt.show()


def augment_structure_eos(
        atoms,
        calc,
        nn_distance_range=(1, 5),
        nn_distance_step=0.1,
        reliability_criteria=KINK,  # "extrapolation" or "kink"
        augmentation_type=EOS,  # "eos" or "zbl"
        epa_reliable_max=None,
        epa_aug_max=None,
        epa_aug_min=None,
        gamma_max=10,
        eos_fit_n_iter=20,
        eos_fit_rmse_threshold=0.5,
        eos_fit_random_state=None,
        plot_verbose=False,
        plot_eos=False,
        plot_zbl=False,
        zbl_r_in=0,
        zbl_r_out=4,
):
    """
    atoms: ASE atoms
    calc: (PyACE) calculator
    nn_distance_range: NN distance range default=(1, 5),
    nn_distance_step: NN step default=0.1,
    reliability_criteria: "extrapolation" or "kink"
    augmentation_type:  "eos" or "zbl"
    epa_reliable_max=None,
    epa_aug_max=None,
    epa_aug_min=None,
    gamma_max=10,
    eos_fit_n_iter=20,
    eos_fit_rmse_threshold=0.5,
    eos_fit_random_state=None,
    plot_verbose=False,
    plot_eos=False,
    plot_zbl=False,
    zbl_r_in=0,
    zbl_r_out=4,
    """

    plot_verbose = plot_verbose or (plot_eos or plot_zbl)
    compute_zbl = augmentation_type == ZBL or plot_zbl
    compute_eos = augmentation_type == EOS or plot_eos

    # plot it, if compute it
    # if plot_verbose:
    #     plot_zbl=compute_zbl
    #     plot_eos=compute_eos
    natoms = len(atoms)
    compute_gamma = reliability_criteria == EXTRAPOLATION
    df = compute_enn_df(atoms, calc, compute_zbl, nn_distance_range, nn_distance_step, compute_gamma,
                        zbl_r_in, zbl_r_out)

    df_reliable = select_reliable_enn_part(df, reliability_criteria, epa_reliable_max, gamma_max)

    epa_reliable_max = df_reliable[EPA_C].max()
    vpa_reliable_min = df_reliable[VPA_C].min()

    # generate prediction over wide range
    # zs_wide = np.arange(0.2, max(nn_distances), 0.25)
    # min_nn_dist = get_min_nn_dist(atoms)
    # v0 = atoms.get_volume() / natoms
    # vs_wide = (zs_wide / min_nn_dist) ** 3 * v0

    if compute_eos:
        best_parsER, e_best_rmse = fit_eos(
            df_reliable[VPA_C],
            df_reliable[EPA_C],
            n_fit_iter=eos_fit_n_iter,
            e_best_rmse_threshold=eos_fit_rmse_threshold,
            random_state=eos_fit_random_state,
        )
        print("BEST E_RMSE:", e_best_rmse)

        df[EPA_EOS_C] = E_ER_pars(df[VPA_C], best_parsER)
        df[FORCES_EOS_C] = df[ASE_ATOMS_C].map(lambda a: np.zeros((len(a), 3)))

        if e_best_rmse > eos_fit_rmse_threshold and not augmentation_type == ZBL:
            if plot_verbose:
                plot_all(
                    df, df_reliable, df_selected=None, plot_eos=True, plot_zbl=plot_zbl
                )
            raise RuntimeError(
                "Cannot reliabley fit EOS-ER to ACE data, E-RMSE={}".format(e_best_rmse)
            )

    # augment data

    df_selected = df.copy()
    if augmentation_type == ZBL:

        df_selected[EPA_CORRECTED_C] = df_selected[EPA_ZBL_C]
        df_selected[FORCES_C] = df_selected[FORCES_ZBL_C]
        df_selected = df_selected[df_selected[EPA_ZBL_C] > epa_reliable_max].copy()
    elif augmentation_type == EOS:
        df_selected[EPA_CORRECTED_C] = df_selected[EPA_EOS_C]
        df_selected[FORCES_C] = df_selected[FORCES_EOS_C]
    else:
        raise NotImplementedError(f"Unknown augmentation_type=`{augmentation_type}`")

    # 1. Volume less than min reliable volume
    df_selected = df_selected[df_selected[VPA_C] < vpa_reliable_min]

    # 2. Epa_aug max criteria
    if epa_aug_max is not None:
        df_selected = df_selected[
            df_selected[EPA_CORRECTED_C] <= epa_aug_max
            ]

    # 3. Epa_aug min criteria
    if epa_aug_min is not None:
        df_selected = df_selected[
            df_selected[EPA_CORRECTED_C] >= epa_aug_min
            ]

    df_selected[E_CORRECTED_C] = df_selected[EPA_CORRECTED_C] * natoms

    df_selected[NAME_C] = (
            "augmented/" + augmentation_type + "/" + atoms.get_chemical_formula()
    )

    if plot_verbose:
        plot_all(
            df,
            df_reliable,
            df_selected,
            plot_zbl=plot_zbl,
            plot_eos=plot_eos,
        )

    # remove calc
    df_selected[ASE_ATOMS_C].map(lambda a: a.set_calculator(None));
    return df_selected[
        [
            NAME_C,
            ASE_ATOMS_C,
            E_CORRECTED_C,
            EPA_CORRECTED_C,
            Z_C,
            FORCES_C,
        ]
    ]


def compute_enn_df(atoms, calc, compute_zbl=False, nn_distance_range=(1, 5),
                   nn_distance_step=0.1, compute_gamma=False,
                   zbl_r_in=0,
                   zbl_r_out=4, ):
    if compute_zbl:
        zblcalc = ZBLCalculator(cut_in=zbl_r_in, cutoff=zbl_r_out)

    nn_distances = list(np.arange(*nn_distance_range, nn_distance_step))
    natoms = len(atoms)
    structs = []
    epas = []
    vpas = []
    zs = []
    gammas = []
    gamma_per_atom = []
    epa_zbls = []
    fzbls = []
    for z, curr_atoms in zip(nn_distances, generate_nndist_atoms(atoms, nn_distances)):
        # compute ACE energy
        curr_atoms.set_calculator(calc)
        epas.append(curr_atoms.get_potential_energy() / natoms)
        vpas.append(curr_atoms.get_volume() / natoms)
        zs.append(z)

        if compute_gamma:
            gammas.append(calc.results[GAMMA_C].max())
            gamma_per_atom.append(calc.results[GAMMA_C])

        structs.append(curr_atoms)

        if compute_zbl:
            curr_atoms.set_calculator(zblcalc)
            epa_zbls.append(curr_atoms.get_potential_energy() / natoms)
            fzbls.append(curr_atoms.get_forces())
    df = (
        pd.DataFrame({VPA_C: vpas, EPA_C: epas, Z_C: zs, ASE_ATOMS_C: structs})
        .sort_values("vpa")
        .reset_index(drop=True)
    )
    if compute_gamma:
        df[GAMMA_C] = gammas
        df[GAMMA_PER_ATOM_C] = gamma_per_atom
    if compute_zbl:
        df[EPA_ZBL_C] = epa_zbls
        df[FORCES_ZBL_C] = fzbls
    return df


def select_reliable_enn_part(df, reliability_criteria=KINK, epa_reliable_max=None, gamma_max=1.5):
    # Selection of reliable part (required for all augmentation types)
    if reliability_criteria == KINK:
        peaks, _ = find_peaks(df[EPA_C])
        if len(peaks) == 0:
            df_reliable = df.copy()
            # if plot_verbose and augmentation_type != ZBL:
            #     plot_all(df, df_reliable, df_selected=None, plot_eos=False, plot_zbl=compute_zbl)
            # print("No kink in ACE-data E(V) found, this structure is good")
        else:
            # get up to last peak
            p = peaks[-1]
            df_reliable = df.iloc[p:].copy()
    elif reliability_criteria == EXTRAPOLATION:
        df_reliable = df[df[GAMMA_C] < gamma_max].copy()
    else:
        raise NotImplementedError(
            f"Reliability_criteria '{reliability_criteria}' is not implemented"
        )
    if epa_reliable_max is not None:
        df_reliable = df_reliable[df_reliable[EPA_C] <= epa_reliable_max].copy()
    return df_reliable
