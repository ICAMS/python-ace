import logging
from collections import defaultdict

import numpy as np
import os
import pandas as pd
import time

from typing import Dict, Union, Tuple, Optional

from ase.calculators.singlepoint import SinglePointCalculator

from pyace.atomicenvironment import aseatoms_to_atomicenvironment, generate_tp_atoms
from pyace.const import *
from pyace.process_df import compute_convexhull_dist, compute_corrected_energy, SINGLE_ATOM_ENERGY_DICT, \
    compute_shifted_scaled_corrected_energy

log = logging.getLogger(__name__)


def sizeof_fmt(file_name_or_size, suffix='B'):
    if isinstance(file_name_or_size, str):
        file_name_or_size = os.path.getsize(file_name_or_size)
    for unit in ['', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi']:
        if abs(file_name_or_size) < 1024.0:
            return "%3.1f%s%s" % (file_name_or_size, unit, suffix)
        file_name_or_size /= 1024.0
    return "%.1f%s%s" % (file_name_or_size, 'Yi', suffix)


def save_dataframe(df: pd.DataFrame, filename: str, protocol: int = 4):
    filename = os.path.abspath(filename)
    log.info("Writing fit pickle file: {}".format(filename))
    if filename.endswith("gzip"):
        compression = "gzip"
    else:
        compression = "infer"
    dirname = os.path.dirname(filename)
    os.makedirs(dirname, exist_ok=True)
    if not isinstance(df, DataFrameWithMetadata):
        log.info("Transforming to DataFrameWithMetadata")
        df = DataFrameWithMetadata(df)
    df.to_pickle(filename, protocol=protocol, compression=compression)
    log.info("Saved to file {} ({})".format(filename, sizeof_fmt(filename)))


def load_dataframe(filename: str, compression: str = "infer") -> pd.DataFrame:
    filesize = os.path.getsize(filename)
    log.info("Loading dataframe from pickle file {} ({})".format(filename, sizeof_fmt(filesize)))
    if filename.endswith(".gzip"):
        compression = "gzip"
    df = pd.read_pickle(filename, compression=compression)
    return df


def attach_single_point_calculator(row):
    atoms = row["ase_atoms"]
    energy = row["energy_corrected"]
    forces = row["forces"]
    calc = SinglePointCalculator(atoms, energy=energy, forces=forces)
    atoms.set_calculator(calc)
    return atoms


# define safe_min function, that return None of input is empty
def safe_min(val):
    try:
        return min((v for v in val if v is not None))
    except (ValueError, TypeError):
        return None


# define function that compute minimal distance
def calc_min_distance(ae):
    atpos = np.array(ae.x)
    nlist = ae.neighbour_list
    return safe_min(safe_min(np.linalg.norm(atpos[nlist[nat]] - atpos[nat], axis=1)) for nat in range(ae.n_atoms_real))


def check_df_non_empty(df: pd.DataFrame):
    if len(df) == 0:
        raise RuntimeError("Couldn't operate with empty dataset. Try to reduce filters and constraints")


def normalize_energy_forces_weights(df: pd.DataFrame):
    if df is None:
        return
    if WEIGHTS_ENERGY_COLUMN not in df.columns:
        raise ValueError("`{}` column not in dataframe".format(WEIGHTS_ENERGY_COLUMN))
    if WEIGHTS_FORCES_COLUMN not in df.columns:
        raise ValueError("`{}` column not in dataframe".format(WEIGHTS_FORCES_COLUMN))

    assert (df[WEIGHTS_FORCES_COLUMN].map(len) == df[FORCES_COL].map(len)).all()

    df[WEIGHTS_ENERGY_COLUMN] = df[WEIGHTS_ENERGY_COLUMN] / df[WEIGHTS_ENERGY_COLUMN].sum()
    # df[WEIGHTS_FORCES_COLUMN] = df[WEIGHTS_FORCES_COLUMN] * df[WEIGHTS_ENERGY_COLUMN]
    w_forces_norm = df[WEIGHTS_FORCES_COLUMN].map(sum).sum()
    df[WEIGHTS_FORCES_COLUMN] = df[WEIGHTS_FORCES_COLUMN] / w_forces_norm

    assert np.allclose(df[WEIGHTS_ENERGY_COLUMN].sum(), 1)
    assert np.allclose(df[WEIGHTS_FORCES_COLUMN].map(sum).sum(), 1)


class StructuresDatasetWeightingPolicy:
    def generate_weights(self, df):
        raise NotImplementedError


# DataFrameWithMetadata is left for backward compatibility only
class DataFrameWithMetadata(pd.DataFrame):
    # normal properties
    _metadata = ["metadata"]

    @property
    def _constructor(self):
        return DataFrameWithMetadata

    @property
    def metadata_dict(self):
        if not hasattr(self, "metadata") or self.metadata is None:
            self.metadata = {}
        return self.metadata

    @metadata_dict.setter
    def metadata_dict(self, value):
        self.metadata = value


class EnergyBasedWeightingPolicy(StructuresDatasetWeightingPolicy):

    def __init__(self,
                 nfit=None,
                 cutoff=None,
                 DElow=1.0,
                 DEup=10.0,
                 DE=1.0,
                 DF=1.0,
                 wlow=None,  # 0.75,
                 DFup=None,
                 n_lower=None,
                 n_upper=None,
                 reftype='all',
                 seed=None,
                 energy="convex_hull"):
        """

        :param nfit:
        :param cutoff:
        :param DElow:
        :param DEup:
        :param DE:
        :param DF:
        :param wlow:
        :param reftype:
        :param seed:
        :param energy: "convex" or "cohesive"
        """
        # #### Data selection and weighting
        # number of structures to be used in fit
        self.nfit = nfit
        # lower threshold: all structures below lower threshold are used in the fit (if fewer than nfit)
        self.DElow = DElow
        # upper threshold: structures between lower and upper threshold are selected randomly (after DElow structures)
        self.DEup = DEup
        # Delta E: energy offset in energy weights
        self.DE = DE
        # Delta F: force offset in force weights
        self.DF = DF

        # maximal forces projection
        self.DFup = DFup
        # relative fraction of structures below lower threshold in energy weights
        if wlow is not None and isinstance(wlow, str) and wlow.lower() == "none":
            wlow = None
        self.wlow = wlow
        # use all/bulk/cluster reference data
        self.reftype = reftype
        # random seed
        self.seed = seed

        self.cutoff = cutoff

        self.energy = energy  # cohesive or convex_hull
        self.n_lower = n_lower
        self.n_upper = n_upper

    def __str__(self):
        return (
                "EnergyBasedWeightingPolicy(nfit={nfit}, n_lower={n_lower}, n_upper={n_upper}, energy={energy}," +
                " DElow={DElow}, DEup={DEup}, DFup={DFup}, DE={DE}, " +
                "DF={DF}, wlow={wlow}, reftype={reftype},seed={seed})").format(nfit=self.nfit,
                                                                               n_lower=self.n_lower,
                                                                               n_upper=self.n_upper,
                                                                               cutoff=self.cutoff,
                                                                               DElow=self.DElow,
                                                                               DEup=self.DEup,
                                                                               DFup=self.DFup,
                                                                               DE=self.DE,
                                                                               DF=self.DF,
                                                                               wlow=self.wlow,
                                                                               energy=self.energy,
                                                                               reftype=self.reftype, seed=self.seed)

    def generate_weights(self, df):
        if self.nfit is None:
            if self.n_upper is None and self.n_lower is None:
                self.nfit = len(df)
                log.info("Set nfit to the dataset size {}".format(self.nfit))
            elif self.n_upper is not None and self.n_lower is not None:
                self.nfit = self.n_upper + self.n_lower
                log.info("Set nfit ({}) = n_lower ({}) + n_upper ({})".format(self.nfit, self.n_lower, self.n_upper))
            else:  # nfit=None, one of n_upper or n_lower not None
                raise ValueError("nfit is None. Please provide both n_lower and n_upper")
        else:  # nfit is not None
            if self.n_upper is not None or self.n_lower is not None:
                raise ValueError("nfit is not None. No n_lower or n_upper is expected")

        if self.reftype == "bulk":
            log.info("Reducing to bulk data")
            df = df[df.pbc]
            log.info("Dataset size after reduction: {}".format(len(df)))
        elif self.reftype == "cluster":
            log.info("Reducing to cluster data")
            df = df[~df.pbc]
            log.info("Dataset size after reduction: {}".format(len(df)))
        else:
            log.info("Keeping bulk and cluster data")

        check_df_non_empty(df)

        if self.cutoff is not None:
            log.info("EnergyBasedWeightingPolicy::cutoff is provided but will be ignored")
        else:
            log.info("No cutoff for EnergyBasedWeightingPolicy is provided, no structures outside cutoff that " +
                     "will now be removed")

        # #### structure selection

        if self.energy == "convex_hull":
            log.info("EnergyBasedWeightingPolicy: energy reference frame - convex hull distance (if possible)")
            # generate "e_chull_dist_per_atom" column
            compute_convexhull_dist(df, energy_per_atom_column=E_CORRECTED_PER_ATOM_COLUMN)
            df[EFFECTIVE_ENERGY] = df[E_CHULL_DIST_PER_ATOM]
        elif self.energy == "cohesive":
            log.info("EnergyBasedWeightingPolicy: energy reference frame - cohesive energy")
            emin = df[E_CORRECTED_PER_ATOM_COLUMN].min()
            df[EFFECTIVE_ENERGY] = df[E_CORRECTED_PER_ATOM_COLUMN] - emin
        else:
            raise ValueError(
                ("Unknown EnergyBasedWeightingPolicy.energy={} settings. Possible values: convex_hull (default) or " +
                 "cohesive").format(self.energy))

        if self.DFup is not None:
            log.info("Maximal allowed on-atom force vector length is DFup = {:.3f}".format(self.DFup))
            fmax_column = df["forces"].map(lambda f: np.max(np.linalg.norm(f, axis=1)))
            size_before = len(df)
            df = df[fmax_column <= self.DFup]
            size_after = len(df)
            log.info("{} structures with higher than dFup forces are removed. Current size: {} structures".format(
                size_before - size_after, size_after))

        check_df_non_empty(df)

        # remove high energy structures
        df = df[df[EFFECTIVE_ENERGY] < self.DEup]  # .reset_index(drop=True)

        check_df_non_empty(df)

        elow_mask = df[EFFECTIVE_ENERGY] < self.DElow
        eup_mask = df[EFFECTIVE_ENERGY] >= self.DElow
        nlow = elow_mask.sum()
        nup = eup_mask.sum()
        log.info("{} structures below DElow={} eV/atom".format(nlow, self.DElow))
        log.info("{} structures between DElow={} eV/atom and DEup={} eV/atom".format(nup, self.DElow, self.DEup))
        log.info("all other structures were removed")

        low_candidate_list = df.index[elow_mask]
        up_candidate_list = df.index[eup_mask]

        np.random.seed(self.seed)
        # lower tier
        if self.n_lower is not None:
            if nlow <= self.n_lower:
                low_selected_list = low_candidate_list
            else:  # nlow>self.n_lower
                low_selected_list = np.random.choice(low_candidate_list, self.n_lower, replace=False)
        else:  # no n_lower provided
            if nlow <= self.nfit:
                low_selected_list = low_candidate_list
            else:  # nlow >nfit
                low_selected_list = np.random.choice(low_candidate_list, self.nfit, replace=False)

        # upper tier
        if self.n_upper is not None:
            if self.n_upper < nup:
                up_selected_list = np.random.choice(up_candidate_list, self.n_upper, replace=False)
            else:
                up_selected_list = up_candidate_list
        else:
            nremain = self.nfit - len(low_selected_list)
            if nremain <= nup:
                up_selected_list = np.random.choice(up_candidate_list, nremain, replace=False)
            else:
                up_selected_list = up_candidate_list

        takelist = np.hstack([low_selected_list, up_selected_list])
        np.random.shuffle(takelist)

        df = df.loc[takelist]  # .reset_index(drop=True)
        check_df_non_empty(df)

        elow_mask = df[EFFECTIVE_ENERGY] < self.DElow
        eup_mask = df[EFFECTIVE_ENERGY] >= self.DElow

        log.info("{} structures were selected".format(len(df)))

        assert elow_mask.sum() + eup_mask.sum() == len(df)
        if len(up_selected_list) == 0 and self.wlow is not None and self.wlow != 1.0:
            log.warning(("All structures were taken from low-tier, but relative weight of low-tier (wlow={}) " +
                         "is less than one. It will be adjusted to one").format(self.wlow))
            self.wlow = 1.0
        # ### energy weights
        log.info("Setting up energy weights")
        DE = abs(self.DE)

        df[WEIGHTS_ENERGY_COLUMN] = 1 / (df[EFFECTIVE_ENERGY] + DE) ** 2
        df[WEIGHTS_ENERGY_COLUMN] = df[WEIGHTS_ENERGY_COLUMN] / df[WEIGHTS_ENERGY_COLUMN].sum()

        e_weights_sum = df[WEIGHTS_ENERGY_COLUMN].sum()
        assert np.allclose(e_weights_sum, 1), "Energy weights doesn't sum up to one: {}".format(e_weights_sum)
        #  ### relative weights of structures below and above threshold DElow
        wlowcur = df.loc[elow_mask, WEIGHTS_ENERGY_COLUMN].sum()
        wupcur = df.loc[eup_mask, WEIGHTS_ENERGY_COLUMN].sum()

        log.info("Current relative energy weights: {:.3f}/{:.3f}".format(wlowcur, wupcur))
        if self.wlow is not None:
            self.wlow = float(self.wlow)
            if 1.0 > wlowcur > 0.:
                log.info("Will be adjusted to            : {:.3f}/{:.3f}".format(self.wlow, 1 - self.wlow))
                flow = self.wlow / wlowcur
                if wlowcur == 1:
                    fup = 0
                else:
                    fup = (1 - self.wlow) / (1 - wlowcur)

                df.loc[elow_mask, WEIGHTS_ENERGY_COLUMN] = flow * df.loc[elow_mask, WEIGHTS_ENERGY_COLUMN]
                df.loc[eup_mask, WEIGHTS_ENERGY_COLUMN] = fup * df.loc[eup_mask, WEIGHTS_ENERGY_COLUMN]
                # log.info('df["w_energy"].sum() after = {}'.format(df["w_energy"].sum()))
                energy_weights_sum = df[WEIGHTS_ENERGY_COLUMN].sum()
                assert np.allclose(energy_weights_sum, 1), "Energy weights sum differs from one and equal to {}".format(
                    energy_weights_sum)
                wlowcur = df.loc[elow_mask, WEIGHTS_ENERGY_COLUMN].sum()
                wupcur = df.loc[eup_mask, WEIGHTS_ENERGY_COLUMN].sum()
                log.info("After adjustment: relative energy weights: {:.3f}/{:.3f}".format(wlowcur, wupcur))
                assert np.allclose(wlowcur, self.wlow)
                assert np.allclose(wupcur, 1 - self.wlow)
            else:
                log.warning("No weights adjustment possible")
        else:
            log.warning("wlow=None, no weights adjustment")

        # ### force weights
        log.info("Setting up force weights")
        DF = abs(self.DF)
        df[FORCES_COL] = df[FORCES_COL].map(np.array)
        assert (df["forces"].map(len) == df["ase_atoms"].map(len)).all(), ValueError(
            "Number of atoms doesn't corresponds to shape of forces")
        df[WEIGHTS_FORCES_COLUMN] = df[FORCES_COL].map(lambda forces: 1 / (np.sum(forces ** 2, axis=1) + DF))
        assert (df[WEIGHTS_FORCES_COLUMN].map(len) == df["NUMBER_OF_ATOMS"]).all()
        df[WEIGHTS_FORCES_COLUMN] = df[WEIGHTS_FORCES_COLUMN] * df[WEIGHTS_ENERGY_COLUMN]
        w_forces_norm = df[WEIGHTS_FORCES_COLUMN].map(sum).sum()
        df[WEIGHTS_FORCES_COLUMN] = df[WEIGHTS_FORCES_COLUMN] / w_forces_norm

        energy_weights_sum = df[WEIGHTS_ENERGY_COLUMN].sum()
        assert np.allclose(energy_weights_sum, 1), "Energy weights sum differs from one and equal to {}".format(
            energy_weights_sum)
        forces_weights_sum = df[WEIGHTS_FORCES_COLUMN].map(sum).sum()
        assert np.allclose(forces_weights_sum, 1), "Forces weights sum differs from one and equal to {}".format(
            forces_weights_sum)
        return df

    def plot(self, df):
        import matplotlib.pyplot as plt
        elist = df[E_CORRECTED_PER_ATOM_COLUMN]
        dminlist = df["dmin"]
        print("Please check that your cutoff makes sense in the following graph")
        xh = [0, self.cutoff + 1]
        yh = [10 ** -3, 10 ** -3]
        yv = [10 ** -10, 10 ** 3]
        xv = [self.cutoff, self.cutoff]
        fig, ax = plt.subplots(1)

        ax.semilogy(dminlist, elist.abs(), '+', label="data")
        ax.semilogy(xh, yh, '--', label="1 meV")
        ax.semilogy(xv, yv, '-', label="cutoff")

        plt.ylabel(r"| cohesive energy | / eV")
        plt.xlabel("dmin / ${\mathrm{\AA}}$")
        plt.title("Reference data overview")
        plt.legend()
        plt.xlim(1, self.cutoff + 0.5)
        plt.ylim(10 ** -4, 10 ** 2)
        plt.show()


class UniformWeightingPolicy(StructuresDatasetWeightingPolicy):

    def __init__(self):
        pass

    def __str__(self):
        return "UniformWeightingPolicy()"

    def generate_weights(self, df):
        df[WEIGHTS_ENERGY_COLUMN] = 1. / len(df)
        df[WEIGHTS_FORCES_COLUMN] = df[FORCES_COL].map(lambda forces: np.ones(len(forces)))
        df[WEIGHTS_FORCES_COLUMN] = df[WEIGHTS_FORCES_COLUMN] * df[WEIGHTS_ENERGY_COLUMN]
        normalize_energy_forces_weights(df)
        return df


class ExternalWeightingPolicy(StructuresDatasetWeightingPolicy):

    def __init__(self, filename_or_df: str):
        """
        :param filename: .pckl.gzip filename of dataframe with index and  `w_energy` and `w_forces` columns
        """
        self.filename_or_df = filename_or_df
        self.weights_df = None

    def __str__(self):
        return "ExternalWeightingPolicy(filename={filename})".format(
            filename=self.filename_or_df if isinstance(self.filename_or_df, str) else "[pd.DataFrame]")

    def generate_weights(self, df):
        if isinstance(self.filename_or_df, str):
            log.info("Loading external weights dataframe {}".format(self.filename_or_df))
            self.weights_df = pd.read_pickle(self.filename_or_df, compression="gzip")
        else:
            self.weights_df = self.filename_or_df
        log.info("External weights dataframe loaded, it contains {} entries".format(len(self.weights_df)))

        # check that columns are presented
        for col in [WEIGHTS_ENERGY_COLUMN, WEIGHTS_FORCES_COLUMN]:
            assert col in self.weights_df.columns, "`{}` column not in external weights dataframe".format(col)

        if not all([w_ind in df.index for w_ind in self.weights_df.index]):
            error_msg = "Not all structure indices from weights dataframe are in original dataframe"
            log.error(error_msg)
            raise ValueError(error_msg)

        # join df and self.weights_df -> df
        for col_to_drop in [WEIGHTS_ENERGY_COLUMN, WEIGHTS_FORCES_COLUMN]:
            if col_to_drop in df.columns:
                df.drop(columns=col_to_drop, inplace=True)

        mdf = pd.merge(df, self.weights_df[[WEIGHTS_ENERGY_COLUMN, WEIGHTS_FORCES_COLUMN]], left_index=True,
                       right_index=True)
        if not (mdf[FORCES_COL].map(len) == mdf[WEIGHTS_FORCES_COLUMN].map(len)).all():
            error_msg = ("Shape of the `{}` column doesn't correspond to the shape of "
                         "`forces` column in original dataframe").format(WEIGHTS_FORCES_COLUMN)
            log.error(error_msg)
            raise ValueError(error_msg)
        log.info("External weights joined to original dataframe")
        normalize_energy_forces_weights(mdf)
        log.info("Weights normalized")
        # check normalization
        energy_weights_sum = mdf[WEIGHTS_ENERGY_COLUMN].sum()
        assert np.allclose(energy_weights_sum, 1), "Energy weights sum differs from one and equal to {}".format(
            energy_weights_sum)
        forces_weights_sum = mdf[WEIGHTS_FORCES_COLUMN].map(sum).sum()
        assert np.allclose(forces_weights_sum, 1), "Forces weights sum differs from one and equal to {}".format(
            forces_weights_sum)
        return mdf


def train_test_split(df, test_size: Union[float, int]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split df into train and test
    :param df: data frame
    :param test_size:
    :return:
    """
    n_samples = len(df)
    if test_size >= 1:
        train_size = n_samples - test_size
    elif 0 < test_size and test_size < 1:
        test_size = int(n_samples * test_size)
        train_size = n_samples - test_size

    inds = np.arange(len(df))
    np.random.shuffle(inds)

    train_inds = sorted(inds[:train_size])
    test_inds = sorted(inds[train_size:])

    return df.iloc[train_inds].copy(), df.iloc[test_inds].copy()


def get_weighting_policy(weighting_policy_spec: Dict) -> StructuresDatasetWeightingPolicy:
    if weighting_policy_spec is None:
        return UniformWeightingPolicy()
    elif isinstance(weighting_policy_spec, StructuresDatasetWeightingPolicy):
        return weighting_policy_spec
    elif not isinstance(weighting_policy_spec, dict):
        raise ValueError(
            "Weighting policy specification ('weighting' option) should be a dictionary " +
            "or StructuresDatasetWeightingPolicy but got " + str(type(weighting_policy_spec)))

    weighting_policy_spec = weighting_policy_spec.copy()

    if WEIGHTING_TYPE_KW not in weighting_policy_spec:
        raise ValueError("Weighting 'type' is not specified")

    if weighting_policy_spec[WEIGHTING_TYPE_KW] == ENERGYBASED_WEIGHTING_POLICY:
        del weighting_policy_spec[WEIGHTING_TYPE_KW]
        weighting_policy = EnergyBasedWeightingPolicy(**weighting_policy_spec)
    elif weighting_policy_spec[WEIGHTING_TYPE_KW] == EXTERNAL_WEIGHTING_POLICY:
        del weighting_policy_spec[WEIGHTING_TYPE_KW]
        weighting_policy = ExternalWeightingPolicy(**weighting_policy_spec)
    else:
        raise ValueError("Unknown weighting 'type': " + weighting_policy_spec[WEIGHTING_TYPE_KW])

    return weighting_policy


def get_elements_mapper_dict(df):
    elements = set()
    for at in df["ase_atoms"]:
        elements.update(at.get_chemical_symbols())
    elements = sorted(elements)
    elements_mapper_dict = {el: i for i, el in enumerate(elements)}
    return elements_mapper_dict


def generate_atomic_env_column(df, cutoff=9, elements_mapper_dict=None, ase_atoms_column="ase_atoms"):
    if elements_mapper_dict is None:
        elements_mapper_dict = get_elements_mapper_dict(df)

    df[ATOMIC_ENV_COLUMN] = df[ase_atoms_column].map(
        lambda at: aseatoms_to_atomicenvironment(at, cutoff=cutoff,
                                                 elements_mapper_dict=elements_mapper_dict))


def get_reference_dataset(evaluator_name, dataframe_fname):
    return ACEDataset(data_config={"filename": dataframe_fname}, evaluator_name=evaluator_name).fitting_data


def apply_weights(df: Optional[pd.DataFrame], weighting_policy_spec, ignore_weights=False) -> Optional[pd.DataFrame]:
    if df is None:
        return
    if WEIGHTS_ENERGY_COLUMN in df.columns and WEIGHTS_FORCES_COLUMN in df.columns and not ignore_weights:
        log.info("Both weighting columns ({} and {}) are found, no another weighting policy will be applied".format(
            WEIGHTS_ENERGY_COLUMN, WEIGHTS_FORCES_COLUMN))
    else:
        if ignore_weights and (WEIGHTS_ENERGY_COLUMN in df.columns or WEIGHTS_FORCES_COLUMN in df.columns):
            log.info("Existing weights are ignored, weighting policy calculation is forced")

        if weighting_policy_spec is None:
            log.info("No weighting policy is specified, setting default weighting policy")
        weighting_policy = get_weighting_policy(weighting_policy_spec)

        log.info("Apply weights policy: " + str(weighting_policy))
        df = weighting_policy.generate_weights(df)
    if WEIGHTS_FACTOR in df.columns:
        log.info("Weights factor column `{}` is found, multiplying energy anf forces weights by this factor".format(
            WEIGHTS_FACTOR))
        df[WEIGHTS_ENERGY_COLUMN] *= df[WEIGHTS_FACTOR]
        df[WEIGHTS_FORCES_COLUMN] *= df[WEIGHTS_FACTOR]
    return df


def adjust_aug_weights(df: Optional[pd.DataFrame], aug_factor) -> Optional[pd.DataFrame]:
    if df is None:
        return
    if "name" in df.columns:
        aug_mask = df["name"].str.startswith("augmented")
        if aug_mask.sum() > 0:
            log.info(f"{aug_mask.sum()} augmented structures found in dataset")

            #  if weights are NAN - fallback to const(median) * aug_factor
            if df.loc[aug_mask, WEIGHTS_ENERGY_COLUMN].isna().any() or df.loc[
                aug_mask, WEIGHTS_FORCES_COLUMN].isna().any():
                log.info(
                    "Augmented weights are not set, probably because real data weights are provided already")
                we_const = df.loc[~aug_mask, WEIGHTS_ENERGY_COLUMN].median()
                wf_const = np.median(np.hstack(df.loc[~aug_mask, WEIGHTS_FORCES_COLUMN]))
                log.info(f"Estimating median weights for real data: {we_const=:.5g}, {wf_const=:.5g}")
                df.loc[aug_mask, WEIGHTS_ENERGY_COLUMN] = we_const
                df.loc[aug_mask, WEIGHTS_FORCES_COLUMN] = wf_const * df.loc[aug_mask, "ase_atoms"].map(
                    lambda at: np.ones((len(at),)))
                if "w_forces_mask" in df.columns:
                    df.loc[aug_mask, WEIGHTS_FORCES_COLUMN] = df.loc[aug_mask, WEIGHTS_FORCES_COLUMN] * df.loc[
                        aug_mask, "w_forces_mask"]
            log.info(f"Decreasing augmented weights by factor {aug_factor:.3g}")
            df.loc[aug_mask, WEIGHTS_ENERGY_COLUMN] *= aug_factor
            df.loc[aug_mask, WEIGHTS_FORCES_COLUMN] *= aug_factor
    return df


def big_warning(msg):
    tot_msg = ""
    lines = msg.split("\n")
    max_len = max(map(len, lines))
    for m in lines:
        tot_msg += f"# " + m + " " * (max_len - len(m)) + " #\n"
    tot_msg = ("\n" + "#" * (max_len + 4) + "\n" +  # first ### line
               "#" + " " * (max_len + 2) + "#\n" +
               tot_msg +
               "#" + " " * (max_len + 2) + "#\n" +
               "#" * (max_len + 4)  # last ### line
               )
    log.warning(tot_msg)


class ACEDataset:
    def __init__(self,
                 data_config,
                 weighting_policy_spec=None,
                 cutoff=10, elements=None,
                 evaluator_name="tensorpot"):
        self.data_config = data_config
        self.weighting_policy_spec = weighting_policy_spec
        self.cutoff = cutoff
        self.elements = elements
        self.evaluator_name = evaluator_name
        self.fitting_data = None
        self.test_data = None
        self.weighting_policy = None

        self.datapath = self.data_config.get("datapath")
        self.reference_energy = self.data_config.get("reference_energy")
        self.ignore_weights = self.data_config.get("ignore_weights")

        # result column name : tuple(mapper function,  kwargs)
        self.ase_atoms_transformers = {}

        # add the transformer depending on the evaluator
        if evaluator_name == PYACE_EVAL:
            elements_mapper_dict = None
            if elements is not None:
                elements_mapper_dict = {el: i for i, el in enumerate(sorted(elements))}
                log.info("Elements-to-indices mapping for 'atomic_env' construction: {}".format(elements_mapper_dict))
            if elements_mapper_dict is None:
                log.info("Elements-to-indices mapping for 'atomic_env' construction is NOT provided")
            self.add_ase_atoms_transformer(ATOMIC_ENV_DF_COLUMN, aseatoms_to_atomicenvironment, cutoff=cutoff,
                                           elements_mapper_dict=elements_mapper_dict)
        elif evaluator_name == TENSORPOT_EVAL:
            self.add_ase_atoms_transformer(TP_ATOMS_DF_COLUMN, generate_tp_atoms, cutoff=cutoff, verbose=False)
        else:
            raise ValueError(f"Unknown evaluator type '{evaluator_name}', only '{TENSORPOT_EVAL}' is supported")

        self.prepare_datasets()

    def prepare_datasets(self):

        train_filenames = self.data_config["filename"]
        self.fitting_data = self.load_dataset(train_filenames)
        # self.reference_energy will be updated for 'auto' option in self.process_dataset
        self.fitting_data = self.process_dataset(self.fitting_data)

        if "test_filename" in self.data_config:
            test_filenames = self.data_config["test_filename"]
            self.test_data = self.load_dataset(test_filenames)
        elif "test_size" in self.data_config:
            test_size = self.data_config["test_size"]
            log.info("Splitting out test dataset (test_size = {}) from main dataset({} samples)".
                     format(test_size, len(self.fitting_data)))
            self.fitting_data, self.test_data = train_test_split(self.fitting_data, test_size=test_size)
        self.test_data = self.process_dataset(self.test_data)

        # apply weights
        if self.test_data is not None:
            # for joint train+test
            self.fitting_data["train"] = True
            self.test_data["train"] = False
            joint_df = pd.concat([self.fitting_data, self.test_data], axis=0)
            joint_df = apply_weights(joint_df, self.weighting_policy_spec, self.ignore_weights)
            self.fitting_data = joint_df.query("train").reset_index(drop=True)
            self.test_data = joint_df.query("~train").reset_index(drop=True)
            # self.test_data = apply_weights(self.test_data, self.weighting_policy_spec, self.ignore_weights)
        else:
            self.fitting_data = apply_weights(self.fitting_data, self.weighting_policy_spec, self.ignore_weights)

        # decrease augmented weights
        aug_factor = self.data_config.get("aug_factor", 1e-4)
        self.fitting_data = adjust_aug_weights(self.fitting_data, aug_factor)
        self.test_data = adjust_aug_weights(self.test_data, aug_factor)

        # normalize
        normalize_energy_forces_weights(self.fitting_data)
        normalize_energy_forces_weights(self.test_data)

    def load_dataset(self, filenames):
        """Load multiple dataframes and concatenate it into one"""
        files_to_load = self.get_actual_filenames(filenames)
        log.info("Search for dataset file(s): " + str(files_to_load))
        if files_to_load is not None:
            dfs = []
            for i, fname in enumerate(files_to_load):
                log.info(f"#{i + 1}/{len(files_to_load)}: try to load {fname}")
                df = load_dataframe(fname)
                log.info(f" {len(df)} structures found")
                if "name" not in df.columns:
                    df["name"] = fname + ":" + df.index.map(str)
                dfs.append(df)
            tot_df = pd.concat(dfs, axis=0).reset_index(drop=True)
        else:  # if ref_df is still not loaded, try to query from DB
            raise RuntimeError("No files to load")
        return tot_df

    def get_actual_filenames(self, filenames):
        """Return list of dataset filenames (with attached datapath)"""
        if filenames is not None:
            if isinstance(filenames, str):
                filename_list = [filenames]
            elif isinstance(filenames, list):
                filename_list = filenames
            else:
                raise ValueError(f"Non-supported type of filename: `{filenames}` (type: {type(filenames)})")

            files_to_load = []
            for f in filename_list:
                if os.path.isfile(f) or self.datapath is None:
                    files_to_load.append(f)
                else:
                    files_to_load.append(os.path.join(self.datapath, f))

        else:
            raise ValueError("Dataset filename is not provided")
        return files_to_load

    def process_dataset(self, df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
        if df is None:
            return
        # check for "ase_atoms", "energy" or "energy_corrected", "forces" columns

        # check for ASE_ATOMS
        if ASE_ATOMS not in df.columns:
            raise ValueError(f"Dataframe is corrupted: no '{ASE_ATOMS}' column found")
        # check for 'energy' or 'energy_corrected'
        if ENERGY not in df.columns and ENERGY_CORRECTED_COL not in df.columns:
            raise ValueError("Column `energy` or `energy_corrected` not found in dataset")
        # check for 'forces'
        if FORCES_COL not in df.columns:
            raise ValueError("Column `forces` not found in dataset")

        df[NUMBER_OF_ATOMS] = df[ASE_ATOMS].map(len)

        # check force shapes [nat,3]
        assert df.apply(lambda row: np.shape(row[FORCES_COL]) == (row[NUMBER_OF_ATOMS], 3),
                        axis=1).all(), f"{FORCES_COL} has wrong shape. It should be [nat,3]"

        tot_atoms_num = df[NUMBER_OF_ATOMS].sum()
        mean_atoms_num = df[NUMBER_OF_ATOMS].mean()
        log.info(f"Processing structures dataframe. Shape: {df.shape}")
        log.info(f"Total number of atoms: {tot_atoms_num}")
        log.info(f"Mean number of atoms per structure: {mean_atoms_num:.1f}")
        df[PBC] = df[ASE_ATOMS].map(lambda atoms: np.all(atoms.pbc))

        if self.reference_energy is not None:  # use ENERGY column to generate
            # possible options in self.config
            # 1. reference_energy: VASP_PBE_500_NM  #, etc...
            # 2. reference_energy: {Al: -0.123, Cu: -0.456, shift: -0.123}
            # 3. reference_energy: auto
            # 3. reference_energy: 0
            log.info("Reference energy is provided, constructing 'energy_corrected'")

            if isinstance(self.reference_energy, str):
                if self.reference_energy in SINGLE_ATOM_ENERGY_DICT:
                    log.info(f"Using {self.reference_energy} as presets calculator name")
                    compute_corrected_energy(df, calculator_name=self.reference_energy)
                elif self.reference_energy == "auto":
                    log.info(f"Computing least-square energy shift and correction")
                    self.reference_energy = compute_shifted_scaled_corrected_energy(df)
                    log.info(
                        f"Computed single-atom reference energy: {self.reference_energy}")
                else:
                    raise ValueError(f"Unsupported data::reference_energy option ('{self.reference_energy}')."
                                     "Must be dict like {Al: -0.123, Cu: -0.456} or one of "
                                     f"{['auto'] + list(SINGLE_ATOM_ENERGY_DICT.keys())}")
            elif isinstance(self.reference_energy, dict):
                log.info(f"Using {self.reference_energy} as single-atom energies")
                compute_corrected_energy(df, esa_dict=self.reference_energy)
            elif isinstance(self.reference_energy, (float, int)):  # just copy energy to energy_corrected
                log.info(f"Using constant reference energy {self.reference_energy}")
                self.reference_energy = defaultdict(lambda: self.reference_energy)
                compute_corrected_energy(df, esa_dict=self.reference_energy)
        elif ENERGY_CORRECTED_COL not in df.columns:
            raise ValueError(f"Neither '{ENERGY_CORRECTED_COL}' column nor 'data::reference_energy' option provided")

        # check ENERGY_CORRECTED_COL is not NAN
        assert not df[ENERGY_CORRECTED_COL].isna().any(), f"{ENERGY_CORRECTED_COL} contains NaNs"

        # enforce calculation of E_CORRECTED_PER_ATOM_COLUMN to avoid mistakes
        df[E_CORRECTED_PER_ATOM_COLUMN] = df[ENERGY_CORRECTED_COL] / df[NUMBER_OF_ATOMS]

        assert not df[
            E_CORRECTED_PER_ATOM_COLUMN].isna().any(), f"{E_CORRECTED_PER_ATOM_COLUMN} column contains NaN"
        assert not df[FORCES_COL].map(lambda f: np.any(np.isnan(f))).any(), f"{FORCES_COL} column contains NaN"

        epa_min = df[E_CORRECTED_PER_ATOM_COLUMN].min()
        epa_max = df[E_CORRECTED_PER_ATOM_COLUMN].max()

        epa_abs_min = df[E_CORRECTED_PER_ATOM_COLUMN].abs().min()
        epa_abs_max = df[E_CORRECTED_PER_ATOM_COLUMN].abs().max()

        log.info(f"Min/max energy per atom: [{epa_min:.3f}, {epa_max:.3f}] eV/atom")
        log.info(f"Min/max abs energy per atom: [{epa_abs_min:.3f}, {epa_abs_max:.3f}] eV/atom")

        # check energy and forces range!
        if epa_min < -20 or epa_max > 250:
            # re-run with self.reference_energy='auto'
            if self.reference_energy is None:
                big_warning(f"Some values of corrected energy (min={epa_min:.3g} eV/atom, max={epa_max:.3g} eV/atom) are too extreme,\n" +
                            "i.e. <-20 eV/atom or >250 eV/atom\n" +
                            "`reference_energy` will be computed automatically.")
                self.reference_energy = 'auto'
                return self.process_dataset(df)
            big_warning(f"Some values of corrected energy (min={epa_min:.3g} eV/atom, max={epa_max:.3g} eV/atom) are too extreme,\n" +
                        "i.e. <-20 eV/atom or >250 eV/atom\n" +
                        "Correct your energy or use data::reference_energy: auto option !!!")
        # enforce attach single point calculator to avoid mistakes
        df[ASE_ATOMS] = df.apply(attach_single_point_calculator, axis=1)
        log.info("Attaching SinglePointCalculator to ASE atoms...done")

        log.info("Construction of neighbour lists...")
        start = time.time()
        self.apply_ase_atoms_transformers(df)
        end = time.time()
        time_elapsed = end - start
        log.info("Construction of neighbour lists...done within {:.3g} sec ({:.3g} ms/atom)".
                 format(time_elapsed, time_elapsed / tot_atoms_num * 1e3))

        return df

    def add_ase_atoms_transformer(self, result_column_name, transformer_func, **kwargs):
        self.ase_atoms_transformers[result_column_name] = (transformer_func, kwargs)

    def apply_ase_atoms_transformers(self, df):
        apply_function = df["ase_atoms"].apply

        for res_column_name, (transformer, kwargs) in self.ase_atoms_transformers.items():
            l1 = len(df)
            cur_cutoff = kwargs["cutoff"]
            log.info(f"Building '{res_column_name}' (dataset size {l1}, cutoff={cur_cutoff:.3f}A)...")
            df[res_column_name] = apply_function(transformer, **kwargs)
            df.dropna(subset=[res_column_name], inplace=True)
            l2 = len(df)
            log.info("Dataframe size after transform: " + str(l2))
