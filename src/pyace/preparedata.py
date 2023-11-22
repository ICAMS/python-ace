import logging
from collections import Counter

import numpy as np
import os
import pandas as pd
import time

from typing import Dict

from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator

from pyace.atomicenvironment import aseatoms_to_atomicenvironment
from pyace.const import *
from scipy.spatial import ConvexHull

DMIN_COLUMN = "dmin"
ATOMIC_ENV_COLUMN = "atomic_env"
FORCES_COLUMN = "forces"
E_CORRECTED_PER_ATOM_COLUMN = "energy_corrected_per_atom"
WEIGHTS_FORCES_COLUMN = "w_forces"
WEIGHTS_ENERGY_COLUMN = "w_energy"
WEIGHTS_FACTOR = "w_factor"
REF_ENERGY_KW = "ref_energy"
E_CHULL_DIST_PER_ATOM = "e_chull_dist_per_atom"
E_FORMATION_PER_ATOM = "e_formation_per_atom"
EFFECTIVE_ENERGY = "effective_energy"

log = logging.getLogger(__name__)

REF_PROP_NAME = '1-body-000001:static'
REF_GENERIC_PROTOTYPE_NAME = '1-body-000001'

# ## QUERY DATA
LATTICE_COLUMNS = ["_lat_ax", "_lat_ay", "_lat_az",
                   "_lat_bx", "_lat_by", "_lat_bz",
                   "_lat_cx", "_lat_cy", "_lat_cz"]


def sizeof_fmt(file_name_or_size, suffix='B'):
    if isinstance(file_name_or_size, str):
        file_name_or_size = os.path.getsize(file_name_or_size)
    for unit in ['', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi']:
        if abs(file_name_or_size) < 1024.0:
            return "%3.1f%s%s" % (file_name_or_size, unit, suffix)
        file_name_or_size /= 1024.0
    return "%.1f%s%s" % (file_name_or_size, 'Yi', suffix)


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


def attach_single_point_calculator(row):
    atoms = row["ase_atoms"]
    energy = row["energy_corrected"]
    forces = row["forces"]
    calc = SinglePointCalculator(atoms, energy=energy, forces=forces)
    atoms.set_calculator(calc)
    return atoms


# ### preprocess and store


def create_ase_atoms(row):
    pbc = row["pbc"]
    if pbc:
        cell = row["cell"]
        if row['COORDINATES_TYPE'] == 'relative':
            atoms = Atoms(symbols=row["_OCCUPATION"], scaled_positions=row["_COORDINATES"], cell=cell, pbc=pbc)
        else:
            atoms = Atoms(symbols=row["_OCCUPATION"], positions=row["_COORDINATES"], cell=cell, pbc=pbc)
    else:
        atoms = Atoms(symbols=row["_OCCUPATION"], positions=row["_COORDINATES"], pbc=pbc)
    e = row["energy_corrected"]
    f = row["_VALUE"]['forces']
    calc = SinglePointCalculator(atoms, energy=np.array(e).reshape(-1, ), forces=np.array(f))
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


def query_data(config: Dict, seed=None, query_limit=None, db_conn_string=None):
    from structdborm import StructSQLStorage, CalculatorType, StructureEntry, StaticProperty, GenericEntry, Property
    from sqlalchemy.orm.exc import NoResultFound

    # validate config
    if "calculator" not in config:
        raise ValueError("'calculator' is not in YAML:data:config, couldn't query")
    if "element" not in config:
        raise ValueError("'element' is not in YAML:data:config, couldn't query")

    log.info("Connecting to database")
    with StructSQLStorage(db_conn_string) as storage:
        log.info("Querying database -- please be patient")
        reference_calculator = storage.query(CalculatorType).filter(
            CalculatorType.NAME == config["calculator"]).one()

        structure_entry_cell = [StructureEntry._lat_ax, StructureEntry._lat_ay, StructureEntry._lat_az,
                                StructureEntry._lat_bx, StructureEntry._lat_by, StructureEntry._lat_bz,
                                StructureEntry._lat_cx, StructureEntry._lat_cy, StructureEntry._lat_cz]
        if REF_ENERGY_KW not in config:
            try:
                # TODO: generalize query of reference property
                ref_energy = query_reference_energy(config["element"], reference_calculator, storage)
            except NoResultFound as e:
                log.error(("No reference energy for {} was found in database. " +
                           "Either add property named `{}` with generic named `{}` to database or use `{}` " +
                           "keyword in data config ").format(config["element"], REF_PROP_NAME,
                                                             REF_GENERIC_PROTOTYPE_NAME, REF_ENERGY_KW))
                raise e
        else:
            ref_energy = config[REF_ENERGY_KW]
        # TODO: join with query with generic-parent-absent structures/properties
        q = storage.query(StaticProperty.id.label("prop_id"),
                          StructureEntry.id.label("structure_id"),
                          GenericEntry.id.label("gen_id"),
                          GenericEntry.PROTOTYPE_NAME,
                          *structure_entry_cell,
                          StructureEntry.COORDINATES_TYPE,
                          StructureEntry._COORDINATES,
                          StructureEntry._OCCUPATION,
                          StructureEntry.NUMBER_OF_ATOMS,
                          StaticProperty._VALUE) \
            .join(StaticProperty.ORIGINAL_STRUCTURE).join(StructureEntry.GENERICPARENT) \
            .filter(Property.CALCULATOR == reference_calculator,
                    StructureEntry.NUMBER_OF_ATOMTYPES == 1,
                    StructureEntry.COMPOSITION.like(config["element"] + "-%"),
                    ).order_by(StaticProperty.id)
        if query_limit is not None:
            q = q.limit(query_limit)
        log.info("Querying entries with defined generic prototype...")
        tot_data = q.all()
        log.info("Queried: {} entries".format(len(tot_data)))

        q_none = storage.query(StaticProperty.id.label("prop_id"),
                               StructureEntry.id.label("structure_id"),
                               StaticProperty.NAME.label("property_name"),
                               *structure_entry_cell,
                               StructureEntry.COORDINATES_TYPE,
                               StructureEntry._COORDINATES,
                               StructureEntry._OCCUPATION,
                               StructureEntry.NUMBER_OF_ATOMS,
                               StaticProperty._VALUE) \
            .join(StaticProperty.ORIGINAL_STRUCTURE) \
            .filter(Property.CALCULATOR == reference_calculator,
                    StructureEntry.NUMBER_OF_ATOMTYPES == 1,
                    StructureEntry.COMPOSITION.like(config["element"] + "-%"),
                    StructureEntry.GENERICPARENT == None
                    ).order_by(StaticProperty.id)

        if query_limit is not None:
            q_none = q_none.limit(query_limit)

        log.info("Querying entries without defined generic prototype...")
        no_generic_tot_data = q_none.all()
        log.info("Queried: {} entries".format(len(no_generic_tot_data)))

        df = DataFrameWithMetadata(tot_data)
        df_no_generic = DataFrameWithMetadata(no_generic_tot_data)

        log.info("Combining both queries together")
        df_total = pd.concat([df, df_no_generic], axis=0)
        log.info("Reseting indices")
        df_total.reset_index(inplace=True, drop=True)

        # shuffle notebook for randomizing parallel processing
        if seed is not None:
            log.info("set numpy random seed = {}".format(seed))
            np.random.seed(seed)
            log.info("Shuffle dataset")
            # pandas should do it inplace, not extra memory allocation
            df_total = df_total.sample(frac=1, random_state=seed)  # .reset_index(drop=True)
        else:
            log.info("Seed is not provided, no shuffling")
        log.info("Total entries obtained from database:" + str(df_total.shape[0]))
        return df_total, ref_energy


def query_reference_energy(element, reference_calculator, storage):
    from structdborm import StructureEntry, StaticProperty, GenericEntry, Property
    ref_prop = storage.query(StaticProperty).join(StructureEntry, GenericEntry).filter(
        Property.CALCULATOR == reference_calculator,
        Property.NAME == REF_PROP_NAME,
        StructureEntry.COMPOSITION.like(element + "-%"),
        StructureEntry.NUMBER_OF_ATOMS == 1,
        GenericEntry.PROTOTYPE_NAME == REF_GENERIC_PROTOTYPE_NAME
    ).one()
    # free atom reference energy
    ref_energy = ref_prop.energy / ref_prop.n_atom
    return ref_energy


class StructuresDatasetWeightingPolicy:
    def generate_weights(self, df):
        raise NotImplementedError


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


class StructuresDatasetSpecification:
    """
    Object to query or load from cache the fitting dataset

        :param config:  dictionary with "element" - the element for which the data will be collected
                                        "calculator" - calculator and
                                        "seed" - random seed
        :param cutoff:
        :param filename:
        :param datapath:
        :param db_conn_string:
        :param force_query:
        :param query_limit:
        :param seed:
        :param cache_ref_df:
    """

    FHI_AIMS_PBE_TIGHT = 'FHI-aims/PBE/tight'

    def __init__(self,
                 config: Dict = None,
                 cutoff: float = 10,
                 filename: str = None,
                 datapath: str = "",
                 db_conn_string: str = None,
                 force_query: bool = False,
                 ignore_weights: bool = False,
                 query_limit: int = None,
                 seed: int = None,
                 cache_ref_df: bool = False,
                 progress_bar: bool = False,
                 df: pd.DataFrame = None,
                 force_rebuild: bool = False,
                 **kwargs
                 ):
        """

        :param config:
        :param cutoff:
        :param filename:
        :param datapath:
        :param db_conn_string:
        :param force_query:
        :param query_limit:
        :param seed:
        :param cache_ref_df:
        """

        # data config
        self.query_limit = query_limit
        if config is None:
            config = {}

        self.config = config
        self.force_query = force_query
        self.force_rebuild = force_rebuild
        self.ignore_weights = ignore_weights
        self.filename = filename
        # ### Path where pickle files will be stored
        self.datapath = datapath

        # random seed
        self.seed = seed
        if self.seed is not None:
            np.random.seed(self.seed)

        # neighbour list cutoff
        self.cutoff = cutoff

        # result column name : tuple(mapper function,  kwargs)
        self.ase_atoms_transformers = {}

        self.db_conn_string = db_conn_string
        self.progress_bar = progress_bar

        self.raw_df = df
        self.df = None
        self.ref_energy = None
        self.weights_policy = None

        self.ref_df_changed = False
        self.cache_ref_df = cache_ref_df

    def set_weights_policy(self, weights_policy):
        self.weights_policy = weights_policy

    def add_ase_atoms_transformer(self, result_column_name, transformer_func, **kwargs):
        self.ase_atoms_transformers[result_column_name] = (transformer_func, kwargs)

    def get_default_ref_filename(self):
        try:
            return "df-{calculator}-{element}-{suffix}.pckl.gzip".format(
                calculator=self.config["calculator"],
                element=self.config["element"],
                suffix="ref").replace("/", "_")
        except KeyError as e:
            log.warning("Couldn't generate default name: " + str(e))
            return None
        except Exception as e:
            raise

    def process_ref_dataframe(self, ref_df: pd.DataFrame, e0_per_atom: float) -> pd.DataFrame:
        if not isinstance(ref_df, DataFrameWithMetadata):
            log.info("Transforming to DataFrameWithMetadata")
            ref_df = DataFrameWithMetadata(ref_df)
            self.ref_df_changed = True

        log.info("Setting up structures dataframe - please be patient...")
        if "NUMBER_OF_ATOMS" not in ref_df.columns:
            if "ase_atoms" in ref_df.columns:
                ref_df["NUMBER_OF_ATOMS"] = ref_df["ase_atoms"].map(len)
                self.ref_df_changed = True
            else:
                raise ValueError("Dataframe is corrupted: neither 'NUMBER_OF_ATOMS' nor 'ase_atoms' columns are found")

        tot_atoms_num = ref_df["NUMBER_OF_ATOMS"].sum()
        mean_atoms_num = ref_df["NUMBER_OF_ATOMS"].mean()
        log.info("Processing structures dataframe. Shape: " + str(ref_df.shape))
        log.info("Total number of atoms: " + str(tot_atoms_num))
        log.info("Mean number of atoms per structure: {:.1f}".format(mean_atoms_num))

        # Extract energies and forces into separate columns
        if "energy" not in ref_df.columns and "energy_corrected" not in ref_df.columns:
            log.info("'energy' columns extraction from '_VALUE'")
            ref_df["energy"] = ref_df["_VALUE"].map(lambda d: d["energy"])
            self.ref_df_changed = True
        else:
            log.info("'energy' columns found")

        if FORCES_COLUMN not in ref_df.columns:
            log.info("'forces' columns extraction from '_VALUE'")
            ref_df[FORCES_COLUMN] = ref_df["_VALUE"].map(lambda d: np.array(d[FORCES_COLUMN]))
            self.ref_df_changed = True
        else:
            log.info("'forces' columns found")

        if "pbc" not in ref_df.columns:
            if "ase_atoms" not in ref_df.columns:
                log.info("'pbc' columns extraction from lattice columns")
                # check the periodicity of coordinates
                ref_df["pbc"] = (ref_df[LATTICE_COLUMNS] != 0).any(axis=1)
            else:
                # check the periodicity from ASE atoms
                log.info("'pbc' columns extraction from 'ase_atoms'")
                ref_df["pbc"] = ref_df["ase_atoms"].map(lambda atoms: np.all(atoms.pbc))
            self.ref_df_changed = True
        else:
            log.info("'pbc' columns found")

        if "cell" not in ref_df.columns and "ase_atoms" not in ref_df.columns:
            log.info("'cell' column extraction from lattice columns")
            ref_df["cell"] = ref_df[LATTICE_COLUMNS].apply(lambda row: row.values.reshape(-1, 3), axis=1)
            ref_df.drop(columns=LATTICE_COLUMNS, inplace=True)
            # ref_df.reset_index(drop=True, inplace=True)
            self.ref_df_changed = True
        else:
            log.info("'cell' column found")

        if "energy_corrected" not in ref_df.columns:
            log.info("'energy_corrected' column extraction from 'energy'")
            if e0_per_atom is not None:
                ref_df["energy_corrected"] = ref_df["energy"] - ref_df["NUMBER_OF_ATOMS"] * e0_per_atom
            else:
                raise ValueError("e0_per_atom is not specified, please re-query the data from database")
            self.ref_df_changed = True
        else:
            log.info("'energy_corrected' column found")

        if E_CORRECTED_PER_ATOM_COLUMN not in ref_df.columns:
            log.info("'{}' column extraction".format(E_CORRECTED_PER_ATOM_COLUMN))
            ref_df[E_CORRECTED_PER_ATOM_COLUMN] = ref_df["energy_corrected"] / ref_df["NUMBER_OF_ATOMS"]
            self.ref_df_changed = True
        else:
            log.info("'{}' column found".format(E_CORRECTED_PER_ATOM_COLUMN))

        log.info("Min energy per atom: {:.3f} eV/atom".format(ref_df[E_CORRECTED_PER_ATOM_COLUMN].min()))
        log.info("Max energy per atom: {:.3f} eV/atom".format(ref_df[E_CORRECTED_PER_ATOM_COLUMN].max()))
        log.info("Min abs energy per atom: {:.3f} eV/atom".format(ref_df[E_CORRECTED_PER_ATOM_COLUMN].abs().min()))
        log.info("Max abs energy per atom: {:.3f} eV/atom".format(ref_df[E_CORRECTED_PER_ATOM_COLUMN].abs().max()))

        if "ase_atoms" not in ref_df.columns:
            log.info("ASE Atoms construction...")
            start = time.time()
            self.apply_create_ase_atoms(ref_df)
            end = time.time()
            time_elapsed = end - start
            log.info("ASE Atoms construction...done within {} sec ({} ms/at)".
                     format(time_elapsed, time_elapsed / tot_atoms_num * 1e3))
            self.ref_df_changed = True
        else:
            log.info("ASE atoms ('ase_atoms' column) are already in dataframe")

        # for tp_atoms: check, that energies and forces in SinglePointCalculator and in energy_corrected,
        # forces are identical

        if self.force_rebuild:
            attach_spc = True
            log.info("Force-rebuild is set, SinglePointCalculator will be reattached")
        else:
            attach_spc = False
        # check ref_df has SinglePointCalculator attached
        test_atoms = ref_df["ase_atoms"].iloc[0]
        if test_atoms.get_calculator() is not None:
            try:
                log.info("Checking stored energies...")
                e = ref_df["energy_corrected"];
                espc = ref_df["ase_atoms"].map(lambda at: at.get_potential_energy())
                de = (e - espc).abs().max()
                if de > 1e-15:
                    log.warning("WARNING! 'energy_corrected' and ase_atoms.SinglePointCalculator.get_potential_energy "
                                "are inconsistent")
                    attach_spc = True

                # check forces
                log.info("Checking stored forces...")
                fspc = np.vstack(ref_df["ase_atoms"].map(lambda at: at.get_forces()))
                f = np.vstack(ref_df["forces"])
                if np.abs(f - fspc).max() > 1e-15:
                    log.warning("WARNING! 'forces' and ase_atoms.SinglePointCalculator.get_forces "
                                "are inconsistent")
                    attach_spc = True
            except KeyboardInterrupt:
                raise
            except Exception as e:
                log.error("Couldn't check ase_atoms.SinglePointCalculator energies/forces, reattaching calculator...")
                attach_spc = True
        else:  # no SinglePointCalculaotr
            attach_spc = True

        # check ref_df has SinglePointCalculator attached

        if attach_spc:
            log.info("Attaching SinglePointCalculator to ASE atoms...")
            start = time.time()
            ref_df["ase_atoms"] = ref_df.apply(attach_single_point_calculator, axis=1)
            end = time.time()
            time_elapsed = end - start
            log.info("Attaching SinglePointCalcualtor to ASE atoms...done within {:.6} sec ({:.6} ms/at)".
                     format(time_elapsed, time_elapsed / tot_atoms_num * 1e3))
            self.ref_df_changed = True

        log.info("Atomic environment representation construction...")
        start = time.time()
        self.apply_ase_atoms_transformers(ref_df)
        end = time.time()
        time_elapsed = end - start
        log.info("Atomic environment representation construction...done within {:.5g} sec ({:.3g} ms/atom)".
                 format(time_elapsed, time_elapsed / tot_atoms_num * 1e3))
        return ref_df

    def apply_create_ase_atoms(self, df):
        df["ase_atoms"] = df.apply(create_ase_atoms, axis=1)

    def apply_ase_atoms_transformers(self, df):
        apply_function = df["ase_atoms"].apply

        for res_column_name, (transformer, kwargs) in self.ase_atoms_transformers.items():
            if res_column_name in df.columns and not self.force_rebuild:
                log.info("'{}' already in dataframe".format(res_column_name))
                # check if cutoff is not smaller than requested now
                try:
                    metadata_kwargs = df.metadata_dict[res_column_name + "_kwargs"]
                    metadata_cutoff = metadata_kwargs["cutoff"]
                    cur_cutoff = kwargs["cutoff"]
                    if metadata_cutoff < cur_cutoff:
                        log.warning("WARNING! Column {} was constructed with smaller cutoff ({}A) "
                                    "that necessary now ({}A). "
                                    "Neighbourlists will be re-built".format(res_column_name, metadata_cutoff,
                                                                             cur_cutoff))
                    else:
                        log.info("Column '{}': existing cutoff ({}A) >= "
                                 "requested  cutoff ({}A), skipping...".format(res_column_name, metadata_cutoff,
                                                                               cur_cutoff))

                        continue
                except KeyboardInterrupt as e:
                    raise e
                except Exception as e:
                    log.info("Could not extract cutoff metadata "
                             "for column '{}' (error: {}). Please ensure the valid cutoff for "
                             "precomputed neighbourlists".format(res_column_name, e))
                    continue

            l1 = len(df)
            cur_cutoff = kwargs["cutoff"]
            log.info("Building '{}' (dataset size {}, cutoff={}A)...".format(res_column_name, l1, cur_cutoff))
            df[res_column_name] = apply_function(transformer, **kwargs)
            df.dropna(subset=[res_column_name], inplace=True)
            # df.reset_index(drop=True, inplace=True)
            l2 = len(df)
            log.info("Dataframe size after transform: " + str(l2))
            df.metadata_dict[res_column_name + "_kwargs"] = kwargs
            self.ref_df_changed = True

    def load_or_query_ref_structures_dataframe(self, force_query=None):
        self.ref_df_changed = False
        if force_query is None:
            force_query = self.force_query

        files_to_load = self.get_actual_filename()  # return list

        log.info("Search for ref-file(s): " + str(files_to_load))
        ref_energy = None
        if self.raw_df is not None and not force_query:
            self.df = self.raw_df
        elif files_to_load is not None and not force_query:  # and os.path.isfile(files_to_load)
            dfs = []
            for i, fname in enumerate(files_to_load):
                log.info(f"#{i + 1}/{len(files_to_load)}: try to load {fname}")
                df = load_dataframe(fname, compression="infer")
                log.info(f" {len(df)} structures found")
                if "name" not in df.columns:
                    df["name"]=fname+":"+df.index.map(str)
                dfs.append(df)
            self.df = pd.concat(dfs, axis=0).reset_index(drop=True)
        else:  # if ref_df is still not loaded, try to query from DB
            raise NotImplementedError("Querying from DB is not supported in pacemaker anymore, use another tools")

        if not isinstance(self.df, DataFrameWithMetadata):
            log.info("Transforming to DataFrameWithMetadata")
            self.df = DataFrameWithMetadata(self.df)
            self.ref_df_changed = True
        # check, that all necessary columns are there
        self.df = self.process_ref_dataframe(self.df, ref_energy)

        return self.df

    def get_actual_filename(self):
        """
        Get actual filename to load dataframe:
        1. If filename is provided
            - try to find file locally
            - else try to find in datapath. If no datapath - switch back to local
        2. If filename is not provided
            - generate standard filename based on element and calculator name (prepend with datapath, if provided)
        :return: filename of dataframe
        """
        if self.filename is not None:
            if isinstance(self.filename, str):
                filename_list = [self.filename]
            elif isinstance(self.filename, list):
                filename_list = self.filename
            else:
                raise ValueError(f"Non-supported type of filename: `{self.filename}` (type: {type(self.filename)})")

            files_to_load = []
            for f in filename_list:
                if os.path.isfile(f) or self.datapath is None:
                    files_to_load.append(f)
                else:
                    files_to_load.append(os.path.join(self.datapath, f))

        else:
            files_to_load = self.get_default_ref_filename()
            files_to_load = os.path.join(self.datapath, files_to_load) if self.datapath is not None else files_to_load
            files_to_load = [files_to_load]  # make a single list
        return files_to_load

    def get_ref_dataframe(self, force_query=None, cache_ref_df=False):
        self.ref_df_changed = False
        if force_query is None:
            force_query = self.force_query
        if self.df is None:
            self.df = self.load_or_query_ref_structures_dataframe(force_query=force_query)
        if cache_ref_df or self.cache_ref_df:
            log.warning("Saving cached dataframe is not supported anymore, you can ignore this warning anyway")
            # if self.ref_df_changed:
            #     # generate filename to save df: if name is provided - try to put it into datapath
            #     filename = self.get_actual_filename()
            #     log.info("Saving processed raw dataframe into " + filename)
            #     save_dataframe(self.df, filename=filename)
            # else:
            #     log.info("Reference dataframe was not changed, nothing to save")
        return self.df

    def get_fit_dataframe(self, force_query=None, weights_policy=None, ignore_weights=None):
        if force_query is None:
            force_query = self.force_query
        self.df = self.get_ref_dataframe(force_query=force_query)

        if ignore_weights is None:
            ignore_weights = self.ignore_weights

        if WEIGHTS_ENERGY_COLUMN in self.df.columns and WEIGHTS_FORCES_COLUMN in self.df.columns and not ignore_weights:
            log.info("Both weighting columns ({} and {}) are found, no another weighting policy will be applied".format(
                WEIGHTS_ENERGY_COLUMN, WEIGHTS_FORCES_COLUMN))
        else:
            if ignore_weights and (
                    WEIGHTS_ENERGY_COLUMN in self.df.columns or WEIGHTS_FORCES_COLUMN in self.df.columns):
                log.info("Existing weights are ignored, weighting policy calculation is forced")

            if weights_policy is not None:
                self.set_weights_policy(weights_policy)

            if self.weights_policy is None:
                log.info("No weighting policy is specified, setting default weighting policy")
                self.set_weights_policy(UniformWeightingPolicy())

            log.info("Apply weights policy: " + str(self.weights_policy))
            self.df = self.weights_policy.generate_weights(self.df)
        if WEIGHTS_FACTOR in self.df.columns:
            log.info("Weights factor column `{}` is found, multiplying energy anf forces weights by this factor".format(
                WEIGHTS_FACTOR))
            self.df[WEIGHTS_ENERGY_COLUMN] *= self.df[WEIGHTS_FACTOR]
            self.df[WEIGHTS_FORCES_COLUMN] *= self.df[WEIGHTS_FACTOR]
        return self.df


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
                 energy="convex_hull",
                 aug_factor=1e-4):
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
                "EnergyBasedWeightingPolicy(nfit={nfit}, n_lower={n_lower}, n_upper={n_upper}, energy={energy}," + \
                " DElow={DElow}, DEup={DEup}, DFup={DFup}, DE={DE}, " + \
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
                # self.nfit = len(df)
                # if self.n_upper is not None:
                #     self.n_lower = self.nfit - self.n_upper
                #     if self.n_lower < 0:
                #         self.n_lower = 0
                #         self.nfit = self.n_upper
                # if self.n_lower is not None:
                #     self.n_upper = self.nfit - self.n_lower
                #     if self.n_upper < 0:
                #         self.n_upper = 0
                #         self.nfit = self.n_lower
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

        self.check_df_non_empty(df)

        if self.cutoff is not None:
            log.info("EnergyBasedWeightingPolicy::cutoff is provided but will be ignored")
        else:
            log.info("No cutoff for EnergyBasedWeightingPolicy is provided, no structures outside cutoff that " +
                     "will now be removed")

        # #### structure selection

        if self.energy == "convex_hull":
            log.info("EnergyBasedWeightingPolicy: energy reference frame - convex hull distance (if possible)")
            compute_convexhull_dist(df)  # generate "e_chull_dist_per_atom" column
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

        self.check_df_non_empty(df)

        # remove high energy structures
        df = df[df[EFFECTIVE_ENERGY] < self.DEup]  # .reset_index(drop=True)

        self.check_df_non_empty(df)

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
        self.check_df_non_empty(df)

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
        df[FORCES_COLUMN] = df[FORCES_COLUMN].map(np.array)
        assert (df["forces"].map(len) == df["ase_atoms"].map(len)).all(), ValueError(
            "Number of atoms doesn't corresponds to shape of forces")
        df[WEIGHTS_FORCES_COLUMN] = df[FORCES_COLUMN].map(lambda forces: 1 / (np.sum(forces ** 2, axis=1) + DF))
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

    def check_df_non_empty(self, df: pd.DataFrame):
        if len(df) == 0:
            raise RuntimeError("Couldn't operate with empty dataset. Try to reduce filters and constraints")

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


def compute_convexhull_dist(df):
    df["comp_dict"] = df["ase_atoms"].map(lambda atoms: Counter(atoms.get_chemical_symbols()))
    df["elements"] = df["comp_dict"].map(lambda cnt: tuple(sorted(cnt.keys())))
    df["NUMBER_OF_ATOMS"] = df["ase_atoms"].map(len);
    elements_set = set()
    for els in df["elements"].unique():
        elements_set.update(els)
    elements = sorted(elements_set)

    for el in elements:
        df[el] = df["comp_dict"].map(lambda dct: dct.get(el, 0))
        df["c_" + el] = df[el] / df["NUMBER_OF_ATOMS"]
    c_elements = ["c_" + el for el in elements]
    df["comp_tuple"] = df[c_elements].apply(lambda r: tuple(r), axis=1)

    element_min_energy_dict = {}
    for el in elements:
        pure_element_df = df[df["elements"] == (el,)]
        e_min = pure_element_df["energy_corrected_per_atom"].min()
        if np.isnan(e_min):
            e_min = 0
            log.warning("No pure element energy for {} is available, assuming 0  eV/atom".format(el))
        else:
            log.info("Pure element lowest energy for {} = {:5f} eV/atom".format(el, e_min))
        element_min_energy_dict[el] = e_min

    element_emin_array = np.array([element_min_energy_dict[el] for el in elements])

    c_conc = df[c_elements].values
    e_formation_ideal = np.dot(c_conc, element_emin_array)
    df[E_FORMATION_PER_ATOM] = df["energy_corrected_per_atom"] - e_formation_ideal

    # check if more than one uniq compositions
    uniq_compositions = df["comp_tuple"].unique()

    if len(uniq_compositions) > 1:
        log.info("Structure dataset: multiple unique compositions found, trying to construct convex hull")
        chull_values = df[c_elements[:-1] + [E_FORMATION_PER_ATOM]].values
        hull = ConvexHull(chull_values)
        ok = hull.equations[:, -2] < 0
        selected_simplices = hull.simplices[ok]
        selected_equations = hull.equations[ok]

        norms = selected_equations[:, :-1]
        offsets = selected_equations[:, -1]

        norms_c = norms[:, :-1]
        norms_e = norms[:, -1]

        e_chull_dist_list = []
        for p in chull_values:
            p_c = p[:-1]
            p_e = p[-1]
            e_simplex_projections = []
            for nc, ne, b, simplex in zip(norms_c, norms_e, offsets, selected_simplices):
                if ne != 0:
                    e_simplex = (-b - np.dot(nc, p_c)) / ne
                    e_simplex_projections.append(e_simplex)
                elif np.abs(b + np.dot(nc, p_c)) < 2e-15:  # ne*e_simplex + b + np.dot(nc,p_c), ne==0
                    e_simplex = p_e
                    e_simplex_projections.append(e_simplex)

            e_simplex_projections = np.array(e_simplex_projections)

            mask = e_simplex_projections < p_e + 1e-15

            e_simplex_projections = e_simplex_projections[mask]

            e_dist_to_chull = np.min(p_e - e_simplex_projections)

            e_chull_dist_list.append(e_dist_to_chull)

        e_chull_dist_list = np.array(e_chull_dist_list)
    else:
        log.info("Structure dataset: only single unique composition found, switching to cohesive energy reference")
        emin = df[E_CORRECTED_PER_ATOM_COLUMN].min()
        e_chull_dist_list = df[E_CORRECTED_PER_ATOM_COLUMN] - emin

    df[E_CHULL_DIST_PER_ATOM] = e_chull_dist_list


class UniformWeightingPolicy(StructuresDatasetWeightingPolicy):

    def __init__(self, aug_factor=1e-4):
        pass

    def __str__(self):
        return "UniformWeightingPolicy()"

    def generate_weights(self, df):
        df[WEIGHTS_ENERGY_COLUMN] = 1. / len(df)

        df[WEIGHTS_FORCES_COLUMN] = df[FORCES_COLUMN].map(lambda forces: np.ones(len(forces)))

        # assert (df[WEIGHTS_FORCES_COLUMN].map(len) == df["NUMBER_OF_ATOMS"]).all()
        df[WEIGHTS_FORCES_COLUMN] = df[WEIGHTS_FORCES_COLUMN] * df[WEIGHTS_ENERGY_COLUMN]

        # w_forces_norm = df[WEIGHTS_FORCES_COLUMN].map(sum).sum()
        # df[WEIGHTS_FORCES_COLUMN] = df[WEIGHTS_FORCES_COLUMN] / w_forces_norm

        # assert np.allclose(df[WEIGHTS_ENERGY_COLUMN].sum(), 1)
        # assert np.allclose(df[WEIGHTS_FORCES_COLUMN].map(sum).sum(), 1)
        normalize_energy_forces_weights(df)
        return df


class ExternalWeightingPolicy(StructuresDatasetWeightingPolicy):

    def __init__(self, filename: str):
        """
        :param filename: .pckl.gzip filename of dataframe with index and  `w_energy` and `w_forces` columns
        """
        self.filename = filename

    def __str__(self):
        return "ExternalWeightingPolicy(filename={filename})".format(filename=self.filename)

    def generate_weights(self, df):
        log.info("Loading external weights dataframe {}".format(self.filename))
        self.weights_df = pd.read_pickle(self.filename, compression="gzip")
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
        if not (mdf[FORCES_COLUMN].map(len) == mdf[WEIGHTS_FORCES_COLUMN].map(len)).all():
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


def normalize_energy_forces_weights(df: pd.DataFrame) -> pd.DataFrame:
    if WEIGHTS_ENERGY_COLUMN not in df.columns:
        raise ValueError("`{}` column not in dataframe".format(WEIGHTS_ENERGY_COLUMN))
    if WEIGHTS_FORCES_COLUMN not in df.columns:
        raise ValueError("`{}` column not in dataframe".format(WEIGHTS_FORCES_COLUMN))

    assert (df[WEIGHTS_FORCES_COLUMN].map(len) == df[FORCES_COLUMN].map(len)).all()

    df[WEIGHTS_ENERGY_COLUMN] = df[WEIGHTS_ENERGY_COLUMN] / df[WEIGHTS_ENERGY_COLUMN].sum()
    # df[WEIGHTS_FORCES_COLUMN] = df[WEIGHTS_FORCES_COLUMN] * df[WEIGHTS_ENERGY_COLUMN]
    w_forces_norm = df[WEIGHTS_FORCES_COLUMN].map(sum).sum()
    df[WEIGHTS_FORCES_COLUMN] = df[WEIGHTS_FORCES_COLUMN] / w_forces_norm

    assert np.allclose(df[WEIGHTS_ENERGY_COLUMN].sum(), 1)
    assert np.allclose(df[WEIGHTS_FORCES_COLUMN].map(sum).sum(), 1)
    return df


def get_weighting_policy(weighting_policy_spec: Dict) -> StructuresDatasetWeightingPolicy:
    weighting_policy = None
    if weighting_policy_spec is None:
        return weighting_policy
    elif isinstance(weighting_policy_spec, StructuresDatasetWeightingPolicy):
        return weighting_policy_spec
    elif not isinstance(weighting_policy_spec, dict):
        raise ValueError(
            "Weighting policy specification ('weighting' option) should be a dictionary " +
            "or StructuresDatasetWeightingPolicy but got " + str(type(weighting_policy_spec)))

    weighting_policy_spec = weighting_policy_spec.copy()
    log.debug("weighting_policy_spec: " + str(weighting_policy_spec))

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


def get_dataset_specification(evaluator_name, data_config: Dict,
                              cutoff=10, elements=None) -> StructuresDatasetSpecification:
    if isinstance(data_config, str):
        spec = StructuresDatasetSpecification(filename=data_config, cutoff=cutoff)
    elif isinstance(data_config, dict):
        spec = StructuresDatasetSpecification(**data_config, cutoff=cutoff)
    else:
        raise ValueError("Unknown data specification type: " + str(type(data_config)))

    # add the transformer depending on the evaluator
    if evaluator_name == PYACE_EVAL:
        elements_mapper_dict = None
        if elements is not None:
            elements_mapper_dict = {el: i for i, el in enumerate(sorted(elements))}
            log.info("Elements-to-indices mapping for 'atomic_env' construction: {}".format(elements_mapper_dict))
        if elements_mapper_dict is None:
            log.info("Elements-to-indices mapping for 'atomic_env' construction is NOT provided")
        spec.add_ase_atoms_transformer(ATOMIC_ENV_DF_COLUMN, aseatoms_to_atomicenvironment, cutoff=cutoff,
                                       elements_mapper_dict=elements_mapper_dict)
    elif evaluator_name == TENSORPOT_EVAL:
        # from tensorpotential.utils.utilities import generate_tp_atoms
        from pyace.atomicenvironment import generate_tp_atoms
        spec.add_ase_atoms_transformer(TP_ATOMS_DF_COLUMN, generate_tp_atoms, cutoff=cutoff, verbose=False)

    return spec


def get_reference_dataset(evaluator_name, data_config: Dict, cutoff=10, elements=None, force_query=False,
                          cache_ref_df=True):
    if isinstance(data_config, dict) and "config" in data_config and elements is None:
        conf = data_config["config"]
        elements = [conf["element"]]

    spec = get_dataset_specification(evaluator_name=evaluator_name, data_config=data_config,
                                     cutoff=cutoff, elements=elements)
    return spec.get_ref_dataframe(force_query=force_query, cache_ref_df=cache_ref_df)


def get_fitting_dataset(evaluator_name, data_config: Dict, weighting_policy_spec: Dict = None,
                        cutoff=10, elements=None, force_query=False, force_weighting=None) -> pd.DataFrame:
    spec = get_dataset_specification(evaluator_name=evaluator_name, data_config=data_config,
                                     cutoff=cutoff, elements=elements)
    spec.set_weights_policy(get_weighting_policy(weighting_policy_spec))
    df = spec.get_fit_dataframe(force_query=force_query, ignore_weights=force_weighting)
    if "name" in df.columns:
        aug_mask = df["name"].str.startswith("augmented")
        if aug_mask.sum()>0:
            log.info(f"{aug_mask.sum()} augmented structures found in dataset")
            aug_factor = data_config.get("aug_factor",
                                         weighting_policy_spec.get("aug_factor", 1e-4) if isinstance(weighting_policy_spec,
                                                                                                     dict) else 1e-4)

            #  if weights are NAN - fallback to const(median) * aug_factor
            if df.loc[aug_mask, WEIGHTS_ENERGY_COLUMN].isna().any() or df.loc[aug_mask, WEIGHTS_FORCES_COLUMN].isna().any():
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
            log.info(f"Decreasing augmented weights by factor {aug_factor}")
            df.loc[aug_mask, WEIGHTS_ENERGY_COLUMN] *= aug_factor
            df.loc[aug_mask, WEIGHTS_FORCES_COLUMN] *= aug_factor

    normalize_energy_forces_weights(df)
    return df


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
