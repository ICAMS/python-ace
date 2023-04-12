import gc
import logging

from datetime import datetime
from functools import partial
from typing import Union, Dict, List, Callable, Tuple

import numpy as np
import pandas as pd

import pyace
from pyace.basis import ACEBBasisSet, BBasisConfiguration
from pyace.basisextension import construct_bbasisconfiguration, get_actual_ladder_step
from pyace.multispecies_basisextension import extend_multispecies_basis, expand_trainable_parameters, \
    compute_bbasisset_train_mask, clean_bbasisconfig, reset_bbasisconfig
from pyace.const import *
from pyace.fitadapter import FitBackendAdapter
from pyace.preparedata import get_fitting_dataset, normalize_energy_forces_weights
from pyace.lossfuncspec import LossFunctionSpecification
from pyace.metrics_aggregator import MetricsAggregator

from pyace.atomicenvironment import calculate_minimal_nn_distance

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

__username = None

FITTING_DATA_INFO_FILENAME = "fitting_data_info.pkl.gz"


def get_username():
    global __username
    if __username is None:
        try:
            import getpass
            __username = getpass.getuser()
            log.info("User name automatically identified: {}".format(__username))
            return __username
        except ImportError:
            log.info("Couldn't automatically identify user name")
    else:
        return __username



def setup_inner_core_repulsion(basisconf, r_in, delta_in=0.1, rho_cut=5, drho_cut=5,
                               core_rep_parameters=(10000, 1)):
    for block in basisconf.funcspecs_blocks:
        block.r_in = r_in
        block.delta_in = delta_in

        if block.number_of_species == 1:
            block.rho_cut = rho_cut
            block.drho_cut = drho_cut
            block.core_rep_parameters = core_rep_parameters
            block.inner_cutoff_type = "distance"
        elif block.number_of_species == 2:
            block.core_rep_parameters = core_rep_parameters
            block.inner_cutoff_type = "distance"


def get_initial_potential(start_potential):
    if isinstance(start_potential, str):
        initial_bbasisconfig = BBasisConfiguration(start_potential)
    elif isinstance(start_potential, BBasisConfiguration):
        initial_bbasisconfig = start_potential
    elif isinstance(start_potential, list):
        total_initial_potential = None
        for pot in start_potential:
            if isinstance(pot, str):
                conf = BBasisConfiguration(pot)
                log.info("Initial potential {} is loaded from {}".format(conf, pot))
            elif isinstance(pot, BBasisConfiguration):
                conf = pot
            else:
                raise ValueError(
                    "potential_config[`{}`] is not a list of str or BBasisConfiguration".format(
                        POTENTIAL_INITIAL_POTENTIAL_KW))
            if total_initial_potential is None:
                total_initial_potential = conf
            else:
                total_initial_potential = total_initial_potential + conf
        initial_bbasisconfig = total_initial_potential
    else:
        raise ValueError("potential_config[`{}`] is neither str nor BBasisConfiguration".format(
            POTENTIAL_INITIAL_POTENTIAL_KW))
    return initial_bbasisconfig


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


class GeneralACEFit:
    """
    Main fitting wrapper class

    :param potential_config:  specification of the potential
                    - configuration dictionary
                    - YAML with BBasisConfiguration potential configuration
                    - BBasisConfiguration
                    - ACEBBasisSet
    :param fit_config:  specification of fitting (loss function, number of iterations, weighting policy, ...)
    :param data_config:  training data specification
    :param backend_config: specification of potential evaluation and fitting backend (pyace / tensorpot)
                    - dict ['evaluator']
    """

    def __init__(self,
                 potential_config: Union[Dict, str, BBasisConfiguration, ACEBBasisSet],
                 fit_config: Dict,
                 data_config: Union[Dict, pd.DataFrame],
                 backend_config: Dict,
                 cutoff=None,
                 seed=None,
                 callbacks=None
                 ):
        self.seed = seed
        if self.seed is not None:
            log.info("Set numpy random seed to {}".format(self.seed))
            np.random.seed(self.seed)

        self.callbacks = [save_interim_potential_callback]
        if callbacks is not None:
            if isinstance(callbacks, (list, tuple)):
                for c in callbacks:
                    if isinstance(c, Callable):
                        self.callbacks.append(c)
                        log.info("{} callback added".format(c))
                    elif isinstance(c, str):
                        log.info("")
                        c = active_import(c)
                        self.callbacks.append(c)
                        log.info("{} callback added".format(c))
            else:
                raise ValueError(
                    "'callbacks' should be list/tuple of importable function name or function with signature: callback" +
                    "(coeffs, bbasisconfig: BBasisConfiguration, current_fit_cycle: int, current_ladder_step: int). " +
                    "But got: {}".format(callbacks)
                )

        self.current_fit_iteration = 0
        self.current_ladder_step = 0
        self.current_fit_cycle = 0
        self.ladder_scheme = False
        self.ladder_type = 'body_order'
        self.initial_bbasisconfig = None
        if isinstance(potential_config, dict):
            # try to identify initial potential
            if POTENTIAL_INITIAL_POTENTIAL_KW in potential_config:
                start_potential = potential_config[POTENTIAL_INITIAL_POTENTIAL_KW]
                self.initial_bbasisconfig = get_initial_potential(start_potential)

                log.info("Initial potential provided: {}, it contains {} functions".format(start_potential,
                                                                                           self.initial_bbasisconfig.total_number_of_functions))
                self.ladder_scheme = True
                log.info("Ladder-scheme fitting is ON")
            elif FIT_LADDER_STEP_KW in fit_config:
                self.initial_bbasisconfig = construct_bbasisconfiguration(potential_config)
                clean_bbasisconfig(self.initial_bbasisconfig)
                self.ladder_scheme = True
                log.info("Ladder-scheme fitting is ON")
                log.info("Initial potential is NOT provided, starting from empty potential")
            # initialize target_bbasisconfig
            # construct potential from initial_bbasisconfig also (copy the blocks if they are not presented)
            if "filename" in potential_config:
                target_pot_filename = potential_config["filename"]
                self.target_bbasisconfig = BBasisConfiguration(target_pot_filename)
                log.info("Target potential loaded from file '{}', it contains {} functions".format(target_pot_filename,
                                                                                                   self.target_bbasisconfig.total_number_of_functions))
                # reset the potential, all funcs=0, crad=delta
                if "reset" in potential_config and potential_config["reset"]:
                    log.info(
                        "Reset target potential, set functions coefficient to zero and radial coefficients to delta_nk")
                    reset_bbasisconfig(self.target_bbasisconfig)
            else:
                self.target_bbasisconfig = construct_bbasisconfiguration(potential_config,
                                                                         initial_basisconfig=self.initial_bbasisconfig)
                if ("functions" in potential_config and
                        "number_of_functions_per_element" in potential_config['functions']):
                    num_block = len(self.target_bbasisconfig.funcspecs_blocks)
                    number_of_functions_per_element = potential_config["functions"]["number_of_functions_per_element"]
                    target_bbasis = ACEBBasisSet(self.target_bbasisconfig)
                    nelements = target_bbasis.nelements

                    log.info(
                        """Number of functions in target potential"""
                        """ is limited to maximum {number_of_functions_per_element}"""
                        """ functions per element  for {nelements} elements ({num_block} blocks)""".format(
                            number_of_functions_per_element=number_of_functions_per_element,
                            nelements=nelements,
                            num_block=num_block))
                    log.info("Resulted potential contains {} functions".format(
                        self.target_bbasisconfig.total_number_of_functions))

                log.info("Target potential shape constructed from dictionary, it contains {} functions".format(
                    self.target_bbasisconfig.total_number_of_functions))

        elif isinstance(potential_config, str):
            self.target_bbasisconfig = BBasisConfiguration(potential_config)
            log.info("Target potential loaded from file '{}', it contains {} functions".format(potential_config,
                                                                                               self.target_bbasisconfig.total_number_of_functions))
        elif isinstance(potential_config, BBasisConfiguration):
            self.target_bbasisconfig = potential_config
            log.info("Target potential provided as `BBasisConfiguration` object, it contains {} functions".format(
                self.target_bbasisconfig.total_number_of_functions))
        elif isinstance(potential_config, ACEBBasisSet):
            self.target_bbasisconfig = potential_config.to_BBasisConfiguration()
            log.info("Target potential provided as `ACEBBasisSet` object, it contains {} functions".format(
                self.target_bbasisconfig.total_number_of_functions))
        else:
            raise ValueError(
                ("Non-supported type: {}. Only dictionary (configuration), " +
                 "str (YAML file name) or BBasisConfiguration are supported").format(
                    type(potential_config)))
        # save target_potential.yaml
        self.target_bbasisconfig.save(TARGET_POTENTIAL_YAML)

        if FIT_LADDER_STEP_KW in fit_config and not self.ladder_scheme:
            if self.initial_bbasisconfig is None:
                self.initial_bbasisconfig = self.target_bbasisconfig.copy()
                clean_bbasisconfig(self.initial_bbasisconfig)
                log.info("Initial potential is NOT provided, starting from empty potential")
            self.ladder_scheme = True
            log.info("Ladder-scheme fitting is ON")

        if cutoff is None:
            rcut = max(
                [block.rcutij for block in self.target_bbasisconfig.funcspecs_blocks if len(block.elements_vec) <= 2])
            self.cutoff = rcut
        else:
            self.cutoff = cutoff

        if self.ladder_scheme:
            if FIT_LADDER_TYPE_KW in fit_config:
                self.ladder_type = str(fit_config[FIT_LADDER_TYPE_KW])
            log.info("Ladder_type: {} is selected".format(self.ladder_type))

        self.fit_config = fit_config
        if FIT_OPTIMIZER_KW not in self.fit_config:
            self.fit_config[FIT_OPTIMIZER_KW] = "BFGS"
            log.warning("'{}' is not provided, switch to default value: {}".format(FIT_OPTIMIZER_KW,
                                                                                   self.fit_config[FIT_OPTIMIZER_KW]))
        if FIT_NITER_KW not in self.fit_config:
            self.fit_config[FIT_NITER_KW] = 100
            log.warning(
                "'{}' is not provided, switch to default value: {}".format(FIT_NITER_KW, self.fit_config[FIT_NITER_KW]))

        if FIT_OPTIONS_KW in self.fit_config:
            log.info(
                "optimizer options are provided: '{}'".format(self.fit_config[FIT_OPTIONS_KW]))

        trainable_parameters = self.fit_config.get("trainable_parameters", [])
        # convert fit_blocks to indices of the blocks to fit
        self.elements = ACEBBasisSet(self.target_bbasisconfig).elements_name
        self.trainable_parameters_dict = expand_trainable_parameters(elements=self.elements,
                                                                     trainable_parameters=trainable_parameters)

        self.data_config = data_config
        self.weighting_policy_spec = self.fit_config.get(FIT_WEIGHTING_KW)
        display_step = backend_config.get('display_step', 20)
        if self.ladder_scheme:
            self.metrics_aggregator = MetricsAggregator(extended_display_step=display_step)
        else:
            self.metrics_aggregator = MetricsAggregator(extended_display_step=display_step,
                                                        ladder_metrics_filename=None)
        self.fit_backend = FitBackendAdapter(backend_config,
                                             fit_metrics_callback=self.fit_metric_callback,
                                             test_metrics_callback=self.test_metric_callback
                                             )
        self.evaluator_name = self.fit_backend.evaluator_name
        versions_dict = self.fit_backend.get_evaluator_version_dict()
        for k, v in versions_dict.items():
            log.info("{}: {}".format(k, v))
        set_general_metadata(self.target_bbasisconfig, **versions_dict)

        self.fitting_data = None
        self.test_data = None
        if isinstance(self.data_config, (dict, str)):
            self.fitting_data = get_fitting_dataset(evaluator_name=self.evaluator_name,
                                                    data_config=self.data_config,
                                                    weighting_policy_spec=self.weighting_policy_spec,
                                                    cutoff=self.cutoff, elements=self.elements
                                                    )
            if isinstance(self.data_config, dict):
                if "test_size" in self.data_config:
                    test_size = self.data_config["test_size"]
                    log.info("Splitting test dataset (test_size = {}) from main dataset({} samples)".format(test_size,
                                                                                                            len(self.fitting_data)))
                    self.fitting_data, self.test_data = train_test_split(self.fitting_data, test_size=test_size)
                elif "test_filename" in self.data_config:
                    test_filename = self.data_config["test_filename"]
                    log.info("Loading test dataset from {}".format(test_filename))
                    self.test_data = get_fitting_dataset(evaluator_name=self.evaluator_name,
                                                         data_config={"filename": test_filename,
                                                                      "datapath": self.data_config.get("datapath"),
                                                                      "force_rebuild": self.data_config.get(
                                                                          "force_rebuild", False)
                                                                      },
                                                         weighting_policy_spec=self.weighting_policy_spec,
                                                         cutoff=self.cutoff, elements=self.elements
                                                         )
        elif isinstance(self.data_config, pd.DataFrame):
            self.fitting_data = self.data_config
        else:
            raise ValueError("'data-config' should be dictionary or pd.DataFrame")

        if self.fitting_data is not None:
            normalize_energy_forces_weights(self.fitting_data)
        if self.test_data is not None:
            normalize_energy_forces_weights(self.test_data)

        self.save_fitting_data_info()

        # automatic repulsion selection
        if "repulsion" in self.fit_config and self.fit_config["repulsion"] == "auto":
            log.info("Auto core-repulsion estimation. Minimal distance calculation...")
            calculate_minimal_nn_distance(self.fitting_data)
            min_distance = self.fitting_data["min_distance"].min()
            log.info("Minimal distance = {:.2f} A ".format(min_distance))
            r_in = min_distance - 0.01
            setup_inner_core_repulsion(self.target_bbasisconfig, r_in)
            if self.initial_bbasisconfig:
                setup_inner_core_repulsion(self.initial_bbasisconfig, r_in)
            log.info("Inner cutoff / core-repulsion initialized")

        # plot self.fitting_data, self.test_data
        if self.fitting_data is not None:
            log.info("Plotting train energy-forces distribution")
            plot_ef_distributions(self.fitting_data, suffix="train_")
        if self.test_data is not None:
            log.info("Plotting test energy-forces distribution")
            plot_ef_distributions(self.test_data, suffix="test_")

        loss_spec_dict = self.fit_config.get(FIT_LOSS_KW, {})
        if loss_spec_dict.get("kappa") == "auto":
            e_std = self.fitting_data["energy_corrected_per_atom"].std()
            f_std = np.std(np.linalg.norm(np.vstack(self.fitting_data["forces"]), axis=1))
            kappa_auto = e_std ** 2 / (e_std ** 2 + f_std ** 2)
            loss_spec_dict["kappa"] = kappa_auto
            log.info("LossFunctionSpecification:kappa automatically selected: kappa = {:.3f}".format(kappa_auto))

        self.loss_spec = LossFunctionSpecification(**loss_spec_dict)

    def fit_metric_callback(self, metrics_dict, extended_display_step=None):
        metrics_dict["cycle_step"] = self.current_fit_cycle
        metrics_dict["ladder_step"] = self.current_ladder_step
        self.metrics_aggregator.fit_metric_callback(metrics_dict, extended_display_step=extended_display_step)

    def test_metric_callback(self, metrics_dict, extended_display_step=None):
        metrics_dict["cycle_step"] = self.current_fit_cycle
        metrics_dict["ladder_step"] = self.current_ladder_step
        self.metrics_aggregator.test_metric_callback(metrics_dict, extended_display_step=extended_display_step)

    def save_fitting_data_info(self):
        # columns to save: w_energy, w_forces, NUMBER_OF_ATOMS, PROTOTYPE_NAME, prop_id,structure_id, gen_id, if any
        # columns_to_save = ["PROTOTYPE_NAME", "NUMBER_OF_ATOMS", "prop_id", "structure_id", "gen_id", "pbc"] + \
        #                   [ENERGY_CORRECTED_COL, EWEIGHTS_COL, FWEIGHTS_COL]
        columns_to_drop = ["tp_atoms", "atomic_env"]
        fitting_data_columns = self.fitting_data.columns

        columns_to_save = [col for col in fitting_data_columns if col not in columns_to_drop]

        self.fitting_data[columns_to_save].to_pickle(FITTING_DATA_INFO_FILENAME,
                                                     protocol=4)
        log.info("Fitting dataset info saved into {}".format(FITTING_DATA_INFO_FILENAME))

    def fit(self) -> BBasisConfiguration:
        gc.collect()
        self.target_bbasisconfig.save(TARGET_POTENTIAL_YAML)

        log.info("Fitting dataset size: {} structures / {} atoms".format(len(self.fitting_data),
                                                                         self.fitting_data["NUMBER_OF_ATOMS"].sum()))
        if self.test_data is not None:
            log.info("Test dataset size: {} structures / {} atoms".format(len(self.test_data),
                                                                          self.test_data["NUMBER_OF_ATOMS"].sum()))
        if not self.ladder_scheme:  # normal "non-ladder" fit
            log.info("'Single-shot' fitting")
            self.target_bbasisconfig = self.cycle_fitting(self.target_bbasisconfig)
        else:  # ladder scheme
            log.info("'Ladder-scheme' fitting")
            self.target_bbasisconfig = self.ladder_fitting(self.initial_bbasisconfig, self.target_bbasisconfig)

        log.info("Fitting done")
        return self.target_bbasisconfig

    def ladder_fitting(self, initial_config, target_config):
        total_number_of_funcs = target_config.total_number_of_functions
        ladder_step_param = self.fit_config.get(FIT_LADDER_STEP_KW)

        current_bbasisconfig = initial_config.copy()
        self.current_ladder_step = 0
        while True:  # ladder loop
            prev_func_num = current_bbasisconfig.total_number_of_functions
            log.info("Current basis set size: {} B-functions".format(prev_func_num))
            ladder_step = get_actual_ladder_step(ladder_step_param, prev_func_num, total_number_of_funcs)
            log.info("Ladder step size: {}".format(ladder_step))
            # TODO: extend basis only for those blocks, where "funcs" is trainable
            current_bbasisconfig, is_extended = extend_multispecies_basis(current_bbasisconfig, target_config,
                                                                          self.ladder_type, ladder_step,
                                                                          return_is_extended=True)
            new_func_num = current_bbasisconfig.total_number_of_functions
            log.info("Extended basis set size: {} B-functions".format(new_func_num))
            current_bbasisconfig.save("current_extended_potential.yaml")
            log.info("Extended basis set saved to current_extended_potential.yaml")
            if not is_extended:
                log.info("No new function added after basis extension. Stopping")
                break

            current_bbasisconfig = self.cycle_fitting(current_bbasisconfig)

            if "_fit_cycles" in current_bbasisconfig.metadata:
                del current_bbasisconfig.metadata["_fit_cycles"]
            log.debug("Update metadata: {}".format(current_bbasisconfig.metadata))

            self.fit_backend.last_fit_metric_data["ladder_step"] = self.current_ladder_step
            self.metrics_aggregator.ladder_step_callback(self.fit_backend.last_fit_metric_data)

            last_test_metric_data = self.fit_backend.last_test_metric_data
            if last_test_metric_data:
                last_test_metric_data["ladder_step"] = self.current_ladder_step
                self.metrics_aggregator.test_ladder_step_callback(last_test_metric_data)

            # save ladder potential
            ladder_final_potential_filename = "interim_potential_ladder_step_{}.yaml".format(self.current_ladder_step)
            log.info("Save current ladder step potential to {}".format(ladder_final_potential_filename))
            save_interim_potential(current_bbasisconfig, potential_filename=ladder_final_potential_filename)
            self.current_ladder_step += 1

        return current_bbasisconfig

    def cycle_fitting(self, bbasisconfig: BBasisConfiguration) -> BBasisConfiguration:
        current_bbasisconfig = bbasisconfig.copy()
        log.info('Cycle fitting loop')

        fit_cycles = int(self.fit_config.get(FIT_FIT_CYCLES_KW, 1))
        noise_rel_sigma = float(self.fit_config.get(FIT_NOISE_REL_SIGMA, 0))
        noise_abs_sigma = float(self.fit_config.get(FIT_NOISE_ABS_SIGMA, 0))

        randomize_func_coeffs_abs_sigma = float(self.fit_config.get("randomize_func_coeffs", 0))

        if "_" + FIT_FIT_CYCLES_KW in current_bbasisconfig.metadata:
            finished_fit_cycles = int(current_bbasisconfig.metadata["_" + FIT_FIT_CYCLES_KW])
        else:
            finished_fit_cycles = 0

        if finished_fit_cycles >= fit_cycles:
            log.warning(
                ("Number of finished fit cycles ({}) >= number of expected fit cycles ({}). " +
                 "Use another potential or remove `{}` from potential metadata")
                .format(finished_fit_cycles, fit_cycles, "_" + FIT_FIT_CYCLES_KW))
            return current_bbasisconfig

        fitting_attempts_list = []
        while finished_fit_cycles < fit_cycles:
            self.current_fit_cycle = finished_fit_cycles
            log.info("Number of fit attempts: {}/{}".format(self.current_fit_cycle, fit_cycles))
            num_of_functions = current_bbasisconfig.total_number_of_functions
            num_of_parameters = len(current_bbasisconfig.get_all_coeffs())
            log.info("Total number of functions: {} / number of parameters: {}".format(num_of_functions,
                                                                                       num_of_parameters))
            log.info("Running fit backend")
            self.current_fit_iteration = 0
            current_bbasisconfig = self.fit_backend.fit(
                current_bbasisconfig,
                dataframe=self.fitting_data, loss_spec=self.loss_spec, fit_config=self.fit_config,
                callback=partial(self.callback_hook, current_fit_cycle=self.current_fit_cycle,
                                 current_ladder_step=self.current_ladder_step),
                test_dataframe=self.test_data
            )

            log.info("Fitting cycle finished, final statistic:")
            self.fit_backend.last_fit_metric_data["cycle_step"] = self.current_fit_cycle
            self.fit_backend.last_fit_metric_data["ladder_step"] = self.current_ladder_step
            self.metrics_aggregator.cycle_step_callback(self.fit_backend.last_fit_metric_data)

            last_test_metric_data = self.fit_backend.last_test_metric_data
            if last_test_metric_data:
                last_test_metric_data["cycle_step"] = self.current_fit_cycle
                last_test_metric_data["ladder_step"] = self.current_ladder_step
                self.metrics_aggregator.test_cycle_step_callback(last_test_metric_data)

            if randomize_func_coeffs_abs_sigma > 0:  # randomize func coeffs mode
                ens_pot_fname = "ensemble_potential_{}.yaml".format(self.current_fit_cycle)
                current_bbasisconfig.save(ens_pot_fname)
                log.info("Ensemble potential is saved to {}".format(ens_pot_fname))

            finished_fit_cycles = self.current_fit_cycle + 1

            current_bbasisconfig.metadata["_" + FIT_FIT_CYCLES_KW] = str(finished_fit_cycles)
            current_bbasisconfig.metadata["_" + FIT_LOSS_KW] = str(self.fit_backend.res_opt.fun)
            log.debug("Update current_bbasisconfig.metadata = {}".format(current_bbasisconfig.metadata))

            # save also current fitting_metric_data
            last_fit_metric_data = self.fit_backend.last_fit_metric_data
            fitting_attempts_list.append(
                (np.sum(self.fit_backend.res_opt.fun), current_bbasisconfig.copy(), last_fit_metric_data))

            # select current_bbasisconfig as a best among all previous
            best_ind = np.argmin([v[0] for v in fitting_attempts_list])
            log.info(
                "Select best fit #{} among all available ({})".format(best_ind + 1, len(fitting_attempts_list)))
            current_bbasisconfig = fitting_attempts_list[best_ind][1].copy()

            if finished_fit_cycles < fit_cycles and randomize_func_coeffs_abs_sigma > 0:
                current_bbasisconfig = self.randomize_func_coeffs(current_bbasisconfig, self.trainable_parameters_dict,
                                                                  randomize_func_coeffs_abs_sigma)
                log.info(
                    "Randomize all trainable functions coefficients using normal distribution N(0;sigma = {:>1.4e})".format(
                        randomize_func_coeffs_abs_sigma))
            elif finished_fit_cycles < fit_cycles and (noise_rel_sigma > 0) or (noise_abs_sigma > 0):
                current_bbasisconfig = self.apply_gaussian_noise(current_bbasisconfig, self.trainable_parameters_dict,
                                                                 noise_abs_sigma, noise_rel_sigma)

        # chose the best fit attempt among fitting_attempts_list
        best_fitting_attempts_ind = np.argmin([v[0] for v in fitting_attempts_list])
        log.info("Best fitting attempt is #{}".format(best_fitting_attempts_ind + 1))
        current_best_bbasisconfig = fitting_attempts_list[best_fitting_attempts_ind][1]
        # restore the best fitting metric data
        best_fitting_metric_data = fitting_attempts_list[best_fitting_attempts_ind][2]
        self.fit_backend.last_fit_metric_data = best_fitting_metric_data
        save_interim_potential(current_best_bbasisconfig, potential_filename="interim_potential_best_cycle.yaml")
        return current_best_bbasisconfig

    @staticmethod
    def apply_gaussian_noise(current_bbasisconfig, trainable_parameters_dict, noise_abs_sigma, noise_rel_sigma):
        cur_bbasis = ACEBBasisSet(current_bbasisconfig)
        trainable_mask = compute_bbasisset_train_mask(cur_bbasis, trainable_parameters_dict)
        all_coeffs = np.array(cur_bbasis.all_coeffs)
        all_trainable_coeffs = all_coeffs[trainable_mask]
        if noise_rel_sigma > 0:
            log.info(
                "Applying Gaussian noise with relative sigma/mean = {:>1.4e} to all trainable parameters".format(
                    noise_rel_sigma))
            noisy_all_coeffs = apply_noise(all_trainable_coeffs, noise_rel_sigma, relative=True)
        elif noise_abs_sigma > 0:
            log.info(
                "Applying Gaussian noise with sigma = {:>1.4e} to all trainable parameters".format(
                    noise_abs_sigma))
            noisy_all_coeffs = apply_noise(all_trainable_coeffs, noise_abs_sigma, relative=False)
        all_coeffs[trainable_mask] = noisy_all_coeffs
        cur_bbasis.all_coeffs = all_coeffs
        current_bbasisconfig = cur_bbasis.to_BBasisConfiguration()
        return current_bbasisconfig

    @staticmethod
    def randomize_func_coeffs(current_bbasisconfig, trainable_parameters_dict, randomize_func_coeffs_abs_sigma):
        randomized_funcs_trainable_parameters = {}
        for k, v in trainable_parameters_dict.items():
            v = [vv for vv in v if vv == "func"]
            randomized_funcs_trainable_parameters[k] = v
        cur_bbasis = ACEBBasisSet(current_bbasisconfig)
        trainable_mask = compute_bbasisset_train_mask(cur_bbasis, randomized_funcs_trainable_parameters)
        all_coeffs = np.array(cur_bbasis.all_coeffs)
        all_trainable_coeffs = all_coeffs[trainable_mask]
        rnd_func_coeffs = np.random.randn(len(all_trainable_coeffs)) * randomize_func_coeffs_abs_sigma
        all_coeffs[trainable_mask] = rnd_func_coeffs
        cur_bbasis.all_coeffs = all_coeffs
        current_bbasisconfig = cur_bbasis.to_BBasisConfiguration()
        return current_bbasisconfig

    def save_optimized_potential(self, potential_filename: str = "output_potential.yaml"):
        if "_" + FIT_FIT_CYCLES_KW in self.target_bbasisconfig.metadata:
            del self.target_bbasisconfig.metadata["_" + FIT_FIT_CYCLES_KW]

        log.debug("Update metadata: {}".format(self.target_bbasisconfig.metadata))
        self.target_bbasisconfig.save(potential_filename)
        log.info("Final potential is saved to {}".format(potential_filename))

    def callback_hook(self, basis_config: BBasisConfiguration, current_fit_cycle: int,
                      current_ladder_step: int):
        # TODO add a list of callbacks

        for callback in self.callbacks:
            callback(
                basis_config=basis_config,
                current_fit_iteration=self.current_fit_iteration,
                current_fit_cycle=current_fit_cycle,
                current_ladder_step=current_ladder_step,
            )
        self.current_fit_iteration += 1

    def predict(self, structures_dataframe=None, bbasisconfig=None):
        return self.fit_backend.predict(structures_dataframe=structures_dataframe, bbasisconfig=bbasisconfig)


def apply_noise(all_coeffs: Union[np.array, List], sigma: float, relative: bool = True) -> np.array:
    coeffs = np.array(all_coeffs)
    noise = np.random.randn(*coeffs.shape)
    if relative:
        base_coeffs = np.abs(coeffs)
        # clip minimal values
        base_coeffs[base_coeffs < 1e-2] = 1e-2
        coeffs = coeffs + noise * sigma * base_coeffs
    else:
        coeffs = coeffs + noise * sigma
    return coeffs


def set_general_metadata(bbasisconfig: BBasisConfiguration, **kwargs) -> None:
    bbasisconfig.metadata[METADATA_STARTTIME_KW] = str(datetime.now())
    if get_username() is not None:
        bbasisconfig.metadata[METADATA_USER_KW] = str(get_username())
    bbasisconfig.metadata["pacemaker_version"] = str(pyace.__version__)
    bbasisconfig.metadata["ace_evaluator_version"] = pyace.get_ace_evaluator_version()
    if kwargs is not None:
        for k, v in kwargs.items():
            bbasisconfig.metadata[str(k)] = str(v)


def safely_update_bbasisconfiguration_coefficients(coeffs: np.array, config: BBasisConfiguration = None) -> None:
    current_coeffs = config.get_all_coeffs()
    for i, c in enumerate(coeffs):
        current_coeffs[i] = c
    config.set_all_coeffs(current_coeffs)


def save_interim_potential(basis_config: BBasisConfiguration, coeffs=None, potential_filename="interim_potential.yaml",
                           verbose=True):
    if coeffs is not None:
        basis_config = basis_config.copy()
        safely_update_bbasisconfiguration_coefficients(coeffs, basis_config)
    basis_config.metadata["intermediate_time"] = str(datetime.now())
    basis_config.save(potential_filename)
    if verbose:
        log.info('Intermediate potential saved in {}'.format(potential_filename))


def save_interim_potential_callback(basis_config: BBasisConfiguration, current_fit_iteration: int,
                                    current_fit_cycle: int,
                                    current_ladder_step: int):
    save_interim_potential(basis_config=basis_config,
                           potential_filename="interim_potential_{}.yaml".format(current_fit_cycle),
                           verbose=False)


def active_import(name):
    """
    This function will import the
    :param name:
    :type name:
    :return:
    :rtype:
    """
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def plot_ef_distributions(df, suffix=""):
    try:
        import matplotlib.pylab as plt

        energies = df["energy_corrected_per_atom"]
        forces = df["forces"].map(lambda f: np.linalg.norm(f, axis=1))
        forces = np.hstack(forces)

        fig, axs = plt.subplots(1, 2, figsize=(7, 2), dpi=150)

        axs[0].hist(energies, bins=100)
        axs[0].set_yscale('log')
        axs[0].set_title('DFT energy distribution')
        axs[0].set_xlabel("E, eV/atom")

        # FORCES
        axs[1].hist(forces, bins=100)
        axs[1].set_yscale('log')
        axs[1].set_title('DFT forces norm distribution')
        axs[1].set_xlabel("|F|, eV/A")

        fig.tight_layout()
        fig.savefig(suffix + "ef-distributions.png")
    except Exception as e:
        log.error("Error while plotting energies/forces distributions: {}".format(e))
