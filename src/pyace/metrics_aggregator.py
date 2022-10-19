import logging
import numpy as np
import os
from pyace.preparedata import E_CHULL_DIST_PER_ATOM

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


class FitMetrics:
    def __init__(self, w_e, w_f, e_scale, f_scale, ncoefs, regs=None):
        self.w_e = w_e
        self.w_f = w_f
        self.e_scale = e_scale
        self.f_scale = f_scale
        self.regs = regs
        self.ncoefs = ncoefs
        self.nfuncs = None
        self.time_history = []

    def record_time(self, time):
        self.time_history.append(time)

    def to_FitMetricsDict(self):
        """
        Store all metric-relevant info into a dictionary
        :return: fit metrics dictionary
        """

        regularization_loss = [float(r_comp * r_weight) for r_comp, r_weight in zip(self.regs, self.reg_weights)]
        l1 = regularization_loss[0]
        l2 = regularization_loss[1]
        smoothness_reg_loss = regularization_loss[2:]
        res_dict = {
            # total loss
            "loss": self.loss,

            # loss contributions
            "e_loss_contrib": self.e_loss * self.e_scale,
            "f_loss_contrib": self.f_loss * self.f_scale,
            "l1_reg_contrib": l1,
            "l2_reg_contrib": l2,
            "extra_regularization_contrib": smoothness_reg_loss,

            # non-weighted e and f losses
            "e_loss": self.e_loss,
            "f_loss": self.f_loss,

            # e and f loss weights (scales)
            "e_scale": self.e_scale,
            "f_scale": self.f_scale,

            # RMSE metrics
            "rmse_epa": self.rmse_epa,
            "low_rmse_epa": self.low_rmse_epa,
            "rmse_f": self.rmse_f,
            "low_rmse_f": self.low_rmse_f,
            "rmse_f_comp": self.rmse_f_comp,
            "low_rmse_f_comp": self.low_rmse_f_comp,

            # MAE metrics
            "mae_epa": self.mae_epa,
            "low_mae_epa": self.low_mae_epa,
            "mae_f": self.mae_f,
            "low_mae_f": self.low_mae_f,
            "mae_f_comp": self.mae_f_comp,
            "low_mae_f_comp": self.low_mae_f_comp,

            # MAX metrics
            "max_abs_epa": self.max_abs_epa,
            "low_max_abs_epa": self.low_max_abs_epa,
            "max_abs_f": self.max_abs_f,
            "low_max_abs_f": self.low_max_abs_f,
            "max_abs_f_comp": self.max_abs_f_comp,
            "low_max_abs_f_comp": self.low_max_abs_f_comp,

            "eval_time": self.eval_time,
            "nat": self.nat,
            "ncoefs": self.ncoefs
        }

        if self.nfuncs is not None:
            res_dict["nfuncs"] = self.nfuncs

        return res_dict

    def from_FitMetricsDict(self, fit_metrics_dict):
        self.loss = fit_metrics_dict["loss"]

        self.e_loss = fit_metrics_dict["e_loss"]
        self.f_loss = fit_metrics_dict["f_loss"]

        self.e_scale = fit_metrics_dict["e_scale"]
        self.f_scale = fit_metrics_dict["f_scale"]

        # RMSE metrics
        self.rmse_epa = fit_metrics_dict["rmse_epa"]
        self.low_rmse_epa = fit_metrics_dict["low_rmse_epa"]
        self.rmse_f = fit_metrics_dict["rmse_f"]
        self.low_rmse_f = fit_metrics_dict["low_rmse_f"]

        self.rmse_f_comp = fit_metrics_dict["rmse_f_comp"]
        self.low_rmse_f_comp = fit_metrics_dict["low_rmse_f_comp"]

        # MAE metrics
        self.mae_epa = fit_metrics_dict["mae_epa"]
        self.low_mae_epa = fit_metrics_dict["low_mae_epa"]
        self.mae_f = fit_metrics_dict["mae_f"]
        self.low_mae_f = fit_metrics_dict["low_mae_f"]
        self.mae_f_comp = fit_metrics_dict["mae_f_comp"]
        self.low_mae_f_comp = fit_metrics_dict["low_mae_f_comp"]

        # MAX metrics
        self.max_abs_epa = fit_metrics_dict["max_abs_epa"]
        self.low_max_abs_epa = fit_metrics_dict["low_max_abs_epa"]
        self.max_abs_f = fit_metrics_dict["max_abs_f"]
        self.low_max_abs_f = fit_metrics_dict["low_max_abs_f"]

        self.max_abs_f_comp = fit_metrics_dict["max_abs_f_comp"]
        self.low_max_abs_f_comp = fit_metrics_dict["low_max_abs_f_comp"]

        self.eval_time = fit_metrics_dict["eval_time"]
        self.nat = fit_metrics_dict["nat"]
        self.ncoefs = fit_metrics_dict["ncoefs"]

        if "nfuncs" in fit_metrics_dict:
            self.nfuncs = fit_metrics_dict["nfuncs"]

    def compute_metrics(self, de, de_pa, df, nat, dataframe=None, de_low=None):
        if de_low is None:
            de_low = 1.
        self.nat = np.sum(nat)
        self.rmse_epa = np.sqrt(np.mean(de_pa ** 2))
        self.rmse_e = np.sqrt(np.mean(de ** 2))
        self.rmse_f = np.sqrt(np.mean(np.sum(df ** 2, axis=1)))
        self.rmse_f_comp = np.sqrt(np.mean(df ** 2))  # per component
        self.mae_epa = np.mean(np.abs(de_pa))
        self.mae_e = np.mean(np.abs(de))
        self.mae_f = np.mean(np.linalg.norm(df, axis=1))
        self.mae_f_comp = np.mean(np.abs(df).flatten())  # per component

        self.e_loss = np.float(np.sum(self.w_e * de_pa ** 2))
        self.f_loss = np.sum(self.w_f * df ** 2)
        self.max_abs_e = np.max(np.abs(de))
        self.max_abs_epa = np.max(np.abs(de_pa))
        self.max_abs_f = np.max(np.abs(df))
        self.max_abs_f_comp = np.max(np.abs(df).flatten())  # per component

        self.low_rmse_epa = 0
        self.low_mae_epa = 0
        self.low_max_abs_epa = 0
        self.low_rmse_f = 0
        self.low_mae_f = 0
        self.low_max_abs_f = 0
        self.low_rmse_f_comp= 0

        if dataframe is not None:
            try:
                if E_CHULL_DIST_PER_ATOM in dataframe.columns:
                    nrgs = dataframe[E_CHULL_DIST_PER_ATOM].to_numpy().reshape(-1, )
                    mask = nrgs <= de_low
                else:
                    nrgs = dataframe['energy_corrected'].to_numpy().reshape(-1, ) / nat.reshape(-1, )
                    emin = min(nrgs)
                    mask = (nrgs <= (emin + de_low))
                nat = nat.astype(int)
                mask_f = np.repeat(mask, nat.reshape(-1, ))
                self.low_rmse_epa = np.sqrt(np.mean(de_pa[mask] ** 2))
                self.low_mae_epa = np.mean(np.abs(de_pa[mask]))
                self.low_max_abs_epa = np.max(np.abs(de_pa[mask]))
                self.low_rmse_f = np.sqrt(np.mean(np.sum(df[mask_f] ** 2, axis=1)))
                self.low_mae_f = np.mean(np.linalg.norm(df[mask_f], axis=1))
                self.low_max_abs_f = np.max(np.abs(df[mask_f]))
                self.low_max_abs_f_comp = np.max(np.abs(df[mask_f]).flatten())  # per component
                self.low_rmse_f_comp = np.sqrt(np.mean(df[mask_f] ** 2))  # per component
                self.low_mae_f_comp = np.mean(np.abs(df[mask_f]).flatten())  # per component
            except Exception as e:
                pass


class MetricsAggregator:
    # format: str (column name) or tuple (column name, col width)
    columns = [("ladder_step", 11), ("cycle_step", 11), ("iter_num", 8),  # reduce column width
               "loss", "e_loss_contrib", "f_loss_contrib", "reg_loss",
               "rmse_epa", "rmse_f_comp", "low_rmse_epa", "low_rmse_f_comp",
               "mae_f_comp", "low_mae_f_comp",
               ("nfuncs", 6), ("ncoefs", 6),  # reduce columns width
               "l1_reg_contrib", "l2_reg_contrib",
               "smooth_orth", "smooth_w1", "smooth_w2", "smooth_w3"
               ]

    def __init__(self, extended_display_step=20,
                 running_metrics_filename="metrics.txt",
                 ladder_metrics_filename="ladder_metrics.txt",
                 cycle_metrics_filename="cycle_metrics.txt",
                 test_running_metrics_filename="test_metrics.txt",
                 test_ladder_metrics_filename="test_ladder_metrics.txt",
                 test_cycle_metrics_filename="test_cycle_metrics.txt",
                 ):
        self.extended_display_step = extended_display_step

        self.running_metrics_filename = running_metrics_filename
        self.ladder_metrics_filename = ladder_metrics_filename
        self.cycle_metrics_filename = cycle_metrics_filename

        self.test_running_metrics_filename = test_running_metrics_filename
        self.test_ladder_metrics_filename = test_ladder_metrics_filename
        self.test_cycle_metrics_filename = test_cycle_metrics_filename

        if os.path.isfile(self.running_metrics_filename):
            os.remove(self.running_metrics_filename)
        self.write_metric_table_title_to_file(self.running_metrics_filename)
        self.write_metric_table_title_to_file(self.test_running_metrics_filename)

        if self.ladder_metrics_filename is not None:
            if os.path.isfile(self.ladder_metrics_filename):
                os.remove(self.ladder_metrics_filename)
            self.ladder_metrics_columns = MetricsAggregator.columns.copy()
            # except cycle_step and iter_num
            self.ladder_metrics_columns = self.ladder_metrics_columns[:1] + self.ladder_metrics_columns[3:]
            self.write_metric_table_title_to_file(self.ladder_metrics_filename, columns=self.ladder_metrics_columns)

        if self.cycle_metrics_filename is not None:
            if os.path.isfile(self.cycle_metrics_filename):
                os.remove(self.cycle_metrics_filename)
            self.cycle_metrics_columns = MetricsAggregator.columns.copy()
            self.cycle_metrics_columns = self.cycle_metrics_columns[:2] + self.cycle_metrics_columns[3:]
            self.write_metric_table_title_to_file(self.cycle_metrics_filename,
                                                  columns=self.cycle_metrics_columns)

        if self.test_ladder_metrics_filename is not None:
            if os.path.isfile(self.test_ladder_metrics_filename):
                os.remove(self.test_ladder_metrics_filename)
            self.ladder_metrics_columns = MetricsAggregator.columns.copy()
            # except cycle_step and iter_num
            self.ladder_metrics_columns = self.ladder_metrics_columns[:1] + self.ladder_metrics_columns[3:]
            self.write_metric_table_title_to_file(self.test_ladder_metrics_filename,
                                                  columns=self.ladder_metrics_columns)

        if self.test_cycle_metrics_filename is not None:
            if os.path.isfile(self.test_cycle_metrics_filename):
                os.remove(self.test_cycle_metrics_filename)
            self.cycle_metrics_columns = MetricsAggregator.columns.copy()
            self.cycle_metrics_columns = self.cycle_metrics_columns[:2] + self.cycle_metrics_columns[3:]
            self.write_metric_table_title_to_file(self.test_cycle_metrics_filename,
                                                  columns=self.cycle_metrics_columns)

    ## FIT/TRAIN metrics writers
    def ladder_step_callback(self, fit_metrics_dict):
        self.print_extended_metrics(fit_metrics_dict, title='LADDER STEP')
        self.append_metric_line_to_file(fit_metrics_dict, filename=self.ladder_metrics_filename,
                                        columns=self.ladder_metrics_columns)

    def cycle_step_callback(self, fit_metrics_dict):
        self.print_extended_metrics(fit_metrics_dict, title='Cycle last iteration:')
        self.append_metric_line_to_file(fit_metrics_dict, filename=self.cycle_metrics_filename,
                                        columns=self.cycle_metrics_columns)

    def fit_metric_callback(self, fit_metrics_dict, extended_display_step=None):
        if extended_display_step is None:
            extended_display_step = self.extended_display_step

        self.append_metric_line_to_file(fit_metrics_dict, filename=self.running_metrics_filename)

        iter_num = fit_metrics_dict["iter_num"]
        if iter_num == 0:
            self.print_detailed_metrics(fit_metrics_dict, title='Initial state:')
            self.print_extended_metrics(fit_metrics_dict, title='INIT STATS')
        elif iter_num % extended_display_step == 0:
            self.print_extended_metrics(fit_metrics_dict, title='FIT STATS')
        else:
            self.print_detailed_metrics(fit_metrics_dict, title="Iteration")

    ## TEST metrics writers
    def test_ladder_step_callback(self, fit_metrics_dict):
        self.print_extended_metrics(fit_metrics_dict, title='TEST LADDER STEP')
        self.append_metric_line_to_file(fit_metrics_dict, filename=self.test_ladder_metrics_filename,
                                        columns=self.ladder_metrics_columns)

    def test_cycle_step_callback(self, fit_metrics_dict):
        self.print_extended_metrics(fit_metrics_dict, title='TEST Cycle last iteration:')
        self.append_metric_line_to_file(fit_metrics_dict, filename=self.test_cycle_metrics_filename,
                                        columns=self.cycle_metrics_columns)

    def test_metric_callback(self, metrics_dict, extended_display_step=None):
        if extended_display_step is None:
            extended_display_step = self.extended_display_step

        self.append_metric_line_to_file(metrics_dict, filename=self.test_running_metrics_filename)

        iter_num = metrics_dict["iter_num"]
        if iter_num == 0:
            self.print_detailed_metrics(metrics_dict, title='Initial(TEST):')
            self.print_extended_metrics(metrics_dict, title='INIT TEST STATS')
        elif iter_num % extended_display_step == 0:
            self.print_extended_metrics(metrics_dict, title='TEST STATS')

    def write_metric_table_title_to_file(self, filename="metrics.log", columns=None):
        if columns is None:
            columns = MetricsAggregator.columns
        fmt_content_list, fmt_string_list = self._prepare_fmt_string_content(columns)

        fmt_string = " ".join(fmt_string_list)
        title = fmt_string.format(*fmt_content_list)

        with open(filename, "w") as f:
            print(title, file=f)

    def _prepare_fmt_string_content(self, columns):
        fmt_string_list = []
        fmt_content_list = []
        for c in columns:
            if isinstance(c, str):
                fmt_string_list.append("{:<22}")
                fmt_content_list.append(c)
            elif isinstance(c, tuple):
                fmt_content_list.append(c[0])
                txt_width = min(len(c[0]), c[1])
                fmt_string_list.append("{:<" + str(txt_width) + "}")
        return fmt_content_list, fmt_string_list

    def append_metric_line_to_file(self, fit_metrics_dict, filename="metrics.log", columns=None):
        if columns is None:
            columns = MetricsAggregator.columns

        fit_metrics_dict = fit_metrics_dict.copy()

        # collect all regularization contributions
        reg_loss = fit_metrics_dict["l1_reg_contrib"] + fit_metrics_dict["l2_reg_contrib"]
        if len(fit_metrics_dict["extra_regularization_contrib"]):
            for reg in fit_metrics_dict["extra_regularization_contrib"]:
                reg_loss += reg
        fit_metrics_dict["reg_loss"] = reg_loss

        smooth_reg = fit_metrics_dict["extra_regularization_contrib"]
        if len(smooth_reg) == 0:
            smooth_reg = [0, 0, 0, 0]
        elif len(smooth_reg) == 1:  # w_orth only
            smooth_reg = smooth_reg + [0, 0, 0]
        elif len(smooth_reg) == 3:  # 3 w_smooth only
            smooth_reg = [0] + smooth_reg
        # smooth_reg: (w_orth, smooth_w0, w1, w2)
        fit_metrics_dict["smooth_orth"] = smooth_reg[0]
        fit_metrics_dict["smooth_w1"] = smooth_reg[1]
        fit_metrics_dict["smooth_w2"] = smooth_reg[2]
        fit_metrics_dict["smooth_w3"] = smooth_reg[3]

        columns, fmt_string_list = self._prepare_fmt_string_content(columns)
        fmt_string = " ".join(fmt_string_list)
        log_message = fmt_string.format(*[fit_metrics_dict.get(col, 0) for col in columns])

        with open(filename, "a") as f:
            print(log_message, file=f)

    @staticmethod
    def print_extended_metrics(fit_metrics_dict, title="FIT_STATS"):
        # (self, iter_num, total_loss, reg_comps, reg_weights, title='FIT STATS', nfuncs=None):

        iter_num = fit_metrics_dict["iter_num"]
        total_loss = fit_metrics_dict["loss"]

        str0 = '\n' + '-' * 44 + title + '-' * 44 + '\n'
        str1 = '{prefix:<11} #{iter_num:<4}'.format(prefix='Iteration:', iter_num=iter_num)
        str1 += '{prefix:<8}'.format(prefix='Loss:')
        str1 += '{prefix:>8} {tot_loss:>1.4e} ({fr:3.0f}%) '.format(prefix='Total: ', tot_loss=total_loss, fr=100)
        str1 += '\n'

        fr = fit_metrics_dict["e_loss_contrib"] / total_loss * 100 if total_loss > 0 else 0
        str2 = '{prefix:>33} {e_loss:>1.4e} ({fr:3.0f}%) '.format(prefix='Energy: ',
                                                                  e_loss=fit_metrics_dict["e_loss_contrib"],
                                                                  fr=fr)
        str2 += '\n'

        fr = fit_metrics_dict["f_loss_contrib"] / total_loss * 100 if total_loss > 0 else 0
        str3 = '{prefix:>33} {f_loss:>1.4e} ({fr:3.0f}%) '.format(prefix='Force: ',
                                                                  f_loss=fit_metrics_dict["f_loss_contrib"],
                                                                  fr=fr)
        str3 += '\n'

        l1 = fit_metrics_dict["l1_reg_contrib"]
        l2 = fit_metrics_dict["l2_reg_contrib"]

        fr = l1 / total_loss * 100 if total_loss != 0 else 0
        str4 = '{prefix:>33} {l1:>1.4e} ({fr:3.0f}%) '.format(prefix='L1: ', l1=l1, fr=fr)
        str4 += '\n'
        fr = l2 / total_loss * 100 if total_loss != 0 else 0
        str4 += '{prefix:>33} {l2:>1.4e} ({fr:3.0f}%) '.format(prefix='L2: ', l2=l2, fr=fr)
        str4 += '\n'

        reg_comps = fit_metrics_dict["extra_regularization_contrib"]
        str5 = ''
        for i, comp in enumerate(reg_comps):
            fr = comp / total_loss * 100 if total_loss != 0 else 0
            str5 += '{prefix:>33} '.format(prefix='Smooth_w{}: '.format(i + 1))
            str5 += '{s1:>1.4e} '.format(s1=comp)
            str5 += '({fr:3.0f}%) '.format(fr=fr)
            str5 += '\n'

        nfuncs = fit_metrics_dict.get('nfuncs')
        ncoefs = fit_metrics_dict.get('ncoefs')

        if nfuncs is None:
            line = 'Number of params.: '
        else:
            line = 'Number of params./funcs: '
        str6 = '{prefix:>20}'.format(prefix=line) + '{ncoefs:>6d}'.format(ncoefs=ncoefs)
        if nfuncs is not None:
            str6 += '/{nfuncs:<6d}'.format(nfuncs=nfuncs)

        avg_t = fit_metrics_dict["eval_time"] / fit_metrics_dict["nat"]  # in sec/atom
        str6 += '{prefix:>42}'.format(prefix='Avg. time: ') + \
                '{avg_t:>10.2f} {un:<6}'.format(avg_t=avg_t * 1e6, un='mcs/at')

        str6 += '\n' + '-' * 97 + '\n'
        str_loss = str0 + str1 + str2 + str3 + str4 + str5 + str6
        ##############################
        er_str_h = '{:>9}'.format('') + \
                   '{:^22}'.format('Energy/at, meV/at') + \
                   '{:^22}'.format('Energy_low/at, meV/at') + \
                   '{:^22}'.format('Force, meV/A') + \
                   '{:^22}\n'.format('Force_low, meV/A')

        er_rmse = '{prefix:>9} '.format(prefix='RMSE: ')
        er_rmse += '{:>14.2f}'.format(fit_metrics_dict["rmse_epa"] * 1e3) + \
                   '{:>21.2f}'.format(fit_metrics_dict["low_rmse_epa"] * 1e3) + \
                   '{:>21.2f}'.format(fit_metrics_dict["rmse_f_comp"] * 1e3) + \
                   '{:>24.2f}\n'.format(fit_metrics_dict["low_rmse_f_comp"] * 1e3)
        er_mae = '{prefix:>9} '.format(prefix='MAE: ')
        er_mae += '{:>14.2f}'.format(fit_metrics_dict["mae_epa"] * 1e3) + \
                  '{:>21.2f}'.format(fit_metrics_dict["low_mae_epa"] * 1e3) + \
                  '{:>21.2f}'.format(fit_metrics_dict["mae_f_comp"] * 1e3) + \
                  '{:>24.2f}\n'.format(fit_metrics_dict["low_mae_f_comp"] * 1e3)
        er_max = '{prefix:>9} '.format(prefix='MAX_AE: ')
        er_max += '{:>14.2f}'.format(fit_metrics_dict["max_abs_epa"] * 1e3) + \
                  '{:>21.2f}'.format(fit_metrics_dict["low_max_abs_epa"] * 1e3) + \
                  '{:>21.2f}'.format(fit_metrics_dict.get("max_abs_f_comp", 0) * 1e3) + \
                  '{:>24.2f}\n'.format(fit_metrics_dict.get("low_max_abs_f_comp", 0) * 1e3)
        er_str = er_str_h + er_rmse + er_mae + er_max + '-' * 97  # + '\n'
        log.info(str_loss + er_str)

    @staticmethod
    def print_detailed_metrics(fit_metrics_dict, title='Iteration:'):
        # fit_metrics_dict
        iter_num = fit_metrics_dict["iter_num"]
        total_loss = fit_metrics_dict["loss"]
        avg_t = fit_metrics_dict["eval_time"] / fit_metrics_dict["nat"]  # in sec/atom
        log.info('{:<12}'.format(title) +
                 "#{iter_num:<5}".format(iter_num=iter_num) +
                 '{:<14}'.format('({numeval} evals):'.format(numeval=fit_metrics_dict["eval_count"])) +
                 '{:>10}'.format('Loss: ') + "{loss: >3.6f}".format(loss=total_loss) +
                 '{str1:>21}{rmse_epa:>.2f} ({low_rmse_e:>.2f}) meV/at' \
                 .format(str1=" | RMSE Energy(low): ",
                         rmse_epa=1e3 * fit_metrics_dict["rmse_epa"],
                         low_rmse_e=1e3 * fit_metrics_dict["low_rmse_epa"]) +
                 '{str3:>16}{rmse_f_comp:>.2f} ({low_rmse_f_comp:>.2f}) meV/A' \
                 .format(str3=" | Forces(low): ",
                         rmse_f_comp=1e3 * fit_metrics_dict["rmse_f_comp"],
                         low_rmse_f_comp=1e3 * fit_metrics_dict["low_rmse_f_comp"]) +
                 ' | Time/eval: {:>6.2f} mcs/at'.format(avg_t * 1e6))
