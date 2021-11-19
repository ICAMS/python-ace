import os
import numpy as np
import matplotlib as mpl
import warnings

# TODO: check if running in interactive mode
mpl.use('Agg')
import matplotlib.pyplot as plt

from scipy import stats


def plot_analyse_error_distributions(df_pred, fig_prefix='train_', fig_path='.', imagetype="png"):
    if not os.path.isdir(fig_path):
        os.makedirs(fig_path)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df_pred["energy_pred"] = df_pred["energy_pred"].astype(float)
        df_pred["energy_corrected_per_atom"] = df_pred["energy_corrected_per_atom"].astype(float)
        df_pred["energy_corrected"] = df_pred["energy_corrected"].astype(float)

        # energies
        df_pred["energy_pred_per_atom"] = df_pred["energy_pred"] / df_pred["NUMBER_OF_ATOMS"]

        df_pred["dE"] = df_pred["energy_pred"] - df_pred["energy_corrected"]
        df_pred["dEpa"] = df_pred["dE"] / df_pred["NUMBER_OF_ATOMS"]

        # forces
        forces = np.vstack(df_pred["forces"])
        forces_pred = np.vstack(df_pred["forces_pred"])
        forces_norm = np.linalg.norm(forces, axis=1)
        dforces = forces_pred - forces
        dforces_norm = np.linalg.norm(dforces, axis=1)
        dforces_flatten = dforces.flatten()
        forces_flatten = forces.flatten()

        energy_corrected_per_atom = df_pred["energy_corrected_per_atom"].values
        energy_pred_per_atom = df_pred["energy_pred_per_atom"].values
        dEpa = df_pred["dEpa"].values

        #####################################
        # 1. E/F pair plots
        #####################################

        fig, (ax1, ax2) = plt.subplots(1, 2, dpi=150, figsize=(8, 4))

        ax1.scatter(energy_corrected_per_atom, energy_pred_per_atom, s=3)
        ax1.set_xlabel('E(DFT), eV/at')
        ax1.set_ylabel('E, eV/at')
        ax1.set_title('Energy')

        xlim = ax1.get_ylim()
        ylim = ax1.get_ylim()
        lim = (min(xlim[0], ylim[0]), max(xlim[1], ylim[1]))
        ax1.set_xlim(lim)
        ax1.set_ylim(lim)
        ax1.set_aspect(1)

        ax2.scatter(forces, forces_pred, s=3)
        ax2.set_xlabel('F$_i$(DFT), eV/$\\AA$')
        ax2.set_ylabel('F$_i$, eV/$\\AA$')
        ax2.set_title('Force components')
        xlim = ax2.get_ylim()
        ylim = ax2.get_ylim()
        lim = (min(xlim[0], ylim[0]), max(xlim[1], ylim[1]))
        ax2.set_xlim(lim)
        ax2.set_ylim(lim)
        ax2.set_aspect(1)
        plt.tight_layout()
        plt.savefig(os.path.join(fig_path, fig_prefix + "EF-pairplots."+imagetype))

        ########################################
        ## 2. E-error dist plot
        ########################################
        fig, ((ax0, ax_inv), (ax1, ax2)) = plt.subplots(2, 2, dpi=150, figsize=(7, 4),
                                                        gridspec_kw={'width_ratios': [7, 1],
                                                                     'height_ratios': [1, 3]
                                                                     })

        ax_inv.set_visible(False)

        energy_hist_res = ax0.hist(energy_corrected_per_atom, bins=100, color="gray")
        ax0.get_xaxis().set_visible(False)
        ax0.get_yaxis().set_visible(False)
        yl = ax0.get_ylim()
        lim = (yl[0] - (yl[1] - yl[0]) * 0.05, yl[1])
        ax0.set_ylim(lim)

        ##########
        # samples
        ax1.scatter(energy_corrected_per_atom, dEpa * 1e3, s=3, label="samples", color="lightsteelblue")
        # aggregated over bins
        bins = energy_hist_res[1]
        bins_inds = np.digitize(energy_corrected_per_atom, bins=bins) - 1

        dEpa_per_bin = np.zeros_like(bins)
        for bin_ind in range(min(bins_inds), max(bins_inds) + 1):
            dEpa_per_bin[bin_ind] = np.sqrt(np.mean(dEpa[bins_inds == bin_ind] ** 2))

        dEpa_per_bin = np.nan_to_num(dEpa_per_bin, copy=False, nan=0)

        ax1.plot(bins, dEpa_per_bin * 1e3, color="red", label="RMSE", lw=2)

        ax1.set_xlabel('E(DFT), eV/atom')
        ax1.set_ylabel('dE, meV/atom')

        ax1.legend()
        ##########
        ax2.get_xaxis().set_visible(False)
        ax2.get_yaxis().set_visible(False)

        hist_res = ax2.hist(dEpa * 1e3, bins=100, density=True, color="gray", orientation="horizontal");
        dE_std = np.std(dEpa)
        dE_mean = np.mean(dEpa)
        dExx = hist_res[1]
        ax2.plot(stats.norm.pdf(dExx, loc=dE_mean * 1e3, scale=dE_std * 1e3), dExx, color='orange', ls='--')

        yl = ax2.get_xlim()
        lim = (yl[0] - (yl[1] - yl[0]) * 0.05, yl[1])
        ax2.set_xlim(lim)
        ax2.set_ylim(ax1.get_ylim())
        plt.subplots_adjust(wspace=0.0, hspace=0.0)

        plt.savefig(os.path.join(fig_path, fig_prefix + "E-dE-dist."+imagetype))

        ##############################
        # 3. Force component dist plot
        ##############################

        fig, ((ax0, ax_inv), (ax1, ax2)) = plt.subplots(2, 2, dpi=150, figsize=(7, 4),
                                                        gridspec_kw={'width_ratios': [7, 1],
                                                                     'height_ratios': [1, 3]
                                                                     })

        ax_inv.set_visible(False)
        force_hist_res = ax0.hist(forces_flatten, bins=150, color="gray")
        ax0.get_xaxis().set_visible(False)
        ax0.get_yaxis().set_visible(False)
        yl = ax0.get_ylim()
        lim = (yl[0] - (yl[1] - yl[0]) * 0.05, yl[1])
        ax0.set_ylim(lim)

        ##############3

        ax1.scatter(forces_flatten, dforces_flatten * 1e3, s=3, label="sample", color="lightsteelblue")
        ax1.set_xlabel('F$_i$, eV/$\\AA$')
        ax1.set_ylabel('dF$_i$, meV/$\\AA$')

        # aggregated over bins
        bins = force_hist_res[1]
        bins_inds = np.digitize(forces_flatten, bins=bins) - 1

        de_per_bin = np.zeros_like(bins)

        for bin_ind in range(min(bins_inds), max(bins_inds) + 1):
            de_per_bin[bin_ind] = np.sqrt(np.mean(dforces_flatten[bins_inds == bin_ind] ** 2))

        de_per_bin = np.nan_to_num(de_per_bin, copy=False, nan=0)
        ax1.plot(bins, de_per_bin * 1e3, color="red", lw=2, label="RMSE")

        ax1.legend()
        #################
        ax2.get_xaxis().set_visible(False)
        ax2.get_yaxis().set_visible(False)
        hist_res = ax2.hist(dforces_flatten * 1e3,
                            orientation='horizontal',
                            bins=100,
                            density=True, color="gray")

        df_mean = np.mean(dforces_flatten)
        df_std = np.std(dforces_flatten)
        dFxx = hist_res[1]
        ax2.plot(stats.norm.pdf(dFxx, loc=df_mean * 1e3, scale=df_std * 1e3), dFxx, color='orange', ls='--',
                 label="N(mean={:.1f}/std={:.1f})".format(df_mean * 1e3, df_std * 1e3)
                 )

        yl = ax2.get_xlim()
        lim = (yl[0] - (yl[1] - yl[0]) * 0.05, yl[1])
        ax2.set_xlim(lim)
        ax2.set_ylim(ax1.get_ylim())
        plt.subplots_adjust(wspace=0.0, hspace=0.0)
        plt.savefig(os.path.join(fig_path, fig_prefix + "Fi-dFi-dist."+imagetype))

        #######################################################################
        # 4. Force-norm error dist plot
        #######################################################################
        fig, ((ax0, ax_inv), (ax1, ax2)) = plt.subplots(2, 2, dpi=150, figsize=(7, 4),
                                                        gridspec_kw={'width_ratios': [7, 1],
                                                                     'height_ratios': [1, 3]
                                                                     })

        ax_inv.set_visible(False)

        force_hist_res = ax0.hist(forces_norm, bins=150, color="gray")
        ax0.get_xaxis().set_visible(False)
        ax0.get_yaxis().set_visible(False)
        yl = ax0.get_ylim()
        lim = (yl[0] - (yl[1] - yl[0]) * 0.05, yl[1])
        ax0.set_ylim(lim)

        ##############3

        ax1.scatter(forces_norm, dforces_norm * 1e3, s=3, label="sample", color="lightsteelblue")
        ax1.set_xlabel('|F|, eV/$\\AA$')
        ax1.set_ylabel('|dF|, meV/$\\AA$')

        # aggregated over bins
        bins = force_hist_res[1]
        bins_inds = np.digitize(forces_norm, bins=bins) - 1

        de_per_bin = np.zeros_like(bins)

        for bin_ind in range(min(bins_inds), max(bins_inds) + 1):
            de_per_bin[bin_ind] = np.sqrt(np.mean(dforces_norm[bins_inds == bin_ind] ** 2))
        de_per_bin = np.nan_to_num(de_per_bin, copy=False, nan=0)
        ax1.plot(bins, de_per_bin * 1e3, color="red", lw=2, label="RMSE")

        ax1.legend()
        #################
        ax2.get_xaxis().set_visible(False)
        ax2.get_yaxis().set_visible(False)
        hist_res = ax2.hist(dforces_norm * 1e3,
                            orientation='horizontal',
                            bins=100,
                            density=True, color="gray")

        df_mean = np.mean(dforces_norm)
        df_std = np.std(dforces_norm)
        dFxx = hist_res[1]
        ax2.plot(stats.norm.pdf(dFxx, loc=df_mean * 1e3, scale=df_std * 1e3), dFxx, color='orange', ls='--'
                 )

        yl = ax2.get_xlim()
        lim = (yl[0] - (yl[1] - yl[0]) * 0.05, yl[1])
        ax2.set_xlim(lim)
        ax2.set_ylim(ax1.get_ylim())

        plt.subplots_adjust(wspace=0.0, hspace=0.0)
        plt.savefig(os.path.join(fig_path, fig_prefix + "F-dF-dist."+imagetype))

        #######################################################################
        # 5. Force-norm error dist plot
        #######################################################################
        fig, (ax1, ax2) = plt.subplots(2, 1, dpi=150, figsize=(7, 4), sharex=True)

        ax1.scatter(df_pred["nn_min"], df_pred["energy_corrected_per_atom"], s=3)
        ax1.set_ylabel('E(DFT), eV/atom')

        ax2.scatter(df_pred["nn_min"], df_pred["dEpa"] * 1e3, s=3)
        ax2.set_xlabel('d$_{NN}$, $\\AA$')
        ax2.set_ylabel('dE, meV/atom')

        fig.tight_layout()
        fig.savefig(os.path.join(fig_path, fig_prefix + "E-dE-nn."+imagetype))
