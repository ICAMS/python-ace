# Author: Lysogorskiy Yury, Antoine Kraych
# originally taken 18.06.2020
# from
# https://git.noc.ruhr-uni-bochum.de/atomicclusterexpansion/pyace/-/blob/6dea4f42636b95798a9133dec41ad7669feb34f5/notebooks/plot_radial_functions.ipynb

import numpy as np
from typing import Union

from pyace import ACECTildeBasisSet, ACEBBasisSet, BBasisConfiguration


def integrate(xs, table):
    # table must be frs / dfrs / ddfrs and corresponding axis (xs or cxs)
    # F is the function to integrate (x**2*sum_nl(R_nlnl))
    # dx=xs[1]-xs[0]
    if table is not None:
        frs = np.abs(table)
        sum_frs = np.sum(frs, axis=(1, 2))
        integrand = sum_frs * xs ** 2
        integral = np.trapz(integrand, x=xs)
        return integral
    else:
        return 0


class RadialFunctionsValues:
    def __init__(self, basis_set: Union[ACEBBasisSet, ACECTildeBasisSet], npoints=None):
        self.basisSet = basis_set
        self.radial_functions = basis_set.radial_functions

        self.cutoff = np.max(self.radial_functions.cut)
        self.dx = self.radial_functions.deltaSplineBins

        if npoints is None:
            npoints = int(np.floor(self.cutoff / self.radial_functions.deltaSplineBins))
        self.npoints = npoints
        # Create the list of R_nl & g_k functions and first derivatives
        self.xs = np.linspace(self.cutoff / self.npoints + 1e-10, self.cutoff, num=self.npoints)

        self.nelements = self.radial_functions.nelements

        # setup grs and its derivatives
        grs_shape = (self.nelements, self.nelements, self.npoints,
                     self.radial_functions.nradbase)
        self.grs = np.zeros(grs_shape)  # (nelements, nelements, npoints, nradbase)
        self.dgrs = np.zeros(grs_shape)
        self.ddgrs = np.zeros(grs_shape)

        # setup frs and its derivatives
        self.frs = None
        self.dfrs = None
        self.ddfrs = None
        frs_shape = (self.nelements, self.nelements, self.npoints,
                     self.radial_functions.nradial, self.radial_functions.lmax + 1)
        if self.radial_functions.nradial > 0:
            self.frs = np.zeros(frs_shape)
            self.dfrs = np.zeros(frs_shape)
            self.ddfrs = np.zeros(frs_shape)

        for mu_i in range(self.nelements):
            for mu_j in range(mu_i, self.nelements):
                self.radial_functions.evaluate_range(self.xs,
                                                     self.radial_functions.nradbase,
                                                     self.radial_functions.nradial,
                                                     mu_i, mu_j)
                # g_k and derivatives
                self.grs[mu_i, mu_j] = self.radial_functions.gr_vec
                self.dgrs[mu_i, mu_j] = self.radial_functions.dgr_vec
                self.ddgrs[mu_i, mu_j] = self.radial_functions.d2gr_vec

                # fill lower triangle with symmetric values
                if mu_i != mu_j:
                    self.grs[mu_j, mu_i] = self.grs[mu_i, mu_j]
                    self.dgrs[mu_j, mu_i] = self.dgrs[mu_i, mu_j]
                    self.ddgrs[mu_j, mu_i] = self.ddgrs[mu_i, mu_j]

                # R_nl and derivatives
                if self.radial_functions.nradial > 0:
                    self.frs[mu_i, mu_j] = self.radial_functions.fr_vec
                    self.dfrs[mu_i, mu_j] = self.radial_functions.dfr_vec
                    self.ddfrs[mu_i, mu_j] = self.radial_functions.d2fr_vec

                    # fill lower triangle with symmetric values
                    if mu_i != mu_j:
                        self.frs[mu_j, mu_i] = self.frs[mu_i, mu_j]
                        self.dfrs[mu_j, mu_i] = self.dfrs[mu_i, mu_j]
                        self.ddfrs[mu_j, mu_i] = self.ddfrs[mu_i, mu_j]


class RadialFunctionSmoothness:
    def __init__(self, radialFunctionsValues: RadialFunctionsValues):
        self.radialFunctionsValues = radialFunctionsValues
        self.xs = self.radialFunctionsValues.xs
        self.frs = self.radialFunctionsValues.frs
        self.dfrs = self.radialFunctionsValues.dfrs
        self.ddfrs = self.radialFunctionsValues.ddfrs
        self.cutoff = self.radialFunctionsValues.cutoff
        self._smooth_quad = None

    def compute(self):
        # delta = 0
        self._smooth_quad = []
        nelements = self.radialFunctionsValues.nelements
        for arr in [self.frs, self.dfrs, self.ddfrs]:
            w_reg = 0
            if arr is not None:
                n_terms = 0
                for mu_i in range(nelements):
                    for mu_j in range(mu_i, nelements):
                        if mu_i == mu_j:
                            prefactor = 1
                        else:
                            prefactor = 2
                        w_reg += prefactor * integrate(self.xs, arr[mu_i, mu_j]) / self.cutoff ** 2
                        n_terms += prefactor
                w_reg /= n_terms

            self._smooth_quad.append(w_reg)

    @property
    def smooth_quad(self):
        if self._smooth_quad is None:
            self.compute()
        return self._smooth_quad


class RadialFunctionsVisualization:

    def __init__(self, radialFunctionsValues: Union[
        RadialFunctionsValues, ACEBBasisSet, ACECTildeBasisSet, BBasisConfiguration],
                 k=None,
                 nl=None,
                 xmin=-0.5,
                 xmax=None,
                 ymin=None,
                 ymax=None):
        if isinstance(radialFunctionsValues, BBasisConfiguration):
            bbasis = ACEBBasisSet(radialFunctionsValues)
            radialFunctionsValues = RadialFunctionsValues(bbasis)
        elif isinstance(radialFunctionsValues, (ACEBBasisSet, ACECTildeBasisSet)):
            radialFunctionsValues = RadialFunctionsValues(radialFunctionsValues)

        if not isinstance(radialFunctionsValues, RadialFunctionsValues):
            raise ValueError("radialFunctionsValues must be one of these type:" +
                             "(RadialFunctionsValues, ACEBBasisSet, ACECTildeBasisSet, BBasisConfiguration)" +
                             "but {} is provided".format(type(radialFunctionsValues)))

        self.radialFunctions = radialFunctionsValues
        self.xs = self.radialFunctions.xs
        # TODO: implement it for multispecies
        self.grs = self.radialFunctions.grs  # shape = (nelements, nelements, npoints, nradbase)
        self.dgrs = self.radialFunctions.dgrs
        self.ddgrs = self.radialFunctions.ddgrs

        self.frs = self.radialFunctions.frs  # shape = (nelements, nelements, npoints, nradial, lmax+1)
        self.dfrs = self.radialFunctions.dfrs
        self.ddfrs = self.radialFunctions.ddfrs
        self.Rc = self.radialFunctions.cutoff

        self.nelements = self.radialFunctions.nelements

        self.init_params(k=k, nl=nl, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)

    def init_params(self, k=None,
                    nl=None,
                    xmin=-0.5,
                    xmax=None,
                    ymin=None,
                    ymax=None):
        self.nl = nl
        self.k = k
        self.xmax = xmax
        self.xmin = xmin
        self.ymax = ymax
        self.ymin = ymin

        if not self.k:
            self.t1 = np.array(self.grs).T
            self.t2 = np.array(self.dgrs).T
            self.t3 = np.array(self.ddgrs).T
            self.klabel = range(np.shape(self.ddgrs)[1])
        else:
            self.t1 = []
            self.t2 = []
            self.t3 = []
            for value in self.k:  # k -nradbase
                self.t1.append(np.array(self.grs).T[value])
                self.t2.append(np.array(self.dgrs).T[value])
                self.t3.append(np.array(self.ddgrs).T[value])
            self.klabel = self.k
        if not self.nl:
            self.t4 = [range(0, len(self.frs[0])), range(0, len(self.frs[0][0]))]
        else:
            self.t4 = [self.nl[0], self.nl[1]]
        if not self.xmax:
            self.xmax = self.Rc + 0.5

    def plot(self,
             k=None,
             nl=None,
             xmin=-0.5,
             xmax=None,
             ymin=None,
             ymax=None):

        self.init_params(k=k, nl=nl, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)

        # Graph options
        import matplotlib.pyplot as plt
        # 6 Plots : R_nl  R'_nl  R''_nl / g_k  g'_k  g''_k
        fig = plt.figure(figsize=(16, 20), facecolor='w')
        # Plot 1 : xs / grs
        ax1 = fig.add_subplot(321)
        for i, gr in enumerate(self.t1):
            ax1.plot(self.xs, gr, label='k={}'.format(self.klabel[i]))
            ax1.legend(ncol=2)
        # Plot 2 : xs / fr
        ax2 = fig.add_subplot(323)
        for dgr in self.t2:
            ax2.plot(self.xs, dgr)
        # Plot 3 : xs / dgrs
        ax3 = fig.add_subplot(325)
        for ddgr in self.t3:
            ax3.plot(self.xs, ddgr)
        # Plot 4 : xs / dfrs
        ax4 = fig.add_subplot(322)
        for ii, i in enumerate(self.t4[0]):
            for jj, j in enumerate(self.t4[1]):
                ax4.plot(self.xs, np.transpose(self.frs, axes=(1, 2, 0))[i][j],
                         label='n={} l={}'.format(self.t4[0][ii], self.t4[1][jj]))
                if self.nl:
                    ax4.legend()
        # Plot 5 : xs / ddgrs
        ax5 = fig.add_subplot(324)
        for i in self.t4[0]:
            for j in self.t4[1]:
                ax5.plot(self.xs, np.transpose(self.dfrs, axes=(1, 2, 0))[i][j])
        # Plot 6 : xs / ddfrs
        ax6 = fig.add_subplot(326)
        for i in self.t4[0]:
            for j in self.t4[1]:
                ax6.plot(self.xs, np.transpose(self.ddfrs, axes=(1, 2, 0))[i][j])

        # General options for the graphs
        Titres = ['$g_k$', '$g\'_k$', '$g\'\'_k$', '$R_{{nl}}$', '$R\'_{{nl}}$', '$R\'\'_{{nl}}$']
        for i, ax in enumerate([ax1, ax2, ax3, ax4, ax5, ax6]):
            ax.set_xlim(self.xmin, self.xmax)
            ax.axhline(y=0, color='black')
            ax.axvline(x=0, color='black')
            # ax.text(self.Rc,0,'$R_c$',fontsize=18)
            ax.set_title(Titres[i], fontsize=22)
            if self.ymin:
                ax.set_ylim(ymin=self.ymin)
            if self.ymax:
                ax.set_ylim(ymax=self.ymax)
