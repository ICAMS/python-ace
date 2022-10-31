/*
 * Performant implementation of atomic cluster expansion and interface to LAMMPS
 *
 * Copyright 2021  (c) Yury Lysogorskiy^1, Cas van der Oord^2, Anton Bochkarev^1,
 * Sarath Menon^1, Matteo Rinaldi^1, Thomas Hammerschmidt^1, Matous Mrovec^1,
 * Aidan Thompson^3, Gabor Csanyi^2, Christoph Ortner^4, Ralf Drautz^1
 *
 * ^1: Ruhr-University Bochum, Bochum, Germany
 * ^2: University of Cambridge, Cambridge, United Kingdom
 * ^3: Sandia National Laboratories, Albuquerque, New Mexico, USA
 * ^4: University of British Columbia, Vancouver, BC, Canada
 *
 *
 * See the LICENSE file.
 * This FILENAME is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.

 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */


// Created by Lysogorskiy Yury on 28.04.2020.

#ifndef ACE_EVALUATOR_ACE_ABSTRACT_BASIS_H
#define ACE_EVALUATOR_ACE_ABSTRACT_BASIS_H

#include <vector>
#include <string>
#include <map>
#include <tuple>

#include "ace-evaluator/ace_c_basisfunction.h"
#include "ace-evaluator/ace_contigous_array.h"
#include "ace-evaluator/ace_radial.h"
#include "ace-evaluator/ace_spherical_cart.h"
#include "ace-evaluator/ace_types.h"

using namespace std;

struct ACEEmbeddingSpecification {
    DENSITY_TYPE ndensity;
    vector<DOUBLE_TYPE> FS_parameters; ///< parameters for cluster functional, see Eq.(3) in implementation notes or Eq.(53) in <A HREF="https://journals.aps.org/prb/abstract/10.1103/PhysRevB.99.014104">  PRB 99, 014104 (2019) </A>
    string npoti = "FinnisSinclair"; ///< FS and embedding function combination

    DOUBLE_TYPE rho_core_cutoff;
    DOUBLE_TYPE drho_core_cutoff;

    //TODO: update method
    string to_string() const {
        stringstream ss;
        ss << "ACEEmbeddingSpecification(npoti=" << npoti << ", ndensity=" << ndensity << ", ";
        ss << "FS_parameter=(";
        for (const auto &p: FS_parameters)
            ss << p << ",";
        ss << "))";
        return ss.str();
    }

    YAML_PACE::Node to_YAML() const {
        YAML_PACE::Node emb_yaml;
        emb_yaml.SetStyle(YAML_PACE::EmitterStyle::Flow);
        emb_yaml["ndensity"] = (int) this->ndensity;
        emb_yaml["FS_parameters"] = this->FS_parameters;

        emb_yaml["npoti"] = this->npoti;
        emb_yaml["rho_core_cutoff"] = this->rho_core_cutoff;
        emb_yaml["drho_core_cutoff"] = this->drho_core_cutoff;

        return emb_yaml;
    }
};

struct ACEBondSpecification {
    NS_TYPE nradmax;
    LS_TYPE lmax;
    NS_TYPE nradbasemax;

    string radbasename;

    std::vector<DOUBLE_TYPE> radparameters; // i.e. lambda
    vector<vector<vector<DOUBLE_TYPE>>> radcoefficients;///< crad_nlk order: [n=0..nradmax-1][l=0..lmax][k=0..nradbase-1]

    //hard-core repulsion
    DOUBLE_TYPE prehc;
    DOUBLE_TYPE lambdahc;

    DOUBLE_TYPE rcut;
    DOUBLE_TYPE dcut;

    //inner cutoff
    DOUBLE_TYPE rcut_in = 0;
    DOUBLE_TYPE dcut_in = 0;
    string inner_cutoff_type = "distance"; // new behaviour is default

    bool operator==(const ACEBondSpecification &another) const {
        return (nradbasemax == another.nradbasemax) && (lmax == another.lmax) &&
               (nradbasemax == another.nradbasemax) && (radbasename == another.radbasename) &&
               (radparameters == another.radparameters) && (radcoefficients == another.radcoefficients) &&
               (prehc == another.prehc) && (lambdahc == another.lambdahc) && (rcut == another.rcut) &&
               (dcut == another.dcut) && (rcut_in == another.rcut_in) && (dcut_in == another.dcut_in) &&
               (inner_cutoff_type == another.inner_cutoff_type);
    }

    bool operator!=(const ACEBondSpecification &another) const {
        return !(*this == (another));
    }

    string to_string() const {
        stringstream ss;
        ss << "ACEBondSpecification(nradmax=" << nradmax << ", lmax=" << lmax << ", nradbasemax=" << nradbasemax
           << ", radbasename=" <<
           radbasename << ", crad=(" << radcoefficients.at(0).at(0).at(0) << "...), ";
        ss << "rcut=" << rcut << ", dcut=" << dcut;
        ss << ", rcut_in=" << rcut_in << ", dcut_in=" << dcut_in;
        ss << ", inner_cutoff_type=" << inner_cutoff_type;
        if (prehc > 0)
            ss << ", core-rep: [prehc=" << prehc << ", lambdahc=" << lambdahc << "]";
        ss << ")";
        return ss.str();
    }

    void from_YAML(YAML_PACE::Node bond_yaml) {
        radbasename = bond_yaml["radbasename"].as<string>();

        if(radbasename=="ACE.jl.base") {

        } else {
            nradmax = bond_yaml["nradmax"].as<NS_TYPE>();
            lmax = bond_yaml["lmax"].as<LS_TYPE>();
            nradbasemax = bond_yaml["nradbasemax"].as<NS_TYPE>();
            radparameters = bond_yaml["radparameters"].as<vector<DOUBLE_TYPE>>();
            radcoefficients = bond_yaml["radcoefficients"].as<vector<vector<vector<DOUBLE_TYPE>>>>();
            prehc = bond_yaml["prehc"].as<DOUBLE_TYPE>();
            lambdahc = bond_yaml["lambdahc"].as<DOUBLE_TYPE>();
            rcut = bond_yaml["rcut"].as<DOUBLE_TYPE>();
            dcut = bond_yaml["dcut"].as<DOUBLE_TYPE>();

            if (bond_yaml["rcut_in"]) rcut_in = bond_yaml["rcut_in"].as<DOUBLE_TYPE>();
            if (bond_yaml["dcut_in"]) dcut_in = bond_yaml["dcut_in"].as<DOUBLE_TYPE>();
            if (bond_yaml["inner_cutoff_type"])
                inner_cutoff_type = bond_yaml["inner_cutoff_type"].as<string>();
            else
                inner_cutoff_type = "density"; // default value to read for backward compatibility
        }
    }

    YAML_PACE::Node to_YAML() const {
        YAML_PACE::Node bond_yaml;
        bond_yaml.SetStyle(YAML_PACE::EmitterStyle::Flow);
        bond_yaml["nradmax"] = (int) this->nradmax;
        bond_yaml["lmax"] = (int) this->lmax;
        bond_yaml["nradbasemax"] = (int) this->nradbasemax;

        bond_yaml["radbasename"] = this->radbasename;
        bond_yaml["radparameters"] = this->radparameters;
        bond_yaml["radcoefficients"] = this->radcoefficients;

        bond_yaml["prehc"] = this->prehc;
        bond_yaml["lambdahc"] = this->lambdahc;
        bond_yaml["rcut"] = this->rcut;
        bond_yaml["dcut"] = this->dcut;

        bond_yaml["rcut_in"] = this->rcut_in;
        bond_yaml["dcut_in"] = this->dcut_in;
        bond_yaml["inner_cutoff_type"] = this->inner_cutoff_type;

        return bond_yaml;
    }
};


/**
 * Abstract basis set class
 */
class ACEAbstractBasisSet {
public:
    SPECIES_TYPE nelements = 0;        ///< number of elements in basis set
    RANK_TYPE rankmax = 0;             ///< maximum value of rank
    DENSITY_TYPE ndensitymax = 0;      ///< maximum number of densities \f$ \rho^{(p)} \f$
    NS_TYPE nradbase = 0; ///< maximum number of radial \f$\textbf{basis}\f$ function \f$ g_{k}(r) \f$
    LS_TYPE lmax = 0;  ///< \f$ l_\textrm{max} \f$ - maximum value of orbital moment \f$ l \f$
    NS_TYPE nradmax = 0;  ///< maximum number \f$ n \f$ of radial function \f$ R_{nl}(r) \f$
    DOUBLE_TYPE cutoffmax = 0;  ///< maximum value of cutoff distance among all species in basis set
    DOUBLE_TYPE deltaSplineBins = 0;  ///< Spline interpolation density

    string *elements_name = nullptr; ///< Array of elements name for mapping from index (0..nelements-1) to element symbol (string)
    map<string, SPECIES_TYPE> elements_to_index_map;


    AbstractRadialBasis *radial_functions = nullptr; ///< object to work with radial functions
    ACECartesianSphericalHarmonics spherical_harmonics; ///< object to work with spherical harmonics in Cartesian representation

    bool is_sort_functions = true; ///< flag to specify, whether to sort the basis functions or preserve the order
    //for multispecies

    map<SPECIES_TYPE, ACEEmbeddingSpecification> map_embedding_specifications;
    map<pair<SPECIES_TYPE, SPECIES_TYPE>, ACEBondSpecification> map_bond_specifications;

    // E0 values
    Array1D<DOUBLE_TYPE> E0vals;

    /**
     * Default empty constructor
     */
    ACEAbstractBasisSet() = default;

    // copy constructor, operator= and destructor (see. Rule of Three)

    /**
     * Copy constructor (see. Rule of Three)
     * @param other
     */
    ACEAbstractBasisSet(const ACEAbstractBasisSet &other);

    /**
     * operator=  (see. Rule of Three)
     * @param other
     * @return
     */
    ACEAbstractBasisSet &operator=(const ACEAbstractBasisSet &other);

    /**
     * virtual destructor (see. Rule of Three)
     */
    virtual ~ACEAbstractBasisSet();

    /**
     * Computing cluster functional \f$ F(\rho_i^{(1)}, \dots, \rho_i^{(P)})  \f$
     * and its derivatives  \f$ (\partial F/\partial\rho_i^{(1)}, \dots, \partial F/\partial \rho_i^{(P)} ) \f$
     * @param rhos array with densities \f$ \rho^{(p)} \f$
     * @param value (out) return value of cluster functional
     * @param derivatives (out) array of derivatives  \f$ (\partial F/\partial\rho_i^{(1)}, \dots, \partial F/\partial \rho_i^{(P)} )  \f$
     * @param ndensity  number \f$ P \f$ of densities to use
     */
    void FS_values_and_derivatives(Array1D<DOUBLE_TYPE> &rhos, DOUBLE_TYPE &value, Array1D<DOUBLE_TYPE> &derivatives,
                                   SPECIES_TYPE mu_i);

    /**
     * Computing hard core pairwise repulsive potential \f$ f_{cut}(\rho_i^{(\textrm{core})})\f$ and its derivative,
     * see Eq.(29) of implementation notes
     * @param rho_core value of \f$ \rho_i^{(\textrm{core})} \f$
     * @param rho_cut  \f$ \rho_{cut}^{\mu_i} \f$ value
     * @param drho_cut \f$ \Delta_{cut}^{\mu_i} \f$ value
     * @param fcut (out) return inner cutoff function
     * @param dfcut (out) return derivative of inner cutoff function
     */
    static void inner_cutoff(DOUBLE_TYPE rho_core, DOUBLE_TYPE rho_cut, DOUBLE_TYPE drho_cut, DOUBLE_TYPE &fcut,
                             DOUBLE_TYPE &dfcut);


    /**
     * Virtual method to save potential to file
     * @param filename file name
     */
    virtual void save(const string &filename) = 0;

    /**
     * Virtual method to load potential from file
     * @param filename file name
     */
    virtual void load(const string filename) = 0;

    /**
     * Get the species index by its element name
     * @param elemname element name
     * @return species index
     */
    SPECIES_TYPE get_species_index_by_name(const string &elemname);


    // routines for copying and cleaning dynamic memory of the class (see. Rule of Three)

    /**
     * Routine for clean the dynamically allocated memory\n
     * IMPORTANT! It must be idempotent for safety.
     */
    virtual void _clean();

    /**
     * Copy dynamic memory from src. Must be override and extended in derived classes!
     * @param src source object to copy from
     */
    void _copy_dynamic_memory(const ACEAbstractBasisSet &src);

    /**
     * Copy scalar values from src. Must be override and extended in derived classes!
     * @param src source object to copy from
     */
    void _copy_scalar_memory(const ACEAbstractBasisSet &src);

    virtual vector<DOUBLE_TYPE> get_all_coeffs() const = 0;

    virtual vector<vector<SPECIES_TYPE>> get_all_coeffs_mask() const = 0;

    virtual void set_all_coeffs(const vector<DOUBLE_TYPE> &coeffs) = 0;
};

void Fexp(DOUBLE_TYPE rho, DOUBLE_TYPE mexp, DOUBLE_TYPE &F, DOUBLE_TYPE &DF);

void FexpShiftedScaled(DOUBLE_TYPE rho, DOUBLE_TYPE mexp, DOUBLE_TYPE &F, DOUBLE_TYPE &DF);

bool compare(const ACEAbstractBasisFunction &b1, const ACEAbstractBasisFunction &b2);

#endif //ACE_EVALUATOR_ACE_ABSTRACT_BASIS_H

