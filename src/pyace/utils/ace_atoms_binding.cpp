#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include <vector>
#include <string>
#include <cmath>

#include "ace-evaluator/ace_types.h"
#include "ace-evaluator/ace_arraynd.h"
#include "ace-evaluator/ace_utils.h"
#include "extra/ace_atoms.h"

#include <pybind11/stl_bind.h>

PYBIND11_MAKE_OPAQUE(std::map<string, string>)

namespace py = pybind11;
using namespace std;
const DOUBLE_TYPE eps = 1e-15;

/*
We could add a pure python version of this class to use it alongside
ase and other tools
*/

string ACEAtomicEnvironment__repr__(ACEAtomicEnvironment &ae) {
    stringstream s;
    s << "ACEAtomicEnvironment(n_atoms_real=" << ae.n_atoms_real << ", "\
 << "n_atoms_extended=" << ae.n_atoms_extended << ""\
 << ")";
    return s.str();
}

pybind11::tuple ACEAtomicEnvironment__getstate__(const ACEAtomicEnvironment &ae) {
    return py::make_tuple(ae.get_x(), ae.get_species_types(), ae.get_neighbour_list(), ae.get_origins());
}

ACEAtomicEnvironment ACEAtomicEnvironment__setstate__(py::tuple t) {
    if (t.size() != 4)
        throw std::runtime_error("Invalid state of ACEAtomicEnvironment-tuple");
    ACEAtomicEnvironment ae;
    auto new_x = t[0].cast<vector<vector<DOUBLE_TYPE>>>();
    auto new_species_types = t[1].cast<vector<SPECIES_TYPE>>();
    auto new_neighbour_list = t[2].cast<vector<vector<int>>>();
    auto new_origins = t[3].cast<vector<int>>();

    ae.set_x(new_x);
    ae.set_species_types(new_species_types);
    ae.set_neighbour_list(new_neighbour_list);
    ae.set_origins(new_origins);

    return ae;
}

DOUBLE_TYPE
get_minimal_nn_distance_tp(py::array_t<DOUBLE_TYPE, py::array::c_style | py::array::forcecast> _positions, //2D[nat][3]
                           py::array_t<DOUBLE_TYPE, py::array::c_style | py::array::forcecast> _cell, //2D[3][3]
                           py::array_t<int, py::array::c_style | py::array::forcecast> _ind_i, // 1D[nbonds]
                           py::array_t<int, py::array::c_style | py::array::forcecast> _ind_j, // 1D[nbonds]
                           py::array_t<DOUBLE_TYPE, py::array::c_style | py::array::forcecast> _offsets // 2D[nbonds][3]
) {
    auto buf_positions = _positions.request();
    if (buf_positions.ndim != 2)
        throw std::runtime_error("Shape of `_positions` should be [n_at][3]");

    auto buf_cell = _cell.request();
    if (buf_cell.ndim != 2 || buf_cell.shape[0] != 3 || buf_cell.shape[1] != 3)
        throw std::runtime_error("Shape of `_cell` should be [3][3]");

    auto buf_int_i = _ind_i.request();
    if (buf_int_i.ndim != 1)
        throw std::runtime_error("Shape of `_int_i` should be [n_bonds]");

    auto buf_int_j = _ind_j.request();
    if (buf_int_j.ndim != 1)
        throw std::runtime_error("Shape of `_int_j` should be [n_bonds]");

    if (buf_int_i.shape[0] != buf_int_j.shape[0])
        throw std::runtime_error("Shape of `_int_i` and `_int_j` should be equal");
    size_t n_bonds = buf_int_i.shape[0];

    auto buf_offsets = _offsets.request();
    if (buf_offsets.ndim != 2 || buf_offsets.shape[0] != n_bonds || buf_offsets.shape[1] != 3)
        throw std::runtime_error("Shape of `_offsets` should be [nbonds][3]");

    DOUBLE_TYPE nn_min_distance = 1e3;
    int i, j;

    for (size_t t = 0; t < n_bonds; t++) {

        i = *_ind_i.data(t);
        j = *_ind_j.data(t);

        const double *off = _offsets.data(t);
        const double *pos_i = _positions.data(i);
        const double *pos_j = _positions.data(j);

        vector<DOUBLE_TYPE> drvec_rel(3, 0.), drvec(3, 0);
        for (int a = 0; a < 3; a++)
            drvec_rel[a] = pos_j[a] + off[a] - pos_i[a];

        for (int alpha = 0; alpha < 3; alpha++)
            for (int lat_vec_ind = 0; lat_vec_ind < 3; lat_vec_ind++) {
                drvec[alpha] += drvec_rel[lat_vec_ind] * (*_cell.data(lat_vec_ind, alpha));
            }
        DOUBLE_TYPE r = sqrt(sqr(drvec[0]) + sqr(drvec[1]) + sqr(drvec[2]));
        if (r < nn_min_distance)
            nn_min_distance = r;
    }
    return nn_min_distance;
}


vector<DOUBLE_TYPE> north(vector<DOUBLE_TYPE> &v1, vector<DOUBLE_TYPE> &v2) {
    vector<DOUBLE_TYPE> vn(3, 0);
    vn[0] = v1[1] * v2[2] - v1[2] * v2[1];
    vn[1] = v1[2] * v2[0] - v1[0] * v2[2];
    vn[2] = v1[0] * v2[1] - v1[1] * v2[0];
    DOUBLE_TYPE norm = sqrt(sqr(vn[0]) + sqr(vn[1]) + sqr(vn[2]));
    vn[0] = vn[0] / norm;
    vn[1] = vn[1] / norm;
    vn[2] = vn[2] / norm;
    return vn;
}

template<typename T1, typename T2>
inline DOUBLE_TYPE dot(const vector<T1> &v1, const vector<T2> &v2) {
    return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2];
}

template<typename T1, typename T2>
vector<DOUBLE_TYPE> dot(const vector<T1> &v1, const vector<vector<T2>> &v2) {
    // v1[1][k] -> v1[k]
    // n = 1;
    size_t k = v1.size();
    size_t k2 = v2.size();
    size_t m = v2.at(0).size();

    if (k != k2)
        throw std::invalid_argument("1D/2D arrays must be of the size (n,k) and (m,k)");

    vector<DOUBLE_TYPE> v(m, 0);

    for (size_t j = 0; j < m; j++) {
        DOUBLE_TYPE s = 0;
        for (size_t t = 0; t < k; t++)
            s += v1.at(t) * v2.at(t).at(j);
        v.at(j) = s;
    }
    return v;
}

template<typename T1, typename T2>
vector<vector<DOUBLE_TYPE>> dot(vector<vector<T1>> &v1, vector<vector<T2>> &v2) {
    // v1[n][k]
    // v2[k][m]
    // v = dot(v1, v2) [n][m]

    size_t n = v1.size();
    size_t k = v1.at(0).size();
    size_t k2 = v2.size();
    size_t m = v2.at(0).size();
    if (k != k2)
        throw std::invalid_argument("2D arrays must be of the size (n,k) and (m,k)");

    vector<vector<DOUBLE_TYPE>> v(n, vector<DOUBLE_TYPE>(m, 0));

    for (size_t i = 0; i < n; i++)
        for (size_t j = 0; j < m; j++) {
            DOUBLE_TYPE s = 0;
            for (size_t t = 0; t < k; t++)
                s += (DOUBLE_TYPE) v1.at(i).at(t) * v2.at(t).at(j);
            v.at(i).at(j) = s;
        }

    return v;
}

template<typename T1, typename T2>
inline vector<DOUBLE_TYPE> operator+(const vector<T1> &v1, const vector<T2> &v2) {
    size_t n1 = v1.size();
    size_t n2 = v2.size();
    if (n1 != n2)
        throw std::runtime_error("operator-: vector sizes are different");

    vector<DOUBLE_TYPE> v(n1, 0);
    for (size_t t = 0; t < n1; t++)
        v.at(t) = v1.at(t) + v2.at(t);
    return v;
}

template<typename T1, typename T2>
inline vector<DOUBLE_TYPE> operator-(const vector<T1> &v1, const vector<T2> &v2) {
    size_t n1 = v1.size();
    size_t n2 = v2.size();
    if (n1 != n2)
        throw std::runtime_error("operator-: vector sizes are different");

    vector<DOUBLE_TYPE> v(n1, 0);
    for (size_t t = 0; t < n1; t++)
        v.at(t) = v1.at(t) - v2.at(t);
    return v;
}

template<typename T1, typename T2>
inline vector<vector<DOUBLE_TYPE>> operator+(const vector<vector<T1>> &v1, const vector<T2> &v2) {
    size_t n1 = v1.size();
    size_t m1 = v1.at(0).size();
    size_t m2 = v2.size();
    if (m1 != m2)
        throw std::runtime_error("operator-: vector sizes are different");

    vector<vector<DOUBLE_TYPE>> v(n1, vector<DOUBLE_TYPE>(m1, 0));
    for (size_t t = 0; t < n1; t++)
        v.at(t) = v1.at(t) + v2;
    return v;
}

template<typename T1, typename T2>
inline vector<vector<DOUBLE_TYPE>> operator-(const vector<vector<T1>> &v1, const vector<T2> &v2) {
    size_t n1 = v1.size();
    size_t m1 = v1.at(0).size();
    size_t m2 = v2.size();
    if (m1 != m2)
        throw std::runtime_error("operator-: vector sizes are different");

    vector<vector<DOUBLE_TYPE>> v(n1, vector<DOUBLE_TYPE>(m1, 0));
    for (size_t t = 0; t < n1; t++)
        v.at(t) = v1.at(t) - v2;
    return v;
}

vector<vector<DOUBLE_TYPE>> inv3x3(vector<vector<DOUBLE_TYPE>> &a) {
    DOUBLE_TYPE det = a[0][0] * (a[2][2] * a[1][1] - a[2][1] * a[1][2])
                      - a[1][0] * (a[2][2] * a[0][1] - a[2][1] * a[0][2])
                      + a[2][0] * (a[1][2] * a[0][1] - a[1][1] * a[0][2]);
    if (fabs(det) < 1e-9)
        throw std::overflow_error("Couldn't invert matrix - determinant is almost zero");

    vector<vector<DOUBLE_TYPE>> ainv = {{0, 0, 0},
                                        {0, 0, 0},
                                        {0, 0, 0}};

    ainv[0][0] = (a[2][2] * a[1][1] - a[2][1] * a[1][2]) / det;
    ainv[0][1] = -(a[2][2] * a[0][1] - a[2][1] * a[0][2]) / det;
    ainv[0][2] = (a[1][2] * a[0][1] - a[1][1] * a[0][2]) / det;

    ainv[1][0] = -(a[2][2] * a[1][0] - a[2][0] * a[1][2]) / det;
    ainv[1][1] = (a[2][2] * a[0][0] - a[2][0] * a[0][2]) / det;
    ainv[1][2] = -(a[1][2] * a[0][0] - a[1][0] * a[0][2]) / det;

    ainv[2][0] = (a[2][1] * a[1][0] - a[2][0] * a[1][1]) / det;
    ainv[2][1] = -(a[2][1] * a[0][0] - a[2][0] * a[0][1]) / det;
    ainv[2][2] = (a[1][1] * a[0][0] - a[1][0] * a[0][1]) / det;

    // check the correctness of inversion
    auto unit_matrix = dot(a, ainv);
    if (abs(unit_matrix[0][0] - 1) > 1e-14 ||
        abs(unit_matrix[1][1] - 1) > 1e-14 ||
        abs(unit_matrix[2][2] - 1) > 1e-14 ||
        abs(unit_matrix[0][1]) > 1e-14 ||
        abs(unit_matrix[0][2]) > 1e-14 ||
        abs(unit_matrix[1][0]) > 1e-14 ||
        abs(unit_matrix[1][2]) > 1e-14 ||
        abs(unit_matrix[2][0]) > 1e-14 ||
        abs(unit_matrix[2][1]) > 1e-14
            )
        throw std::runtime_error("inv3x3: Matrix inversion is wrong");
    return ainv;
};

struct NeighbourCluster {
    int n_atoms_real; // number of real atoms
    vector<DOUBLE_TYPE> min_r, max_r;
    vector<vector<DOUBLE_TYPE>> positions;
    vector<vector<DOUBLE_TYPE>> scaled_positions;
    vector<SPECIES_TYPE> species_types;
    vector<int> origins;
    vector<vector<int>> nshifts;
};

NeighbourCluster
build_cluster_pbc(py::array_t<DOUBLE_TYPE, py::array::c_style | py::array::forcecast> positions_, //2D[nat][3]
                  py::array_t<int, py::array::c_style | py::array::forcecast> species_type_, //1D[nat])
                  py::array_t<DOUBLE_TYPE, py::array::c_style | py::array::forcecast> cell_, //2D[3][3]
                  DOUBLE_TYPE rcut
) {
    vector<DOUBLE_TYPE> v1(3), v2(3), v3(3), v(3);
    DOUBLE_TYPE x;


    auto buf_cell = cell_.request();
    if (buf_cell.ndim != 2 || buf_cell.shape[0] != 3 || buf_cell.shape[1] != 3)
        throw std::runtime_error("Shape of `_cell` should be [3][3]");

    auto buf_positions = positions_.request();
    if (buf_positions.ndim != 2 || buf_positions.shape[1] != 3)
        throw std::runtime_error("Shape of `posin` should be [n_at][3]");
    int natoms = buf_positions.shape[0];

    auto buf_species_type = species_type_.request();
    if (buf_species_type.ndim != 1 || buf_species_type.shape[0] != natoms)
        throw std::runtime_error("Shape of `species_type_` should be [n_at]");

    vector<vector<DOUBLE_TYPE>> cell = {{0, 0, 0},
                                        {0, 0, 0},
                                        {0, 0, 0}};
    for (size_t i = 0; i < 3; i++)
        for (size_t j = 0; j < 3; j++) {
            cell[i][j] = *cell_.data(i, j);
        }

    for (size_t t = 0; t < 3; t++) {
        v1[t] = cell[0][t];
        v2[t] = cell[1][t];
        v3[t] = cell[2][t];
    }
    //# 1. calculate size of cluster by adding a layer of thickness
    //#    rcut parallel to the faces of the unit cell
    // v1/v2
    v = north(v1, v2);
    x = dot(v3, v);
    if (x < 1e-3)
        throw std::runtime_error("Extreme shape of unit cell");
    DOUBLE_TYPE dxhc3 = 1 + rcut / x;
    DOUBLE_TYPE dxlc3 = -dxhc3 + 1.0;
    int hc3 = int(ceil(dxhc3));
    int lc3 = int(floor(dxlc3));

    // v3/v1
    v = north(v3, v1);
    x = dot(v2, v);
    if (x < 1e-3)
        throw std::runtime_error("Extreme shape of unit cell");
    DOUBLE_TYPE dxhc2 = 1 + rcut / x;
    DOUBLE_TYPE dxlc2 = -dxhc2 + 1.0;
    int hc2 = int(ceil(dxhc2));
    int lc2 = int(floor(dxlc2));

    // v2/v3
    v = north(v2, v3);
    x = dot(v1, v);
    if (x < 1e-3)
        throw std::runtime_error("Extreme shape of unit cell");
    DOUBLE_TYPE dxhc1 = 1 + rcut / x;
    DOUBLE_TYPE dxlc1 = -dxhc1 + 1.0;
    int hc1 = int(ceil(dxhc1));
    int lc1 = int(floor(dxlc1));

    //# 2. evaluate size of orthorombic box that contains unit cell
    //#    plus distance rcutmax around unit cell
    for (int n = 0; n < 3; n++) {
        DOUBLE_TYPE rout = 0.;
        for (int n1 = 0; n1 <= 1; n1++)
            for (int n2 = 0; n2 <= 1; n2++)
                for (int n3 = 0; n3 <= 1; n3++) {
                    x = rcut + n1 * cell[0][n] + n2 * cell[1][n] + n3 * cell[2][n];
                    if (x > rout)
                        rout = x;
                }
        v1[n] = rout;
    }

    //# now in negative direction
    for (int n = 0; n < 3; n++) {
        DOUBLE_TYPE rout = 0.;
        for (int n1 = 0; n1 <= 1; n1++)
            for (int n2 = 0; n2 <= 1; n2++)
                for (int n3 = 0; n3 <= 1; n3++) {
                    x = -rcut + n1 * cell[0][n] + n2 * cell[1][n] + n3 * cell[2][n];
                    if (x < rout)
                        rout = x;
                }
        v2[n] = rout;
    }

    auto dxhcc1 = v1[0];
    auto dxhcc2 = v1[1];
    auto dxhcc3 = v1[2];

    auto dxlcc1 = v2[0];
    auto dxlcc2 = v2[1];
    auto dxlcc3 = v2[2];
    auto cellinv = inv3x3(cell);

    vector<vector<DOUBLE_TYPE>> scaledatompos(natoms, vector<DOUBLE_TYPE>(3, 0));
    vector<DOUBLE_TYPE> r(3, 0);
    vector<DOUBLE_TYPE> rscaled(3, 0);
    for (int n = 0; n < natoms; n++) {
        r[0] = *positions_.data(n, 0);
        r[1] = *positions_.data(n, 1);
        r[2] = *positions_.data(n, 2);

        rscaled = dot(r, cellinv);


        for (int k = 0; k < 3; k++) {
            DOUBLE_TYPE t = rscaled.at(k);
            if (t < -eps || t > (1 + eps)) {
                t = fmod(t, 1.0); //(-1,1)
                if (t < -eps) t += 1; // TODO: increase cutoff by uncertainty
                if (t < -eps || t > 1 + eps) {
                    stringstream ss;
                    ss << "scaled_pos0 " << rscaled.at(k) << " is still outside [0,1): " << t;
                    throw std::runtime_error(ss.str());
                }
            }
            rscaled.at(k) = t;
        }
        scaledatompos.at(n) = rscaled;
    }

    //?#! build cluster including periodic images
    NeighbourCluster neigh_cluster;

    vector<DOUBLE_TYPE> r1(3, 0), r2(3, 0);

    //inbounds of the cluster, initialized with extreme large/small values
    neigh_cluster.min_r.resize(3, 1e9);
    neigh_cluster.max_r.resize(3, -1e9);
    neigh_cluster.scaled_positions = scaledatompos;
    for (int n = 0; n < natoms; n++) {
        r1 = scaledatompos.at(n);

        r2[0] = cell[0][0] * r1[0] + cell[1][0] * r1[1] + cell[2][0] * r1[2];
        r2[1] = cell[0][1] * r1[0] + cell[1][1] * r1[1] + cell[2][1] * r1[2];
        r2[2] = cell[0][2] * r1[0] + cell[1][2] * r1[1] + cell[2][2] * r1[2];

        neigh_cluster.positions.push_back(r2);
        neigh_cluster.origins.push_back(n);
        neigh_cluster.species_types.push_back(*species_type_.data(n));
        neigh_cluster.nshifts.push_back({0, 0, 0});

        //update inbounds
        for (int a = 0; a < 3; a++) {
            if (neigh_cluster.min_r[a] > r2[a]) neigh_cluster.min_r[a] = r2[a];
            if (neigh_cluster.max_r[a] < r2[a]) neigh_cluster.max_r[a] = r2[a];
        }
    }

    for (int k1 = lc1; k1 <= hc1; k1++)
        for (int k2 = lc2; k2 <= hc2; k2++)
            for (int k3 = lc3; k3 <= hc3; k3++) {
                if (k1 == 0 && k2 == 0 && k3 == 0) continue;

                for (int n = 0; n < natoms; n++) {
                    r1 = scaledatompos[n];//scaled
                    r1[0] += k1;
                    r1[1] += k2;
                    r1[2] += k3;

                    if (r1[0] > dxlc1 && r1[0] <= dxhc1 &&
                        r1[1] > dxlc2 && r1[1] <= dxhc2 &&
                        r1[2] > dxlc3 && r1[2] <= dxhc3) {

                        r2[0] = cell[0][0] * r1[0] + cell[1][0] * r1[1] + cell[2][0] * r1[2];
                        r2[1] = cell[0][1] * r1[0] + cell[1][1] * r1[1] + cell[2][1] * r1[2];
                        r2[2] = cell[0][2] * r1[0] + cell[1][2] * r1[1] + cell[2][2] * r1[2];

                        if (r2[0] > dxlcc1 && r2[0] <= dxhcc1 &&
                            r2[1] > dxlcc2 && r2[1] <= dxhcc2 &&
                            r2[2] > dxlcc3 && r2[2] <= dxhcc3) {
                            neigh_cluster.positions.push_back(r2);
                            neigh_cluster.origins.push_back(n);
                            neigh_cluster.species_types.push_back(*species_type_.data(n));
                            neigh_cluster.nshifts.push_back({k1, k2, k3});

                            //update inbounds
                            for (int a = 0; a < 3; a++) {
                                if (neigh_cluster.min_r[a] > r2[a]) neigh_cluster.min_r[a] = r2[a];
                                if (neigh_cluster.max_r[a] < r2[a]) neigh_cluster.max_r[a] = r2[a];
                            }
                        }
                    }
                }
            }
    // now we have:
    // 1. double postmp[n_extended][3]
    // 2. int origintmp[n_extended]
    // 3. int nshifttmp[n_extended][3]
    neigh_cluster.n_atoms_real = natoms;
    return neigh_cluster;
}

NeighbourCluster
build_cluster_nonpbc(py::array_t<DOUBLE_TYPE, py::array::c_style | py::array::forcecast> positions_, //2D[nat][3]
                     py::array_t<int, py::array::c_style | py::array::forcecast> species_type_ //1D[nat])
) {

    vector<DOUBLE_TYPE> v1(3);


    auto buf_positions = positions_.request();
    if (buf_positions.ndim != 2 || buf_positions.shape[1] != 3)
        throw std::runtime_error("Shape of `posin` should be [n_at][3]");
    int natom = buf_positions.shape[0];

    auto buf_species_type = species_type_.request();
    if (buf_species_type.ndim != 1 || buf_species_type.shape[0] != natom)
        throw std::runtime_error("Shape of `species_type_` should be [n_at]");


    //?#! build cluster including periodic images
    NeighbourCluster neigh_cluster;
    //inbounds of the cluster, initialized with extreme large/small values
    neigh_cluster.min_r.resize(3, 1e9);
    neigh_cluster.max_r.resize(3, -1e9);

    for (int n = 0; n < natom; n++) {
        v1[0] = *positions_.data(n, 0);
        v1[1] = *positions_.data(n, 1);
        v1[2] = *positions_.data(n, 2);

        neigh_cluster.positions.push_back(v1);
        neigh_cluster.origins.push_back(n);
        neigh_cluster.species_types.push_back(*species_type_.data(n));
        neigh_cluster.nshifts.push_back({0, 0, 0});

        //update inbounds
        for (int a = 0; a < 3; a++) {
            if (neigh_cluster.min_r[a] > v1[a]) neigh_cluster.min_r[a] = v1[a];
            if (neigh_cluster.max_r[a] < v1[a]) neigh_cluster.max_r[a] = v1[a];
        }
    }

    // now we have:
    // 1. double postmp[n_extended][3]
    // 2. int origintmp[n_extended]
    // 3. int nshifttmp[n_extended][3]
    neigh_cluster.n_atoms_real = natom;
    return neigh_cluster;
}

inline DOUBLE_TYPE distance(vector<DOUBLE_TYPE> &r1, vector<DOUBLE_TYPE> &r2) {
    return sqrt(sqr(r1[0] - r2[0]) + sqr(r1[1] - r2[1]) + sqr(r1[2] - r2[2]));
}


ACEAtomicEnvironment build_atomic_env_from_cluster_nogrid(NeighbourCluster &cluster, DOUBLE_TYPE r_cut) {
    DOUBLE_TYPE eps = 1e-14;
    // first n_atoms_real of cluster.positions/species_types/origins are for REAL atoms
    // remaining atoms - from ghost images
    int n_atoms_real = cluster.n_atoms_real;

    vector<vector<DOUBLE_TYPE>> x;
    vector<SPECIES_TYPE> species_type;
    vector<int> origins;
    vector<vector<int>> neighbour_list;
    vector<int> num_neighbours;

    // put all real atoms
    for (int i = 0; i < n_atoms_real; i++) {
        auto &r_i = cluster.positions[i];
        x.push_back(r_i);
        species_type.push_back(cluster.species_types[i]);
        origins.push_back(cluster.origins[i]);
    }

    for (int i = 0; i < n_atoms_real; i++) {
        auto &r_i = cluster.positions[i];
        vector<int> current_nl;

        for (int j = 0; j < cluster.positions.size(); j++) {

            if (i == j) continue;
            auto &r_j = cluster.positions[j];
            DOUBLE_TYPE r_ij = distance(r_j, r_i);
            int jj;
            if (r_ij < r_cut) {
                // if r_j in x
                auto it = std::find(x.begin(), x.end(), r_j);
                if (it != x.end()) {
                    /* x contains r_j */
                    jj = it - x.begin();
                } else {
                    /* x does not contain r_j =>  add */
                    x.push_back(r_j);
                    species_type.push_back(cluster.species_types[j]);
                    origins.push_back(cluster.origins[j]);
                    jj = x.size() - 1; // index of newly added element
                }
                current_nl.push_back(jj);
            }
        }

        neighbour_list.emplace_back(current_nl);
    }

    ACEAtomicEnvironment ae;
    ae.n_atoms_real = n_atoms_real;
    ae.set_x(x);
    ae.set_species_types(species_type);
    ae.set_origins(origins);
    ae.set_neighbour_list(neighbour_list);

    return ae;
}

ACEAtomicEnvironment build_atomic_env_from_cluster(NeighbourCluster &cluster, DOUBLE_TYPE r_cut) {

    // first n_atoms_real of cluster.positions/species_types/origins are for REAL atoms
    // remaining atoms - from ghost images
    int n_atoms_real = cluster.n_atoms_real;


    //construct grid
    vector<int> grid_size(3, 0);
    for (int a = 0; a < 3; a++) {
        grid_size[a] = ceil((cluster.max_r[a] - cluster.min_r[a]) / r_cut);
        if (grid_size[a] == 0)
            grid_size[a] = 1;
    }
    //[gx][gy][gz] -> vector[n_atom_in_grid_box] of indices clusters.position in this grid box
//    vector<vector<vector<vector<int>>>> indices_grid(grid_size[0], vector<vector<vector<int>>>(
//            grid_size[1], vector<vector<int>>(
//                    grid_size[2])));
    Array3D<vector<int>> indices_grid(grid_size[0], grid_size[1], grid_size[2]);
    vector<int> cur_ind(3, 0);
    for (int j = 0; j < cluster.positions.size(); j++) {
        auto &rj = cluster.positions[j];
        // identify the grid box index
        for (int a = 0; a < 3; a++) {
            cur_ind[a] = floor((rj[a] - cluster.min_r[a]) / r_cut);
            if (cur_ind[a] >= grid_size[a])
                cur_ind[a] = grid_size[a] - 1;
        }
        //add atom index to this grid box
        indices_grid(cur_ind[0], cur_ind[1], cur_ind[2]).push_back(j);
    }

    vector<vector<DOUBLE_TYPE>> x;
    vector<SPECIES_TYPE> species_type;
    vector<int> origins;
    vector<vector<int>> neighbour_list;
    vector<int> num_neighbours;

    // put all real atoms
    for (int i = 0; i < n_atoms_real; i++) {
        auto &r_i = cluster.positions[i];
        x.push_back(r_i);
        species_type.push_back(cluster.species_types[i]);
        origins.push_back(cluster.origins[i]);
    }

    for (int i = 0; i < n_atoms_real; i++) {
        auto &r_i = cluster.positions[i];

        // identify the grid box index
        for (int a = 0; a < 3; a++) {
            cur_ind[a] = floor((r_i[a] - cluster.min_r[a]) / r_cut);
            if (cur_ind[a] >= grid_size[a])
                cur_ind[a] = grid_size[a] - 1;
        }
        vector<int> current_nl;

        vector<int> lo_ind(3), hi_ind(3);
        for (int a = 0; a < 3; a++) {
            lo_ind[a] = cur_ind[a] - 1;
            if (lo_ind[a] < 0)lo_ind[a] = 0;

            hi_ind[a] = cur_ind[a] + 1;
            if (hi_ind[a] >= grid_size[a]) hi_ind[a] = grid_size[a] - 1;
        }

        for (int ix = lo_ind[0]; ix <= hi_ind[0]; ix++)
            for (int iy = lo_ind[1]; iy <= hi_ind[1]; iy++)
                for (int iz = lo_ind[2]; iz <= hi_ind[2]; iz++) {
                    for (auto j:  indices_grid(ix, iy, iz)) {
                        if (i == j) continue;
                        auto &r_j = cluster.positions[j];
                        DOUBLE_TYPE r_ij = distance(r_j, r_i);
                        int jj;
                        if (r_ij < r_cut) {
                            // if r_j in x
                            auto it = std::find(x.begin(), x.end(), r_j);
                            if (it != x.end()) {
                                /* x contains r_j */
                                jj = it - x.begin();
                            } else {
                                /* x does not contain r_j =>  add */
                                x.push_back(r_j);
                                species_type.push_back(cluster.species_types[j]);
                                origins.push_back(cluster.origins[j]);
                                jj = x.size() - 1; // index of newly added element
                            }
                            current_nl.push_back(jj);
                        }
                    }
                }
        neighbour_list.emplace_back(current_nl);
    }

    ACEAtomicEnvironment ae;
    ae.n_atoms_real = n_atoms_real;
    ae.set_x(x);
    ae.set_species_types(species_type);
    ae.set_origins(origins);
    ae.set_neighbour_list(neighbour_list);

    return ae;
}

ACEAtomicEnvironment
build_atomic_env(py::array_t<DOUBLE_TYPE, py::array::c_style | py::array::forcecast> positions_, //2D[nat][3]
                 py::array_t<DOUBLE_TYPE, py::array::c_style | py::array::forcecast> cell_, //2D[3][3]
                 py::array_t<int, py::array::c_style | py::array::forcecast> species_type_, //1D[nat])
                 bool pbc,
                 DOUBLE_TYPE rcut) {
    NeighbourCluster cluster;
    if (pbc) {
        cluster = build_cluster_pbc(positions_, species_type_, cell_, rcut);
    } else {
        cluster = build_cluster_nonpbc(positions_, species_type_);
    }

//    ACEAtomicEnvironment ae = build_atomic_env_from_cluster(cluster, rcut);
    ACEAtomicEnvironment ae = build_atomic_env_from_cluster_nogrid(cluster, rcut);
    return ae;
}

template<typename T>
vector<size_t> sort_indexes(const vector<T> &v) {

    // initialize original index locations
    vector<size_t> idx(v.size());
    iota(idx.begin(), idx.end(), 0);

    // sort indexes based on comparing values in v
    // using std::stable_sort instead of std::sort
    // to avoid unnecessary index re-orderings
    // when v contains elements of equal values
    stable_sort(idx.begin(), idx.end(),
                [&v](size_t i1, size_t i2) { return v[i1] < v[i2]; });

    return idx;
}

py::tuple
get_nghbrs_tp_atoms(py::array_t<DOUBLE_TYPE, py::array::c_style | py::array::forcecast> positions_, //2D[nat][3]
                    py::array_t<DOUBLE_TYPE, py::array::c_style | py::array::forcecast> cell_, //2D[3][3]
                    py::array_t<int, py::array::c_style | py::array::forcecast> species_type_, //1D[nat])
                    bool pbc,
                    DOUBLE_TYPE rcut) {
    NeighbourCluster cluster;
    if (pbc) {
        cluster = build_cluster_pbc(positions_, species_type_, cell_, rcut);
    } else {
        //cluster = build_cluster_nonpbc(positions_, species_type_);
        throw std::invalid_argument("Could not generate tp_atoms for non-pbc cell");
    }

    // first n_atoms_real of cluster.positions/species_types/origins are for REAL atoms
    // remaining atoms - from ghost images
    int n_atoms_real = cluster.n_atoms_real;
    //    ind_i [n_bonds]
    //    ind_j [n_bonds]
    //    mu_i [n_bonds]
    //    mu_j [n_bonds]
    //    offsets [n_bonds][3]
    vector<int> _ind_i, _ind_j, _mu_i, _mu_j;
    vector<vector<int>> _offsets;
    for (int i = 0; i < n_atoms_real; i++) {
        auto &r_i = cluster.positions[i];

        vector<int> cur_ind_j, cur_mu_j;
        vector<vector<int>> cur_offsets;

        bool neighbour_found = false;

        for (int j = 0; j < cluster.positions.size(); j++) {
            if (i == j) continue;
            auto &r_j = cluster.positions[j];
            DOUBLE_TYPE r_ij = distance(r_j, r_i);

            if (r_ij < rcut) {
                neighbour_found = true;

                _ind_i.push_back(i);
                _mu_i.push_back(cluster.species_types[i]);

                cur_ind_j.push_back(cluster.origins[j]);
                cur_mu_j.push_back(cluster.species_types[j]);
                cur_offsets.push_back(cluster.nshifts[j]);
            }
        }

        if (!neighbour_found) {
            //return all false
            return py::make_tuple(false, false, false, false, false, false, false);
        }
        // sort by cur_ind_j
        for (auto j: sort_indexes(cur_ind_j)) {
            _ind_j.push_back(cur_ind_j.at(j));
            _mu_j.push_back(cur_mu_j.at(j));
            _offsets.push_back(cur_offsets.at(j));
        }

    }

    size_t nbonds = _ind_i.size();
    vector<size_t> shape{nbonds};
    vector<size_t> shape2D{nbonds, 3};
    py::array_t<int> _ind_i_np(shape), _ind_j_np(shape),
            _mu_i_np(shape), _mu_j_np(shape);

    py::array_t<double> _offsets_np(shape2D);

    for (size_t t = 0; t < nbonds; t++) {
        _ind_i_np.mutable_at(t) = _ind_i.at(t);
        _ind_j_np.mutable_at(t) = _ind_j.at(t);
        _mu_i_np.mutable_at(t) = _mu_i.at(t);
        _mu_j_np.mutable_at(t) = _mu_j.at(t);

        for (int k = 0; k < 3; k++) {
            _offsets_np.mutable_at(t, k) = _offsets.at(t).at(k);
        }
    }

    py::array_t<double> _scaled_positions_np({n_atoms_real, 3});
    for (size_t t = 0; t < n_atoms_real; t++)
        for (int k = 0; k < 3; k++)
            _scaled_positions_np.mutable_at(t, k) = cluster.scaled_positions.at(t).at(k);

    return py::make_tuple(_ind_i_np, _ind_j_np, _mu_i_np, _mu_j_np, _offsets_np, true, _scaled_positions_np);
}

PYBIND11_MODULE(catomicenvironment, m) {
    py::options options;
    options.disable_function_signatures();

    py::class_<ACEAtomicEnvironment>(m, "ACEAtomicEnvironment", R"mydelimiter(

    Atomic environment class

    Attributes
    ----------
    n_atoms_real
    n_atoms_extended
    x
    species_type
    neighbour_list
    )mydelimiter")
            .def(py::init())
            .def(py::init<int>())
            .def_readwrite("n_atoms_real", &ACEAtomicEnvironment::n_atoms_real)
            .def_readwrite("n_atoms_extended", &ACEAtomicEnvironment::n_atoms_extended)
            .def_property("x", &ACEAtomicEnvironment::get_x, &ACEAtomicEnvironment::set_x)
            .def_property("species_type", &ACEAtomicEnvironment::get_species_types,
                          &ACEAtomicEnvironment::set_species_types)
            .def_property("neighbour_list", &ACEAtomicEnvironment::get_neighbour_list,
                          &ACEAtomicEnvironment::set_neighbour_list)
            .def_property("origins", &ACEAtomicEnvironment::get_origins, &ACEAtomicEnvironment::set_origins)
            .def("__repr__", &ACEAtomicEnvironment__repr__)
            .def("load_full", &ACEAtomicEnvironment::load_full)
            .def("save_full", &ACEAtomicEnvironment::save_full)
            .def("get_minimal_nn_distance", &ACEAtomicEnvironment::get_minimal_nn_distance)
            .def(py::pickle(&ACEAtomicEnvironment__getstate__, &ACEAtomicEnvironment__setstate__));

    m.def("get_minimal_nn_distance_tp", &get_minimal_nn_distance_tp);
    m.def("build_atomic_env", &build_atomic_env);
    m.def("get_nghbrs_tp_atoms", &get_nghbrs_tp_atoms);


#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}
