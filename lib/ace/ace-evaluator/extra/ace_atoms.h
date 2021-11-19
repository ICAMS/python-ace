//
// Created by Yury Lysogorskiy on 31.01.20.
//

#ifndef ACE_ATOMS_H
#define ACE_ATOMS_H

#include <cmath>
#include <vector>
#include "ace_types.h"
#include "ace_arraynd.h"

#define sqr(x) (x)*(x)

using namespace std;

struct ACEAtomicEnvironment {
    int n_atoms_real; // number of real atoms
    int n_atoms_extended; // number of extended atoms (incl. periodic images)

    DOUBLE_TYPE **x = nullptr; // of shape (n_atoms_extended,3)
    SPECIES_TYPE *species_type = nullptr; // of shape (n_atoms_extended,)
    int *origins = nullptr; // of shape (n_atoms_extended,)

    int *num_neighbours = nullptr; // of shape (n_atoms_real,)
    int **neighbour_list = nullptr; // of shape (n_atoms_real,...)

    ACEAtomicEnvironment() = default;

    explicit ACEAtomicEnvironment(int n_atoms) {
        n_atoms_real = n_atoms;
        n_atoms_extended = n_atoms;
        x = new DOUBLE_TYPE *[n_atoms_extended];
        for (int i = 0; i < n_atoms_extended; i++) {
            x[i] = new DOUBLE_TYPE[3];
        }

        species_type = new SPECIES_TYPE[n_atoms_extended];
        for (int i = 0; i < n_atoms_extended; i++) {
            species_type[i] = 0;
        }
    }

    void load_full(const string &filename);

    void save_full(const string &filename);

    void load(const string &filename);

    void save(const string &filename);

    void _load(FILE *fin);

    void _load(FILE *fin, const string &filename);

    void compute_neighbour_list(DOUBLE_TYPE cutoff = 100.) {
        _clean_neighbour_list();

        num_neighbours = new int[n_atoms_real];
        neighbour_list = new int *[n_atoms_real];

        for (int i = 0; i < n_atoms_real; i++) {
            int num_neigh = 0;
            for (int j = 0; j < n_atoms_extended; j++) {
                if (i != j) {
                    if (sqrt(sqr(x[i][0] - x[j][0]) + sqr(x[i][1] - x[j][1]) + sqr(x[i][2] - x[j][2])) <= cutoff)
                        num_neigh++;
                }
            }

            num_neighbours[i] = num_neigh;
            neighbour_list[i] = new int[num_neigh];

            num_neigh = 0;
            for (int j = 0; j < n_atoms_extended; j++) {
                if (i != j) {
                    if (sqrt(sqr(x[i][0] - x[j][0]) + sqr(x[i][1] - x[j][1]) + sqr(x[i][2] - x[j][2])) <= cutoff) {
                        neighbour_list[i][num_neigh] = j;
                        num_neigh++;
                    }
                }
            }
        }

    }

    void _clean_x() {
        if (x != nullptr)
            for (int i = 0; i < n_atoms_extended; i++) {
                delete[] x[i];
                x[i] = nullptr;
            }
        delete[] x;
        x = nullptr;
    }

    void _clean_neighbour_list() {
        if (neighbour_list != nullptr)
            for (int i = 0; i < n_atoms_real; i++) {
                delete[] neighbour_list[i];
            }
        delete[] neighbour_list;
        neighbour_list = nullptr;

        delete[] num_neighbours;
        num_neighbours = nullptr;
    }

    void _clean_species_types(){
        delete[] species_type;
        species_type = nullptr;
    }

    void _clean() {
        _clean_x();
        _clean_neighbour_list();
        _clean_species_types();

        delete[] origins;
        origins = nullptr;
    }

    void _copy_from(const ACEAtomicEnvironment &other) {
        n_atoms_real = other.n_atoms_real;
        n_atoms_extended = other.n_atoms_extended;

        x = new DOUBLE_TYPE *[n_atoms_extended];
        species_type = new SPECIES_TYPE[n_atoms_extended];
        for (int i = 0; i < n_atoms_extended; i++) {
            x[i] = new DOUBLE_TYPE[3];
            x[i][0] = other.x[i][0];
            x[i][1] = other.x[i][1];
            x[i][2] = other.x[i][2];
            species_type[i] = other.species_type[i];
        }

        if (other.origins != nullptr) {
            origins = new int[n_atoms_extended];
            for (int i = 0; i < n_atoms_extended; i++)
                origins[i] = other.origins[i];
        }

        neighbour_list = new int *[n_atoms_real];
        num_neighbours = new int[n_atoms_real];
        for (int i = 0; i < n_atoms_real; i++) {
            num_neighbours[i] = other.num_neighbours[i];
            neighbour_list[i] = new int[num_neighbours[i]];
            for (int j = 0; j < num_neighbours[i]; j++) {
                neighbour_list[i][j] = other.neighbour_list[i][j];
            }
        }
    }

    ACEAtomicEnvironment(const ACEAtomicEnvironment &other) {
        _copy_from(other);
    }

    ACEAtomicEnvironment &operator=(const ACEAtomicEnvironment &other) {
        if (&other != this) {
            _clean();
            _copy_from(other);
        }
        return *this;
    }

    ~ACEAtomicEnvironment() {
        _clean();
    }

    void set_x(vector<vector<DOUBLE_TYPE>> &new_x);

    vector<vector<DOUBLE_TYPE>> get_x() const;

    void set_species_types(vector<SPECIES_TYPE> &new_species_types);

    vector<SPECIES_TYPE> get_species_types() const;

    void set_origins(vector<int> &new_origins);

    vector<int> get_origins() const;

    void set_neighbour_list(vector<vector<int>> &new_neighbour_list);

    vector<vector<int>> get_neighbour_list() const;

    DOUBLE_TYPE get_minimal_nn_distance() const;
};

ACEAtomicEnvironment create_linear_chain(int n, int axis = 2, double scale_factor = 1.);

ACEAtomicEnvironment create_cube(const DOUBLE_TYPE dr, const DOUBLE_TYPE cube_side_length);

ACEAtomicEnvironment create_bcc(const DOUBLE_TYPE lat);

ACEAtomicEnvironment
create_supercell(ACEAtomicEnvironment &simple_cell, DOUBLE_TYPE lx, DOUBLE_TYPE ly, DOUBLE_TYPE lz, int nx, int ny,
                 int nz);

typedef Array2D<DOUBLE_TYPE> Matrix;

Matrix rotation_matrix(DOUBLE_TYPE theta, DOUBLE_TYPE theta1, DOUBLE_TYPE theta2);

void rotate_structure(ACEAtomicEnvironment &env, Matrix &rotation_matrix);

#endif //ACE_ATOMS_H
