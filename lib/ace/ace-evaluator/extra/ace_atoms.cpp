//
// Created by Yury Lysogorskiy on 31.01.20.
//
#include "ace_atoms.h"
#include "ace_arraynd.h"
#include "ace_types.h"


void ACEAtomicEnvironment::set_x(vector<vector<DOUBLE_TYPE>> &new_x) {
    _clean_x();
    n_atoms_extended = new_x.size();
    x = new DOUBLE_TYPE *[n_atoms_extended];
    for (int i = 0; i < n_atoms_extended; i++) {
        x[i] = new DOUBLE_TYPE[3];
        for (int j = 0; j < 3; j++)
            x[i][j] = new_x.at(i).at(j);
    }
}

vector<vector<DOUBLE_TYPE>> ACEAtomicEnvironment::get_x() const {
    vector<vector<DOUBLE_TYPE>> ret_x(n_atoms_extended);
    for (int i = 0; i < n_atoms_extended; i++) {
        ret_x.at(i).resize(3);
        for (int j = 0; j < 3; j++)
            ret_x[i][j] = x[i][j];
    }
    return ret_x;
}

void ACEAtomicEnvironment::set_species_types(vector<SPECIES_TYPE> &new_species_types) {
    _clean_species_types();
    n_atoms_extended = new_species_types.size();
    species_type = new SPECIES_TYPE[n_atoms_extended];
    for (int i = 0; i < n_atoms_extended; i++) {
        species_type[i] = new_species_types[i];
    }
}

vector<SPECIES_TYPE> ACEAtomicEnvironment::get_species_types() const {
    vector<SPECIES_TYPE> ret_species_types(n_atoms_extended);
    for (int i = 0; i < n_atoms_extended; i++) {
        ret_species_types[i] = species_type[i];
    }
    return ret_species_types;
}

void ACEAtomicEnvironment::set_neighbour_list(vector<vector<int>> &new_neighbour_list) {
    _clean_neighbour_list();
    n_atoms_real = new_neighbour_list.size();
    num_neighbours = new int[n_atoms_real];
    neighbour_list = new int *[n_atoms_real];

    for (int i = 0; i < n_atoms_real; i++) {
        int num_neigh = new_neighbour_list.at(i).size();
        num_neighbours[i] = num_neigh;
        neighbour_list[i] = new int[num_neigh];
        for (int j = 0; j < num_neigh; j++)
            neighbour_list[i][j] = new_neighbour_list.at(i).at(j);
    }
}

vector<vector<int>> ACEAtomicEnvironment::get_neighbour_list() const {
    vector<vector<int>> ret_neighbour_list(n_atoms_real);
    for (int i = 0; i < n_atoms_real; i++) {
        int num_neigh = num_neighbours[i];
        ret_neighbour_list.at(i).resize(num_neigh);

        for (int j = 0; j < num_neigh; j++)
            ret_neighbour_list.at(i).at(j) = neighbour_list[i][j];
    }

    return ret_neighbour_list;
}

DOUBLE_TYPE ACEAtomicEnvironment::get_minimal_nn_distance() const {
    DOUBLE_TYPE nn_min_distance=1e3;
    for (int i = 0; i<this->n_atoms_real; i++) {
        auto r_i = this->x[i];
        auto num_neighbours = this->num_neighbours[i];
        auto cur_neighbour_list = this->neighbour_list[i];
        for(int j=0;j<num_neighbours;j++) {
            auto r_j = this->x[cur_neighbour_list[j]];
            DOUBLE_TYPE r = sqrt(sqr(r_j[0]-r_i[0]) + sqr(r_j[1]-r_i[1]) + sqr(r_j[2]-r_i[2]));
            if (r<nn_min_distance)
                nn_min_distance = r;
        }
    }
    return nn_min_distance;
}


/**
 * Read the structure from the file. File format is:
 *
 * n_atoms=2
 * # type x y z
 * 0 0.0 0.0 0.0
 * 1 1.0 2.0 3.0
 * ...
 *
 *
 * @param filename
 */
void ACEAtomicEnvironment::load(const string &filename) {
    FILE *fin = fopen(filename.c_str(), "rt");
    if (fin == NULL)
        throw invalid_argument("Could not open file " + filename);
    this->_load(fin, filename);
    fclose(fin);
}

void ACEAtomicEnvironment::_load(FILE *fin) {
    this->_load(fin, "<unknown>");
}


void ACEAtomicEnvironment::_load(FILE *fin, const string &filename) {
    int res;
    char buffer[1024]={0};

    // read natoms
    int n_atoms_real = 0;
    res = fscanf(fin, "%*s = %s\n", buffer);
    if (res != 1) {
        fclose(fin);
        throw runtime_error("Couldn't read n_atoms from file " + filename);
    }
    n_atoms_real = stoi(buffer);
//    printf("n_atoms_real = %d\n", n_atoms_real);
    res = fscanf(fin, "%[^\n]\n", buffer);
//    printf("res=%d\n", res);
//    printf("buffer=%s\n", buffer);
    if (res != 1) {
        fclose(fin);
        throw runtime_error("Couldn't read separation line '# type x y z' from file " + filename);
    }
    SPECIES_TYPE type;
    DOUBLE_TYPE coords[3];
    this->n_atoms_real = n_atoms_real;
    this->n_atoms_extended = n_atoms_real;
    this->species_type = new SPECIES_TYPE[this->n_atoms_extended];
    this->x = new DOUBLE_TYPE *[this->n_atoms_extended];
    for (int i = 0; i < this->n_atoms_extended; i++) {
        this->x[i] = new DOUBLE_TYPE[3];
    }

    for (int i = 0; i < n_atoms_real; i++) {
        res = fscanf(fin, "%d %lf %lf %lf\n", &type, &coords[0], &coords[1], &coords[2]);
        if (res != 4) {
            fclose(fin);
            throw runtime_error("Couldn't read 'species_type x y z' data from file " + filename);
        }
//        printf("%d: %f %f %f\n", type, coords[0], coords[1], coords[2]);
        species_type[i] = type;
        x[i][0] = coords[0];
        x[i][1] = coords[1];
        x[i][2] = coords[2];
    }
    compute_neighbour_list();
}

void ACEAtomicEnvironment::save(const string &filename) {
    FILE *fout = fopen(filename.c_str(), "wt");
    if (fout == NULL)
        throw invalid_argument("Could not open file " + filename);
    fprintf(fout, "n_atoms = %d\n", n_atoms_real);
    fprintf(fout, "# type x y z\n");

    for (int i = 0; i < n_atoms_real; i++) {
        fprintf(fout, "%d %lf %lf %lf\n", species_type[i], x[i][0], x[i][1], x[i][2]);
    }
    fclose(fout);
}

void ACEAtomicEnvironment::load_full(const string &filename) {
    _clean();
    FILE *fin = fopen(filename.c_str(), "rt");
    if (fin == NULL)
        throw invalid_argument("Could not open file " + filename);

    int res;
    char buffer[1024] = {0};
    SPECIES_TYPE type;
    DOUBLE_TYPE coords[3];

    // read natoms
    res = fscanf(fin, "%*s = %s\n", buffer);
    if (res != 1) {
        fclose(fin);
        throw runtime_error("Couldn't read n_atoms_real from file " + filename);
    }
    n_atoms_real = stoi(buffer);

    // read natoms
    res = fscanf(fin, "%*s = %s\n", buffer);
    if (res != 1) {
        fclose(fin);
        throw runtime_error("Couldn't read n_atoms_extended from file " + filename);
    }
    n_atoms_extended = stoi(buffer);

    //# type x y z
    res = fscanf(fin, "%[^\n]\n", buffer);
    if (res != 1) {
        fclose(fin);
        throw runtime_error("Couldn't read separation line '# type x y z' from file " + filename);
    }

    species_type = new SPECIES_TYPE[n_atoms_extended];
    x = new DOUBLE_TYPE *[n_atoms_extended];
    for (int i = 0; i < n_atoms_extended; i++) {
        x[i] = new DOUBLE_TYPE[3];
    }

    for (int i = 0; i < n_atoms_extended; i++) {
        res = fscanf(fin, "%d %lf %lf %lf\n", &type, &coords[0], &coords[1], &coords[2]);
        if (res != 4) {
            fclose(fin);
            throw runtime_error("Couldn't read 'species_type x y z' data from file " + filename);
        }
        species_type[i] = type;
        x[i][0] = coords[0];
        x[i][1] = coords[1];
        x[i][2] = coords[2];
    }

    //# neighbour list
    res = fscanf(fin, "%[^\n]\n", buffer);
    if (res != 1) {
        fclose(fin);
        throw runtime_error("Couldn't read separation line '# neighbour list' from file " + filename);
    }

    int curr_num_neigh = 0, jj;
    num_neighbours = new int[n_atoms_real];
    neighbour_list = new int *[n_atoms_real];

    for (int i = 0; i < n_atoms_real; i++) {
        res = fscanf(fin, " %d", &curr_num_neigh);
        if (res != 1) {
            fclose(fin);
            throw runtime_error("Couldn't read 'species_type x y z' data from file " + filename);
        }
        num_neighbours[i] = curr_num_neigh;
        neighbour_list[i] = new int[curr_num_neigh];
        for (int j = 0; j < curr_num_neigh; j++) {
            res = fscanf(fin, " %d", &jj);
            if (res != 1) {
                fclose(fin);
                throw runtime_error("Couldn't read 'species_type x y z' data from file " + filename);
            }
            neighbour_list[i][j] = jj;
        }

    }

    // # origin list
    res = fscanf(fin, " %[^\n]\n", buffer);
    if (res == 1) {
        origins = new int[n_atoms_extended];
        for (int i = 0; i < n_atoms_extended; i++) {
            res = fscanf(fin, " %d", &jj);
            if (res != 1) {
                fclose(fin);
                throw runtime_error("Couldn't read origins data from file " + filename);
            }
            origins[i] = jj;
        }

    }

    fclose(fin);
}

void ACEAtomicEnvironment::save_full(const string &filename) {
    FILE *fout = fopen(filename.c_str(), "wt");
    if (fout == NULL)
        throw invalid_argument("Could not open file " + filename);
    fprintf(fout, "n_atoms_real = %d\n", n_atoms_real);
    fprintf(fout, "n_atoms_extended = %d\n", n_atoms_extended);
    fprintf(fout, "# type x y z\n");

    for (int i = 0; i < n_atoms_extended; i++) {
        fprintf(fout, "%d %lf %lf %lf\n", species_type[i], x[i][0], x[i][1], x[i][2]);
    }

    fprintf(fout, "# neighbour list\n");
    for (int i = 0; i < n_atoms_real; i++) {
        fprintf(fout, "%d ", num_neighbours[i]);
        for (int j = 0; j < num_neighbours[i]; j++)
            fprintf(fout, "%d ", neighbour_list[i][j]);
        fprintf(fout, "\n");
    }

    if (origins != nullptr) {
        fprintf(fout, "# origin list\n");
        for (int i = 0; i < n_atoms_extended; i++) {
            fprintf(fout, "%d ", origins[i]);
        }
    }
    fclose(fout);
}

void ACEAtomicEnvironment::set_origins(vector<int> &new_origins) {
    if (new_origins.size() != n_atoms_extended)
        throw invalid_argument("Length of origins is inconsistent with n_atoms_extended");
    if (origins != nullptr)
        delete[] origins;
    origins = new int[n_atoms_extended];
    for (int i = 0; i < n_atoms_extended; i++)
        origins[i] = new_origins[i];
}

vector<int> ACEAtomicEnvironment::get_origins() const {
    vector<int> res;
    if (origins != nullptr) {
        res.resize(n_atoms_extended);
        for (int i = 0; i < n_atoms_extended; i++)
            res[i] = origins[i];
    }
    return res;
}

ACEAtomicEnvironment create_cube(const DOUBLE_TYPE dr, const DOUBLE_TYPE cube_side_length) {
    int n_atoms = 0;
    for (DOUBLE_TYPE x = -cube_side_length / 2; x <= cube_side_length / 2 + dr / 2; x += dr)
        for (DOUBLE_TYPE y = -cube_side_length / 2; y <= cube_side_length / 2 + dr / 2; y += dr)
            for (DOUBLE_TYPE z = -cube_side_length / 2; z <= cube_side_length / 2 + dr / 2; z += dr)
                n_atoms++;

    ACEAtomicEnvironment a(n_atoms);
    int i = 0;
    for (DOUBLE_TYPE x = -cube_side_length / 2; x <= cube_side_length / 2 + dr / 2; x += dr)
        for (DOUBLE_TYPE y = -cube_side_length / 2; y <= cube_side_length / 2 + dr / 2; y += dr)
            for (DOUBLE_TYPE z = -cube_side_length / 2; z <= cube_side_length / 2 + dr / 2; z += dr) {
                a.x[i][0] = x;
                a.x[i][1] = y;
                a.x[i][2] = z;
                i++;
            }

    a.compute_neighbour_list();

    return a;

}

ACEAtomicEnvironment create_linear_chain(const int n, const int axis, double scale_factor) {
    ACEAtomicEnvironment a(n);
    for (int i = 0; i < a.n_atoms_real; i++) {
        //a.x[i] = new DOUBLE_TYPE[3];
        a.x[i][0] = 0;
        a.x[i][1] = 0;
        a.x[i][2] = 0;
        a.x[i][axis] = scale_factor * i - (double) n / 2.;
    }

    a.compute_neighbour_list();

    return a;
}

ACEAtomicEnvironment
create_supercell(ACEAtomicEnvironment &simple_cell, DOUBLE_TYPE lx, DOUBLE_TYPE ly, DOUBLE_TYPE lz, int nx, int ny,
                 int nz) {
    int number_of_cells = nx * ny * nz;
    ACEAtomicEnvironment a(simple_cell.n_atoms_real * number_of_cells);

    int at_i = 0;
    for (int ix = 0; ix < nx; ix++)
        for (int iy = 0; iy < ny; iy++)
            for (int iz = 0; iz < nz; iz++) {
                for (int simple_at_i = 0; simple_at_i < simple_cell.n_atoms_real; simple_at_i++) {
                    a.x[at_i][0] = simple_cell.x[simple_at_i][0] + lx * ix;
                    a.x[at_i][1] = simple_cell.x[simple_at_i][1] + ly * iy;
                    a.x[at_i][2] = simple_cell.x[simple_at_i][2] + lz * iz;
                    at_i++;
                }
            }
    a.compute_neighbour_list();
    return a;
}

ACEAtomicEnvironment create_bcc(const DOUBLE_TYPE lat) {
    ACEAtomicEnvironment a(9);

    a.x[0][0] = -lat / 2.;
    a.x[0][1] = -lat / 2.;
    a.x[0][2] = -lat / 2.;


    a.x[1][0] = lat / 2.;
    a.x[1][1] = -lat / 2.;
    a.x[1][2] = -lat / 2.;

    a.x[2][0] = -lat / 2.;
    a.x[2][1] = lat / 2.;
    a.x[2][2] = -lat / 2.;

    a.x[3][0] = lat / 2.;
    a.x[3][1] = lat / 2.;
    a.x[3][2] = -lat / 2.;


    a.x[4][0] = -lat / 2.;
    a.x[4][1] = -lat / 2.;
    a.x[4][2] = lat / 2.;

    a.x[5][0] = lat / 2.;
    a.x[5][1] = -lat / 2.;
    a.x[5][2] = lat / 2.;

    a.x[6][0] = -lat / 2.;
    a.x[6][1] = lat / 2.;
    a.x[6][2] = lat / 2.;

    a.x[7][0] = lat / 2.;
    a.x[7][1] = lat / 2.;
    a.x[7][2] = lat / 2.;

    a.x[8][0] = 0;
    a.x[8][1] = 0;
    a.x[8][2] = 0;

    a.compute_neighbour_list();

    return a;
}

typedef Array2D<DOUBLE_TYPE> Matrix;

Matrix rotation_matrix(DOUBLE_TYPE theta, DOUBLE_TYPE theta1, DOUBLE_TYPE theta2) {
    DOUBLE_TYPE Rx[3][3] = {{1, 0,          0},
                            {0, cos(theta), -sin(theta)},
                            {0, sin(theta), cos(theta)},
    };

    DOUBLE_TYPE Ry[3][3] = {{cos(theta1),  0, sin(theta1)},
                            {0,            1, 0},
                            {-sin(theta1), 0, cos(theta1)}
    };

    DOUBLE_TYPE Rz[3][3] = {{cos(theta2), -sin(theta2), 0},
                            {sin(theta2), cos(theta2),  0},
                            {0,           0,            1}
    };

    DOUBLE_TYPE R[3][3] = {0};
    int i, j, k;
    for (i = 0; i < 3; i++)
        for (j = 0; j < 3; j++) {
            for (k = 0; k < 3; k++)
                R[i][j] += Rx[i][k] * Ry[k][j];
        }

    DOUBLE_TYPE R2[3][3] = {0};
    for (i = 0; i < 3; i++)
        for (j = 0; j < 3; j++) {
            for (k = 0; k < 3; k++)
                R2[i][j] += R[i][k] * Rz[k][j];
        }

    Matrix res(3, 3);
    for (i = 0; i < 3; i++) {
        for (j = 0; j < 3; j++) {
            res(i, j) = R2[i][j];
        }
    }

    return res;
}


void rotate_structure(ACEAtomicEnvironment &env, Matrix &rotation_matrix) {
    int nat, i, j, k;

    for (nat = 0; nat < env.n_atoms_real; nat++) {
        DOUBLE_TYPE r[3] = {0};
        for (i = 0; i < 3; i++) {
            for (j = 0; j < 3; j++) {
                r[i] += rotation_matrix(i, j) * env.x[nat][j];
            }
        }
        for (i = 0; i < 3; i++)
            env.x[nat][i] = r[i];
    }
}
