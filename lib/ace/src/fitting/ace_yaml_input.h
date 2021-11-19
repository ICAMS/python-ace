/** \file ace_yaml_input.h
C++ file which would contain the code for reading in an input file
*/
#ifndef ACE_YAML_INPUT_H
#define ACE_YAML_INPUT_H

#include <fstream>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "ace_types.h"
#include "ace_b_basis.h"

using namespace std;

template<typename K, typename V>
bool is_key_in_map(const K &name, const map<K, V> m) {
    return m.find(name) != m.end();
}

template<class T>
set<T> unique(vector<T> vector) {
    set<T> _set;
    for (auto val: vector) {
        if (_set.count(val) == 0)
            _set.insert(val);
    }
    return _set;
};

template<class T>
int num_of_unique(vector<T> vector) {
    return unique(vector).size();
};

inline bool if_file_exist(const std::string &name) {
    ifstream f(name.c_str());
    return f.good();
}

/**
A structure which contains the global input values
For now it is only DeltaSplineBins, maybe if required we can add
more members later.
*/
struct GlobalPotentialParameters {
public:
    DOUBLE_TYPE DeltaSplineBins;
    SPECIES_TYPE nelements;
    LS_TYPE lmax;
    NS_TYPE nradmax;
    NS_TYPE nradbase;
    int nblocks;
    RANK_TYPE rankmax;
    vector<string> element_names;
    map<string, int> elements_to_number_map;

    DENSITY_TYPE ndensitymax;
    DOUBLE_TYPE cutoffmax;

    map<string, string> metadata;

    //auxiliary data
    AuxiliaryData auxdata;
};


class Input {
    vector<string> split_key(string);

    void read_radcoefficients(const YAML_PACE::Node &YAML_input_species_block,
                              BBasisFunctionsSpecificationBlock &b_basisfunc_spec_block) const;

    void read_core_rho_drho_cut(const YAML_PACE::Node &YAML_input_species_block,
                                BBasisFunctionsSpecificationBlock &b_basisfunc_spec_block) const;

    void read_core_repulsion(const YAML_PACE::Node &YAML_input_species_block,
                             BBasisFunctionsSpecificationBlock &b_basisfunc_spec_block) const;

    void copy_radial_basis_from_to_block(const BBasisFunctionsSpecificationBlock &from_spec_block,
                                         BBasisFunctionsSpecificationBlock &to_spec_block) const;

    void check_radial_basis_consistency(const BBasisFunctionsSpecificationBlock &block1,
                                        const BBasisFunctionsSpecificationBlock &block2) const;

    void check_core_repulsion_consistency(const BBasisFunctionsSpecificationBlock &block1,
                                          const BBasisFunctionsSpecificationBlock &block2) const;

    void copy_core_repulsion_from_to_block(const BBasisFunctionsSpecificationBlock &from_spec_block,
                                           BBasisFunctionsSpecificationBlock &to_spec_block) const;

public:
    string inputfile;
    GlobalPotentialParameters global;
    unsigned short int number_of_species_block;

    vector<BBasisFunctionsSpecificationBlock> bbasis_func_spec_blocks_vector; // [2-species-1, 2-species-2, 2-species-3, ..] , [3-species-1,.. ]

    Input() = default;

    ~Input() = default;

    void parse_input(const string &ff);


};


/** \class Input "src/read_input.h"

Input class is the high-level class to parse input information from a yaml file. The usage of this class is as
illustrated below. Two samples of the yaml input file format are provided in `examples/` folder, and are named
`input_example1.yaml` and `input_example2.yaml`. Yaml files can be checked for syntax issues [here](http://www.yamllint.com/).

Parsing the input file should be carried out as follows. The first step is adding the header file by `#include "read_input.h"`.
Once this is done, the input can be parsed by-

\code
Input input = Input();
input.parse_input(filename);
\endcode

where `filename` is the name of the input yaml file. For example, for an input file as shown below-
\code
global:
  DeltaSplineBins: 0.001
species:
  - speciesblock: Al
    nradmaxi: 3
    lmaxi: 3
    ndensityi: 2
    npoti: FinnisSinclair
    parameters: [1.0, 2.0, 7.0, 9.0]
    rcutij: 7.5
    dcutij: 0.1
    NameOfCutoffFunctionij: cos
    radbase: ChebExpCos
    radparameters: [7.0]
    nradbaseij: 6
    radcoefficients:
      [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
       0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
       0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
       0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
       0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
       0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ]
   density:
   - nd:   1
     nbody:
       - {type: Al Al, nr: [1], nl: [0], c: 1.0}
       - {type: Al Al Al, nr: [1,1], nl: [1, 1], c: 1.0}
       - {type: Al Al Al Al, nr: [2,2,2], nl: [1, 1, 2], c: 1.0, lint: [2]}
       - {type: Al Al Al Al Al, nr: [1, 1, 1, 1], nl: [1, 1, 1, 1], c: 1.0, lint: [1, 1]}
       - {type: Al Al Al Al Al, nr: [1, 1, 1, 1], nl: [1, 1, 1, 1], c: 1.0, lint: [2, 2]}
\endcode
All the global parameters can be accessed by `input.global`. Global parameters also include additional
parameters that are calculated during the parsing of input, such as `lmax, nradmax, nradbase, rankmax`
which are the maximum values in case more than one species are present. These values can be accessed
in the same way as `input.global.lmax` etc.

The species specific information is stored in the `Input.species` attribute of the `Input` class, which
in turn  is an independent class `SpeciesBlock`. The `species`
attribute is two dimensional. For example, if two species `Al` and `Ti` are present, the pure elements `Al` and
`Ti` would be available by `input.species[0][0]` and `input.species[0][1]` respectively. `Al Ti` and `Ti Al`
would then be accessible by `input.species[1][0]` and `input.species[1][1]`.

Considering the above example file, all attributes for Al can be accessed through `input.species[0][0]`.
All the properties such as `nradmaxi, lmaxi, rcutij` etc can be accessed as `input.species[0][0].nradmaxi,
input.species[0][0].lmaxi, input.species[0][0].rcutij` etc. The `nbody` information is handled in a slightly
different way. `nbody` information is stored in the `nbody` attribute, which is represented by the `NBody` class.
The `nbody` attribute has three indices, the first one being density, the second is the rank, and the third index
is the number of terms of each rank. In the above case, the term of rank 2 and density 1 can be accessed as,
\code
input.species[0][0].nbody[0][0][0]
\endcode
This stands for,
\code
input.species[0][0].nbody[nd-1][rank-1][count]
\endcode
When there are two terms of same rank, for example, `rank = 4` in the above example,
\code
\\first term
input.species[0][0].nbody[0][3][0]
\\second term
input.species[0][0].nbody[0][3][1]
\endcode
The other attributes, therefore are accessed as `input.species[0][0].nbody[0][3][1].nr, input.species[0][0].nbody[0][3][1].nl`
etc.

To Do
-----
- Extend for multi-species system: Now even if multi-species input is provided, it is ignored.
- Error checks for the input scheme.
- Cutoff block which contains all cutoffs.
*/
#endif
