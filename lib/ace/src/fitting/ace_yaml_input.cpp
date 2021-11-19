#include "ace_yaml_input.h"

#include <algorithm>
#include <iostream>
#include <set>
#include <string>

#include "yaml-cpp/yaml.h"

#include "ace_couplings.h"

using namespace std;


/**
Split a given string by tabs or space

@param mainkey, string - input string

@returns splitted, string - the string after splitting
*/
vector<string> Input::split_key(string mainkey) {

    vector<string> splitted;
    istringstream ins(mainkey);

    for (string mainkey; ins >> mainkey;)
        splitted.emplace_back(mainkey);

    return splitted;
}


/**
Main function to parse the yaml file and read the input data
*/
void Input::parse_input(const string &ff) {
    //set the input file - first thing to do
    inputfile = ff;
    if (!if_file_exist(inputfile)) {
        stringstream s;
        s << "Potential file " << inputfile << " doesn't exists";
        cerr << "Exception: " << s.str();
        throw invalid_argument(s.str());
    }

    //load the file with yaml
    YAML_PACE::Node YAML_input = YAML_PACE::LoadFile(inputfile);

    //all the raw data is now available in rawinput

    //first step - parse the global data
    global.DeltaSplineBins = YAML_input["global"]["DeltaSplineBins"].as<DOUBLE_TYPE>();
    auto YAML_input_species = YAML_input["species"];
    //now get the number of species blocks
    number_of_species_block = static_cast<unsigned short>(YAML_input_species.size());
    global.nblocks = number_of_species_block;

    if (YAML_input["metadata"]) {
        global.metadata = YAML_input["metadata"].as<map<string, string>>();
    }

    if (YAML_input["auxdata"]) {
        YAML_PACE::Node node = YAML_input["auxdata"];
        global.auxdata.from_YAML(node);
    }

    //find the names of all species block, and use that to count
    //single elements, pairs and so on
    vector<string> elements_splitted;

    global.nelements = 0;
    // use set to keep track of the available elements
    set<string> elements_set;
    //loop over each block and find elements.
    for (unsigned short int i = 0; i < number_of_species_block; i++) {
        string tosplit = YAML_input_species[i]["speciesblock"].as<string>();
        elements_splitted = split_key(tosplit);
        for (const auto &element: elements_splitted) {
            if (elements_set.count(element) == 0) {
                elements_set.insert(element);
            }
        }
    }

    //now check
    global.nelements = elements_set.size();
    // copy to the global.elements_names section and sort it
    std::copy(elements_set.begin(), elements_set.end(), std::back_inserter(global.element_names));
    std::sort(global.element_names.begin(), global.element_names.end());
#ifdef DEBUG_READ_YAML
    cout << "global.element_names" << endl;
    for (auto element: global.element_names) {
        cout << element << endl;
    }
#endif

    map<string, int> elements_ndensity_map;

    for (int species_block_ind = 0; species_block_ind < number_of_species_block; species_block_ind++) {
        //create temp block
        //vector<SpeciesBlock> temp_species;
        auto YAML_input_species_block = YAML_input_species[species_block_ind];
        string block_name = YAML_input_species_block["speciesblock"].as<string>();
#ifdef DEBUG_READ_YAML
        cout << "block_name:" << block_name << endl;
#endif
        //create a species_block
        vector<string> elements_vec = split_key(block_name);
        string mu0 = elements_vec[0];
        BBasisFunctionsSpecificationBlock b_basisfunc_spec_block;


        b_basisfunc_spec_block.block_name = block_name;
        b_basisfunc_spec_block.elements_vec = split_key(block_name);

        b_basisfunc_spec_block.mu0 = mu0;

        //count number of unique elements in elements_vec
        set<string> elements_set;
        for (auto const &el: b_basisfunc_spec_block.elements_vec) elements_set.insert(el);
        b_basisfunc_spec_block.number_of_species = elements_set.size();

        // double the element for 1-species block, i.e. Al -> Al Al
        if (b_basisfunc_spec_block.number_of_species == 1 & b_basisfunc_spec_block.elements_vec.size() == 1)
            b_basisfunc_spec_block.elements_vec.emplace_back(mu0);

        // Element-specific values:
        //  - embedding block: ndensityi, npoti, parameters
        //  - (optional) rho_core_cut and drho_core_cut
        if (b_basisfunc_spec_block.number_of_species == 1) {
            //embedding block for 1-species
            b_basisfunc_spec_block.ndensityi = YAML_input_species_block["ndensityi"].as<DENSITY_TYPE>();

            // check and validate ndensity
            // if ndensity for element mu0 is not defined
            if (elements_ndensity_map.count(b_basisfunc_spec_block.mu0) == 0)
                elements_ndensity_map[b_basisfunc_spec_block.mu0] = b_basisfunc_spec_block.ndensityi;
            else {
                // else if ndensity for element mu0 already defined -> error
                stringstream s;
                s << "ndensity for " << b_basisfunc_spec_block.mu0 << " = " << b_basisfunc_spec_block.ndensityi <<
                  " is already defined" << endl;
                cerr << "Exception: " << s.str();
                throw invalid_argument(s.str());
            }

            b_basisfunc_spec_block.npoti = YAML_input_species_block["npoti"].as<string>();
            b_basisfunc_spec_block.fs_parameters = YAML_input_species_block["parameters"].as<vector<DOUBLE_TYPE>>();

            //optional - energy cutoff parameters
            read_core_rho_drho_cut(YAML_input_species_block, b_basisfunc_spec_block);
        } else {
            b_basisfunc_spec_block.ndensityi = elements_ndensity_map[b_basisfunc_spec_block.mu0];
        } // end of 1-specie if-else conditions

        //======================
        //bond basis::bond block
        //======================

        // radial basis functions (defined for 1-species and optionally defined for 2-species (with A-B <=> B-A elements duality)
        if (YAML_input_species_block["lmaxi"]) { // if lmaxi is defined

            b_basisfunc_spec_block.lmaxi = YAML_input_species_block["lmaxi"].as<LS_TYPE>();
            b_basisfunc_spec_block.nradmaxi = YAML_input_species_block["nradmaxi"].as<NS_TYPE>();
            b_basisfunc_spec_block.nradbaseij = YAML_input_species_block["nradbaseij"].as<NS_TYPE>();
            b_basisfunc_spec_block.radbase = YAML_input_species_block["radbase"].as<string>();
            b_basisfunc_spec_block.radparameters = YAML_input_species_block["radparameters"].as<vector<DOUBLE_TYPE>>();
            read_radcoefficients(YAML_input_species_block, b_basisfunc_spec_block);
            b_basisfunc_spec_block.validate_radcoefficients();

            b_basisfunc_spec_block.rcutij = YAML_input_species_block["rcutij"].as<DOUBLE_TYPE>();
            b_basisfunc_spec_block.dcutij = YAML_input_species_block["dcutij"].as<DOUBLE_TYPE>();
            //TODO: deprecated?
            if (YAML_input_species_block["NameOfCutoffFunctionij"])
                b_basisfunc_spec_block.NameOfCutoffFunctionij = YAML_input_species_block["NameOfCutoffFunctionij"].as<string>();

            b_basisfunc_spec_block.is_radial_basis_defined = true;
#ifdef DEBUG_READ_YAML
            cout << b_basisfunc_spec_block.block_name << ": b_basisfunc_spec_block.is_radial_basis_defined = true"
                 << endl;
#endif
        } else if (YAML_input_species_block["nradmaxi"] || YAML_input_species_block["nradbaseij"] ||
                   YAML_input_species_block["radbase"] || YAML_input_species_block["radparameters"] ||
                   YAML_input_species_block["radcoefficients"]) {
            // else if radial basis is not completely defined -> error
            stringstream s;
            s << "Radial basis functions are partially defined for  " << b_basisfunc_spec_block.block_name;
            cerr << "Exception: " << s.str();
            throw invalid_argument(s.str());
        }

        //hard-core repulsion parameters
        read_core_repulsion(YAML_input_species_block, b_basisfunc_spec_block);

        //this is just the number of entries - it can have multiple terms of rank 1 and so on
        auto num_of_basis_functions = static_cast<SHORT_INT_TYPE>(YAML_input_species_block["nbody"].size());
        if (num_of_basis_functions == 0) {
            stringstream s;
            s << "Potential yaml file '" << inputfile << "' has no <nbody> section in <speciesblock>";
            if (YAML_input_species_block["density"].size() > 0)
                s << "Section <density> is presented. It seems that this is old file format. Please use new format";
            cerr << "Exception: " << s.str();
//            throw std::invalid_argument(s.str());
        }

        //this is a vector to store the nbodys temporarily
        vector<BBasisFunctionSpecification> temp_bbasisfunc_spec_vector;
        for (auto YAML_input_current_basisfunc_spec:  YAML_input_species_block["nbody"]) {
            BBasisFunctionSpecification bBasisFunctionSpec = BBasisFunctionSpecification();

            string basisfunc_species_str = YAML_input_current_basisfunc_spec["type"].as<string>();
            vector<string> curr_basisfunc_species_vec = split_key(basisfunc_species_str);
            bBasisFunctionSpec.elements = curr_basisfunc_species_vec;
            //check that:
            // 1. func.El0 == block El0

            string central_atom_element = bBasisFunctionSpec.elements[0];
            if (central_atom_element != b_basisfunc_spec_block.mu0) {
                stringstream s;
                s << "Function specification with type: '" << basisfunc_species_str << "' is inconsistent with"
                  << " block specification '" << block_name << "'";
                cerr << "Exception: " << s.str();
                throw std::invalid_argument(s.str());
            }

            // check that
            // 2. rest of the func elements consistent with rest of block elements
            for (int ii = 1; ii < curr_basisfunc_species_vec.size(); ii++) {
                string func_neigh_el = curr_basisfunc_species_vec[ii];
                if (elements_set.count(func_neigh_el) == 0) {
                    stringstream s;
                    s << "Function specification with type: '" << basisfunc_species_str << "' is inconsistent with"
                      << " block specification '" << block_name << "': element " << func_neigh_el
                      << " is not in block's expected elements";
                    cerr << "Exception: " << s.str();
                    throw std::invalid_argument(s.str());
                }
            }

            // decrease by 1 because func::type str has form "El0 El El El"
            RANK_TYPE rank = curr_basisfunc_species_vec.size() - 1;

            bBasisFunctionSpec.rank = rank;

            if (rank > 0) {
                bBasisFunctionSpec.ns = YAML_input_current_basisfunc_spec["nr"].as<vector<NS_TYPE>>();
                bBasisFunctionSpec.ls = YAML_input_current_basisfunc_spec["nl"].as<vector<LS_TYPE>>();

                if (YAML_input_current_basisfunc_spec["c"].Type() == YAML_PACE::NodeType::Sequence)
                    bBasisFunctionSpec.coeffs = YAML_input_current_basisfunc_spec["c"].as<vector<DOUBLE_TYPE>>();
                else if (YAML_input_current_basisfunc_spec["c"].Type() == YAML_PACE::NodeType::Scalar) {
                    vector<DOUBLE_TYPE> c_vec(1);
                    c_vec[0] = YAML_input_current_basisfunc_spec["c"].as<DOUBLE_TYPE>();
                    bBasisFunctionSpec.coeffs = c_vec;
                }

                // if actual number of coefficients is not equal to declared in embedding functions
                // then throw an exception
                if (bBasisFunctionSpec.coeffs.size() != elements_ndensity_map[central_atom_element]) {
                    stringstream s;
                    s << "'n-body' function has " << bBasisFunctionSpec.coeffs.size() << " coefficients, " <<
                      "but the embedding function for " << central_atom_element << " expects "
                      << elements_ndensity_map[central_atom_element];
                    cerr << "Exception: " << s.str() << endl;
                    throw std::invalid_argument(s.str());
                }
            }

            if (rank > 2)
                bBasisFunctionSpec.LS = YAML_input_current_basisfunc_spec["lint"].as<vector<LS_TYPE>>();

            //check and read skip flag
            if (YAML_input_current_basisfunc_spec["skip"]) {
                bBasisFunctionSpec.skip = YAML_input_current_basisfunc_spec["skip"].as<bool>();
            }

            //validation according to ls-LS relations
            try {
                validate_ls_LS(bBasisFunctionSpec.ls, bBasisFunctionSpec.LS);
            } catch (invalid_argument &e) {
                cerr << "Exception: " << e.what();
                throw e;
            }

            b_basisfunc_spec_block.funcspecs.emplace_back(bBasisFunctionSpec);
        }//end loop over basis functions "nbody"




        //collect b_basisfunc_spec_block into vector
        bbasis_func_spec_blocks_vector.emplace_back(b_basisfunc_spec_block);
    } //end of loop over species block

    // now all data are read

    // extend the radial basis functions description for pair-blocks from its dual (A-B <=> B-A)
    for (auto &b_basisfunc_spec_block: bbasis_func_spec_blocks_vector) {
        if (b_basisfunc_spec_block.number_of_species == 2) {//if pair-block and 2-species
            // check if radial basis functions are defined with b_basisfunc_spec_block.is_radial_basis_defined
            //find dual pair-block:
#ifdef DEBUG_READ_YAML
            cout << "Pair 2-species block: " << b_basisfunc_spec_block.block_name << endl;
#endif
            vector<string> dual_elements_vec = {b_basisfunc_spec_block.elements_vec[1],
                                                b_basisfunc_spec_block.elements_vec[0]};
            for (auto &dual_b_basisfunc_spec_block: bbasis_func_spec_blocks_vector) {

                if (dual_b_basisfunc_spec_block.elements_vec == dual_elements_vec) {
#ifdef DEBUG_READ_YAML
                    cout << "Dual 2-species block found: " << dual_b_basisfunc_spec_block.block_name << endl;
#endif

                    //checking radial basis settings
                    if (!b_basisfunc_spec_block.is_radial_basis_defined) {
                        if (dual_b_basisfunc_spec_block.is_radial_basis_defined) { // copy the radial basis definition from dual block
                            copy_radial_basis_from_to_block(dual_b_basisfunc_spec_block, b_basisfunc_spec_block);
                        } else {
                            stringstream s;
                            s << "Radial basis is not defined neither in '" << b_basisfunc_spec_block.block_name
                              << "' nor in '"
                              << dual_b_basisfunc_spec_block.block_name << "' 2-species blocks";
                            cerr << "Exception: " << s.str();
                            throw invalid_argument(s.str());
                        }
                    } else { //b_basisfunc_spec_block.is_radial_basis_defined
                        if (!dual_b_basisfunc_spec_block.is_radial_basis_defined)  //if radial basis is defined in first block
                            copy_radial_basis_from_to_block(b_basisfunc_spec_block, dual_b_basisfunc_spec_block);
                        else // radial basis in both blocks are defined
                            check_radial_basis_consistency(b_basisfunc_spec_block, dual_b_basisfunc_spec_block);
                    }

                    //checking core repulsion?
                    if (!b_basisfunc_spec_block.is_core_repulsion_defined &
                        dual_b_basisfunc_spec_block.is_core_repulsion_defined) {
                        copy_core_repulsion_from_to_block(dual_b_basisfunc_spec_block, b_basisfunc_spec_block);
                    } else if (b_basisfunc_spec_block.is_core_repulsion_defined &
                               !dual_b_basisfunc_spec_block.is_core_repulsion_defined) {
                        copy_core_repulsion_from_to_block(b_basisfunc_spec_block, dual_b_basisfunc_spec_block);
                    } else if (b_basisfunc_spec_block.is_core_repulsion_defined &
                               dual_b_basisfunc_spec_block.is_core_repulsion_defined) {
                        check_core_repulsion_consistency(b_basisfunc_spec_block, dual_b_basisfunc_spec_block);
                    }
                }
            }
        } // now 2-species b_basisfunc_spec_block has correct settings for radial basis

        // let us do the consistency check between radial basis settings and BFunctions
        if (b_basisfunc_spec_block.number_of_species <= 2) {
            NS_TYPE max_nradbase = 0;
            NS_TYPE max_nrad = 0;
            LS_TYPE max_l = 0;
            LS_TYPE max_LS = 0;
            RANK_TYPE max_rank = 0;

            for (const auto &bBasisFunctionSpec: b_basisfunc_spec_block.funcspecs) {
                RANK_TYPE rank = bBasisFunctionSpec.rank;

                if (rank > max_rank) max_rank = rank;

                NS_TYPE bfunc_max_ns = *max_element(bBasisFunctionSpec.ns.begin(), bBasisFunctionSpec.ns.end());
                LS_TYPE bfunc_max_ls = *max_element(bBasisFunctionSpec.ls.begin(), bBasisFunctionSpec.ls.end());

                if (bfunc_max_ls > max_l) max_l = bfunc_max_ls; //update maximum l

                if (rank == 1) {
                    if (bfunc_max_ns > max_nradbase) max_nradbase = bfunc_max_ns; //update maximum nradbase

                    if (b_basisfunc_spec_block.nradbaseij < bfunc_max_ns) {
                        stringstream s;
                        s << "Given nradbaseij = " << b_basisfunc_spec_block.nradbaseij <<
                          " is less than the max(nr) = " << bfunc_max_ns << " for rank=1 "
                          << bBasisFunctionSpec.to_string();
                        cerr << "Exception: " << s.str();
                        throw invalid_argument(s.str());
                    }

                    if (bfunc_max_ls > 0) {
                        stringstream s;
                        s << "Given current max(nl) = " << bfunc_max_ls <<
                          " is not equal to 0 for rank = 1 function " << bBasisFunctionSpec.to_string();
                        cerr << "Exception: " << s.str();
                        throw invalid_argument(s.str());
                    }
                } else { //rank > 1
                    if (bfunc_max_ns > max_nrad) max_nrad = bfunc_max_ns;  //update maximum nrad

                    if (bfunc_max_ns > b_basisfunc_spec_block.nradmaxi) {
                        stringstream s;
                        s << "Given nradmaxi = " << b_basisfunc_spec_block.nradmaxi <<
                          " is less than the max(nr) = " << bfunc_max_ns << " for rank>1 "
                          << bBasisFunctionSpec.to_string();
                        cerr << "Exception: " << s.str();
                        throw invalid_argument(s.str());
                    }

                    if (bfunc_max_ls > b_basisfunc_spec_block.lmaxi) {
                        stringstream s;
                        s << "Given current max(nl) = " << bfunc_max_ls <<
                          " is larger than block lmaxi = " << b_basisfunc_spec_block.lmaxi << " for function "
                          << bBasisFunctionSpec.to_string();
                        cerr << "Exception: " << s.str();
                        throw invalid_argument(s.str());
                    }

                    if (rank > 2) {
                        LS_TYPE bfunc_max_LS = *max_element(bBasisFunctionSpec.LS.begin(), bBasisFunctionSpec.LS.end());
                        if (bfunc_max_LS > max_LS) max_LS = bfunc_max_LS; // update  max_LS
                        if (2 * b_basisfunc_spec_block.lmaxi < max_LS) {
                            stringstream s;
                            s << "Given 2*lmaxi = " << 2 * b_basisfunc_spec_block.lmaxi
                              << " is less than the max(lint) = "
                              << max_LS << " in function " << bBasisFunctionSpec.to_string();
                            cerr << "Exception: " << s.str();
                            throw invalid_argument(s.str());
                        }
                    }
                }
            } // end loop over b-basis func specs

            //update maximums
            if (max_rank > b_basisfunc_spec_block.rankmax)
                b_basisfunc_spec_block.rankmax = max_rank;

            //TODO: update this maximums ?
            if (max_l > b_basisfunc_spec_block.lmaxi)
                b_basisfunc_spec_block.lmaxi = max_l;

            if (max_nrad > b_basisfunc_spec_block.nradmaxi)
                b_basisfunc_spec_block.nradmaxi = max_nrad;

            if (max_nradbase > b_basisfunc_spec_block.nradbaseij)
                b_basisfunc_spec_block.nradbaseij = max_nradbase;
        } //enf if for 1- or 2- species block

        // TODO: checking consistencies for >=3-species blocks
        // WARNING: currently we implicitly assume that 3-species blocks are consistent with pair interactions
    } // end loop over blocks

    global.lmax = 0;
    global.nradmax = 0;
    global.nradbase = 0;
    global.rankmax = 0;
    global.ndensitymax = 0;
    global.cutoffmax = 0;

    // collect global: [lmax, nradmaxi, nradbaseij, rankmax, ndensityi, rcutij
    // over all primary blocks
    for (const auto &pair_block: bbasis_func_spec_blocks_vector) {
        if (pair_block.number_of_species <= 2) {
            if (pair_block.lmaxi > global.lmax) global.lmax = pair_block.lmaxi;
            if (pair_block.nradmaxi > global.nradmax) global.nradmax = pair_block.nradmaxi;
            if (pair_block.nradbaseij > global.nradbase) global.nradbase = pair_block.nradbaseij;
            if (pair_block.rankmax > global.rankmax) global.rankmax = pair_block.rankmax;
            if (pair_block.ndensityi > global.ndensitymax) global.ndensitymax = pair_block.ndensityi;
            if (pair_block.rcutij > global.cutoffmax) global.cutoffmax = pair_block.rcutij;
        }
    }

}

void Input::copy_radial_basis_from_to_block(const BBasisFunctionsSpecificationBlock &from_spec_block,
                                            BBasisFunctionsSpecificationBlock &to_spec_block) const {
    to_spec_block.lmaxi = from_spec_block.lmaxi;
    to_spec_block.nradmaxi = from_spec_block.nradmaxi;
    to_spec_block.nradbaseij = from_spec_block.nradbaseij;
    to_spec_block.radbase = from_spec_block.radbase;
    to_spec_block.radparameters = from_spec_block.radparameters;
    to_spec_block.radcoefficients = from_spec_block.radcoefficients;
    to_spec_block.rcutij = from_spec_block.rcutij;
    to_spec_block.dcutij = from_spec_block.dcutij;
}

void Input::check_radial_basis_consistency(const BBasisFunctionsSpecificationBlock &block1,
                                           const BBasisFunctionsSpecificationBlock &block2) const {
    if (block1.lmaxi != block2.lmaxi ||
        block1.nradmaxi != block2.nradmaxi ||
        block1.nradbaseij != block2.nradbaseij ||
        block1.radbase != block2.radbase ||
        block1.radparameters != block2.radparameters ||
        block1.radcoefficients != block2.radcoefficients ||
        block1.rcutij != block2.rcutij ||
        block1.dcutij != block2.dcutij ||
        block1.r_in != block2.r_in ||
        block1.delta_in != block2.delta_in ||
        block1.inner_cutoff_type != block2.inner_cutoff_type
            ) {
        stringstream s;
        s << "Radial basis in blocks '" << block1.block_name << "' and '"
          << block2.block_name << "' is not consistent";
        cerr << "Exception: " << s.str();
        throw invalid_argument(s.str());
    }
}

void Input::read_core_repulsion(const YAML_PACE::Node &YAML_input_species_block,
                                BBasisFunctionsSpecificationBlock &b_basisfunc_spec_block) const {
    if (YAML_input_species_block["core-repulsion"]) {
        b_basisfunc_spec_block.core_rep_parameters = YAML_input_species_block["core-repulsion"].as<vector<DOUBLE_TYPE>>();
        b_basisfunc_spec_block.is_core_repulsion_defined = true;
    } else
        b_basisfunc_spec_block.core_rep_parameters.resize(2, 0);

    if (YAML_input_species_block["inner_cutoff_type"])
        b_basisfunc_spec_block.inner_cutoff_type = YAML_input_species_block["inner_cutoff_type"].as<string>();
    else
        b_basisfunc_spec_block.inner_cutoff_type = "density"; // for backward compatibility
    if (YAML_input_species_block["r_in"])
        b_basisfunc_spec_block.r_in = YAML_input_species_block["r_in"].as<DOUBLE_TYPE>();
    if (YAML_input_species_block["delta_in"])
        b_basisfunc_spec_block.delta_in = YAML_input_species_block["delta_in"].as<DOUBLE_TYPE>();
}

void Input::copy_core_repulsion_from_to_block(const BBasisFunctionsSpecificationBlock &from_spec_block,
                                              BBasisFunctionsSpecificationBlock &to_spec_block) const {
    to_spec_block.core_rep_parameters = from_spec_block.core_rep_parameters;
    to_spec_block.inner_cutoff_type = from_spec_block.inner_cutoff_type;
    to_spec_block.r_in = from_spec_block.r_in;
    to_spec_block.delta_in = from_spec_block.delta_in;
}

void Input::check_core_repulsion_consistency(const BBasisFunctionsSpecificationBlock &block1,
                                             const BBasisFunctionsSpecificationBlock &block2) const {
    if (block1.core_rep_parameters != block2.core_rep_parameters) {
        stringstream s;
        s << "Core-repulsion in blocks '" << block1.block_name << "' and '"
          << block2.block_name << "' is not consistent";
        cerr << "Exception: " << s.str();
        throw invalid_argument(s.str());
    }

    if (block1.r_in != block2.r_in || block1.delta_in != block2.delta_in) {
        stringstream s;
        s << "Inner cutoff (r_in, delta_in) in blocks '" << block1.block_name << "' and '"
          << block2.block_name << "' is not consistent";
        cerr << "Exception: " << s.str();
        throw invalid_argument(s.str());
    }
}


void Input::read_core_rho_drho_cut(const YAML_PACE::Node &YAML_input_species_block,
                                   BBasisFunctionsSpecificationBlock &b_basisfunc_spec_block) const {
    if (YAML_input_species_block["rho_core_cut"])
        b_basisfunc_spec_block.rho_cut = YAML_input_species_block["rho_core_cut"].as<DOUBLE_TYPE>();
    else
        b_basisfunc_spec_block.rho_cut = 100000.0;


    if (YAML_input_species_block["drho_core_cut"])
        b_basisfunc_spec_block.drho_cut = YAML_input_species_block["drho_core_cut"].as<DOUBLE_TYPE>();
    else
        b_basisfunc_spec_block.drho_cut = 250.0;
}

void Input::read_radcoefficients(const YAML_PACE::Node &YAML_input_species_block,
                                 BBasisFunctionsSpecificationBlock &b_basisfunc_spec_block) const {
    try {
        b_basisfunc_spec_block.radcoefficients = YAML_input_species_block["radcoefficients"].as<vector<vector<vector<DOUBLE_TYPE>>>>();
    } catch (YAML_PACE::RepresentationException &exc) {
        cout
                << "DEPRECATION WARNING!!! Old (flatten) radcoefficients parameter encounterd, whereas it should be three-dimensional with [nradmax][lmax+1][nradbase] shape."
                << endl;
        cout << "Automatic reshaping will be done" << endl;
        auto radcoefficients = YAML_input_species_block["radcoefficients"].as<vector<DOUBLE_TYPE>>();
        size_t j = 0;

        //initialize array
        b_basisfunc_spec_block.radcoefficients = vector<vector<vector<DOUBLE_TYPE>>>(
                b_basisfunc_spec_block.nradmaxi,
                vector<vector<DOUBLE_TYPE>>(b_basisfunc_spec_block.lmaxi + 1,
                                            vector<DOUBLE_TYPE>(b_basisfunc_spec_block.nradbaseij)
                )
        );

        for (NS_TYPE k = 0; k < b_basisfunc_spec_block.nradbaseij; k++) {
            for (NS_TYPE n = 0; n < b_basisfunc_spec_block.nradmaxi; n++) {
                for (LS_TYPE l = 0; l <= b_basisfunc_spec_block.lmaxi; l++, j++) {
                    b_basisfunc_spec_block.radcoefficients.at(n).at(l).at(k) = radcoefficients.at(j);
                }
            }
        }

    } catch (exception &exc) {
        printf("Error: %s\n", exc.what());
    }
}


