import numpy as np
from ase.neighborlist import NewPrimitiveNeighborList, PrimitiveNeighborList


class ACENeighborList:

    def __init__(self, cutoff=9, skin=0):
        self.cutoff = cutoff
        self.skin = skin
        self._species_type = []
        self._types_mapper_dict = {}
        self._reset_nl()

    def _reset_nl(self):
        self._ri = []  # coordinates on atom i, i.e. atomic positions
        self.x = []  # coordinates of all neighbors
        self.origins = []  # origin atoms indices for PBC
        self.types = []
        self.jlists = []  ##list of neighbor indicies for each atom

    def make_neighborlist(self, atoms, neigh_class=None):
        # if neigh_class is None, choose best NeighbourListBuilder
        if neigh_class is None:
            neigh_class = self._suggest_best_neighbour_list_class(atoms)

        self._reset_nl()
        self._nl = neigh_class(cutoffs=[self.cutoff * 0.5] * len(atoms), skin=self.skin,
                               self_interaction=False, bothways=True, use_scaled_positions=False)
        atoms_positions = atoms.get_positions()
        atoms_types = atoms.get_chemical_symbols()
        atoms_cell = atoms.get_cell()
        atoms_pbc = atoms.get_pbc()
        self._nl.update(atoms_pbc, atoms_cell, atoms_positions)

        # initialize array with real atoms
        rs_to_index_type_dict = {
            tuple(pos): (ind, neigh_type, ind)
            for ind, (pos, neigh_type) in enumerate(zip(atoms_positions, atoms_types))
        }

        for cur_at_ind in range(len(atoms)):
            cur_at_neigh_indices, cur_at_neigh_offsets = self._nl.get_neighbors(cur_at_ind)
            # extend neigh positions with periodic images
            cur_at_neigh_rs = np.take(atoms_positions, cur_at_neigh_indices, axis=0) + np.dot(cur_at_neigh_offsets,
                                                                                              atoms_cell)
            cur_at_neigh_types = np.take(atoms_types, cur_at_neigh_indices, axis=0)

            cur_at_neighb_list = []

            for neigh_r, neigh_type, neigh_origin in zip(cur_at_neigh_rs, cur_at_neigh_types, cur_at_neigh_indices):
                neigh_r = tuple(neigh_r)
                if neigh_r not in rs_to_index_type_dict:
                    new_ind = len(rs_to_index_type_dict)
                    rs_to_index_type_dict[neigh_r] = (new_ind, neigh_type, neigh_origin)
                cur_at_neighb_list.append(rs_to_index_type_dict[neigh_r][0])  # append index to current neighbour list

            self.jlists.append(cur_at_neighb_list)

        # inverting position -> index dict into indexed array of positions
        self.x = np.zeros((len(rs_to_index_type_dict), 3))
        self.origins = np.zeros(len(rs_to_index_type_dict), dtype=int)
        # create empty array of strings of len(rs_to_index_type_dict) by len(2) strings
        self.types = np.empty(len(rs_to_index_type_dict), dtype="S2")
        for neigh_r, (cur_at_ind, neigh_type, neigh_origin) in rs_to_index_type_dict.items():
            self.x[cur_at_ind] = neigh_r
            self.types[cur_at_ind] = neigh_type
            self.origins[cur_at_ind] = neigh_origin

    def _suggest_best_neighbour_list_class(self, atoms):
        if len(atoms) < 8 or np.any(atoms.get_pbc() == 0):
            return PrimitiveNeighborList
        else:  # else use NewPrimitiveNeighborList
            return NewPrimitiveNeighborList

    @property
    def types_mapper_dict(self):
        if not self._types_mapper_dict:
            for i, t in enumerate(sorted(np.unique(self.types))):
                self._types_mapper_dict[t] = i
        return self._types_mapper_dict

    @types_mapper_dict.setter
    def types_mapper_dict(self, new_types_mapper_dict):
        self._types_mapper_dict = {}
        els = np.array(list(new_types_mapper_dict.keys()), dtype="S2")
        inds = np.array(list(new_types_mapper_dict.values()), dtype=int)
        self._types_mapper_dict = {el: ind for el, ind in zip(els, inds)}

    @property
    def species_type(self):

        self._species_type = np.zeros(len(self.types))
        for i, t in enumerate(self.types):
            self._species_type[i] = self.types_mapper_dict[t]
        return self._species_type.astype(int)
