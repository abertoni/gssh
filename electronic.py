#!/usr/bin/env python
# -*- coding: utf-8 -*-

from .hamiltonian import build_hamiltonian, diagonalize_hamiltonian

def electronic_minimization(positions, parameters):
    raise NotImplementedError()
    #return state_energies, state_vectors

def get_electronic_solutions(positions, parameters):
    lat_opt_mode = parameters["lattice_optimization_mode"].lower()
    if "numerical" in lat_opt_mode:
        state_energies, state_vectors = electronic_minimization(positions, parameters)
    elif "analytical" in lat_opt_mode:
        hamiltonian = build_hamiltonian(positions, parameters)
        state_energies, state_vectors = diagonalize_hamiltonian(hamiltonian, parameters)
    return state_energies, state_vectors

def get_neighbour_sites_projection(states_vectors, occupations, shift, parameters):
    n_sites = parameters["number_of_sites"]
    is_periodic = parameters["periodic_boundaries"]
    states_dotprod = np.sum(occupations * np.conj(state_vectors) * np.roll(state_vectors, -shift, axis=0), axis=1)
    if np.linalg.norm(np.imag(states_dotprod)) > 0: print("Warning! Imaginary part of neighbor sites projection is non-zero.")
    states_dotprod = 2 * np.real(states_dotprod) # h.c.
    return states_dotprod

###############################################################
#       Leandro Manuel Arancibia & Andrés Ignacio Bertoni     #
# (leandro.arancibia9@gmail.com)   (andresibertoni@gmail.com) #
###############################################################
