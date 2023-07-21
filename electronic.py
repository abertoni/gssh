#!/usr/bin/env python
# -*- coding: utf-8 -*-

from .hamiltonian import build_hamiltonian, diagonalize_hamiltonian

def electronic_minimization(phaselinks, parameters):
    raise NotImplementedError()
    #return state_energies, state_vectors

def get_electronic_solutions(phaselinks, parameters):
    lat_opt_mode = parameters["lattice_optimization_mode"].lower()
    if "numerical" in lat_opt_mode:
        state_energies, state_vectors = electronic_minimization(phaselinks, parameters)
    elif "analytical" in lat_opt_mode:
        hamiltonian = build_hamiltonian(phaselinks, parameters)
        state_energies, state_vectors = diagonalize_hamiltonian(hamiltonian, parameters)
    return state_energies, state_vectors

def neighbour_sites_projection(n_shift, states_vectors, occupations, parameters):
    """Projection between n+n_shift and n+n_shift+1."""
    n_sites = parameters["number_of_sites"]
    is_periodic = parameters["periodic_boundaries"]
    phaselinks_size = n_sites - (0 if is_periodic else 1)
    states_dot_product = np.sum(occupations * np.conj(np.roll(state_vectors, -(n_shift), axis=0)) * np.roll(state_vectors, -(n_shift+1), axis=0), axis=1)
    if np.linalg.norm(np.imag(states_dot_product)) > 0: print("Warning! Imaginary part of neighbor sites projection is non-zero.")
    states_dot_product = 2 * np.real(states_dot_product)[:phaselinks_size] # h.c.
    return states_dot_product

###############################################################
#       Leandro Manuel Arancibia & Andrés Ignacio Bertoni     #
# (leandro.arancibia9@gmail.com)   (andresibertoni@gmail.com) #
###############################################################
