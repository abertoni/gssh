#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from .hamiltonian import build_hamiltonian, diagonalize_hamiltonian

def electronic_minimization(positions, parameters):
    raise NotImplementedError()
    #return state_energies, state_vectors

def get_electronic_solutions(time, positions, parameters):
    lat_opt_mode = parameters["lattice_optimization_mode"].lower()
    if "numerical" in lat_opt_mode:
        state_energies, state_vectors = electronic_minimization(positions, parameters)
    elif "analytical" in lat_opt_mode:
        hamiltonian = build_hamiltonian(time, positions, parameters)
        state_energies, state_vectors = diagonalize_hamiltonian(hamiltonian, parameters)
    return state_energies, state_vectors

def get_neighbour_sites_projection(state_vectors, occupations, shift, parameters):
    n_sites = parameters["number_of_sites"]
    is_periodic = parameters["periodic_boundaries"]
    states_dotprod = np.sum(occupations * np.conj(state_vectors) * np.roll(state_vectors, -shift, axis=0), axis=1)
    return states_dotprod

def sum_hermitian_conjugate(complex_expression):
    real_expression = np.real(complex_expression + np.conj(complex_expression.T))
    return real_expression

###############################################################
#       Leandro Manuel Arancibia & Andrés Ignacio Bertoni     #
# (leandro.arancibia9@gmail.com)   (andresibertoni@gmail.com) #
###############################################################
