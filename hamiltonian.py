#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from .latice_tools import get_neighbour_distances
from .sparse_tools import sp_diag, sp_roll
from .perturbations import get_vector_potential

def build_hamiltonian(time, positions, parameters):
    # Initialize diagonal and non-diagonal elements
    n_sites = parameters["number_of_sites"]
    H_ons = np.zeros(n_sites, dtype=complex)
    H_hop = np.zeros(n_sites, dtype=complex)
    # Electron-lattice coupling
    a = parameters["lattice_parameter"]
    hopping_params = parameters["hopping_parameters"]
    dr_n = get_neighbour_distances(positions, +1, parameters)
    for idx,t_i in enumerate(hopping_params):
        sign = (-1)**(idx+1) # Ensures electron-site attraction
        H_hop[:y_n.size] += sign * t_i * ((dr_n - a)**idx)
    # Includes coupling with external vector potential
    # (related to derivative of external electric field)
    if "external_perturbation" in parameters:
        vector_potential = get_vector_potential(time, parameters)
        H_hop *= np.exp(+1j * a * vector_potential)
    # Includes on-site perturbation
    if "onsite_perturbation" in parameters:
        onsite_energies = get_onsite_energies(time, parameters)
        H_ons += get_onsite_energies(time, parameters)
    # Builds sparse matrix
    # (non-diagonal elements)
    H = sp_diag(H_hop)
    H = sp_roll(H, shift=-1) # (upper diagonal)
    H = H + np.conjugate(H.T)
    # (diagonal elements)
    H += sp_diag(H_ons)
    return H

def diagonalize_hamiltonian(hamiltonian, parameters):
    eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian.toarray())
    # (sorts states by energy)
    energy_sort = np.argsort(eigenvalues)
    state_energies = eigenvalues[energy_sort]
    state_vectors = eigenvectors[:, energy_sort]
    # (ensures state vectors normalization)
    state_vectors /= np.linalg.norm(state_vectors, axis=0)
    return state_energies, state_vectors

###############################################################
#       Leandro Manuel Arancibia & Andrés Ignacio Bertoni     #
# (leandro.arancibia9@gmail.com)   (andresibertoni@gmail.com) #
###############################################################
