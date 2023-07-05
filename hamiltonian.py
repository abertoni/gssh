#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from .sparse_tools import sp_diag, sp_roll

def build_hamiltonian(phaselinks, parameters):
    y_n = phaselinks
    a = parameters["lattice_parameter"]
    # Initialize diagonal and non-diagonal elements
    H_ons = np.zeros(y_n.size, dtype=complex)
    H_hop = np.zeros(y_n.size, dtype=complex)
    # Electron-lattice coupling
    hopping_params = parameters["hopping_parameters"]
    for idx,t_i in enumerate(hopping_params):
        sign = (-1)**(idx+1) # Ensures electron-site attraction
        H_hop += sign * t_i * (y_n**idx)
    # Includes coupling with external field
    if "external_field_perturbation" in parameters:
        raise NotImplementedError()
        E_ext = parameters["external_field"]
        H_hop *= np.exp(-1j*E_ext)
    # Includes on-site perturbation
    if "onsite_perturbation" in parameters:
        raise NotImplementedError()
        H_ons += parameters["onsite_perturbation"]
    # Includes open-boundary corrections
    if "open_boundary_corrections" in parameters:
        raise NotImplementedError()
        is_periodic = parameters["periodic_boundaries"]
        if is_periodic: H_ons += parameters["open_boundary_corrections"]
    # Builds sparse matrix
    # (non-diagonal elements)
    H = sp_diag(H_hop)
    H = sp_roll(H, shift=-1)
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
