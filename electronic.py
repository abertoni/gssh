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

###############################################################
#       Leandro Manuel Arancibia & Andrés Ignacio Bertoni     #
# (leandro.arancibia9@gmail.com)   (andresibertoni@gmail.com) #
###############################################################
