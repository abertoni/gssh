#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from .lattice_tools import get_sum_relative_positions
from gssh.hamiltonian import build_hamiltonian, diagonalize_hamiltonian
from gssh.occupations import get_occupations

def calculate_static_electronic_energy(positions, parameters, time = 0):
    hamiltonian = build_hamiltonian(time, positions, parameters)
    state_energies, state_vectors = diagonalize_hamiltonian(hamiltonian, parameters)
    occupations = get_occupations(state_energies, parameters)
    electronic_energy = 0
    for idx in range(len(['spin_up','spin_down'])):
        electronic_energy += np.sum(occupations[:,idx] * state_energies)
    return electronic_energy

def calculate_static_lattice_energy(positions, parameters):
    # (only harmonic oscillation is implemented)
    oscillator_params = parameters["oscillator_parameters"]
    is_periodic = parameters["periodic_boundaries"]
    if len(oscillator_params) > 1: raise NotImplementedError()
    K = oscillator_params[0] # harmonic constant
    lattice_energy = K/2 * np.sum( positions_to_phaselinks(positions, parameters)**2 )
    if not is_periodic:
        if "open_boundary_stretching" in parameters.keys():
            G = parameters["open_boundary_stretching"]
            a = parameters["lattice_parameter"]
            N = parameters["number_of_sites"]
            lattice_energy += -G * sum(positions_to_phaselinks(positions, parameters))
    return lattice_energy

def calculate_static_total_energy(positions, parameters):
    electronic_energy = calculate_static_electronic_energy(positions, parameters)
    lattice_energy = calculate_static_lattice_energy(positions, parameters)
    return electronic_energy + lattice_energy

def positions_to_phaselinks(positions, parameters):
    a = parameters["lattice_parameter"]
    n_sites = parameters["number_of_sites"]
    is_periodic = parameters["periodic_boundaries"]
    # y_n = u_{n+1} - u_{n} = Δr_{n+1} - Δr_{n}
    # y_n = r_{n+1} - r_{n} - a
    r_n = positions
    y_n = np.roll(r_n, -1) - r_n - a
    if is_periodic: y_n[-1] += n_sites * a # u_{N+1} = u_{n}
    else: y_n = y_n[:-1]
    # Return phaselinks
    phaselinks = y_n
    return phaselinks

###############################################################
#       Leandro Manuel Arancibia & Andrés Ignacio Bertoni     #
# (leandro.arancibia9@gmail.com)   (andresibertoni@gmail.com) #
###############################################################
