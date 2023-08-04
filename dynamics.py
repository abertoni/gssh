#!/usr/bin/env python
# -*- coding: utf-8 -*-

from .time_tools import get_time_grid
from .lattice_propagation import propagate_lattice
from .hamiltonian import build_hamiltonian
from .occupations import compute_occupations
from .electronic_propagation import propagate_electrons
from .output import initialize_output, store_to_output, finalize_output

def time_propagation(positions, velocities, accelerations, state_vectors, parameters):
    # Avoids overwritting input positions
    positions = positions.copy()
    # Initialize output
    initialize_output(parameters)
    # Time propagation
    time_grid = get_time_grid(parameters)
    store_to_output(time_grid[0], positions, velocities, accelerations, state_vectors, parameters)
    # Build initial time Hamiltonian
    hamiltonian = build_hamiltonian(time_grid[0], positions, parameters)
    state_energies, state_vectors = diagonalize_hamiltonian(hamiltonian, parameters)
    occupations = compute_occupations(state_energies, parameters)
    # Propagate in time
    for time in time_grid[1:]:
        positions, velocities, accelerations = propagate_lattice(time, positions, velocities, accelerations, state_vectors, occupations, parameters)
        state_vectors, hamiltonian = propagate_electrons(time, positions, state_vectors, occupations, hamiltonian, parameters)
        store_to_output(positions, velocities, accelerations, state_vectors, parameters)
    # Finalize the output
    finalize_output(parameters)
    # Return last 
    return positions, velocities, accelerations, state_vectors

###############################################################
#       Leandro Manuel Arancibia & Andrés Ignacio Bertoni     #
# (leandro.arancibia9@gmail.com)   (andresibertoni@gmail.com) #
###############################################################
