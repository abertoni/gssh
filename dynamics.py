#!/usr/bin/env python
# -*- coding: utf-8 -*-

from .time_tools import get_time_grid
from .lattice_propagation import propagate_lattice
from .electronic_propagation import propagate_electrons
from .output import initialize_output, store_to_output, finalize_output

def time_propagation(positions, velocities, accelerations, state_vectors, parameters):
    # Initialize output
    initialize_output(parameters)
    # Time propagation
    time_grid = get_time_grid(parameters)
    store_to_output(time_grid[0], positions, velocities, accelerations, state_vectors, parameters)
    for time in time_grid[1:]:
        positions, velocities, accelerations = propagate_lattice(time, positions, velocities, accelerations, state_vectors, parameters)
        state_vectors = propagate_electrons(time, positions, state_vectors, parameters)
        store_to_output(positions, velocities, accelerations, state_vectors, parameters)
    # Finalize the output
    finalize_output(parameters)
    # Return last 
    return positions, velocities, accelerations, state_vectors

###############################################################
#       Leandro Manuel Arancibia & Andrés Ignacio Bertoni     #
# (leandro.arancibia9@gmail.com)   (andresibertoni@gmail.com) #
###############################################################
