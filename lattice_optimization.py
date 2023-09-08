#!/usr/bin/env python
# -*- coding: utf-8 -*-

from .forces import get_forces, analytical_relaxation
from .electronic import get_electronic_solutions
from .lattice_tools import get_neighbour_matrix, check_lattice_minimum
from .occupations import get_occupations

def update_lattice(positions, state_vectors, occupations, parameters):
    lat_opt_mode = parameters["lattice_optimization_mode"].lower()
    # Via numerical force computation
    if "numerical" in lat_opt_mode:
        raise NotImplementedError()
    # Via Helmann-Feynman analytic solution
    elif "analytical" in lat_opt_mode:
        new_positions = analytical_relaxation(positions, state_vectors, occupations, parameters)
    return new_positions

def lattice_optimization(positions, parameters):
    # Avoids overwritting input positions
    positions = positions.copy()
    # Optimization is a recursive procedure
    max_steps = parameters["maximum_lattice_optimization_steps"]
    is_optimized = False
    for idx in range(max_steps):
        # Compute shift of lattice towards minimum
        time = 0 # (optimization --> ground state)
        state_energies, state_vectors = get_electronic_solutions(time, positions, parameters)
        if idx == 0: # (initial occupations should not change --> computed once)
            occupations = get_occupations(state_energies, parameters)
        new_positions = update_lattice(positions, state_vectors, occupations, parameters)
        # Checks end criteria
        is_optimized = check_lattice_minimum(positions, new_positions, parameters)
        if is_optimized: break
        else: positions = new_positions
    # Check if optimization is done
    if is_optimized: print(f"Optimization finished after {idx+1}!")
    else: print(f"The lattice is not yet optimized after {max_steps}!")
    # Returns positions and electronic states (+ energies)
    return positions, state_energies, state_vectors, is_optimized

###############################################################
#       Leandro Manuel Arancibia & Andrés Ignacio Bertoni     #
# (leandro.arancibia9@gmail.com)   (andresibertoni@gmail.com) #
###############################################################
