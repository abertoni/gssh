#!/usr/bin/env python
# -*- coding: utf-8 -*-

from .forces import compute_forces, compute_lattice_shift, steepest_descent_step
from .electronic import get_electronic_solutions
from .lattice_tools import get_neighbour_matrix, check_lattice_minimum
from .occupations import compute_occupations

def update_lattice(positions, state_vectors, occupations, parameters):
    lat_opt_mode = parameters["lattice_optimization_mode"].lower()
    # Via numerical force computation
    if "numerical" in lat_opt_mode:
        forces = compute_forces(positions, state_vectors, occupations, parameters)
        positions_shift = compute_lattice_shift(positions, forces, parameters)
    # Via Helmann-Feynman analytic solution
    elif "analytical" in lat_opt_mode:
        positions_shift = steepest_descent_step(positions, state_vectors, occupations, parameters)
    return positions_shift

def lattice_optimization(positions, parameters):
    # Optimization is a recursive procedure
    max_steps = parameters["maximum_lattice_optimization_steps"]
    is_optimized = False
    for idx in range(max_steps):
        # Compute shift of lattice towards minimum
        state_energies, state_vectors = get_electronic_solutions(positions, parameters)
        if idx == 0: occupations = compute_occupations(state_energies, parameters)
        positions_shift = update_lattice(positions, state_vectors, occupations, parameters)
        # Checks end criteria
        is_optimized = check_lattice_minimum(positions_shift, parameters)
        if is_optimized: break
        else: positions += positions_shift
    # Check if optimization is done
    if is_optimized: print(f"Optimization finished after {idx+1}!")
    else: print(f"The lattice is not yet optimized after {max_steps}!")
    # Returns positions and electronic states (+ energies)
    return positions, state_energies, state_vectors

###############################################################
#       Leandro Manuel Arancibia & Andrés Ignacio Bertoni     #
# (leandro.arancibia9@gmail.com)   (andresibertoni@gmail.com) #
###############################################################
