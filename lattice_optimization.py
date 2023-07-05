#!/usr/bin/env python
# -*- coding: utf-8 -*-

from .electronic import get_electronic_solutions
from .lattice import update_lattice
from .lattice_tools import positions_to_phaselinks, phaselinks_to_positions, check_lattice_minimum
from .occupations import compute_occupations

def lattice_optimization(positions, parameters):
    # Converts to phaselinks
    phaselinks = positions_to_phaselinks(positions, parameters)
    # Optimization is a recursive procedure
    max_steps = parameters["maximum_lattice_optimization_steps"] ; is_optimized = False
    for idx in range(max_steps):
        # Compute shift of lattice towards minimum
        state_energies, state_vectors = get_electronic_solutions(phaselinks, parameters)
        occupations = compute_occupations(state_energies, parameters)
        phaselinks_shift = update_lattice(phaselinks, state_vectors, occupations, parameters)
        # Updates the lattice
        phaselinks += phaselinks_shift
        # Checks end criteria
        is_optimized = check_lattice_minimum(phaselinks_shift, parameters)
        if is_optimized: break
    # Check if optimization is done
    if is_optimized: print(f"Optimization finished after {idx+1}!")
    else: print(f"The lattice is not yet optimized after {max_steps}!")
    # Restore positions
    optimized_positions = phaselinks_to_positions(phaselinks, parameters)
    # Returns positions and electronic states (+ energies)
    return optimized_positions, state_energies, state_vectors

###############################################################
#       Leandro Manuel Arancibia & Andrés Ignacio Bertoni     #
# (leandro.arancibia9@gmail.com)   (andresibertoni@gmail.com) #
###############################################################
