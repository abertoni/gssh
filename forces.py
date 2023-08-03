#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from .lattice_tools import get_paired_matrix
from .electronic import get_neighbour_sites_projection, sum_hermitian_conjugate
from .sparse_tools import sp_roll

def compute_lattice_shift(positions, forces, parameters):
    raise NotImplementedError()
    #return positions_shift

def get_lattice_matrix(neighbours, parameters):
    # (only harmonic oscillation is implemented)
    oscillator_params = parameters["oscillator_parameters"]
    if len(oscillator_params) > 1: raise NotImplementedError()
    K = oscillator_params[0] # harmonic constant
    # Lattice matrix is partially used to compute lattice forces
    neighbours = [-1, +1]
    P = get_paired_matrix(neighbours, parameters)
    lattice_matrix = - K * P
    return lattice_matrix

def get_lattice_forces(positions, velocities, parrameters):
    """Forces from classical potential for inter-nuclear interactions."""
    # (only harmonic oscillation is implemented)
    oscillator_params = parameters["oscillator_parameters"]
    if len(oscillator_params) > 1: raise NotImplementedError()
    K = oscillator_params[0] # harmonic constant
    # Compute forces from other nuclei
    neighbours = [-1, +1]
    sum_rel_positions = get_sum_relative_positions(positions, neighbours, parameters)
    lattice_forces = - K * sum_rel_positions
    return lattice_forces

def open_boundary_forces(parameters):
    n_sites = parameters["number_of_sites"]
    is_periodic = parameters["periodic_boundaries"]
    # Open boundary stretching
    open_boundary_forces = np.zeros(n_sites)
    if not is_periodic and "open_boundary_stretching" in parameters.keys():
        G = parameters["open_boundary_stretching"]
        open_boundary_forces[0] = -G
        open_boundary_forces[-1] = +G
    return open_boundary_forces

def get_electronic_forces(time, state_vectors, occupations, parameters):
    """Mean field forces from electrons."""
    # (only first order hopping is implemented)
    hopping_params = parameters["hopping_parameters"]
    if len(hopping_params) > 2: raise NotImplementedError()
    A = hopping_params[1] # first-order coupling
    # Includes coupling with external vector potential
    # (related to derivative of external electric field)
    if "external_perturbation" in parameters:
        vector_potential = get_vector_potential(time, parameters)
        ext_pertubation_factor = np.exp(+1j * a * vector_potential)
    else: ext_pertubation_factor = 1
    # Compute forces from electrons
    electronic_forces = np.zeros(n_sites)
    for jdx,spin in enumerate(["up","down"]):
        states_dotprod_left = get_neighbour_sites_projection(states_vectors, occupations[:,jdx], -1, parameters)
        states_dotprod_right = get_neighbour_sites_projection(states_vectors, occupations[:,jdx], +1, parameters)
        if is_periodic:
            states_dotprod_left[0] = 0
            states_dotprod_right[-1] = 0
        spin_forces += - A * ext_pertubation_factor * (states_dotprod_left - states_dotprod_right)
        spin_forces = sum_hermitian_conjugate(spin_forces)
        electronic_forces += spin_forces
    return electronic_forces

def compute_forces(time, positions, velocities, state_vectors, occupations, parameters):
    forces = lattice_forces(positions, velocities, parameters)
    forces += open_boundary_forces(parameters)
    forces += electronic_forces(time, state_vectors, occupations, parameters)
    return forces

def steepest_descent_step(positions, state_vectors, occupations, parameters):
    n_sites = parameters["number_of_sites"]
    is_periodic = parameters["periodic_boundaries"]
    # Derived analytically using the Hellmann–Feynman theorem
    # Lattice matrix
    neighbours = [-1, +1]
    lattice_matrix = get_lattice_matrix(neighbours, parameters)
    # Electronic forces
    electronic_forces = get_electronic_forces(state_vectors, occupations, parameters)
    # Open boundary correction forces
    open_boundary_forces = open_boundary_forces(parameters)
    # New positions computed by solving system of linear equations
    new_positions = np.linalg.solve(-lattice_matrix, electronic_forces + open_boundary_forces)
    # Compute the shift in positions
    positions_shift = new_positions - positions
    return positions_shift

###############################################################
#       Leandro Manuel Arancibia & Andrés Ignacio Bertoni     #
# (leandro.arancibia9@gmail.com)   (andresibertoni@gmail.com) #
###############################################################
