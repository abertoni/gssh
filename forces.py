#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from .lattice_tools import get_paired_matrix, get_sum_relative_positions
from .electronic import get_neighbour_sites_projection, sum_hermitian_conjugate
from .sparse_tools import sp_roll

# ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ #
# For general use and dynamics  #
# ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ #

def get_lattice_forces(positions, velocities, parameters):
    """Forces from classical potential for inter-nuclear interactions."""
    # (only harmonic oscillation is implemented)
    oscillator_params = parameters["oscillator_parameters"]
    if len(oscillator_params) > 1: raise NotImplementedError()
    K = oscillator_params[0] # harmonic constant
    # Compute forces from other nuclei
    neighbours = [-1, +1] # (hardcoded)
    sum_rel_positions = get_sum_relative_positions(positions, neighbours, parameters)
    lattice_forces = - K * sum_rel_positions
    return lattice_forces

def get_lattice_matrix(neighbours, parameters):
    # (only harmonic oscillation is implemented)
    oscillator_params = parameters["oscillator_parameters"]
    if len(oscillator_params) > 1: raise NotImplementedError()
    K = oscillator_params[0] # harmonic constant
    # Lattice matrix is partially used to compute lattice forces
    P = get_paired_matrix(neighbours, parameters)
    lattice_matrix = - K * P
    return lattice_matrix

def get_periodic_boundary_forces(neighbours, parameters):
    n_sites = parameters["number_of_sites"]
    a = parameters["lattice_parameter"]
    # (only harmonic oscillation is implemented)
    oscillator_params = parameters["oscillator_parameters"]
    if len(oscillator_params) > 1: raise NotImplementedError()
    K = oscillator_params[0] # harmonic constant
    # Initializes forces array
    pbcorr_forces = np.zeros(n_sites)
    # Periodic boundary corrections
    for neig_idx in neighbours:
        position_correction = np.sign(neig_idx) * a * n_sites
        force_correction = - K * position_correction
        pbcorr_forces[::np.sign(neig_idx)][-abs(neig_idx):] += force_correction
    return pbcorr_forces

def get_open_boundary_forces(parameters):
    n_sites = parameters["number_of_sites"]
    a = parameters["lattice_parameter"]
    # (only harmonic oscillation is implemented)
    oscillator_params = parameters["oscillator_parameters"]
    if len(oscillator_params) > 1: raise NotImplementedError()
    K = oscillator_params[0] # harmonic constant
    # Initializes forces array
    open_boundary_forces = np.zeros(n_sites)
    # Open boundary corrections
    open_boundary_forces[(0,-1)] = +K * a
        # Open boundary stretching
    if "open_boundary_stretching" in parameters.keys():
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
    n_sites = parameters["number_of_sites"]
    is_periodic = parameters["periodic_boundaries"]
    electronic_forces = np.zeros(n_sites)
    for jdx,spin in enumerate(["up","down"]):
        states_dotprod_left = get_neighbour_sites_projection(-1, state_vectors, occupations[:,jdx], parameters)
        states_dotprod_right = get_neighbour_sites_projection(+1, state_vectors, occupations[:,jdx], parameters)
        if not is_periodic:
            states_dotprod_left[0] = 0
            states_dotprod_right[-1] = 0
        spin_forces = - A * ext_pertubation_factor * (states_dotprod_left - states_dotprod_right)
        spin_forces = sum_hermitian_conjugate(spin_forces)
        electronic_forces += spin_forces
    return electronic_forces

def get_forces(time, positions, velocities, state_vectors, occupations, parameters):
    forces = get_lattice_forces(positions, velocities, parameters)
    forces += get_open_boundary_forces(parameters)
    forces += get_electronic_forces(time, state_vectors, occupations, parameters)
    return forces

def analytical_relaxation(positions, state_vectors, occupations, parameters):
    n_sites = parameters["number_of_sites"]
    a = parameters["lattice_parameter"]
    is_periodic = parameters["periodic_boundaries"]
    # Derived analytically using the Hellmann–Feynman theorem
    # Lattice matrix
    neighbours = [-1, +1] # (hardcoded)
    lattice_matrix = get_lattice_matrix(neighbours, parameters)
    # Electronic forces
    time = 0 # (this function is for optimization only)
    electronic_forces = get_electronic_forces(time, state_vectors, occupations, parameters)
    position_independent_forces = electronic_forces.copy()
    # Open boundary correction forces
    # (stretching forces are needed on the sides to prevent implosion/condensate)
    if not is_periodic:
        open_boundary_forces = get_open_boundary_forces(parameters)
        position_independent_forces += open_boundary_forces
    # Periodic boundary correction forces
    # (lattice matrix does not include periodic boundary corrections)
    if is_periodic:
        periodic_boundary_forces = get_periodic_boundary_forces(neighbours, parameters)
        position_independent_forces += periodic_boundary_forces
    # New positions computed by solving system of linear equations
    # (solve using the Moore-Penrose pseudoinverse, via SVD)
    new_positions = np.linalg.pinv(-lattice_matrix) @ position_independent_forces
    # (generates translated solutions)
    # (center of geometry is translated to origin)
    new_positions += - np.mean(new_positions)
    # (sum_n r_n = 0)
    return new_positions

###############################################################
#       Leandro Manuel Arancibia & Andrés Ignacio Bertoni     #
# (leandro.arancibia9@gmail.com)   (andresibertoni@gmail.com) #
###############################################################
