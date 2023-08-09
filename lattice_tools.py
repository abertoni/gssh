#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from gssh.sparse_tools import sp_roll

def initialize_positions(parameters):
    n_sites = parameters["number_of_sites"]
    a = parameters["lattice_parameter"]
    positions = a * np.arange(n_sites)
    if "initialize_positions_noise" in parameters:
        noise = parameters["initialize_positions_noise"]
        if "random_seed" in parameters: np.random.seed(seed=parameters["random_seed"])
        positions += (noise/100) * np.random.uniform(-1, 1, n_sites)
    # (sum_n r_n = 0)
    positions += - np.mean(positions)
    return positions

def get_neighbour_matrix(neighbours, parameters):
    n_sites = parameters["number_of_sites"]
    a = parameters["lattice_parameter"]
    is_periodic = parameters["periodic_boundaries"]
    N = np.zeros((n_sites, n_sites))
    for nidx in neighbours:
        n = np.ones(n_sites)
        if not is_periodic: n[::np.sign(nidx)][-abs(nidx):] = 0
        N += sp_roll(np.diag(n), -nidx).toarray()
    return N

def get_paired_matrix(neighbours, parameters):
    N = get_neighbour_matrix(neighbours, parameters)
    paired_matrix = N - np.diag(np.sum(N, axis=-1))
    return paired_matrix

def get_sum_relative_positions(positions, neighbours, parameters):
    n_sites = parameters["number_of_sites"]
    a = parameters["lattice_parameter"]
    is_periodic = parameters["periodic_boundaries"]
    P = get_paired_matrix(neighbours, parameters)
    sum_rel_positions = P @ positions
    if is_periodic:
        for neig_idx in neighbours:
            correction = np.sign(neig_idx) * a * n_sites
            sum_rel_positions[::np.sign(neig_idx)][-abs(neig_idx):] += correction
    return sum_rel_positions

def positions_to_bondlengths(positions, parameters):
    is_periodic = parameters["periodic_boundaries"]
    neighbours = [+1]
    bondlengths = np.abs(get_sum_relative_positions(positions, neighbours, parameters))
    if not is_periodic: bondlengths = bondlengths[:-1]
    return bondlengths

def check_lattice_minimum(old_positions, new_positions, parameters):
    tolerance = parameters["lattice_optimization_tolerance"]
    is_periodic = parameters["periodic_boundaries"]
    old_bondlengths = positions_to_bondlengths(old_positions, parameters)
    new_bondlengths = positions_to_bondlengths(new_positions, parameters)
    if is_periodic: new_bondlengths = new_bondlengths[::-1]
    bondlength_diffs = new_bondlengths - old_bondlengths
    rmse = np.sqrt(np.mean(bondlength_diffs**2))
    is_optimized = (rmse <= tolerance)
    # If necessary, checks for two alternating minima   
    if not is_optimized:
        if "__old_old_positions" in parameters.keys():
            old_old_positions = parameters["__old_old_positions"]
            old_old_bondlengths = positions_to_bondlengths(old_old_positions, parameters)
            bondlength_diffs = new_bondlengths - old_old_bondlengths
            rmse = np.sqrt(np.mean(bondlength_diffs**2))
            is_optimized = (rmse <= tolerance)
        if is_optimized:
            print("Found second minimum!\n(Stored in parameters as 'alt_opt_positions'!)")
            parameters["alt_opt_positions"] = new_positions
        else: parameters["__old_old_positions"] = old_positions
    if is_optimized: del parameters["__old_old_positions"]
    return is_optimized

###############################################################
#       Leandro Manuel Arancibia & Andrés Ignacio Bertoni     #
# (leandro.arancibia9@gmail.com)   (andresibertoni@gmail.com) #
###############################################################
