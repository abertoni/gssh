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
    positions -= np.mean(positions)
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

def check_lattice_minimum(positions_shift, parameters):
    tolerance = parameters["lattice_optimization_tolerance"]
    rmse = np.sqrt(np.mean(positions_shift**2))
    is_optimized = (rmse <= tolerance)
    return is_optimized

###############################################################
#       Leandro Manuel Arancibia & Andrés Ignacio Bertoni     #
# (leandro.arancibia9@gmail.com)   (andresibertoni@gmail.com) #
###############################################################
