#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

def initialize_positions(parameters):
    n_sites = parameters["number_of_sites"]
    a = parameters["lattice_parameter"]
    positions = a * np.arange(M)
    if "initialize_positions_noise" in parameters:
        noise = parameters["initialize_positions_noise"]
        if "random_seed" in parameters: np.random.seed(seed=parameters["random_seed"])
        positions += (noise/100) * np.random.uniform(-1, 1, n_sites)
    return positions

def get_neighbour_distances(positions, shift, parameters):
    n_sites = parameters["number_of_sites"]
    is_periodic = parameters["periodic_boundaries"]
    dr_n = np.roll(r_n, -(shift)) - r_n
    if is_periodic:
        a = parameters["lattice_parameter"]
        dr_n[-1] += n_sites * a
    else: dr_n = dr_n[:-1]
    return dr_n

def check_lattice_minimum(phaselinks_shift, parameters):
    tolerance = parameters["lattice_optimization_tolerance"]
    rmse = np.sqrt(np.mean(phaselinks_shift**2))
    is_optimized = (rmse <= tolerance)
    return is_optimized

###############################################################
#       Leandro Manuel Arancibia & Andrés Ignacio Bertoni     #
# (leandro.arancibia9@gmail.com)   (andresibertoni@gmail.com) #
###############################################################
