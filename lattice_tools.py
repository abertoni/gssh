#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

def initialize_positions(parameters):
    M = parameters["number_of_sites"]
    a = parameters["lattice_parameter"]
    positions = a * np.arange(M)
    if "initialize_positions_noise" in parameters:
        noise = parameters["initialize_positions_noise"]
        if "random_seed" in parameters: np.random.seed(seed=parameters["random_seed"])
        positions += (noise/100) * np.random.uniform(-1,1,M)
    return positions

def positions_to_phaselinks(positions, parameters):
    # y_n = u_{n+1} - u_{n} = Δr_{n+1} - Δr_{n}
    # y_n = r_{n+1} - r_{n} - a
    r_n = positions
    a = parameters["lattice_parameter"]
    y_n = np.roll(r_n, -1) - r_n - a
    is_periodic = parameters["periodic_boundaries"]
    if is_periodic: y_n[-1] += len(r_n) * a # u_{N+1} = u_{n}
    else: y_n = y_n[:-1]
    # Return phaselinks
    phaselinks = y_n
    return phaselinks

def phaselinks_to_positions(phaselinks, parameters):
    # r_{n+1} = r_{n} + y_n + a
    # r_{n} = 0 for n = 0
    y_n = phaselinks
    a = parameters["lattice_parameter"]
    is_periodic = parameters["periodic_boundaries"]
    M = len(y_n)
    if not is_periodic: M -= 1
    r_n = np.zeros(M)
    r_n[1:] = np.cumsum(y_n + a)[:M-1]
    # Return positions
    positions = r_n
    return positions

def check_lattice_minimum(phaselinks_shift, parameters):
    tolerance = parameters["lattice_optimization_tolerance"]
    rmse = np.sqrt(np.mean(phaselinks_shift**2))
    is_optimized = (rmse <= tolerance)
    return is_optimized

###############################################################
#       Leandro Manuel Arancibia & Andrés Ignacio Bertoni     #
# (leandro.arancibia9@gmail.com)   (andresibertoni@gmail.com) #
###############################################################
