#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from .electronic import neighbour_sites_projection

def steepest_descent_step(phaselinks, state_vectors, occupations, parameters):
    # Derived analytically using the Hellmann–Feynman theorem
    # (only first order hopping is implemented)
    hopping_params = parameters["hopping_parameters"]
    if len(hopping_params) > 2: raise NotImplementedError()
    A = hopping_params[1] # first-order coupling
    # (only harmonic oscillation is implemented)
    oscillator_params = parameters["oscillator_parameters"]
    if len(oscillator_params) > 1: raise NotImplementedError()
    K = oscillator_params[0] # harmonic constant
    # Compute new phaselinks
    new_phaselinks = np.zeros(phaselinks.size)
    for jdx,spin in enumerate(["up","down"]):
        states_dot_product = neighbour_sites_projection(n_shift, states_vectors, occupations[:,jdx], parameters)
        new_phaselinks += - A/K * states_dot_product
    # Apply correction for periodic boundary conditions
    is_periodic = parameters["periodic_boundaries"]
    if is_periodic: new_phaselinks -= np.mean(new_phaselinks)
    elif "open_boundary_stretching" in parameters.keys():
        G = parameters["open_boundary_stretching"]
        new_phaselinks += G/K
    # Compute the shift in phaselinks
    phaselinks_shift = new_phaselinks - phaselinks
    return phaselinks_shift

###############################################################
#       Leandro Manuel Arancibia & Andrés Ignacio Bertoni     #
# (leandro.arancibia9@gmail.com)   (andresibertoni@gmail.com) #
###############################################################
