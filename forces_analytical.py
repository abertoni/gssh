#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

def solve_Helmann_Feynman(phaselinks, state_vectors, occupations, parameters):
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
        states_dot_product = np.sum(occupations[:,jdx] * np.conj(state_vectors) * np.roll(state_vectors, -1, axis=0), axis=1)
        states_dot_product += np.sum(occupations[:,jdx] * state_vectors * np.conj(np.roll(state_vectors, -1, axis=0)), axis=1)
        if np.linalg.norm(np.imag(states_dot_product)) > 0: print("Warning! Imaginary part of neighbor sites projection is non-zero.")
        states_dot_product = np.real(states_dot_product)[:phaselinks.size]
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
