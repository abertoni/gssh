#!/usr/bin/env python
# -*- coding: utf-8 -*-

from .forces import compute_forces, compute_lattice_shift, steepest_descent_step

def update_lattice(phaselinks, state_vectors, occupations, parameters):
    lat_opt_mode = parameters["lattice_optimization_mode"].lower()
    # Via numerical force computation
    if "numerical" in lat_opt_mode:
        forces = compute_forces(phaselinks, state_vectors, occupations, parameters)
        phaselinks_shift = compute_lattice_shift(phaselinks, forces, parameters)
    # Via Helmann-Feynman analytic solution
    elif "analytical" in lat_opt_mode:
        phaselinks_shift = steepest_descent_step(phaselinks, state_vectors, occupations, parameters)
    return phaselinks_shift

###############################################################
#       Leandro Manuel Arancibia & Andrés Ignacio Bertoni     #
# (leandro.arancibia9@gmail.com)   (andresibertoni@gmail.com) #
###############################################################
