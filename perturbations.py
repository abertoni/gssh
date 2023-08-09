#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

def get_vector_potential(time, parameters):
    t_grid, A_t = parameters["external_perturbation"]
    vector_potential = np.interp(time, t_grid, A_t)
    return vector_potential

def get_onsite_energies(time, parameters):
    try:
        t_grid, onsites_t = parameters["onsite_perturbation"]
        onsite_energies = np.interp(time, t_grid, onsites_t)
    except: onsite_energies = parameters["onsite_perturbation"]
    return onsite_energies

###############################################################
#       Leandro Manuel Arancibia & Andrés Ignacio Bertoni     #
# (leandro.arancibia9@gmail.com)   (andresibertoni@gmail.com) #
###############################################################
