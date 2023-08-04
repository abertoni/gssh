#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

def positions_to_phaselinks(positions, parameters):
    a = parameters["lattice_parameter"]
    n_sites = parameters["number_of_sites"]
    is_periodic = parameters["periodic_boundaries"]
    # y_n = u_{n+1} - u_{n} = Δr_{n+1} - Δr_{n}
    # y_n = r_{n+1} - r_{n} - a
    r_n = positions
    y_n = np.roll(r_n, -1) - r_n - a
    if is_periodic: y_n[-1] += n_sites * a # u_{N+1} = u_{n}
    else: y_n = y_n[:-1]
    # Return phaselinks
    phaselinks = y_n
    return phaselinks

###############################################################
#       Leandro Manuel Arancibia & Andrés Ignacio Bertoni     #
# (leandro.arancibia9@gmail.com)   (andresibertoni@gmail.com) #
###############################################################
