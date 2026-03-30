#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from .lattice_tools import check_lattice_minimum
from .hamiltonian import build_hamiltonian
from .postprocessing import positions_to_phaselinks
def propagate_electrons(time, positions, state_vectors, occupations, hamiltonian, parameters):
    elec_prop_mode = parameters["electronic_propagator"].lower()
    if "crank" in elec_prop_mode:
        state_vectors = crank_nicolson_propagator(time, positions, state_vectors, occupations, hamiltonian, parameters)
    return state_vectors

def crank_nicolson_propagator(time, propagated_positions, state_vectors, occupations, hamiltonian, parameters):
    dt = parameters['dynamics_time_step']
    N = parameters["number_of_sites"]
    # Forward half step (t+dt/2)
    state_vectors_half = (np.identity(N) - (1j*dt/2) * hamiltonian) @ state_vectors
    # Build future Hamitonian (t+dt)
    new_hamiltonian = build_hamiltonian(time, propagated_positions, parameters)
    # Backward half step propagator from future (t+dt-dt/2 --> t+dt/2)
    propagator_half = (np.identity(N) + (1j*dt/2) * new_hamiltonian)
    # Solve system of linear equations to get future states
    # A.x = b : x <-- solve(A,b)
    new_state_vectors = np.linalg.solve(propagator_half, state_vectors_half)
    return new_state_vectors, new_hamiltonian

###############################################################
#       Leandro Manuel Arancibia & Andrés Ignacio Bertoni     #
# (leandro.arancibia9@gmail.com)   (andresibertoni@gmail.com) #
###############################################################
