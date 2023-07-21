#!/usr/bin/env python
# -*- coding: utf-8 -*-

def propagate_electrons(time, positions, state_vectors, parameters):
    elec_prop_mode = parameters["electronic_propagator"].lower()
    if "crank" in elec_prop_mode:
        state_vectors = crank_nicolson_propagator(time, positions, state_vectors, parameters)
    return state_vectors

def crank_nicolson_propagator(time, positions, state_vectors, parameters):
    # Scherer Computational Physcis (2017) p.528
    dt = parameters['dynamics_time_step']
    N = parameters["number_of_sites"]
    H = build_hamiltonian(phaselinks, parameters)
    chi = np.linalg.solve(np.identity(N) + (1j*dt/2) * H, state_vectors)
    new_state_vectors = 2 * chi - state_vectors
    return new_state_vectors

###############################################################
#       Leandro Manuel Arancibia & Andrés Ignacio Bertoni     #
# (leandro.arancibia9@gmail.com)   (andresibertoni@gmail.com) #
###############################################################
