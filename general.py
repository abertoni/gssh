#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

def delta_displacement(site_positions, lattice_param):
    # y_n = Δr_n - Δr_0
    # Δr_0 = r_{n+1}_0 - r_n_0 = a : lattice parameter (equillibrium reference)
    # Δr_n = r_{n+1} - r_n
    # r_n = n . a + u_n
    # y_n = u_{n+1} - u_n
    r_n = site_positions
    a = lattice_param
    y_n = np.roll(r_n, -1) - r_n - a
    # Periodic Boundary Condition (PBC)
    # u_{N+1} = u_{n}
    N = len(r_n)
    y_n[-1] += N*a
    return y_n

def init_site_positions(N_sites, lattice_param, mode="equillibrium"):
    a = lattice_param
    if mode in ["equillibrium", "eq"]: r_n = a * np.arange(N_sites)
    elif mode.startswith("noise_"):
        noise_percent = float(mode.split("_")[-1])/100
        r_n = a * (np.arange(N_sites) + noise_percent * np.random.uniform(-1,1,N_sites))
    return r_n

def get_states(Hamiltonian):
    H = Hamiltonian
    eigvals, eigvecs = diagonalize(H)
    order = np.argsort(eigvals)
    energies = eigvals[order]
    states = eigvecs[:,order] / np.linalg.norm(eigvecs[:,order], axis=-1)
    return energies, states<

##############################
#   Andrés Ignacio Bertoni   #
# (andresibertoni@gmail.com) #
##############################
