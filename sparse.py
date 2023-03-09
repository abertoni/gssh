#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import scipy.constants as spc
import scipy.sparse as sp_sparse
import scipy.sparse.linalg as sp_sparse_linalg

def sp_roll(matrix, shift):
    M = sp_sparse.csr_matrix(matrix)
    s = shift
    rolled_M = sp_sparse.hstack((M[:,s:], M[:,:s]), format='csr')
    return rolled_M
    
def build_SSH_Hamiltonian(site_positions, lattice_param, hopping_params, vector_potential, sho_force_constant=None, site_masses=None, time=0, mode="all"):
    r_n = site_positions
    a = lattice_param
    y_n = delta_displacement(r_n, a)
    # H_{SSH} = H_{el} + H_{lat}
    # H_{el} : Electronic Hamiltonian
    # H_{el} = sum_s sum_{n=1}^N |Ψ_{n,s}> (-[t_0 - α . y_n] . e^{-i.A(t).(e.a/h.c).}) <Ψ_{n+1,s}|
    # A(t) : Vector potential --> E.g.: A(t) = -E_0 . c . t 
    # REVISAR UNIDADES DE POTENCIAL VECTOR A(t)
    if mode == "all" or mode == "electronic":
        A_t = vector_potential(time)
        norm_constant = a * spc.e / (spc.hbar * spc.c)
        H_el_hop = -hopping_params[0] # -t_0 (hopping_zero_param)
        for idx,t_i in enumerate(hopping_params[1:]):
            H_el_hop += t_i * y_n**(idx+1) # idx+1 = 1 --> t_i = alpha (hopping_elecphonon_param)
        H_el_hop = H_el_hop.astype(complex) * np.exp(1j * norm_constant * A_t) # Upper hopping elements array
        H_el = sp_sparse.diags(H_el_hop) # Upper as diagonal
        H_el = sp_roll(H_el, shift=-1) # Upper only
        H_el = H_el + np.conjugate(H_el.T) # Upper + lower
    else: H_el = 0
    # H_{lat} : Lattice Hamiltonian
    # H_{lat} = (k/2) sum_{n=1}^N (y_n)^2 + (m/2) sum_{n=1}^N (u°_n)^2
    if mode == "all" or mode == "lattice":
        if None in [sho_force_constant, site_masses]:
            print("Is not possible to build the lattice contribution to the SSH Hamiltonian! (Missing arguments!)")
            H_lat = 0
        else:
            site_velocities = get_site_velocities(site_positions)
            u_dot = site_velocities
            k = sho_force_constant
            m = site_masses
            H_lat = (k/2) * y_n**2 + (m/2) * u_dot**2
            H_lat = np.sum(H_lat)
    else: H_lat = 0
    # H_{SSH} = H_{el} + H_{lat}
    H = H_el + H_lat
    return H

def diagonalize(matrix, mode="array"):
    # eigsh is an ARPACK implement. for Hermitian matrices (fastest)
    if mode == "array":
        # WARNING: might require too much RAM!
        eigvals, eigvecs = np.linalg.eigh(matrix.toarray())
    elif mode == "sparse":
        # WARNING: too slow!
        bot_N = N//2
        top_N = N-bot_N
        eigvals_small, eigvecs_small = sp_sparse_linalg.eigsh(matrix, k=bot_N, which="SM") # ARPACK for smallest
        eigvals_large, eigvecs_large = sp_sparse_linalg.eigsh(matrix, k=top_N, which="LM") # ARPACK for largest
        eigvals = np.concatenate([eigvals_small, eigvals_large])
        eigvecs = np.hstack([eigvecs_small, eigvecs_large])
    return eigvals, eigvecs

##############################
#   Andrés Ignacio Bertoni   #
# (andresibertoni@gmail.com) #
##############################
