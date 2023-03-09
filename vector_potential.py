#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

# This is experimental (not tested)

def constant_field(time, phonon_frequency, lattice_param):
    """(External electric field as vector potential)."""
	def _constant_field(time, w_0=phonon_frequency, a=lattice_param):
		# phonon_frequency [1/s]
		# lattice_param [Å]
		# E_0 : external electric field strength
		E_0 = spc.hbar * w_0 / (spc.e * a) # J/(C*Å) = V/Å
		# h_bar [J.s] | e [C]
		return E_0 * time
    return _constant_field

##############################
#   Andrés Ignacio Bertoni   #
# (andresibertoni@gmail.com) #
##############################
