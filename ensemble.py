#!/usr/bin/env python
# -*- coding: utf-8 -*-

from .lattice_optimization import lattice_optimization
from .output import to_dictionary, initialize_output, store_to_output, finalize_output

def static_systematic_explore(initial_positions, changing_parameter_name, changing_parameter_values, parameters):
    parameters["initial_positions"] = initial_positions
    initialize_output(changing_parameter_name, changing_parameter_values, parameters)
    for param_value in changing_parameter_values:
        print(f">> Optimizing with {changing_parameter_name} = '{param_value}' ...", end = "\t")
        parameters[changing_parameter_name] = param_value
        opt_positions, opt_energies, opt_states, opt_result = lattice_optimization(initial_positions, parameters)
        to_store = to_dictionary(positions=opt_positions, state_energies=opt_energies, state_vectors=opt_states, is_optimized=opt_result)
        store_to_output(to_store, parameters)
    finalize_output(parameters)

###############################################################
#       Leandro Manuel Arancibia & Andrés Ignacio Bertoni     #
# (leandro.arancibia9@gmail.com)   (andresibertoni@gmail.com) #
###############################################################
