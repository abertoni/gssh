#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

def get_time_grid(parameters):
    # Checks if the time grid was already given
    if "dynamics_time_grid" in parameters.keys():
        time_grid = parameters["dynamics_time_grid"]
    # Creates one if not
    else:
        if "dynamics_initial_time" not in parameters.keys():
            parameters["dynamics_initial_time"] = 0
        t0 = parameters["dynamics_initial_time"]
        T = parameters["dynamics_total_time"]
        dt = parameters["dynamics_time_step"]
        time_grid = np.arange(t0, T+dt, dt)
        # Corrects the save_every parameter to match the timestep
        save_dt = parameters["output_save_every"]
        save_dt = dt * (save_dt/dt)
        parameters["output_save_every"] = save_dt
        # Stores the time grid as parameter
        parameters["dynamics_time_grid"] = time_grid
    # Returns the time grid
    return time_grid

def get_output_time_grid(parameters):
    # Checks if the time grid was already given
    if "output_time_grid" in parameters.keys():
        output_time_grid = parameters["output_time_grid"]
    # Creates one if not
    else:
        time_grid = get_time_grid(parameters)
        shifted_time_grid = time_grid - time_grid[0]
        save_dt = parameters["output_save_every"]
        idxs_to_save = np.where(shifted_time_grid%save_dt == 0)[0]
        output_time_grid = time_grid[idxs_to_save]
        # Stores the output time grid as parameter
        parameters["output_time_grid"] = output_time_grid
    # Returns the output time grid
    return output_time_grid

###############################################################
#       Leandro Manuel Arancibia & Andrés Ignacio Bertoni     #
# (leandro.arancibia9@gmail.com)   (andresibertoni@gmail.com) #
###############################################################
