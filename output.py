#!/usr/bin/env python
# -*- coding: utf-8 -*-

import h5py
import os
from .time_tools import get_output_time_grid

def initialize_output(parameters):
    # Initialize datastructures in the HDF5 format
    output_path = os.path.join(parameters["output_folder"], parameters["output_filename"])
    output = h5py.File(output_path, "w")
    # Time grid (for output datapoints only)
    output_time_grid = get_output_time_grid(parameters)
    output.create_dataset("output_time_grid", data=output_time_grid)
    # General parameters
    n_times = output_time_grid.size
    n_sites = parameters["number_of_sites"]
    # Positions
    if "lattice_positions" in parameters["output_to_save"]:
        positions_shape = (n_times, n_sites)
        output.create_dataset("lattice_positions", shape=positions_shape)
    # Velocities
    elif "lattice_velocities" in parameters["output_to_save"]:
        velocities_shape = (n_times, n_sites)
        output.create_dataset("lattice_velocities", shape=velocities_shape)
    # Accelerations
    elif "lattice_accelerations" in parameters["output_to_save"]:
        accelerations_shape = (n_times, n_sites)
        output.create_dataset("lattice_accelerations", shape=accelerations_shape)
    # State vectors
    elif "electronic_states" in parameters["output_to_save"]:
        states_shape = (n_times, n_sites, n_sites)
        output.create_dataset("electronic_states", shape=states_shape)
    # Stores the hdf5 file handle in the parameters dictionary
    parameters["output_handle"] = output

def store_to_output(time, positions, velocities, accelerations, state_vectors, parameters):
    # Get index for given time
    output_time_grid = get_output_time_grid(parameters)
    if time not in output_time_grid: return
    tdx = output_time_grid.tolist().index(time)
    # Stores data using handle
    output = parameters["output_handle"]
    if "lattice_positions" in parameters["output_to_save"]:
        output["lattice_positions"][tdx] = positions
    # Velocities
    elif "lattice_velocities" in parameters["output_to_save"]:
        output["lattice_velocities"][tdx] = velocities
    # Accelerations
    elif "lattice_accelerations" in parameters["output_to_save"]:
        output["lattice_accelerations"][tdx] = accelerations
    # State vectors
    elif "electronic_states" in parameters["output_to_save"]:
        output["electronic_states"][tdx] = state_vectors

def finalize_output(parameters):
    parameters["output_handle"].close()
    del parameters["output_handle"]

###############################################################
#       Leandro Manuel Arancibia & Andrés Ignacio Bertoni     #
# (leandro.arancibia9@gmail.com)   (andresibertoni@gmail.com) #
###############################################################
