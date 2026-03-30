#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import h5py
import numpy as np

def to_dictionary(**kwargs):
    return kwargs

def initialize_output(changing_parameter_name, changing_parameter_values, parameters):
    # Deals with defaults
    if "output_folder" not in parameters: parameters["output_folder"] = "."
    if "output_filename" not in parameters:
        random_tag = "".join(np.random.choice(list("0123456789abcdef"),8))
        parameters["output_filename"] = f"output_{changing_parameter_name}_{random_tag}.hdf5"
        print(f"No file name has been specified for the output. Saving your output into '{parameters['output_filename']}'...")
    # Initialize datastructures in the HDF5 format
    output_path = os.path.join(parameters["output_folder"], parameters["output_filename"])
    output = h5py.File(output_path, "w")
    # Updates parameters dictionary with output grid
    # (for output datapoints only)
    parameters["changing_parameter_name"] = changing_parameter_name
    parameters["changing_parameter_values"] = np.array(changing_parameter_values)
    # Stores fixed parameters as metadata
    fixed_parameters = parameters.copy()
    try: del fixed_parameters[changing_parameter_name]
    except: pass
    output.attrs.update(fixed_parameters)
    # General
    n_frames = len(changing_parameter_values)
    n_sites = parameters["number_of_sites"]
    # Optimization result
    if "optimization_result" in parameters["output_to_store"]:
        output.create_dataset("optimization_result", shape=(n_frames,))
    # Positions
    if "lattice_positions" in parameters["output_to_store"]:
        positions_shape = (n_frames, n_sites)
        output.create_dataset("lattice_positions", shape=positions_shape)
    # Velocities
    if "lattice_velocities" in parameters["output_to_store"]:
        velocities_shape = (n_frames, n_sites)
        output.create_dataset("lattice_velocities", shape=velocities_shape)
    # Accelerations
    if "lattice_accelerations" in parameters["output_to_store"]:
        accelerations_shape = (n_frames, n_sites)
        output.create_dataset("lattice_accelerations", shape=accelerations_shape)
    # State energies
    if "electronic_energies" in parameters["output_to_store"]:
        energies_shape = (n_frames, n_sites)
        output.create_dataset("electronic_energies", shape=energies_shape)
    # State vectors
    if "electronic_states" in parameters["output_to_store"]:
        states_shape = (n_frames, n_sites, n_sites)
        output.create_dataset("electronic_states", shape=states_shape, dtype="complex128")
    # Stores the hdf5 file handle in the parameters dictionary
    parameters["output_handle"] = output

def store_to_output(frame_idx, to_store, parameters):
    # Get index for given frame
    ch_param_name = parameters["changing_parameter_name"]
    ch_param_val = parameters[ch_param_name]
    ch_param_values = parameters["changing_parameter_values"]
    # Stores data using handle
    output = parameters["output_handle"]
    # Optimization result
    if "optimization_result" in parameters["output_to_store"]:
        output["optimization_result"][frame_idx] = to_store["is_optimized"]
    # Positions
    if "lattice_positions" in parameters["output_to_store"]:
        output["lattice_positions"][frame_idx] = to_store["positions"]
    # Velocities
    if "lattice_velocities" in parameters["output_to_store"]:
        output["lattice_velocities"][frame_idx] = to_store["velocities"]
    # Accelerations
    if "lattice_accelerations" in parameters["output_to_store"]:
        output["lattice_accelerations"][frame_idx] = to_store["accelerations"]
    # State energies
    if "electronic_energies" in parameters["output_to_store"]:
        output["electronic_energies"][frame_idx] = to_store["state_energies"]
    # State vectors
    if "electronic_states" in parameters["output_to_store"]:
        output["electronic_states"][frame_idx] = to_store["state_vectors"]

def finalize_output(parameters):
    parameters["output_handle"].close()
    del parameters["output_handle"]

def load_output(output_path):
    loaded_file = h5py.File(output_path, mode="r")
    output_data = {data_key:loaded_file[data_key] for data_key in loaded_file.keys()}
    parameters = dict(loaded_file.attrs.items())
    parameters["output_handle"] = loaded_file
    return output_data, parameters

###############################################################
#       Leandro Manuel Arancibia & Andrés Ignacio Bertoni     #
# (leandro.arancibia9@gmail.com)   (andresibertoni@gmail.com) #
###############################################################
