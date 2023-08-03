#!/usr/bin/env python
# -*- coding: utf-8 -*-

def propagate_lattice(time, positions, velocities, accelerations, state_vectors, occupations, parameters):
    latt_prop_mode = parameters["lattice_propagator"].lower()
    if "verlet" in latt_prop_mode:
        positions, velocities, accelerations = velocity_verlet_propagator(time, positions, velocities, accelerations, state_vectors, occupations, parameters)
    return positions, velocities, accelerations

def compute_accelerations(time, positions, velocities, state_vectors, occupations, parameters):    
    masses = parameters["lattice_site_masses"]
    # Compute accelerations
    forces = compute_forces(time, positions, velocities, state_vectors, occupations, parameters)
    accelerations = forces/masses
    return accelerations

def half_update_velocities(velocities, accelerations, parameters):
    time_step = parameters["dynamics_time_step"]
    # Half-step velocities
    new_velocities = velocities + 0.5 * accelerations * (time_step**2)
    # Returns velocities
    return new_velocities

def update_positions(positions, velocities, parameters):
    time_step = parameters["dynamics_time_step"]
    new_positions = positions + velocities * time_step
    return new_positions

def velocity_verlet_propagator(time, positions, velocities, accelerations, state_vectors, occupations, parameters):
    # Compute first-half velocities
    velocities = half_update_velocities(velocities, accelerations, parameters)
    # Update positions
    positions = update_positions(positions, velocities, parameters)
    # Compute new accelerations
    accelerations = compute_accelerations(time, positions, velocities, state_vectors, occupations, parameters)
    # Compute second-half velocities
    velocities = half_update_velocities(velocities, accelerations, parameters)
    # Return positions and velocities
    return positions, velocities, accelerations

###############################################################
#       Leandro Manuel Arancibia & Andrés Ignacio Bertoni     #
# (leandro.arancibia9@gmail.com)   (andresibertoni@gmail.com) #
###############################################################
