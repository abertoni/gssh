# SSH-based Self-Consistent Tight-Binding Code

This repository contains the code used to obtain the results published in:

https://journals.aps.org/prb/abstract/10.1103/PhysRevB.111.045405

## Overview

This code implements a self-consistent tight-binding scheme based on the Su-Schrieffer-Heeger (SSH) model and its generalizations. It is designed to study conducting polymers such as polyacetylene and can be extended to more complex Hamiltonians.

## Features

- Flexible initialization of the tight-binding chain geometry (atomic positions).
- User-defined parameters:
  - Hopping amplitudes
  - On-site energies
  - Nearest-neighbor interactions
- Self-consistent energy minimization using the Hellmann–Feynman theorem (see reference paper).

## Methodology

The code performs the following iterative procedure:

1. Initialize a configuration of atomic positions for the tight-binding chain.
2. Define model parameters (hopping, on-site, nearest-neighbor terms).
3. Construct the Hamiltonian.
4. Diagonalize the Hamiltonian to obtain eigenvalues and eigenfunctions.
5. Update atomic positions using the Hellmann–Feynman forces derived from the wavefunctions of the time-independent Schrödinger equation.
6. Compare the updated positions with those from the previous iteration.
7. Repeat the process until a convergence threshold (tolerance) is reached.

Once convergence is achieved, the code outputs:

- Equilibrium atomic positions
- Energy eigenvalues
- Energy eigenfunctions

## Post-processing

Post-processing routines are included to compute physical observables discussed in the reference paper.

## Extensibility

The structure of this code allows for straightforward generalizations of the SSH Hamiltonian, enabling the exploration of:

- Modified hopping schemes
- Additional interaction terms
- Coupling to external systems or fields

## Requirements

(Python version 3.12.3 and libraries in code)

## Usage

(To be completed: include basic example or instructions to run the code)

## Reference

If you use this code, please cite:

[Phys. Rev. B 111, 045405 (2025)]
https://journals.aps.org/prb/abstract/10.1103/PhysRevB.111.045405
