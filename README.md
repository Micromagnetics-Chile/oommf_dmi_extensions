# OOMMF DMI Extensions

This repository contains code extensions for the micromagnetic package OOMMF, for the calculation of the symmetric exchange interaction, and the antisymmetric Dzyaloshinskii-Moriya interaction (DMI), using higher order finite difference schemes, and different kinds of DMIs. Mathematical details about these extensions are published in *Conical-helix magnetic textures stabilized in thin films with different kinds of
Dzyaloshinskii-Moriya interaction*, by M. Cepeda-Arancibia et al. (2025)

Micromagnetic codes usually approximate the calculation of the exchange interactions using the nearest 6-neighbours. The modules in this repository use the next-nearest neighbours, making a 12-neighbour finite difference scheme. Boundary conditions are treated in two ways:

- **Free boundaries**: where virtual sites outside the sample are set to zero, and finite differences use zero-spins for these neighbours.

- **Robin boundaries**: where the Robin boundary condition from the combination of exchange and DMI are set explicitly at the boundaries of the sample. This allows calculating a magnetization point exactly at the boundary with the right boundary conditions, which is used for asymmetric finite difference schemes at the sample edges. For interfacial DMI, for instance, the Robin condition reads,

  ```
   ->               ->    ^   ^
  d m / dn - (D/2A) m  x (z x n) = 0
  ```

  For the computation, the DMI part can be written in matrix form. This matrix, of course, depends on the DMI class.


The following DMI symmetry classes (Schoenflies notation mostly) are implemented: 

- T, O  (bulk)
- D_2d
- C_nv, n>2  (interfacial)
- Anisotropic
- D_n
- C_n

The code works only for homogeneous DMI, i.e. not spatially-dependent.

## Code

For the development of these modules, we implemented functions to compute the nearest and next-nearest neighbors for each spin site. These functions are located in the `src/mesh_neighbors.h` header file, which is imported in every `.cc` module. This approach enables computing the interactions easily around boundaries and holes.

Robin conditions are only implemented for the T and C_nv classes (initially).

## Installation

The modules can be installed by copying the files in `src` into OOMMF's `app/oxs/local` directory, and compiling OOMMF again (`tclsh oommf.tcl pimake` from its root directory).

## Ubermag


## Tests


## Citation
