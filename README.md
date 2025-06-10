[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15475605.svg)](https://doi.org/10.5281/zenodo.15475605)

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


The following DMI symmetry classes (Schoenflies notation) are implemented: 

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

For the 12 neighbors code for the `Cn` class, with free boundaries, the OOMMF script code can be specified as

```tcl
# Uniform Exchange and DMI
Specify Oxs_ExchangeAndDMI_Cn_12ngbrs {
  Aex 10e-12
  D1 0.001
  D2 0.002
  atlas :atlas
  mesh :mesh
}
```

with `atlas` and `mesh` according to the geometry specified in the script. Notice that both exchange and DMI are specified in this function. In the case of Robin conditions:

```tcl
# Uniform Exchange and DMI
Specify Oxs_ExchangeAndDMI_Cnv_12ngbrs_RobinBC {
  Aex 10e-12
  D 0.003
  atlas :atlas
  mesh :mesh
}
```


## Installation

The modules can be installed by copying the files in `src` into OOMMF's `app/oxs/local` directory, and compiling OOMMF again (`tclsh oommf.tcl pimake` from its root directory).

## Ubermag


## Tests



## Citation

```tex
@Misc{OOMMFextensions,
  author       = {Cortés-Ortuño, David and Brevis, Felipe},
  title        = {{Micromagnetics Chile: oommf\_dmi\_extensions}},
  publisher    = {Zenodo},
  note         = {Github: \url{https://github.com/Micromagnetics-Chile/oommf_dmi_extensions}},
  year         = {2025},
  doi          = {10.5281/zenodo.15475605},
  url          = {https://doi.org/10.5281/zenodo.15475605},
}
```
