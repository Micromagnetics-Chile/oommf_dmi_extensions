# OOMMF DMI Extensions

This repository contains code extensions for the micromagnetic package OOMMF, for the calculation of the symmetric exchange interaction, and the antisymmetric Dzyaloshinskii-Moriya interaction (DMI), using higher order finite difference schemes, and different kinds of DMIs.

Micromagnetic codes usually approximate the calculation of the exchange interactions using the nearest 6-neighbours. The modules in this repository use the next-nearest neighbours, making a 12-neighbour finite difference scheme. Boundary conditions are treated in two ways:

- Free boundaries: where virtual sites outside the sample are set to zero, and finite differences use zero-spins for these neighbours.

- Robin boundaries: where the Robin boundary condition from the combination of exchange and DMI are set explicitly at the boundaries of the sample. This allows calculating a magnetization point exactly at the boundary with the right boundary conditions, which is used for asymmetric finite difference schemes at the sample edges.
