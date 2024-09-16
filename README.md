<!--
SPDX-FileCopyrightText: 2024 Amgen

SPDX-License-Identifier: BSD-3-Clause
-->

![](./img/gumps_logo.PNG)  
G eneric  
U niversal  
M odeling  
P latform  
S oftware

Designed to remove redundancy in computational modeling
and allowing the creation of pipelines to chain multiple
models together.

## Installing

You can create a gumps environment using
mamba env create -f environment_XXX.yml where XXX is the environment you want to create.
We offer Python 3.10 and 3.11 with a test environent for 3.12 in both CPU and GPU accelerated version.

Activate the enivironment and then run
python setup.py sdist bdist_wheeel
pip install .

## GUMPS

The goal of GUMPS is two-fold:

1. Provide an interface for Modeling Engineers to integrate new and existing model code into GUMPS
2. Provide library functions in order to perform higher order mathematical operations.

## Kernel

First Entry Point to GUMPS. This is where Users can take their existing models
and adapt them to the Framework. Users are responsible for defining the following functions:

| Function              | Purpose                                                                  |
| --------------------- | ------------------------------------------------------------------------ |
| user_defined_function | Wrapper function to define what should be executed in a single iteration |
| get_states            | Define a dictionary of variable names and their default values           |
| initialize            | Responsible for any initialization the kernel requires                   |

## Solver

Class that represents different classes of mathematical solvers to run on top of Kernels

## Study

Bridge between the Solver and Kernel, responsible for confirming that the problem statement of the Solver
matches the states of the Kernel

# Development Workflow

See the jupyter notebook included here for an overview of how to construct different studies.

## Authors

- Bill Heymann
- Joao Alberto de Faria
- Pablo Rolandi
- Will Johnson
- Martin Carcamo
- DTI/DIPT

## License

BSD-3-Clause
