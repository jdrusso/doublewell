# Double-well Simulation

This repo contains some sample code for running a simulation of a particle in a double-well
potential under overdamped Langevin dynamics.

The `weighted_ensemble` directory contains the necessary files to initialize and run a simulation under WESTPA.

The `brute_force` directory contains code I've written to do a brute-force simulation.

### Requirements

The WESTPA code, of course, requires WESTPA, which can be installed via Anaconda.

`conda install -c conda-forge westpa`

The brute-force code requires Numpy, Sympy, Ray, and Rich. Numpy and Sympy can be installed in the usual ways.
Ray and Rich can be installed by `pip -U ray rich`.

### Running

The WESTPA code can be run with `./init.sh` and `./run.sh`.

The brute-force code can be run with simply `python brute_force.py`.