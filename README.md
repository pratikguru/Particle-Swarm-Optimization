
# Particle Swarm Optimization with Sphere Function

This Python script demonstrates the Particle Swarm Optimization (PSO) algorithm applied to optimize the Sphere function. It includes a Docker configuration to simplify setup and execution.

## Overview

Particle Swarm Optimization (PSO) is a powerful optimization algorithm inspired by the social behavior of birds and fish. It is commonly used to find the global minimum of a function by iteratively adjusting a swarm of particles' positions based on their own and their peers' best-known solutions. This script applies PSO to optimize the Sphere function, a classic test function used in optimization tasks.

The Sphere function is defined as follows:

`f(x) = sum(xi^2) for i = 1 to n`

where `n` is the dimensionality of the problem, and `xi` represents individual components of the solution vector `x`. The goal is to minimize this function and find the optimal solution where `f(x)` equals zero.

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Docker Commands](#usage)


## Overview

Particle Swarm Optimization (PSO) is a popular optimization algorithm used to find the global minimum of a function. In this example, we apply PSO to optimize the Sphere function, a well-known test function used in optimization tasks.

## Prerequisites

Before running the script, ensure you have the following prerequisites installed:

- Python (3.x recommended)
- NumPy
- Matplotlib
- Docker (if running inside a Docker container)


## Docker Commands

 - Make sure you have docker deamon runnning. 
 - `docker build -t pos .`
 - `docker run -v output:/app/output pso`