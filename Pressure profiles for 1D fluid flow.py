# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 01:07:33 2024

@author: M Nur Alam Zico
"""
# Pressure profiles for 1D fluid flow

# Getting the numpy module
import numpy as np
# Getting the matplotlib
import matplotlib.pyplot as plt

# Parameters
L = 1.0  # Length of the domain
nx = 100  # Number of grid points
dx = L / nx  # Grid spacing
dt = 0.001  # Time step
nu = 0.01  # Kinematic viscosity
rho = 1.0  # Density

# Initialize velocity field
u = np.zeros(nx + 1)
u_new = np.zeros(nx + 1)

# Initialize pressure field
p = np.zeros(nx + 1)

# Main time loop
for n in range(1000):
    # Boundary conditions
    u[0] = 1.0  # Inlet velocity
    u[-1] = 0.0  # No-slip wall

    # Compute convective fluxes
    F = np.zeros(nx + 1)
    for i in range(1, nx):
        F[i] = 0.5 * (u[i] + u[i + 1]) * (u[i] - u[i - 1])

    # Compute viscous fluxes
    du_dx2 = (u[:-2] - 2 * u[1:-1] + u[2:]) / dx ** 2
    viscous_flux = nu * du_dx2

    # Update solution
    u_new[1:-1] = u[1:-1] - dt * (F[1:-1] - F[:-2]) / dx + dt * viscous_flux

    # Compute pressure field update
    velocity_gradient = (u_new[1:] - u_new[:-1]) / dx
    pressure_change = dt * rho * velocity_gradient
    p[1:-1] -= pressure_change[:-1]

    # Update time step
    u[:] = u_new

# Plot pressure
plt.plot(np.linspace(0, L, nx + 1), p)
plt.xlabel('Position')
plt.ylabel('Pressure')
plt.title('Pressure Distribution')
plt.show()
