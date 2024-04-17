# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 00:32:00 2024

@author: M Nur Alam Zico
"""
# Pressure Contour for 2D fluid flow by Navier-Stokes equations.

# Getting the numpy module
import numpy as np
# Getting the matplotlib
import matplotlib.pyplot as plt

# Parameters
nx = 50  # Number of grid points in x direction
ny = 50  # Number of grid points in y direction
Lx = 1.0  # Length of the domain in x direction
Ly = 1.0  # Length of the domain in y direction
dx = Lx / (nx - 1)  # Grid spacing in x direction
dy = Ly / (ny - 1)  # Grid spacing in y direction
nt = 1000  # Number of time steps
dt = 0.001  # Time step size
Re = 100  # Reynolds number
rho = 1.0  # Density
nu = Lx / Re  # Viscosity
x = np.linspace(0, 2, nx)
y = np.linspace(0, 2, ny)
X, Y = np.meshgrid(x, y)

# Initialization
u = np.zeros((nx, ny))
v = np.zeros((nx, ny))
p = np.zeros((nx, ny))

# Function to solve pressure Poisson equation
def solve_pressure_poisson(p, dx, dy, b):
    for q in range(50):
        pn = p.copy()
        p[1:-1, 1:-1] = ((pn[1:-1, 2:] + pn[1:-1, 0:-2]) * dy**2 +
                         (pn[2:, 1:-1] + pn[0:-2, 1:-1]) * dx**2) / (2 * (dx**2 + dy**2)) - \
                        dx**2 * dy**2 / (2 * (dx**2 + dy**2)) * b[1:-1, 1:-1]
        # Boundary conditions
        p[:, -1] = p[:, -2]  # dp/dy = 0 at y = Ly
        p[0, :] = p[1, :]    # dp/dx = 0 at x = 0
        p[-1, :] = p[-2, :]  # dp/dx = 0 at x = Lx

# Function to solve Navier-Stokes equations
def solve_navier_stokes(u, v, p, nu, rho, dt, dx, dy):
    # Calculate source term
    b = np.zeros((nx, ny))
    b[1:-1, 1:-1] = (rho * (1 / dt * ((u[1:-1, 2:] - u[1:-1, 0:-2]) / (2 * dx) +
                                       (v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * dy)) -
                            ((u[1:-1, 2:] - u[1:-1, 0:-2]) / (2 * dx))**2 -
                            2 * ((u[2:, 1:-1] - u[0:-2, 1:-1]) / (2 * dy) *
                                 (v[1:-1, 2:] - v[1:-1, 0:-2]) / (2 * dx)) -
                            ((v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * dy))**2))

    # Solve pressure Poisson equation
    solve_pressure_poisson(p, dx, dy, b)

    # Update velocity
    u[1:-1, 1:-1] = (u[1:-1, 1:-1] -
                     u[1:-1, 1:-1] * dt / dx *
                     (u[1:-1, 1:-1] - u[1:-1, 0:-2]) -
                     v[1:-1, 1:-1] * dt / dy *
                     (u[1:-1, 1:-1] - u[0:-2, 1:-1]) -
                     dt / (2 * rho * dx) * (p[1:-1, 2:] - p[1:-1, 0:-2]) +
                     nu * (dt / dx**2 *
                           (u[1:-1, 2:] - 2 * u[1:-1, 1:-1] + u[1:-1, 0:-2]) +
                           dt / dy**2 *
                           (u[2:, 1:-1] - 2 * u[1:-1, 1:-1] + u[0:-2, 1:-1])))

    v[1:-1, 1:-1] = (v[1:-1, 1:-1] -
                     u[1:-1, 1:-1] * dt / dx *
                     (v[1:-1, 1:-1] - v[1:-1, 0:-2]) -
                     v[1:-1, 1:-1] * dt / dy *
                     (v[1:-1, 1:-1] - v[0:-2, 1:-1]) -
                     dt / (2 * rho * dy) * (p[2:, 1:-1] - p[0:-2, 1:-1]) +
                     nu * (dt / dx**2 *
                           (v[1:-1, 2:] - 2 * v[1:-1, 1:-1] + v[1:-1, 0:-2]) +
                           dt / dy**2 *
                           (v[2:, 1:-1] - 2 * v[1:-1, 1:-1] + v[0:-2, 1:-1])))

    # Boundary conditions
    u[:, 0] = 0
    u[:, -1] = 0
    u[0, :] = 0
    u[-1, :] = 1  # velocity boundary condition (inlet)

    v[:, 0] = 0
    v[:, -1] = 0
    v[0, :] = 0
    v[-1, :] = 0

# Main loop
for n in range(nt):
    solve_navier_stokes(u, v, p, nu, rho, dt, dx, dy)

# Plotting
fig = plt.figure(figsize=(11,7), dpi=100)
plt.contourf(X, Y, p, alpha=0.5, cmap='jet') 
plt.colorbar(label='Pressure')
# plotting velocity field
plt.quiver(X[::2, ::2], Y[::2, ::2], u[::2, ::2], v[::2, ::2])  
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Pressure Contour')
plt.show()

