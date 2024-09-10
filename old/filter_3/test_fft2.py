from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import LineCollection
from numpy import pi, cosh, exp, round, zeros, identity, arange, real, cos, sin, multiply, outer
from numpy.fft import fft,ifft, fftfreq, fft2, ifft2
from matplotlib.pyplot import figure

import numpy as np  # Import numpy
import math
from numpy.linalg import inv
###########################################
'''
Code to test 2d fft

The code is written on the transformed co-ordinates xc x yc = [0, 2\pi] x [0, 2\pi]
Physical coordinates [0, Lx] x [0, Ly]

- Pratik Aghor
'''
###########################################
print("_____________________________________________________________________\n")
print("Running python test_fft2.py... \n")
# Grid:
# c stands for computational domain
xc_min = 0; xc_max = 2*pi;
yc_min = 0; yc_max = 2*pi;
Lx = xc_max - xc_min;
Ly = yc_max - yc_min;

Nx = 8; Ny = 8;
print("Nx x Ny = ", Nx, " x ", Ny, "\n")

kx = zeros(Nx);
kx[0:Nx//2] = arange(0, Nx//2); kx[Nx//2+1:] = arange(-Nx//2+1, 0, 1);
ky = zeros(Ny);
ky[0:Ny//2] = arange(0, Ny//2); ky[Ny//2+1:] = arange(-Ny//2+1, 0, 1);

print(f'{kx}')
dx = 1.0 / (Nx-1);
dx = 1.0 / (Ny-1);

xc = (1.0*Lx/Nx)*arange(0, Nx);
yc = (1.0*Ly/Ny)*arange(0, Ny);
print("xc = \n", xc)
print("yc = \n", yc)

u = zeros((Ny, Nx))
for i in range(0, Ny):
    for j in range(0, Nx):
        u[i, j] = sin(xc[j])*cos(yc[i]);
print("u(x, y) = sin(xc)cos(yc) \n")
# for i in range(0, Ny):
#     for j in range(0, Nx):
#         u[i, j] = sin(xc[j])*cos(2.0*yc[i]);

print("u(x, y) = sin(xc)cos(2yc) \n")

v = fft2(u)/(Nx*Ny); # normalized fft

# if abs(value) < tol, set it to zero
tol = 1e-10
v.real[abs(v.real) < tol] = 0.0
v.imag[abs(v.imag) < tol] = 0.0
# v = v[range(int(len(u)/2))] # Exclude sampling frequency

print("v = fft2(u)/(Nx*Ny) = \n", v)
print("\n done!\n")
print("_____________________________________________________________________\n")
