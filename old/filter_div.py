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
given a 2d field u_vec = [u, v] on xc, yc in the physical space

filter out divergent part to make del . u_vec = 0

using Helmholtz decomposition


The code is written on the transformed co-ordinates xc x yc = [0, 2\pi) x [0, 2\pi)

Physical coordinates [0, Lx] x [0, Ly]

- Pratik Aghor
'''
###########################################
def filter_div(u, v):
    uHat = fft2(u)
    vHat = fft2(v)

    [Ny, Nx] = np.shape(u)
    
    # see test_fft2.py to see how k's are arranged in python3
    kx = zeros(Nx);
    kx[0:Nx//2] = arange(0, Nx//2); kx[Nx//2+1:] = arange(-Nx//2+1, 0, 1);
    ky = zeros(Ny);
    ky[0:Ny//2] = arange(0, Ny//2); ky[Ny//2+1:] = arange(-Ny//2+1, 0, 1);

    divuHat = zeros((Ny, Nx), dtype=complex)
    phiHat = zeros((Ny, Nx), dtype=complex)
    du0Hat = zeros((Ny, Nx), dtype=complex)
    dv0Hat = zeros((Ny, Nx), dtype=complex)
    # divuHat = - Hat{(del.u_vec)}
    for ii in range(0, Ny):
        for jj in range(0, Nx):
            k = kx[jj]
            l = ky[ii]
            divuHat[ii, jj] = -1j*k*u[ii, jj] - 1j*l*v[ii, jj]

    for ii in range(0, Ny):
        for jj in range(0, Nx):
            k = kx[jj]
            l = ky[ii]

            if(k == 0 and l == 0):
                phiHat[ii, jj] = 0. # phase fixing condition, zero mean condition
            else:
                phiHat[ii, jj] = divuHat[ii, jj]/(- k**2 - l**2)

    # du0_vec = grad phi => {du0Hat, dv0Hat} = {1j*k*phiHat, 1j*l*phiHat}
    for ii in range(0, Ny):
        for jj in range(0, Nx):
            k = kx[jj]
            l = ky[ii]
            du0Hat[ii, jj] = 1j*k*phiHat[ii, jj]
            dv0Hat[ii, jj] = 1j*l*phiHat[ii, jj]

    u0 = real(ifft2(du0Hat)) # probably need scaling factor for ifft2
    u0 = u0 + u

    v0 = real(ifft2(dv0Hat))
    v0 = v0 + v
    
    return u0, v0