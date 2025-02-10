from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import LineCollection
from numpy import pi, cosh, exp, round, zeros, identity, arange, real, cos, sin, multiply, outer
from numpy.fft import fft,ifft, fftfreq, fft2, ifft2
from matplotlib.pyplot import figure

import numpy as np  # Import numpy
import math
from numpy.linalg import inv

import torch
import matplotlib.pyplot as plt
import h5py
import sys
import numpy as np
# import plotly.graph_objects as go


# import filter_div
from filter_div import *

# Code to test filter_div
# The code is written on the transformed co-ordinates xc x yc = [0, 2\pi) x [0, 2\pi)
# Physical coordinates [0, Lx] x [0, Ly]

# load data
n_train = 10
T = 5

# TRAIN_PATH = '/home/atif/datasets/dpsl_Re1e4_dx_256/dpsl_data_N5000_T201_ux_uy_vort.h5'
# f = h5py.File(TRAIN_PATH, 'r')

# u = torch.tensor(f['vel_x'][0:n_train,0:T,:,:]).type(torch.float32)
# v = torch.tensor(f['vel_y'][0:n_train,0:T,:,:]).type(torch.float32)
u = torch.load('./vel_x.pt')
v = torch.load('./vel_y.pt')

u = u[0,0,:,:]
v = v[0,0,:,:]

u = np.array(u)
v = np.array(v)
print(f'{np.shape(u)}')

[Ny, Nx] = np.shape(u)

L = 2*pi
# kx = zeros(Nx);
# kx[0:Nx//2] = arange(0, Nx//2); kx[Nx//2+1:] = arange(-Nx//2+1, 0, 1);
# ky = zeros(Ny);
# ky[0:Ny//2] = arange(0, Ny//2); ky[Ny//2+1:] = arange(-Ny//2+1, 0, 1);

kx = fftfreq(Nx, d =L/Nx) * 2 * pi
ky = fftfreq(Ny, d =L/Ny) * 2 * pi

KY, KX = np.meshgrid(ky, kx)
# KX, KY = np.meshgrid(kx, ky)

# print(f'kx = {kx}')

# print(f'kx ky {kx} {ky}')
[u0, v0] = filter_div(u, v)

uHat = fft2(u)
vHat = fft2(v)
u0Hat = fft2(u0)
v0Hat = fft2(v0)
divu0 = zeros((Ny, Nx))
divu0Hat = zeros((Ny, Nx), dtype=complex)
vort0Hat = zeros((Ny, Nx), dtype=complex)
vortHat = zeros((Ny, Nx), dtype=complex)
vortFD = zeros((Ny, Nx))

#vort = 1j * (KX * v_hat - KY * u_hat)

# for ii in range(0, Ny):
#     for jj in range(0, Nx):
#         k = kx[jj]
#         l = ky[ii]
#         vortHat[ii, jj]  = 1j*k*vHat[ii, jj]  - 1j*l*uHat[ii, jj]
#         vort0Hat[ii, jj] = 1j*k*v0Hat[ii, jj] - 1j*l*u0Hat[ii, jj]
#         divu0Hat[ii, jj] = 1j*k*u0Hat[ii, jj] + 1j*l*v0Hat[ii, jj]

vortHat = 1j*(KX*vHat - KY*uHat)

# divu0 = real(ifft2(divu0Hat))
# vort0 = real(ifft2(vort0Hat))
vort = (real(ifft2(vortHat)))
print("max(max(divu0))", np.max(abs(divu0)))

for ii in range(1, Ny-1):
    for jj in range(1, Nx-1):
        vortFD[ii, jj]   = (v[ii,jj+1] - v[ii,jj-1]) - (u[ii-1,jj] - u[ii+1,jj])
        # vortFD[ii, jj]   = (v[ii+1,jj] - v[ii-1,jj]) - (u[ii,jj+1] - u[ii,jj-1])



###########################################

for index in range(0,1):
    
    # Plotting a sample vorticity heat map
    #vortmax = (np.max(vortFD))
    #vortmin = (np.min(vortFD))
    #umax = (torch.max(u))
    #umin = (torch.min(u))
    #vmax = (torch.max(v))
    #vmin = (torch.min(v))
     
    fig, axs = plt.subplots(figsize=(150, 40))
    
    for a in range (0,1):
        ax2 = fig.add_subplot(2,1,a+1)
        im  = ax2.imshow(torch.from_numpy(vortFD[:,:]), cmap='bwr', interpolation='nearest')
        #im  = ax2.imshow(torch.from_numpy(u0[:,:]), vmin=umin, vmax=umax, cmap='bwr', interpolation='nearest')
        #im  = ax2.imshow(u[index,a,:,:].cpu(), vmin=umin, vmax=umax, cmap='bwr', interpolation='nearest')
        plt.axis('off')
        ax2.set_xticks([])
        ax2.set_yticks([])
    
        ax2 = fig.add_subplot(2,1,a+2)
        im  = ax2.imshow(torch.from_numpy(vort.copy()), cmap='bwr', interpolation='nearest')
        #im  = ax2.imshow(torch.from_numpy(v0[:,:]), vmin=vmin, vmax=vmax, cmap='bwr', interpolation='nearest')
        #im  = ax2.imshow(v[index,a,:,:].cpu(), vmin=vmin, vmax=vmax, cmap='bwr', interpolation='nearest')
        plt.axis('off')
        ax2.set_xticks([])
        ax2.set_yticks([])
    
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.86, 0.15, 0.01, 0.7])
    cbar_ax.tick_params(labelsize=20)
    fig.colorbar(im,cax=cbar_ax)
    
    fig.savefig(f'test_{index}_3d_channels_uv.png')
    plt.close()




