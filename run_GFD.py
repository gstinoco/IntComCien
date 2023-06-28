# All the codes were developed by:
#   Dr. Gerardo Tinoco Guerrero
#   Universidad Michoacana de San Nicolás de Hidalgo
#   gerardo.tinoco@umich.mx
#
# With the funding of:
#   National Council of Science and Technology, CONACyT (Consejo Nacional de Ciencia y Tecnología, CONACyT). México.
#   Coordination of Scientific Research, CIC-UMSNH (Coordinación de la Investigación Científica de la Universidad Michoacana de San Nicolás de Hidalgo, CIC-UMSNH). México
#   Aula CIMNE-Morelia. México
#
# Date:
#   January, 2023.
#
# Last Modification:
#   June, 2023.

import numpy as np
from scipy.io import loadmat
import Graphs
import GFD_Diffusion

# Diffusion coefficient
v = 0.2

# Boundary conditions
# The boundary conditions are defined as
#   f = e^{-2*\pi^2vt}\cos(\pi x)cos(\pi y)

def fDIF(x, y, t, v):
    fun = np.exp(-2*np.pi**2*v*t)*np.cos(np.pi*x)*np.cos(np.pi*y)
    return fun

# Names of the regions
regi = 'CAB_1'

# Number of Time Steps
t = 1000

# All data is loaded from the file
mat = loadmat('Data/' + regi + '.mat')

# Node data is saved
x  = mat['x']
y  = mat['y']

# Poisson 2D computed in a logically rectangular mesh
u_ap, u_ex = GFD_Diffusion.Mesh(x, y, fDIF, v, t)
Graphs.Mesh_Transient(x, y, u_ap, x, y, u_ex)