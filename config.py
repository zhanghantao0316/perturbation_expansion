import numpy as np
class ScatteringConfig:
    """ Stores basic configuration parameters for computational physics simulations"""
    def __init__(self):
        self.E=1.136863396929620 # scattering energy
        self.L=0 # angular momentum
        self.omega=1 # omega parameter in harmonic oscillator 
        self.HBar_c = 197.327053 
        self.e2 = 1 / (8.854187817e-12 * 4 * np.pi) * 1.6021766e-19 * 1e9
        self.amu = 938.918 
        self.m1 = 1
        self.m2 = 1
        self.mu = (self.m1*self.m2)*self.amu/(self.m1+self.m2) # reduced mass
        self.Z1 = 1
        self.Z2 = 1
        self.Z = -self.Z1 * self.Z2 * self.e2
        self.rmax1 = 1e3  # First-order perturbation term integration upper limit (unit: fm)
        self.rmax = 1e2   # Higher-order terms integration upper limit
        self.neval = 1e3  # Number of sampling points (affects calculation accuracy)