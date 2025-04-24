import numpy as np
from mpmath import mp
from config import ScatteringConfig
from scipy.special import gamma, jv, yv
mp.dps = 20

config = ScatteringConfig()

class GreenFunctionCalculator:
    """Compute various Green's functions and their limits"""
    
    @staticmethod
    def k(E):
        return np.sqrt(2 * config.mu * E) / config.HBar_c

    @staticmethod
    def gamma_creduced(E):
        return (-config.Z * config.mu) / (GreenFunctionCalculator.k(E) * config.HBar_c**2)
    
    @staticmethod
    def Cl(E, L):
        return (2**L) * np.abs(gamma(L + 1 + 1j * GreenFunctionCalculator.gamma_creduced(E))) / gamma(2*L + 2) * np.exp(-np.pi * GreenFunctionCalculator.gamma_creduced(E)/2)

    @staticmethod
    def Cl0(L):
        return (2**L) * np.abs(gamma(L + 1)) / gamma(2*L + 2)


    @staticmethod
    def spherical_bessel(E, r, L):
        return np.sqrt(np.pi / (2 * GreenFunctionCalculator.k(E) * r)) * jv(L + 0.5, GreenFunctionCalculator.k(E)*r)

    @staticmethod
    def spherical_neumann(E, r, L):
        return (-1)**(L + 1) * np.sqrt(np.pi / (2 * GreenFunctionCalculator.k(E) * r))* yv(L + 0.5, GreenFunctionCalculator.k(E)*r)

    @staticmethod
    def hl_plus(E, r, L):
        return GreenFunctionCalculator.spherical_bessel(E, r, L) + 1j * GreenFunctionCalculator.spherical_neumann(E, r, L)

    @staticmethod
    def hl_minus(E, r, L):
        return GreenFunctionCalculator.spherical_bessel(E, r, L) - 1j * GreenFunctionCalculator.spherical_neumann(E, r, L)

    @classmethod
    def GHO(cls, E, omega, L, r, rp):
        """Harmonic oscillator Green's function"""
        max_r = max(r, rp)
        min_r = min(r, rp)
        coeff = -1 / (omega * (r*rp)**1.5)
        gamma_term = gamma(L/2 + 3/4 - E/(2*omega)) / gamma(L + 3/2)
        W = mp.whitw(E/(2*omega), L/2 + 0.25, (config.mu*omega*max_r**2)/config.HBar_c**2)
        M = mp.whitm(E/(2*omega), L/2 + 0.25, (config.mu*omega*min_r**2)/config.HBar_c**2)
        return coeff * gamma_term * W * M

    @classmethod
    def GC0(cls, E, r, rp, L):
        """Green's function of free particle"""
        return -2j * config.mu * cls.k(E) * cls.spherical_bessel(E, min(r, rp), L) * cls.hl_plus(E, max(r, rp), L) / config.HBar_c**2
    
    @classmethod
    def limitGHO(cls,E, omega, L, r):
        """Limitation of harmonic oscillator Green's function"""
        coeff = -1 / (omega * r**1.5) * (config.mu*omega/config.HBar_c**2)**(L/2 + 3/4)
        gamma_term = gamma(L/2 + 3/4 - E/(2*omega)) / gamma(L + 3/2)
        W = mp.whitw(E/(2*omega), L/2 + 0.25, (config.mu*omega*r**2)/config.HBar_c**2)
        return coeff * gamma_term * W

    @classmethod
    def limitGC0(cls,E, r, L):
        """Limitation of Green's function of free particle"""
        return -2j * config.mu * cls.k(E) * cls.Cl0(L) *  cls.k(E)**L *  cls.hl_plus(E, r, L) / config.HBar_c**2