# Coulomb-corrected BERW formula 
Relevant content can be found in the paper "Charged particle scattering in harmonic traps"  by Hantao Zhang, Dong Bai, Zhen Wang, Zhongzhou Ren and the paper "Coupled-channels reactions for charged particles in harmonic traps"
by Hantao Zhang, Dong Bai, Zhongzhou Ren
修改为：
# Trap optimization for Nuclear-scattering with Generalized Harmonic Algorithm Interface (TONGHAI)

## PERTURBATION EXPANSION
### Written by Hantao Zhang, Dong Bai and Zhongzhou Ren
For specific applications of the Coulomb-corrected BERW formula with perturbation expansion to evaluate the  scattering phase shifts of charged particles in a harmonic oscillator trap can be  seen in the
two papers above

# Phase Shift Calculation with VEGAS Monte Carlo Integration

This program calculates quantum mechanical phase shifts using perturbation expansion up to 12th order, implemented with the VEGAS Monte Carlo integration algorithm.

## 1. Installation

### Required Packages
```bash
pip install vegas==6.2.1 
```

## 2. Physical Constants
Constants are defined in natural units (MeV and fm):
```bash
HBar_c = 197.327053  # Reduced Planck constant 
e2 = 1 / (8.854187817e-12 * 4 * np.pi) * 1.6021766e-19 * 1e9  # Coulomb constant 
amu = 938.918  # Atomic mass unit 
mu = (m1*m2)*amu/(m1+m2)  # Reduced mass
Z = -Z1 * Z2 * e2  # Coulomb interaction strength
```
If you want to change these constants, please directly modify them in the  file


## 3. Green's functions and their limitations at r ->0 
Please note that we have rigorously handled the limits of the Green's functions GC0 (Coulomb Green's funtion) and GHO (harmonic  oscillator Green's function) as r approaches zero, with the analytical results defined limitGC0 and limitGHO

## 4. Integration Parameters
| Parameter   | Value  | Description                          |
|-------------|--------|--------------------------------------|
| `rmax1`     | 1e3 fm | Upper limit for 1st-order (quad)     |
| `rmax`      | 1e2 fm | Upper limit for 2–12th orders (VEGAS) |
| `nitn`      | 20     | VEGAS adaptive iterations            |
| `nevalvalue`| 1e4    | Function evaluations per iteration (sampling points)   |
