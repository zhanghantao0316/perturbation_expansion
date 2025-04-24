# Trap Optimization for Nuclear-scattering with Generalized Harmonic Algorithm Interface (TONGHAI)
Relevant content can be found in the paper "Charged particle scattering in harmonic traps"  by Hantao Zhang, Dong Bai, Zhen Wang, Zhongzhou Ren and the paper "Coupled-channels reactions for charged particles in harmonic traps"
by Hantao Zhang, Dong Bai, Zhongzhou Ren

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
Constants are defined in common units (MeV and fm), if you want to change these constants, please directly modify them in the config.py file


## 3. Green's functions and their limitations at r ->0 
Please note that we have rigorously handled the limits of the Green's functions GC0 (Coulomb Green's funtion) and GHO (harmonic  oscillator Green's function) as r approaches zero, with the analytical results defined limitGC0 and limitGHO

## 4. Integration Parameters
| Parameter   | Value  | Description                          |
|-------------|--------|--------------------------------------|
| `rmax1`     | 1e3 fm | Upper limit for 1st-order (quad)     |
| `rmax`      | 1e2 fm | Upper limit for 2–12th orders (VEGAS) |
| `nitn`      | 20     | VEGAS adaptive iterations            |
| `nevalvalue`| 1e4    | Function evaluations per iteration (sampling points)   |
