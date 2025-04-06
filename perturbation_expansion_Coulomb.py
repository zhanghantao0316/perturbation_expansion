"""
vegas   6.2.1  (latest)
"""
import vegas
import numpy as np
from scipy.integrate import quad
from mpmath import  mp
from scipy.special import gamma, jv, yv
mp.dps = 20  


# inpouts  angular momentum L, frequency omega of harmonic oscillator trap, binding energy E in the trap
# s-wave data for test (and also benchmark)
L=0
E=1.136863396929620
omega=1


# constant
HBar_c = 197.327053
e2 = 1 / (8.854187817e-12 * 4 * np.pi) * 1.6021766e-19 * 1e9
amu = 938.918
m1 = 1
m2 = 1
mu = (m1*m2)*amu/(m1+m2)
Z1 = 1
Z2 = 1
Z = -Z1 * Z2 * e2


# basic functions
def k(E):
    return np.sqrt(2 * mu * E) / HBar_c

def gammaCreduced(E):
    return (-Z * mu) / (k(E) * HBar_c**2)

def Cl(Ek, L):
    return (2**L) * np.abs(gamma(L + 1 + 1j * gammaCreduced(Ek))) / gamma(2*L + 2) * np.exp(-np.pi * gammaCreduced(Ek)/2)

def Cl0(L):
    return (2**L) * np.abs(gamma(L + 1)) / gamma(2*L + 2)

def jl(E, r, L):
    return np.sqrt(np.pi / (2 * k(E) * r)) * jv(L + 0.5, k(E)*r)

def nl(E, r, L):
    return (-1)**(L + 1) * np.sqrt(np.pi / (2 * k(E) * r)) * yv(L + 0.5, k(E)*r)

def hlplus(E, r, L):
    return jl(E, r, L) + 1j * nl(E, r, L)

def hlminus(E, r, L):
    return jl(E, r, L) - 1j * nl(E, r, L)

# GHO and its limitation
def GHO(E, omega, L, r, rp):
    max_r = max(r, rp)
    min_r = min(r, rp)
    coeff = -1 / (omega * (r*rp)**1.5)
    gamma_term = gamma(L/2 + 3/4 - E/(2*omega)) / gamma(L + 3/2)
    W = mp.whitw(E/(2*omega), L/2 + 0.25, (mu*omega*max_r**2)/HBar_c**2)
    M = mp.whitm(E/(2*omega), L/2 + 0.25, (mu*omega*min_r**2)/HBar_c**2)
    return coeff * gamma_term * W * M

def limitGHO(E, omega, L, r):
    coeff = -1 / (omega * r**1.5) * (mu*omega/HBar_c**2)**(L/2 + 3/4)
    gamma_term = gamma(L/2 + 3/4 - E/(2*omega)) / gamma(L + 3/2)
    W = mp.whitw(E/(2*omega), L/2 + 0.25, (mu*omega*r**2)/HBar_c**2)
    return coeff * gamma_term * W

# GC0 and its limitation

def GC0(E, r, rp, L):
    return -2j * mu * k(E) * jl(E, min(r, rp), L) * hlplus(E, max(r, rp), L) / HBar_c**2

def limitGC0(E, r, L):
    return -2j * mu * k(E) * Cl0(L) * k(E)**L * hlplus(E, r, L) / HBar_c**2




# perturbation expansion
def integrand1order(rpp):
      gc =  limitGC0(E, rpp, L)
      gho = limitGHO(E, omega, L, rpp)
      return np.real((gc**2 - gho**2) * Z * rpp)
  

def integrand2order(r2,r3):
        rpp=r2
        rp3=r3
        term1 = limitGC0(E, rpp, L) * GC0(E, rpp, rp3, L) * limitGC0(E, rp3, L)
        term2 = limitGHO(E, omega, L, rpp) * GHO(E, omega, L, rpp, rp3) * limitGHO(E, omega, L, rp3)
        return np.real((term1 - term2) * Z**2 * rpp * rp3)

def integrand2orderMC(r):
        rpp=r[0]
        rp3=r[1]
        term1 = limitGC0(E, rpp, L) * GC0(E, rpp, rp3, L) * limitGC0(E, rp3, L)
        term2 = limitGHO(E, omega, L, rpp) * GHO(E, omega, L, rpp, rp3) * limitGHO(E, omega, L, rp3)
        return np.real((term1 - term2) * Z**2 * rpp * rp3)
    
    
def integrand3orderMC(r):
            rp2=r[0]
            rp3=r[1]
            rp4=r[2]
            term1 = limitGC0(E, rp2, L) * GC0(E, rp2, rp3, L) * GC0(E, rp3, rp4, L) * limitGC0(E, rp4, L)
            term2 = (limitGHO(E, omega, L, rp2) * GHO(E, omega, L, rp2, rp3) 
                      * GHO(E, omega, L, rp3, rp4) * limitGHO(E, omega, L, rp4))
            return np.real((term1 - term2) * Z**3 * rp2 * rp3 * rp4)
        

def integrand4orderMC(r):
            rp2=r[0]
            rp3=r[1]
            rp4=r[2]
            rp5=r[3]
            term1 = limitGC0(E, rp2, L) * GC0(E, rp2, rp3, L) * GC0(E, rp3, rp4, L) * GC0(E, rp4, rp5, L) * limitGC0(E, rp5, L)
            term2 = (limitGHO(E, omega, L, rp2) * GHO(E, omega, L, rp2, rp3) 
                      * GHO(E, omega, L, rp3, rp4)  * GHO(E, omega, L, rp4, rp5) * limitGHO(E, omega, L, rp5))
            return np.real((term1 - term2) * Z**4 * rp2 * rp3 * rp4 * rp5)   
    
    
    
def integrand5orderMC(r):
            rp2=r[0]
            rp3=r[1]
            rp4=r[2]
            rp5=r[3]
            rp6=r[4]
            term1 = limitGC0(E, rp2, L) * GC0(E, rp2, rp3, L) * GC0(E, rp3, rp4, L) * GC0(E, rp4, rp5, L)* GC0(E, rp5, rp6, L) * limitGC0(E, rp6, L)
            term2 = (limitGHO(E, omega, L, rp2) * GHO(E, omega, L, rp2, rp3) 
                      * GHO(E, omega, L, rp3, rp4)  * GHO(E, omega, L, rp4, rp5) * GHO(E, omega, L, rp5, rp6) * limitGHO(E, omega, L, rp6))
            return np.real((term1 - term2) * Z**5 * rp2 * rp3 * rp4 * rp5 * rp6)     
        
        
        
def integrand6orderMC(r):
            rp2=r[0]
            rp3=r[1]
            rp4=r[2]
            rp5=r[3]
            rp6=r[4]
            rp7=r[5]
            term1 = limitGC0(E, rp2, L) * GC0(E, rp2, rp3, L) * GC0(E, rp3, rp4, L) * GC0(E, rp4, rp5, L)* GC0(E, rp5, rp6, L) * GC0(E, rp6, rp7, L)* limitGC0(E, rp7, L)
            term2 = (limitGHO(E, omega, L, rp2) * GHO(E, omega, L, rp2, rp3) 
                      * GHO(E, omega, L, rp3, rp4)  * GHO(E, omega, L, rp4, rp5) * GHO(E, omega, L, rp5, rp6) * GHO(E, omega, L, rp6, rp7) * limitGHO(E, omega, L, rp7))
            return np.real((term1 - term2) * Z**6 * rp2 * rp3 * rp4 * rp5 * rp6 * rp7)     


def integrand7orderMC(r):
            rp2=r[0]
            rp3=r[1]
            rp4=r[2]
            rp5=r[3]
            rp6=r[4]
            rp7=r[5]
            rp8=r[6]
            term1 = limitGC0(E, rp2, L) * GC0(E, rp2, rp3, L) * GC0(E, rp3, rp4, L) * GC0(E, rp4, rp5, L)* GC0(E, rp5, rp6, L) * GC0(E, rp6, rp7, L) * GC0(E, rp7, rp8, L)* limitGC0(E, rp8, L)
            term2 = (limitGHO(E, omega, L, rp2) * GHO(E, omega, L, rp2, rp3) 
                      * GHO(E, omega, L, rp3, rp4)  * GHO(E, omega, L, rp4, rp5) * GHO(E, omega, L, rp5, rp6) * GHO(E, omega, L, rp6, rp7) * GHO(E, omega, L, rp7, rp8) * limitGHO(E, omega, L, rp8))
            return np.real((term1 - term2) * Z**7 * rp2 * rp3 * rp4 * rp5 * rp6 * rp7* rp8)     
        
        
        
def integrand8orderMC(r):
            rp2=r[0]
            rp3=r[1]
            rp4=r[2]
            rp5=r[3]
            rp6=r[4]
            rp7=r[5]
            rp8=r[6]
            rp9=r[7]
            term1 = limitGC0(E, rp2, L) * GC0(E, rp2, rp3, L) * GC0(E, rp3, rp4, L) * GC0(E, rp4, rp5, L)* GC0(E, rp5, rp6, L) * GC0(E, rp6, rp7, L) * GC0(E, rp7, rp8, L)* GC0(E, rp8, rp9, L)* limitGC0(E, rp9, L)
            term2 = (limitGHO(E, omega, L, rp2) * GHO(E, omega, L, rp2, rp3) 
                      * GHO(E, omega, L, rp3, rp4)  * GHO(E, omega, L, rp4, rp5) * GHO(E, omega, L, rp5, rp6) * GHO(E, omega, L, rp6, rp7) * GHO(E, omega, L, rp7, rp8)* GHO(E, omega, L, rp8, rp9) * limitGHO(E, omega, L, rp9))
            return np.real((term1 - term2) * Z**8 * rp2 * rp3 * rp4 * rp5 * rp6 * rp7* rp8* rp9)     


def integrand9orderMC(r):
            rp2=r[0]
            rp3=r[1]
            rp4=r[2]
            rp5=r[3]
            rp6=r[4]
            rp7=r[5]
            rp8=r[6]
            rp9=r[7]
            rp10=r[8]
            term1 = limitGC0(E, rp2, L) * GC0(E, rp2, rp3, L) * GC0(E, rp3, rp4, L) * GC0(E, rp4, rp5, L)* GC0(E, rp5, rp6, L) * GC0(E, rp6, rp7, L) * GC0(E, rp7, rp8, L)* GC0(E, rp8, rp9, L)* GC0(E, rp9, rp10, L)* limitGC0(E, rp10, L)
            term2 = (limitGHO(E, omega, L, rp2) * GHO(E, omega, L, rp2, rp3) 
                      * GHO(E, omega, L, rp3, rp4)  * GHO(E, omega, L, rp4, rp5) * GHO(E, omega, L, rp5, rp6) * GHO(E, omega, L, rp6, rp7) * GHO(E, omega, L, rp7, rp8)* GHO(E, omega, L, rp8, rp9)* GHO(E, omega, L, rp9, rp10)  * limitGHO(E, omega, L, rp10))
            return np.real((term1 - term2) * Z**9 * rp2 * rp3 * rp4 * rp5 * rp6 * rp7* rp8* rp9* rp10)   
        
        
def integrand10orderMC(r):
            rp2=r[0]
            rp3=r[1]
            rp4=r[2]
            rp5=r[3]
            rp6=r[4]
            rp7=r[5]
            rp8=r[6]
            rp9=r[7]
            rp10=r[8]
            rp11=r[9]
            term1 = limitGC0(E, rp2, L) * GC0(E, rp2, rp3, L) * GC0(E, rp3, rp4, L) * GC0(E, rp4, rp5, L)* GC0(E, rp5, rp6, L) * GC0(E, rp6, rp7, L) * GC0(E, rp7, rp8, L)* GC0(E, rp8, rp9, L)* GC0(E, rp9, rp10, L)* GC0(E, rp10, rp11, L)* limitGC0(E, rp11, L)
            term2 = (limitGHO(E, omega, L, rp2) * GHO(E, omega, L, rp2, rp3) 
                      * GHO(E, omega, L, rp3, rp4)  * GHO(E, omega, L, rp4, rp5) * GHO(E, omega, L, rp5, rp6) * GHO(E, omega, L, rp6, rp7) * GHO(E, omega, L, rp7, rp8)* GHO(E, omega, L, rp8, rp9)* GHO(E, omega, L, rp9, rp10)* GHO(E, omega, L, rp10, rp11)  * limitGHO(E, omega, L, rp11))
            return np.real((term1 - term2) * Z**10 * rp2 * rp3 * rp4 * rp5 * rp6 * rp7* rp8* rp9* rp10* rp11)    

def integrand11orderMC(r):
            rp2=r[0]
            rp3=r[1]
            rp4=r[2]
            rp5=r[3]
            rp6=r[4]
            rp7=r[5]
            rp8=r[6]
            rp9=r[7]
            rp10=r[8]
            rp11=r[9]
            rp12=r[10]
            term1 = limitGC0(E, rp2, L) * GC0(E, rp2, rp3, L) * GC0(E, rp3, rp4, L) * GC0(E, rp4, rp5, L)* GC0(E, rp5, rp6, L) * GC0(E, rp6, rp7, L) * GC0(E, rp7, rp8, L)* GC0(E, rp8, rp9, L)* GC0(E, rp9, rp10, L)* GC0(E, rp10, rp11, L)* GC0(E, rp11, rp12, L)* limitGC0(E, rp12, L)
            term2 = (limitGHO(E, omega, L, rp2) * GHO(E, omega, L, rp2, rp3) 
                      * GHO(E, omega, L, rp3, rp4)  * GHO(E, omega, L, rp4, rp5) * GHO(E, omega, L, rp5, rp6) * GHO(E, omega, L, rp6, rp7) * GHO(E, omega, L, rp7, rp8)* GHO(E, omega, L, rp8, rp9)* GHO(E, omega, L, rp9, rp10)* GHO(E, omega, L, rp10, rp11) * GHO(E, omega, L, rp11, rp12)  * limitGHO(E, omega, L, rp12))
            return np.real((term1 - term2) * Z**11 * rp2 * rp3 * rp4 * rp5 * rp6 * rp7* rp8* rp9* rp10* rp11* rp12)    

def integrand12orderMC(r):
            rp2=r[0]
            rp3=r[1]
            rp4=r[2]
            rp5=r[3]
            rp6=r[4]
            rp7=r[5]
            rp8=r[6]
            rp9=r[7]
            rp10=r[8]
            rp11=r[9]
            rp12=r[10]
            rp13=r[11]
            term1 = limitGC0(E, rp2, L) * GC0(E, rp2, rp3, L) * GC0(E, rp3, rp4, L) * GC0(E, rp4, rp5, L)* GC0(E, rp5, rp6, L) * GC0(E, rp6, rp7, L) * GC0(E, rp7, rp8, L)* GC0(E, rp8, rp9, L)* GC0(E, rp9, rp10, L)* GC0(E, rp10, rp11, L)* GC0(E, rp11, rp12, L)* GC0(E, rp12, rp13, L)* limitGC0(E, rp13, L)
            term2 = (limitGHO(E, omega, L, rp2) * GHO(E, omega, L, rp2, rp3) 
                      * GHO(E, omega, L, rp3, rp4)  * GHO(E, omega, L, rp4, rp5) * GHO(E, omega, L, rp5, rp6) * GHO(E, omega, L, rp6, rp7) * GHO(E, omega, L, rp7, rp8)* GHO(E, omega, L, rp8, rp9)* GHO(E, omega, L, rp9, rp10)* GHO(E, omega, L, rp10, rp11) * GHO(E, omega, L, rp11, rp12) * GHO(E, omega, L, rp12, rp13)   * limitGHO(E, omega, L, rp13))
            return np.real((term1 - term2) * Z**12 * rp2 * rp3 * rp4 * rp5 * rp6 * rp7* rp8* rp9* rp10* rp11* rp12* rp13)   

# M0 function (zero-order term)
def M0(E, L, omega):
    return Cl0(L)**2 / Cl(E, L)**2 * (-1)**(L + 1) * (1 / (E / (2 * omega))**(L + 1/2)) * (gamma(L/2 + 3/4 - E/(2 * omega))/ gamma(1/4 - L/2 - E/(2 * omega)))

# Mi function (ith-order term, i>1)
def Mi(E,L):
      result = (Cl0(L) ** 2 / Cl(E,L) ** 2) * ( -(2 ** (2 * L + 2)) * ( gamma(L + 3/2) ** 2) /((2 * mu) / (HBar_c ** 2)) /(k(E) ** (2 * L + 1)) / np.pi) 
      return result

#  uplimit of integral(1 order correction)
rmax1=1e3
#  uplimit of integral(2-12 order corrections)
rmax=1e2 # fm 
# number of ampling points (for Monte Carlo integration)
nevalvalue=1e4

# quad for 1st-order correction (using quad for higher accuracy)
result1, error1 = quad(integrand1order, 0, rmax1,epsabs=10e-6,epsrel=10e-6,limit=150)
# 1order correction
print(f"Result for order {1}: mean = {result1}, error= {error1}")

# #  2-12order corrections
for indexM in range(2, 12+1): 
  if indexM == 2:
   integrator = vegas.Integrator(indexM*[[0, rmax]])
   integrandfun=integrand2orderMC
#
   result2 = integrator(integrandfun, nitn=20, neval=nevalvalue)
   print(f"Result for order {indexM}: mean = {result2.mean}, standard deviation = {result2.sdev}")
  elif indexM==3:
   integrator = vegas.Integrator(indexM*[[0, rmax]])
   integrandfun=integrand3orderMC
#
   result3 = integrator(integrandfun, nitn=20, neval=nevalvalue)
   print(f"Result for order {indexM}: mean = {result3.mean}, standard deviation = {result3.sdev}")
  elif indexM==4:
   integrator = vegas.Integrator(indexM*[[0, rmax]])
   integrandfun=integrand4orderMC
#
   result4 = integrator(integrandfun, nitn=20, neval=nevalvalue)
   print(f"Result for order {indexM}: mean = {result4.mean}, standard deviation = {result4.sdev}")
  elif indexM==5:
   integrator = vegas.Integrator(indexM*[[0, rmax]])
   integrandfun=integrand5orderMC
#
   result5 = integrator(integrandfun, nitn=20, neval=nevalvalue)
   print(f"Result for order {indexM}: mean = {result5.mean}, standard deviation = {result5.sdev}")
  elif indexM==6:
   integrator = vegas.Integrator(indexM*[[0, rmax]])
   integrandfun=integrand6orderMC
#
   result6 = integrator(integrandfun, nitn=20, neval=nevalvalue)
   print(f"Result for order {indexM}: mean = {result6.mean}, standard deviation = {result6.sdev}")
  elif indexM==7:
   integrator = vegas.Integrator(indexM*[[0, rmax]])
   integrandfun=integrand7orderMC
#
   result7 = integrator(integrandfun, nitn=20, neval=nevalvalue)
   print(f"Result for order {indexM}: mean = {result7.mean}, standard deviation = {result7.sdev}")
  elif indexM==8:
   integrator = vegas.Integrator(indexM*[[0, rmax]])
   integrandfun=integrand8orderMC
#
   result8 = integrator(integrandfun, nitn=20, neval=nevalvalue)
   print(f"Result for order {indexM}: mean = {result8.mean}, standard deviation = {result8.sdev}")
  elif indexM==9:
     integrator = vegas.Integrator(indexM*[[0, rmax]])
     integrandfun=integrand9orderMC
#
     result9 = integrator(integrandfun, nitn=20, neval=nevalvalue)
     print(f"Result for order {indexM}: mean = {result9.mean}, standard deviation = {result9.sdev}")
  elif indexM==10:
   integrator = vegas.Integrator(indexM*[[0, rmax]])
   integrandfun=integrand10orderMC
#
   result10 = integrator(integrandfun, nitn=20, neval=nevalvalue)
   print(f"Result for order {indexM}: mean = {result10.mean}, standard deviation = {result10.sdev}")
  elif indexM==11:
   integrator = vegas.Integrator(indexM*[[0, rmax]])
   integrandfun=integrand11orderMC
#
   result11 = integrator(integrandfun, nitn=20, neval=nevalvalue)
   print(f"Result for order {indexM}: mean = {result11.mean}, standard deviation = {result11.sdev}")
  elif indexM==12:
   integrator = vegas.Integrator(indexM*[[0, rmax]])
   integrandfun=integrand12orderMC
#
   result12 = integrator(integrandfun, nitn=20, neval=nevalvalue)
   print(f"Result for order {indexM}: mean = {result12.mean}, standard deviation = {result12.sdev}")
  
  
# array of perturbation expansion results
results=[result1,-result2.mean,result3.mean,-result4.mean,result5.mean,-result6.mean
          ,result7.mean,-result8.mean,result9.mean,-result10.mean,result11.mean,-result12.mean]
# final summation
Mtotalsum=M0(E, L,omega) + np.sum(results)*Mi(E,L)
# phase shift (deg)
phaseshift=mp.atan(1/Mtotalsum)*180/np.pi
print(f"phase shift at {E} MeV = {phaseshift} deg")


