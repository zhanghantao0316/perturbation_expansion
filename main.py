from MC_vegas import IntegrationManager
from  Greens_functions import GreenFunctionCalculator
from scipy.special import gamma
import numpy as np
import mpmath as mp
from config import ScatteringConfig


if __name__ == "__main__":
    # Initialize physical parameters
    params=ScatteringConfig()
    # Create integration manager for Monte Carlo integration  
    manager = IntegrationManager(params)
    
    # Compute perturbative corrections and store results(corrections)  
    results = []
    # maximum order considered
    maxorder=6
    for order in range(1, maxorder+1):
        if order == 1:
            result, error = manager.perform_integration(order)
            print(f"first order result: {result:.6e} ± {error:.6e}")
            results.append(result)
        else:
            mean, std = manager.perform_integration(order)
            sign = (-1)**(order-1)  # Apply sign convention rules for perturbation expansion
            results.append(sign * mean)
            print(f"{order}order result (without sign): {mean:.6e} ± {std:.6e}")
    # M0 function (zero-order term)
    def M0(E, L, omega):
        M0=GreenFunctionCalculator.Cl0(L)**2 / GreenFunctionCalculator.Cl(E, L)**2 * (-1)**(L + 1) * (1 / (E / (2 * omega))**(L + 1/2)) * (gamma(L/2 + 3/4 - E/(2 * omega))/ gamma(1/4 - L/2 - E/(2 * omega)))
        return M0
    # Mi function (ith-order term, i>1)
    def Mi(E,L):
          Mi = (GreenFunctionCalculator.Cl0(L) ** 2 / GreenFunctionCalculator.Cl(E,L) ** 2) * ( -(2 ** (2 * L + 2)) * ( gamma(L + 3/2) ** 2) /((2 * params.mu) / (params.HBar_c ** 2)) /(GreenFunctionCalculator.k(E) ** (2 * L + 1)) / np.pi) 
          return Mi
    # 计算总相移
    M0val = M0(params.E, params.L, params.omega)  # value of M0
    Mival = Mi(params.E,params.L)  # value of Mi
    M_total = M0val + np.sum(results) * Mival # summation as final result
    deltaL = mp.atan(1/M_total) * 180/np.pi #  phase shift
    print(f" final phase shift : {deltaL :} (deg)")