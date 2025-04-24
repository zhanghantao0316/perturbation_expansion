import numpy as np
from config import ScatteringConfig
from  Greens_functions import GreenFunctionCalculator


params = ScatteringConfig()
class IntegrandGenerator:
    """Generate integration kernels for various orders """
    
    @staticmethod
    def generate_integrand(order):
        """Generate integration kernels for certain orders """
        def integrand(r):
            # 生成GC0项
            gc_chain = [GreenFunctionCalculator.GC0(params.E, r[i], r[i+1], params.L) 
                       for i in range(order-1)]
            term1 = GreenFunctionCalculator.limitGC0(params.E, r[0], params.L)
            term1 *= np.prod(gc_chain)
            term1 *= GreenFunctionCalculator.limitGC0(params.E, r[-1], params.L)
            
            # 生成GHO项
            gho_chain = [GreenFunctionCalculator.GHO(params.E, params.omega, params.L, r[i], r[i+1]) 
                        for i in range(order-1)]
            term2 = GreenFunctionCalculator.limitGHO(params.E, params.omega, params.L, r[0])
            term2 *= np.prod(gho_chain)
            term2 *= GreenFunctionCalculator.limitGHO(params.E, params.omega, params.L, r[-1])
            
            # 组合项
            z_factor = (params.Z**order) * np.prod(r)
            return np.real((term1 - term2) * z_factor)
        
        return integrand

    @staticmethod
    def first_order_integrand(rpp):
        """First-order special integral kernel"""
        gc = GreenFunctionCalculator.limitGC0(params.E, rpp, params.L)
        gho = GreenFunctionCalculator.limitGHO(params.E, params.omega, params.L, rpp)
        return np.real((gc**2 - gho**2) * params.Z * rpp)