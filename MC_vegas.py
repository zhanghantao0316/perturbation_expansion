import vegas # 6.2.1 version
from scipy.integrate import quad
from integrand_kernel import IntegrandGenerator
class IntegrationManager:
    """Manage the integration process"""
    
    def __init__(self, config):
        self.config = config
        self.results = []
        
    def perform_integration(self, order):
        """Perform integration of a specified order"""
        if order == 1:
            return self._first_order_integration()
        else:
            return self._monte_carlo_integration(order)
    
    def _first_order_integration(self):
        """Handle first-order special integral"""
        result, error = quad(IntegrandGenerator.first_order_integrand, 
                           0, self.config.rmax1,
                           epsabs=1e-6, epsrel=1e-6, limit=150)
        return result, error
    
    def _monte_carlo_integration(self, order):
        """Handle high-order Monte Carlo integration with vegas"""
        integrand =IntegrandGenerator.generate_integrand(order)
        integrator = vegas.Integrator(order * [[0, self.config.rmax]])
        result = integrator(integrand, nitn=20, neval=int(self.config.neval))
        return result.mean, result.sdev