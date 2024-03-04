"""
This module tests dynamic parameter functionality
"""
from collections import OrderedDict
import jax.numpy as jnp
import matplotlib.pyplot as plt
from run_time_assurance.zoo.cwh.docking_2d import Docking2dExplicitSwitchingRTA, Docking2dImplicitSwitchingRTA, Docking2dExplicitOptimizationRTA, Docking2dImplicitOptimizationRTA
from run_time_assurance.constraint import DirectInequalityConstraint
from test_docking_rta_2d import Env


class MyDirectInequalityConstraint(DirectInequalityConstraint):
    """Constraint Fx > val"""
    def ineq_weight(self, state: jnp.ndarray, params: dict) -> jnp.ndarray:
        return jnp.array([1., 0.])

    def ineq_constant(self, state: jnp.ndarray, params: dict) -> float:
        return params['val']
    

class TestDocking2dExplicitOptimizationRTA(Docking2dExplicitOptimizationRTA):
    def _setup_direct_constraints(self):
        return OrderedDict([('u1', MyDirectInequalityConstraint(params={'val': -1.}))])

class TestEnv(Env):
    def _update_status(self, state, time):
        if time == 200:
            try:
                # Set Fx > -0.5 (previously -1)
                self.rta.direct_inequality_constraints["u1"].params['val'] = -0.5
            except: pass
        if time == 1000:
            # Set docking radius to 2 (previously 0.2)
            self.rta.constraints["rel_vel"].params['v0'] = 2.
            try:
                # Disable direct Fx constraint (actuation limits still apply)
                self.rta.direct_inequality_constraints["u1"].enabled = False
            except: pass
        if time == 1250:
            # Disable rel_vel constraint
            self.rta.constraints["rel_vel"].enabled = False


if __name__ == '__main__':
    rtas = [Docking2dExplicitSwitchingRTA(), Docking2dImplicitSwitchingRTA(), 
            TestDocking2dExplicitOptimizationRTA(), Docking2dImplicitOptimizationRTA()]
    output_names = ['rta_test_docking_2d_explicit_switching', 'rta_test_docking_2d_implicit_switching',
                    'rta_test_docking_2d_explicit_optimization', 'rta_test_docking_2d_implicit_optimization']

    for rta, output_name in zip(rtas, output_names):
        env = TestEnv(rta)
        x, u, i = env.simulate_episode()
        env.plotter(x, u, i)
        plt.show()
