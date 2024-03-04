"""
This module defines the base RTA constraints and constraint strengthener classes and includes some standard implementations
"""
from __future__ import annotations

import abc
import numbers
from typing import Any, Union

import jax.numpy as jnp
from jax import grad, jacrev, jit
from pydantic import BaseModel  # pylint: disable=no-name-in-module


class SubsampleConfigValidator(BaseModel):
    """
    Validator for ConstraintModule's subsample_config.
    Specifies subsampling parameters for implicit ASIF methods.
    See ImplicitASIFModule.subsample_constraints for more information.

    Parameters
    ----------
    num_check_all : int
        Number of points at beginning of backup trajectory to check at every sequential simulation timestep.
        Should be <= backup_window.
        Defaults to -1 as skip_length defaults to 1 resulting in all backup trajectory points being checked.
    skip_length : int
        After num_check_all points in the backup trajectory are checked, the remainder of the backup window is filled by
        skipping every skip_length points to reduce the number of backup trajectory constraints.
        Defaults to 1, resulting in no skipping.
    subsample_constraints_num_least : int
        subsample the backup trajectory down to the points with the N least constraint function outputs
        i.e. the n points closest to violating a safety constraint. Default None
    keep_first : bool
        keep the first point along the trajectory, regardless of prior subsampling. Default False.
    keep_last : bool
        keep the last point along the trajectory, regardless of prior subsampling. Default False.
    """
    num_check_all: int = -1
    skip_length: int = 1
    subsample_constraints_num_least: Union[int, None] = None
    keep_first: bool = False
    keep_last: bool = False


class ConstraintModule(abc.ABC):
    """Base class implementation for safety constraints
    Implements constraint with form of h(x) >= 0

    Parameters
    ----------
    alpha : ConstraintStrengthener
        Constraint Strengthener object used for ASIF methods. Required for ASIF methods.
    alpha_negative : ConstraintStrengthener
        Constraint Strengthener object used for ASIF methods. Required for ASIF methods.
        Used when the constraint value is negative, to stabilize the system back to the safe set.
        By default, if alpha_negative is None, 10x the value of alpha(x) will be used.
    bias : float
        Value that adds a bias to the boundary of the constraint.
        Use a small negative value to make the constraint slightly more conservative.
    bias_negative : float
        Value that adds a bias to the boundary of the constraint.
        Used when the constraint value is negative, to stabilize the system back to the safe set.
        By default -0.01, which helps stabilize the system back to the safe set rather than the boundary.
    jit_enable: bool, optional
        Flag to enable or disable JIT compiliation. Useful for debugging
    slack_priority_scale: float, optional
        A scaling constant for the constraint's slack variable, for use during optimization.
        Value should be large (ex. 1e12). Higher values correspond to higher priority.
        By default None -> no slack variable will be used.
    params : dict, optional
        Dict of parameters that can be changed during a simulation.
        These parameters do not have dynamics and are only adjusted by the user.
        These parameters cannot be class attributes due to jit.
    enabled : bool, optional
        Flag to enable/disable the constraint from being enforced by RTA.
        Default enabled.
    """

    def __init__(
        self,
        alpha: Union[ConstraintStrengthener, None] = None,
        alpha_negative: Union[ConstraintStrengthener, None] = None,
        bias: float = 0,
        bias_negative: float = -0.01,
        jit_enable: bool = True,
        slack_priority_scale: float = None,
        params: Union[dict, None] = None,
        enabled: bool = True,
        **kwargs
    ):
        self._alpha = alpha
        self._alpha_negative = alpha_negative
        self.bias = bias
        self.bias_negative = bias_negative
        self.jit_enable = jit_enable
        self.slack_priority_scale = slack_priority_scale
        self.params = params
        if self.params is None:
            self.params = {}
        self.enabled = enabled
        self.subsample_config = SubsampleConfigValidator(**kwargs)
        self._compose()

    def _compose(self):
        if self.jit_enable:
            self._compute_fn = jit(self._compute)
            self.phi = jit(self._phi)
            self._grad_fn = jit(grad(self._compute))
        else:
            self._compute_fn = self._compute
            self.phi = self._phi
            self._grad_fn = grad(self._compute)

    def __call__(self, state: jnp.ndarray, params: dict) -> float:
        """Evaluates constraint function h(x)
        Considered satisfied when h(x) >= 0

        Parameters
        ----------
        state : jnp.ndarray
            current rta state of the system
        params : dict
            dict of user defined dynamically changing parameters

        Returns
        -------
        float:
            result of inequality constraint function
        """
        return self.compute(state, params)

    def compute(self, state: jnp.ndarray, params: dict) -> float:
        """Evaluates constraint function h(x)
        Considered satisfied when h(x) >= 0

        Parameters
        ----------
        state : jnp.ndarray
            current rta state of the system
        params : dict
            dict of user defined dynamically changing parameters

        Returns
        -------
        float:
            result of inequality constraint function
        """
        return self._compute_fn(state, params) + self.bias

    @abc.abstractmethod
    def _compute(self, state: jnp.ndarray, params: dict) -> float:
        """Custom implementation of constraint function

        !!! Note: To be compatible with jax jit compilation, must not rely on external states that are overwritten here
            or elsewhere after initialization

        Parameters
        ----------
        state : jnp.ndarray
            current rta state of the system
        params : dict
            dict of user defined dynamically changing parameters

        Returns
        -------
        float:
            result of inequality constraint function
        """
        raise NotImplementedError()

    def grad(self, state: jnp.ndarray, params: dict) -> jnp.ndarray:
        """
        Computes Gradient of Safety Constraint Function wrt x
        Required for ASIF methods

        Parameters
        ----------
        state : jnp.ndarray
            current rta state of the system
        params : dict
            dict of user defined dynamically changing parameters

        Returns
        -------
        jnp.ndarray:
            gradient of constraint function wrt x. Shape of (n, n) where n = state.vector.size.
        """
        return self._grad_fn(state, params)

    def alpha(self, x: float) -> float:
        """Evaluates Strengthing function to soften Nagumo's condition outside of constraint set boundary
        Pass through for class member alpha

        Parameters
        ----------
        x : float
            output of constraint function

        Returns
        -------
        float
            Strengthening Function output
        """
        assert isinstance(self._alpha, ConstraintStrengthener), "alpha must be an instance/sub-class of ConstraintStrenthener"
        return self._alpha(x)

    def alpha_negative(self, x: float) -> float:
        """Strengthing function when the constraint value is negative.
        Used to stabilize the system back to the safe set.

        Parameters
        ----------
        x : float
            output of constraint function

        Returns
        -------
        float
            Strengthening Function output
        """
        if self._alpha_negative is None:
            assert isinstance(self._alpha, ConstraintStrengthener), "alpha must be an instance/sub-class of ConstraintStrenthener"
            return self._alpha(x) * 10

        return self._alpha_negative(x)

    def _phi(self, state: jnp.ndarray, params: dict) -> float:
        """Evaluates constraint function phi(x).
        Considered satisfied when phi(x) >= 0, where phi is not guaranteed to be control invariant.
        Not used by RTA to enforce the constraint, but rather is useful for logging and plotting.
        By default, returns the value of _compute without the bias.

        Parameters
        ----------
        state : jnp.ndarray
            current rta state of the system
        params : dict
            dict of user defined dynamically changing parameters

        Returns
        -------
        float:
            result of inequality constraint function
        """
        return self._compute_fn(state, params)


class ConstraintStrengthener(abc.ABC):
    """Strengthing function used to soften Nagumo's condition outside of constraint set boundary

    Required for ASIF methods
    """

    @abc.abstractmethod
    def __call__(self, x) -> float:
        """
        Compute Strengthening Function (Required for ASIF):
        Must be monotonically decreasing with f(0) = 0

        !!! Note: To be compatible with jax jit compilation, must not rely on external states that are overwritten here
            or elsewhere after initialization

        Returns
        -------
        float
            output of monotonically decreasing constraint strengther function
        """
        raise NotImplementedError


class ConstraintMagnitudeStateLimit(ConstraintModule):
    """
    Generic state vector element magnitude limit constraint
    Builds a constraint function for |state[state_index]| <= limit_val

    Parameters
    ----------
    limit_val : float
        state vector element limit constraint value
    state_index: int
        index/indices of state vector element to apply limit constraint to
        Currently only supports single indices
    alpha : ConstraintStrengthener
        Constraint Strengthener object used for ASIF methods. Required for ASIF methods.
        Defaults to PolynomialConstraintStrengthener([0, 0.0005, 0, 0.001])
    """

    def __init__(self, limit_val: float, state_index: int, alpha: ConstraintStrengthener = None, **kwargs: Any):
        self.state_index = state_index

        if alpha is None:
            alpha = PolynomialConstraintStrengthener([0, 0.0005, 0, 0.001])
        super().__init__(alpha=alpha, params={'limit_val': limit_val}, **kwargs)

    def _compute(self, state: jnp.ndarray, params: dict) -> float:
        return params['limit_val']**2 - state[self.state_index]**2


class ConstraintMaxStateLimit(ConstraintModule):
    """
    Generic state vector element maximum limit constraint
    Builds a constraint function for state[state_index] <= limit_val

    Parameters
    ----------
    limit_val : float
        state vector element limit constraint value
    state_index: int
        index/indices of state vector element to apply limit constraint to
        Currently only supports single indices
    alpha : ConstraintStrengthener
        Constraint Strengthener object used for ASIF methods. Required for ASIF methods.
        Defaults to PolynomialConstraintStrengthener([0, 0.0005, 0, 0.001])
    """

    def __init__(self, limit_val: float, state_index: int, alpha: ConstraintStrengthener = None, **kwargs: Any):
        self.state_index = state_index

        if alpha is None:
            alpha = PolynomialConstraintStrengthener([0, 0.0005, 0, 0.001])
        super().__init__(alpha=alpha, params={'limit_val': limit_val}, **kwargs)

    def _compute(self, state: jnp.ndarray, params: dict) -> float:
        return params['limit_val'] - state[self.state_index]


class ConstraintMinStateLimit(ConstraintModule):
    """
    Generic state vector element minimum limit constraint
    Builds a constraint function for state[state_index] >= limit_val

    Parameters
    ----------
    limit_val : float
        state vector element limit constraint value
    state_index: int
        index/indices of state vector element to apply limit constraint to
        Currently only supports single indices
    alpha : ConstraintStrengthener
        Constraint Strengthener object used for ASIF methods. Required for ASIF methods.
        Defaults to PolynomialConstraintStrengthener([0, 0.0005, 0, 0.001])
    """

    def __init__(self, limit_val: float, state_index: int, alpha: ConstraintStrengthener = None, **kwargs: Any):
        self.state_index = state_index

        if alpha is None:
            alpha = PolynomialConstraintStrengthener([0, 0.0005, 0, 0.001])
        super().__init__(alpha=alpha, params={'limit_val': limit_val}, **kwargs)

    def _compute(self, state: jnp.ndarray, params: dict) -> float:
        return state[self.state_index] - params['limit_val']


class PolynomialConstraintStrengthener(ConstraintStrengthener):
    """Implements strengthing function as polynomial function of x

    Parameters
    ----------
    coefs: list
        list of polynomial coefs. Arbitrary length.
        Results in strengthening function sum(coefs[i]*(x**i)) for i in range(0, len(coefs))
    """

    def __init__(self, coefs: list = None):
        assert isinstance(coefs, list) or coefs is None, "coefs must be a list of numbers"

        assert coefs is None or all((isinstance(i, numbers.Number) for i in coefs)), "coefs must be a list of numbers"

        if coefs is None:
            coefs = [0, 1]

        self.coefs = coefs

    def __call__(self, x: float) -> float:
        """Evaluates strengthening function

        Parameters
        ----------
        x : float
            output of inequality constraint function h(x)

        Returns
        -------
        float
            output of monotonically decreasing constraint strengther function
        """
        output = 0
        for n, c in enumerate(self.coefs):
            output += c * x**n
        return output


class HOCBFConstraint(ConstraintModule):
    """Constraint for use with Higher Order Control Barrier Functions
    Uses Jax to compute gradients

    Parameters
    ----------
    constraint: ConstraintModule
        Current constraint to transform into higher order CBF
    relative_degree: int
        Defines the relative degree of the constraint with respect to the system
    state_transition_system:
        Function to compute the control input matrix contribution to the system state's time derivative
    alpha: ConstraintStrengthener
        Constraint Strengthener object used for ASIF methods. Required for ASIF methods.
    """

    def __init__(
        self, constraint: ConstraintModule, relative_degree: int, state_transition_system, alpha: ConstraintStrengthener, **kwargs: Any
    ):
        self.initial_constraint = constraint
        hocbf_constraint_dict = {"constraint_0": constraint}
        for i in range(1, relative_degree):
            psi = HOCBFConstraintHelper(
                constraint=hocbf_constraint_dict["constraint_" + str(i - 1)], state_transition_system=state_transition_system, alpha=alpha
            )
            hocbf_constraint_dict["constraint_" + str(i)] = psi
        self.final_constraint = psi
        super().__init__(alpha=alpha, params=self.final_constraint.params, **kwargs)

    def _compute(self, state: jnp.ndarray, params: dict) -> float:
        return self.final_constraint(state, params)

    def _phi(self, state: jnp.ndarray, params: dict) -> float:
        return self.initial_constraint.phi(state, params)


class HOCBFConstraintHelper(ConstraintModule):
    """Constraint helper class for use with Higher Order Control Barrier Functions

    Parameters
    ----------
    constraint: ConstraintModule
        Current constraint to transform into higher order CBF
    state_transition_system:
        Function to compute the control input matrix contribution to the system state's time derivative
    alpha: ConstraintStrengthener
        Constraint Strengthener object used for ASIF methods. Required for ASIF methods.
    """

    def __init__(self, constraint: ConstraintModule, state_transition_system, alpha: ConstraintStrengthener, **kwargs: Any):
        self.constraint = constraint
        self.state_transition_system = state_transition_system
        super().__init__(alpha=alpha, params=self.constraint.params, **kwargs)

    def _compute(self, state: jnp.ndarray, params: dict) -> float:
        return self.constraint.grad(state, params) @ self.state_transition_system(state) + self.alpha(self.constraint(state, params))


class DirectInequalityConstraint():
    """Base class for inequality constraints, to be used in the QP for ASIF
    Constraints in the form: ineq_weight * u >= ineq_constant

    Parameters
    ----------
    params : dict, optional
        dict of parameters that can be changed during a simulation.
        These parameters do not have dynamics and are only adjusted by the user.
        These parameters cannot be class attributes due to jit.
    enabled : bool, optional
        Flag to enable/disable the constraint from being enforced by RTA.
        Default enabled.
    """

    def __init__(self, params: Union[dict, None] = None, enabled: bool = True):
        self.params = params
        if self.params is None:
            self.params = []
        self.enabled = enabled

    @abc.abstractmethod
    def ineq_weight(self, state: jnp.ndarray, params: dict) -> jnp.ndarray:
        """Inequality constraint weight array

        Parameters
        ----------
        state : jnp.ndarray
            system state
        params : dict
            dict of user defined dynamically changing parameters

        Returns
        -------
        jnp.ndarray
            1 x m array, where m is the length of the control vector
            ineq_weight * u >= ineq_constant
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def ineq_constant(self, state: jnp.ndarray, params: dict) -> float:
        """Inequality constraint constant

        Parameters
        ----------
        state : jnp.ndarray
            system state
        params : dict
            dict of user defined dynamically changing parameters

        Returns
        -------
        float
            ineq_weight * u >= ineq_constant
        """
        raise NotImplementedError()


class DiscreteCBFConstraint():
    """Constraint helper class for converting constraints into discrete Control Barrier Functions (CBFs)

    Parameters
    ----------
    constraint: ConstraintModule
        Current constraint to transform into a discrete CBF
    next_state_fn: Any
        Function to compute the next state of the system. Must be differentiable using Jax.
    control_dim : int
        length of control vector
    slack_idx : int
        Index of slack variable to be used. None if no slack variable.
    """

    def __init__(self, constraint: ConstraintModule, next_state_fn: Any, control_dim: int, slack_idx: int = None):
        self.constraint = constraint
        self.next_state_fn = next_state_fn
        self.control_dim = control_dim
        self.slack_idx = slack_idx
        self.cbf = jit(self.cbf_fn)
        self.jac = jit(jacrev(self.cbf_fn))

    def cbf_fn(self, control: jnp.ndarray, state: jnp.ndarray, step_size: float, params: dict):
        """Discrete CBF

        Parameters
        ----------
        control: jnp.ndarray
            Control vector of the system
        state: jnp.ndarray
            Current rta state of the system
        step_size: float
            Time duration over which filtered control will be applied to actuators
        params : dict
            dict of user defined dynamically changing parameters

        Returns
        -------
        float:
            value of the discrete CBF
        """
        next_state = self.next_state_fn(state, step_size, control[0:self.control_dim])
        h = (self.constraint(next_state, params) - self.constraint(state, params)) / step_size + self.constraint.alpha(
            self.constraint(state, params)
        )
        if self.slack_idx is not None:
            h -= control[self.control_dim + self.slack_idx]
        return h
