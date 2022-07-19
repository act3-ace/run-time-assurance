"""
This module defines the base RTA constraints and constraint strengthener classes and includes some standard implementations
"""
from __future__ import annotations

import abc
import numbers

import jax.numpy as jnp
from jax import grad, jit


class ConstraintModule(abc.ABC):
    """Base class implementation for safety constraints
    Implements constraint with form of h(x) >= 0

    Parameters
    ----------
    alpha : ConstraintStrengthener
        Constraint Strengthener object used for ASIF methods. Required for ASIF methods.
    """

    def __init__(self, alpha: ConstraintStrengthener = None):
        assert isinstance(alpha, ConstraintStrengthener), "alpha must be an instance/sub-class of ConstraintStrenthener"
        self._alpha = alpha
        self._compose()

    def _compose(self):
        self._compute_fn = jit(self._compute)
        self._grad_fn = jit(grad(self._compute))

    def __call__(self, state: jnp.ndarray) -> float:
        """Evaluates constraint function h(x)
        Considered satisfied when h(x) >= 0

        Parameters
        ----------
        state : jnp.ndarray
            current rta state of the system

        Returns
        -------
        float:
            result of inequality constraint function
        """
        return self._compute_fn(state)

    def compute(self, state: jnp.ndarray) -> float:
        """Evaluates constraint function h(x)
        Considered satisfied when h(x) >= 0

        Parameters
        ----------
        state : jnp.ndarray
            current rta state of the system

        Returns
        -------
        float:
            result of inequality constraint function
        """
        return self._compute_fn(state)

    @abc.abstractmethod
    def _compute(self, state: jnp.ndarray) -> float:
        """Custom implementation of constraint function

        !!! Note: To be compatible with jax jit compilation, must not rely on external states that are overwritten here
            or elsewhere after initialization

        Parameters
        ----------
        state : jnp.ndarray
            current rta state of the system

        Returns
        -------
        float:
            result of inequality constraint function
        """
        raise NotImplementedError()

    def grad(self, state: jnp.ndarray) -> jnp.ndarray:
        """
        Computes Gradient of Safety Constraint Function wrt x
        Required for ASIF methods

        Parameters
        ----------
        state : jnp.ndarray
            current rta state of the system

        Returns
        -------
        jnp.ndarray:
            gradient of constraint function wrt x. Shape of (n, n) where n = state.vector.size.
        """
        return self._grad_fn(state)

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
        if self._alpha is None:
            return None

        return self._alpha(x)


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

    def __init__(self, limit_val: float, state_index: int, alpha: ConstraintStrengthener = None):
        self.limit_val = limit_val
        self.state_index = state_index

        if alpha is None:
            alpha = PolynomialConstraintStrengthener([0, 0.0005, 0, 0.001])
        super().__init__(alpha=alpha)

    def _compute(self, state: jnp.ndarray) -> float:
        return self.limit_val**2 - state[self.state_index]**2


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

    def __init__(self, limit_val: float, state_index: int, alpha: ConstraintStrengthener = None):
        self.limit_val = limit_val
        self.state_index = state_index

        if alpha is None:
            alpha = PolynomialConstraintStrengthener([0, 0.0005, 0, 0.001])
        super().__init__(alpha=alpha)

    def _compute(self, state: jnp.ndarray) -> float:
        return self.limit_val - state[self.state_index]


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

    def __init__(self, limit_val: float, state_index: int, alpha: ConstraintStrengthener = None):
        self.limit_val = limit_val
        self.state_index = state_index

        if alpha is None:
            alpha = PolynomialConstraintStrengthener([0, 0.0005, 0, 0.001])
        super().__init__(alpha=alpha)

    def _compute(self, state: jnp.ndarray) -> float:
        return state[self.state_index] - self.limit_val


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
