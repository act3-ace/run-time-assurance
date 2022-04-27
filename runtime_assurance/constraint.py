"""
This module defines the base RTA constraints and constraint strengthener classes and includes some standard implementations
"""
from __future__ import annotations

import abc
import numbers

import numpy as np

from runtime_assurance.state import RTAState


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

    def __call__(self, state: RTAState) -> float:
        """Evaluates constraint function h(x)
        Considered satisfied when h(x) >= 0

        Parameters
        ----------
        state : np.ndarray
            current rta state of the system

        Returns
        -------
        float:
            result of inequality constraint function
        """
        return self._compute(state)

    @abc.abstractmethod
    def _compute(self, state: RTAState) -> float:
        """Custom implementation of constraint function

        Parameters
        ----------
        state : np.ndarray
            current rta state of the system

        Returns
        -------
        float:
            result of inequality constraint function
        """
        raise NotImplementedError()

    def grad(self, state: RTAState) -> np.ndarray:
        """
        Computes Gradient of Safety Constraint Function wrt x
        Required for ASIF methods

        Parameters
        ----------
        state : np.ndarray
            current rta state of the system

        Returns
        -------
        np.ndarray:
            gradient of constraint function wrt x. Shape of (n, n) where n = state.vector.size.
        """
        raise NotImplementedError()

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

        Returns
        -------
        float
        """
        raise NotImplementedError


class ConstraintStateLimit(ConstraintModule):
    """
    Generic state vector element limit constraint
    Builds a constraint function for |state[state_index]| <= limit_val

    Parameters
    ----------
    limit_val : float
        state vector element limit constraint value
    state_index: int
        index/indices of state vector element to apply limit constraint to
        Currently only supports since indices
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

    def _compute(self, state: RTAState) -> float:
        state_vec = state.vector
        return self.limit_val**2 - state_vec[self.state_index]**2

    def grad(self, state: RTAState) -> np.ndarray:
        state_vec = state.vector

        gh = np.zeros((state_vec.size, state_vec.size), dtype=float)
        gh[self.state_index, self.state_index] = -2
        g = gh @ state_vec
        return g


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
        """
        output = 0
        for n, c in enumerate(self.coefs):
            output += c * x**n
        return output
