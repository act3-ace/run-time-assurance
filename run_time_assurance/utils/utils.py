"""Provides util functions for the run-time-assurance library"""
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax import jit


@partial(jit, static_argnames=['delta'])
def norm_with_delta(x: jnp.ndarray, delta: float):
    """
    Computes the norm of the vector with a small positive delta factor added inside the square root.
    Allows norm function to be differentiable at x = 0

    Parameters
    ----------
    x : jnp.ndarray
        input vector to compute norm of
    delta : float
        Small positive delta value to add offset to norm square root

    Returns
    -------
    float
        vector norm value
    """
    return jnp.sqrt(jnp.sum(jnp.square(x)) + delta)


@jit
def to_jnp_array_jit(x: np.ndarray) -> jnp.ndarray:
    """
    Converts a numpy array to a jax numpy array with a jit compiled function
    Allows significantly faster jax numpy array conversion when called repeatedly with an input of the same shape

    Parameters
    ----------
    x : np.ndarray
        input numpy array to be converted

    Returns
    -------
    jnp.ndarray
        jax numpy version of the input array
    """
    return jnp.array(x)


@partial(jit, static_argnames=['axis'])
def jnp_stack_jit(arrays, axis: int = 0) -> jnp.ndarray:
    """Apples a jit compiled version of jax numpy stack

    Parameters
    ----------
    arrays : Sequence of array_likes
        Array to be stacked together into a single jnp ndarray
    axis : int, optional
        axis across which to stack arrays, by default 0

    Returns
    -------
    jnp.ndarray
        stack array of input array sequence
    """
    return jnp.stack(arrays, axis=axis)


@jit
def add_dim_jit(x: jnp.ndarray) -> jnp.ndarray:
    """
    Add a dimension to a 1d jax array

    Parameters
    ----------
    x : np.ndarray
        input array of shape (N,)

    Returns
    -------
    jnp.ndarray
        output array of shape (1, N)
    """
    return x[None, :]


def disable_all_jax_jit():
    """Disables all jit compilation, useful for debugging
    """
    jax.config.update('jax_disable_jit', True)


class SolverError(Exception):
    """Exception for when solver does not find a solution
    """

    def __str__(self):
        return "SolverError: Solver could not find a solution"


class SolverWarning(UserWarning):
    """Warning for when solver does not find a solution
    """

    def __str__(self):
        return "**Warning! Solver could not find a solution, passing desired control**"
