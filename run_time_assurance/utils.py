"""Provides util functions for the run-time-assurance library"""
import jax.numpy as jnp
from jax import jit


@jit
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
