"""
This module defines the base RTAState class for wrapping state vectors with custom class interfaces
"""
from __future__ import annotations

import jax.numpy as jnp


class RTAStateWrapper:
    """rta state wrapper used by rta algorithms to access/modify jax ndarray elements via named getter/setters

    Parameters
    ----------
    vector : jnp.ndarray
        numpy vector representation of state to copy into an RTA state
    """

    def __init__(self, vector: jnp.ndarray):
        self.vector = jnp.copy(vector)

    @property
    def size(self) -> int:
        """returns size of state vector"""
        return self.vector.size

    def __len__(self) -> int:
        return self.size
