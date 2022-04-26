from __future__ import annotations
import numpy as np

class RTAState:
    """rta state used by rta algorithms to determine safety and compute safe actions
    wraps a 1d numpy vector providing an access point for getters/setters

    Parameters
    ----------
    vector : np.ndarray
        numpy vector representation of state to copy into an RTA state
    """

    def __init__(self, vector: np.ndarray):
        self._vector = np.copy(vector)

    def copy(self) -> RTAState:
        """Returns a deep copy of itself

        Returns
        -------
        RTAState
            deep copy of the object
        """
        cls = self.__class__
        return cls(vector=np.copy(self._vector))

    @property
    def vector(self) -> np.ndarray:
        """getter for raw rta state numpy vector. Provides a deep copy"""
        return np.copy(self._vector)

    @vector.setter
    def vector(self, val):
        """sets rta state with a raw numpy vector"""
        self._vector = np.copy(val)

    @property
    def size(self) -> int:
        """returns size of state vector"""
        return self._vector.size

    def __len__(self) -> int:
        return self.size()