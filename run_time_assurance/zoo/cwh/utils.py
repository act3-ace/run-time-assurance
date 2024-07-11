"""Utility functions for the CWH Zoo example."""

import typing
import numpy as np


def generate_cwh_matrices(
    m: float, n: float, mode: str = "2d"
) -> typing.Tuple[np.ndarray, np.ndarray]:
    """Generates A and B Matrices from Clohessy-Wiltshire linearized dynamics of dx/dt = Ax + Bu

    Parameters
    ----------
    m : float
        mass in kg of spacecraft
    n : float
        orbital mean motion in rad/s of current Hill's reference frame
    mode : str, optional
        dimensionality of dynamics matrices. '2d' or '3d', by default '2d'

    Returns
    -------
    np.ndarray
        A dynamics matrix
    np.ndarray
        B dynamics matrix
    """
    assert mode in ["2d", "3d"], "mode must be on of ['2d', '3d']"
    if mode == "2d":
        A = np.array(
            [
                [0, 0, 1, 0],
                [0, 0, 0, 1],
                [3 * n**2, 0, 0, 2 * n],
                [0, 0, -2 * n, 0],
            ],
            dtype=np.float64,
        )

        B = np.array(
            [
                [0, 0],
                [0, 0],
                [1 / m, 0],
                [0, 1 / m],
            ],
            dtype=np.float64,
        )
    else:
        A = np.array(
            [
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
                [3 * n**2, 0, 0, 0, 2 * n, 0],
                [0, 0, 0, -2 * n, 0, 0],
                [0, 0, -(n**2), 0, 0, 0],
            ],
            dtype=np.float64,
        )

        B = np.array(
            [
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
                [1 / m, 0, 0],
                [0, 1 / m, 0],
                [0, 0, 1 / m],
            ],
            dtype=np.float64,
        )

    return A, B
