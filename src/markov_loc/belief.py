"""The belief ``Bel(L)`` over poses and operations on it.

The belief is a 3D array ``bel[iy, ix, ith]`` approximating the density
``P(L = (x, y, theta))`` on the grid (Sec. 3.4 of the paper).
"""

from __future__ import annotations

import numpy as np

from .occ_map import OccupancyMap


def uniform_belief(m: OccupancyMap, n_theta: int) -> np.ndarray:
    """Uniform over all free poses -- the prior for global localization."""
    bel = np.zeros((m.ny, m.nx, n_theta))
    bel[m.free_mask, :] = 1.0
    return normalize(bel)


def gaussian_belief(m: OccupancyMap, n_theta: int, pose, sigma_xy: float = 0.3,
                    sigma_th: float = 0.25) -> np.ndarray:
    """A belief peaked at ``pose = (x, y, theta)`` (known-start prior / banana demo)."""
    x, y, th = pose
    iy, ix = np.mgrid[0:m.ny, 0:m.nx]
    xs, ys = (ix + 0.5) * m.res, (iy + 0.5) * m.res
    g_xy = np.exp(-((xs - x) ** 2 + (ys - y) ** 2) / (2 * sigma_xy ** 2))
    ths = np.arange(n_theta) * (2 * np.pi / n_theta)
    d_th = np.angle(np.exp(1j * (ths - th)))                 # wrapped angular diff
    g_th = np.exp(-d_th ** 2 / (2 * sigma_th ** 2))
    bel = g_xy[:, :, None] * g_th[None, None, :]
    bel[~m.free_mask, :] = 0.0
    return normalize(bel)


def normalize(bel: np.ndarray) -> np.ndarray:
    s = bel.sum()
    if s <= 0:
        raise ValueError("belief has zero mass; cannot normalize")
    return bel / s


def entropy(bel: np.ndarray) -> float:
    """Shannon entropy H(L) = -sum_l Bel(l) log Bel(l)  (Eq. 33, natural log)."""
    p = bel[bel > 0]
    return float(-(p * np.log(p)).sum())


def xy_marginal(bel: np.ndarray) -> np.ndarray:
    """Marginal over orientation -> [iy, ix], for visualisation as a heatmap."""
    return bel.sum(axis=2)


def map_estimate(bel: np.ndarray, res: float, n_theta: int):
    """Most likely pose (argmax cell) in world coordinates (x, y, theta)."""
    iy, ix, ith = np.unravel_index(int(np.argmax(bel)), bel.shape)
    return (ix + 0.5) * res, (iy + 0.5) * res, ith * (2 * np.pi / n_theta)


def mean_xy(bel: np.ndarray, res: float):
    """Expected (x, y) under the belief (useful when the belief is unimodal)."""
    marg = xy_marginal(bel)
    iy, ix = np.mgrid[0:marg.shape[0], 0:marg.shape[1]]
    return (float(((ix + 0.5) * res * marg).sum()),
            float(((iy + 0.5) * res * marg).sum()))
