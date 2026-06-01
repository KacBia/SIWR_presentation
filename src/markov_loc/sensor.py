"""Perception model for proximity sensors -- the correction step (Sec. 3.2).

For a single beam the likelihood ``P(s^k | l)`` is a mixture (Eqs. 22-23):

* **hit**   -- Gaussian about the expected distance ``o_l`` (known map obstacle);
* **short** -- exponential favouring readings *shorter* than ``o_l`` (an unknown
               / dynamic obstacle, e.g. a person);
* **max**   -- spike at the maximum range (no return);
* **rand**  -- small uniform term (sensor glitches; also a probability floor).

A full scan factorises under conditional independence given the pose:
``P(s | l) = prod_k P(s^k | l)`` -- we accumulate it in log-space for stability.
This independence is exactly what people violate, motivating the filters.

``exp_dist[iy, ix, idir]`` (the precomputed cache) gives ``o_l`` for every cell in
every *absolute* direction. Beam ``k`` of a pose at orientation ``ith`` points in
absolute direction ``ith + beam_offsets[k]``, so its per-pose likelihood field is
the absolute-direction field rolled by ``-beam_offsets[k]`` along the theta axis.
"""

from __future__ import annotations

import numpy as np

# Default beam (perception) model parameters.
DEFAULT_PARAMS = dict(
    sigma=0.20,        # hit-Gaussian std [m]
    lambda_short=1.0,  # short-reading exponential rate [1/m]
    z_hit=0.80,
    z_short=0.10,
    z_max=0.05,
    z_rand=0.05,
    max_range=8.0,
)

_LOG_EPS = 1e-300


def beam_likelihood_field(measured: float, exp_dist: np.ndarray, p=None) -> np.ndarray:
    """``P(measured | l)`` for every cell and *absolute* beam direction.

    Returns an array shaped like ``exp_dist`` ([iy, ix, n_theta]).
    """
    if p is None:
        p = DEFAULT_PARAMS
    o = exp_dist
    sigma, mr = p["sigma"], p["max_range"]

    p_hit = np.exp(-0.5 * ((measured - o) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))

    lam = p["lambda_short"]
    eta = 1.0 / (1.0 - np.exp(-lam * np.maximum(o, 1e-6)))    # normaliser on [0, o]
    p_short = np.where(measured <= o, eta * lam * np.exp(-lam * measured), 0.0)

    p_max = 1.0 if measured >= mr - 1e-9 else 0.0
    p_rand = 1.0 / mr

    return (p["z_hit"] * p_hit + p["z_short"] * p_short
            + p["z_max"] * p_max + p["z_rand"] * p_rand)


def scan_log_likelihood(scan, exp_dist, beam_offsets, p=None, beam_mask=None) -> np.ndarray:
    """Log of ``P(s | l)`` over the whole pose grid: sum of per-beam log fields.

    ``beam_mask`` (optional) selects which beams to include (used by the filters).
    """
    if p is None:
        p = DEFAULT_PARAMS
    L = np.zeros(exp_dist.shape)
    for k, (m, off) in enumerate(zip(scan, beam_offsets)):
        if beam_mask is not None and not beam_mask[k]:
            continue
        field = beam_likelihood_field(m, exp_dist, p)
        L += np.roll(np.log(field + _LOG_EPS), -off, axis=2)
    return L
