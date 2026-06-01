"""Filtering techniques for dynamic environments (Sec. 3.3) -- the paper's key
contribution. Both return a boolean mask over beams: True = keep, False = discard.
The localizer then runs the normal product update over kept beams only.
"""

from __future__ import annotations

import numpy as np
from scipy.special import ndtr                      # standard normal CDF

from .belief import entropy
from .sensor import beam_likelihood_field, DEFAULT_PARAMS, _LOG_EPS


def entropy_filter(bel, scan, exp_dist, beam_offsets, p=None) -> np.ndarray:
    """Entropy filter (Eqs. 33-34), sensor-agnostic.

    For each beam, tentatively fold it into the *current* belief and measure the
    entropy change ``dH = H(L|s) - H(L)``. **Keep iff dH <= 0** (the reading
    confirms the belief / does not increase uncertainty); discard if dH > 0
    (it raises uncertainty -> probably an unmodeled object).

    NOTE: the paper's literal text says "exclude dH < 0", which contradicts its
    own next sentence ("only uses readings confirming the belief"). We implement
    the *intent* (discard dH > 0). Do not flip this sign.
    """
    if p is None:
        p = DEFAULT_PARAMS
    h0 = entropy(bel)
    keep = np.ones(len(scan), dtype=bool)
    for k, (m, off) in enumerate(zip(scan, beam_offsets)):
        field = beam_likelihood_field(m, exp_dist, p)
        log_l = np.roll(np.log(field + _LOG_EPS), -off, axis=2)
        b2 = bel * np.exp(log_l - log_l.max())
        s = b2.sum()
        if s <= 0:
            keep[k] = False
            continue
        keep[k] = (entropy(b2 / s) - h0) <= 0.0
    return keep


def distance_filter(bel, scan, exp_dist, beam_offsets, sigma, gamma=0.99) -> np.ndarray:
    """Distance filter (Eqs. 35-36), for range sensors.

    ``P_short(d | l) = P(o_l > d | l) = 1 - Phi((d - o_l)/sigma)`` is the
    probability that the measurement is shorter than expected at pose ``l``;
    averaged over the belief, ``P_short(d) = sum_l P_short(d|l) Bel(l)``.
    **Discard the beam if P_short(d) > gamma** (default 0.99): the reading is,
    with near certainty, shorter than the map predicts -> an unmapped object.
    """
    keep = np.ones(len(scan), dtype=bool)
    for k, (m, off) in enumerate(zip(scan, beam_offsets)):
        o = np.roll(exp_dist, -off, axis=2)            # o_l per pose for beam k
        p_short_field = 1.0 - ndtr((m - o) / sigma)
        p_short = float((p_short_field * bel).sum())
        keep[k] = p_short <= gamma
    return keep
