"""Grid-based Markov localizer -- the Bayes filter of Table 1 tied together.

Loop: ``predict(odometry)`` (motion) and ``correct(scan, filt=...)`` (perception).
The belief is masked to free cells and a small uniform floor is mixed in after
each correction so no cell reaches exactly zero -- this is what lets the robot
recover from the "kidnapped robot" problem.
"""

from __future__ import annotations

import numpy as np

from . import motion, sensor, filters, belief
from .occ_map import OccupancyMap


def fan_offsets(n_theta: int, fov_deg: float = 180.0) -> list[int]:
    """Beam offsets (in angular-grid units) for a forward fan of width ``fov_deg``,
    one beam per orientation bin. Offsets are relative to the robot heading."""
    half = int(round((fov_deg / 360.0) * n_theta) // 2)
    return list(range(-half, half + 1))


class MarkovLocalizer:
    def __init__(self, occ_map: OccupancyMap, n_theta: int = 36, max_range: float = 8.0,
                 beam_offsets=None, sensor_params=None, motion_noise=None,
                 fov_deg: float = 180.0, floor: float = 1e-4, exp_dist=None):
        self.map = occ_map
        self.nth = n_theta
        self.res = occ_map.res
        self.max_range = max_range
        self.p = dict(sensor.DEFAULT_PARAMS)
        if sensor_params:
            self.p.update(sensor_params)
        self.p["max_range"] = max_range
        self.noise = dict(motion.DEFAULT_NOISE)
        if motion_noise:
            self.noise.update(motion_noise)
        self.beam_offsets = (list(beam_offsets) if beam_offsets is not None
                             else fan_offsets(n_theta, fov_deg))
        self.floor = floor

        # static, computed once and shared by sensor model + both filters
        # (may be passed in to reuse across localizers on the same map)
        self.exp_dist = (exp_dist if exp_dist is not None
                         else occ_map.precompute_expected_distances(n_theta, max_range))
        self.free3 = np.broadcast_to(occ_map.free_mask[:, :, None],
                                     (occ_map.ny, occ_map.nx, n_theta))
        self._uniform = belief.uniform_belief(occ_map, n_theta)
        self.bel = self._uniform.copy()

    # -- belief setup -------------------------------------------------------
    def set_uniform(self):
        self.bel = self._uniform.copy()

    def set_gaussian(self, pose, sigma_xy=0.3, sigma_th=0.25):
        self.bel = belief.gaussian_belief(self.map, self.nth, pose, sigma_xy, sigma_th)

    # -- Bayes filter steps -------------------------------------------------
    def predict(self, control):
        b = motion.predict(self.bel, control, self.res, self.nth, self.noise)
        self.bel = self._clean(b)

    def correct(self, scan, filt=None):
        """Fold a scan into the belief. ``filt`` in {None, 'entropy', 'distance'}.
        Returns the beam keep-mask (or None)."""
        mask = None
        if filt == "entropy":
            mask = filters.entropy_filter(self.bel, scan, self.exp_dist,
                                          self.beam_offsets, self.p)
        elif filt == "distance":
            mask = filters.distance_filter(self.bel, scan, self.exp_dist,
                                           self.beam_offsets, self.p["sigma"])
        elif filt is not None:
            raise ValueError(f"unknown filter {filt!r}")

        log_l = sensor.scan_log_likelihood(scan, self.exp_dist, self.beam_offsets,
                                           self.p, beam_mask=mask)
        b = self.bel * np.exp(log_l - log_l.max())
        self.bel = self._clean(b, with_floor=True)
        return mask

    # -- estimates ----------------------------------------------------------
    def map_estimate(self):
        return belief.map_estimate(self.bel, self.res, self.nth)

    def entropy(self):
        return belief.entropy(self.bel)

    def xy_marginal(self):
        return belief.xy_marginal(self.bel)

    # -- internal -----------------------------------------------------------
    def _clean(self, b, with_floor=False):
        b = np.where(self.free3, b, 0.0)
        s = b.sum()
        b = b / s if s > 0 else self._uniform.copy()
        if with_floor and self.floor > 0:
            b = (1.0 - self.floor) * b + self.floor * self._uniform
        return b
