"""Occupancy-grid map + ray-casting for grid-based Markov localization.

Conventions used throughout the package
----------------------------------------
* World is a rectangle ``width x height`` metres, discretised into ``nx x ny``
  square cells of side ``resolution`` (metres).
* 2D arrays are indexed ``[iy, ix]`` (row = y, column = x) so that
  ``matplotlib.imshow(..., origin="lower")`` shows them the right way up.
* The belief is a 3D array ``[iy, ix, ith]`` over poses ``(x, y, theta)``.
* ``theta`` is measured counter-clockwise from the +x axis; orientation bin
  ``ith`` corresponds to angle ``ith * 2*pi/n_theta``.
* Cell ``(ix, iy)`` has its centre at world coordinates
  ``((ix + 0.5) * res, (iy + 0.5) * res)``.
"""

from __future__ import annotations

import numpy as np


class OccupancyMap:
    """A static 2D occupancy grid the robot localises against."""

    def __init__(self, occ: np.ndarray, resolution: float):
        self.occ = np.asarray(occ, dtype=bool)          # [ny, nx], True = obstacle
        self.ny, self.nx = self.occ.shape
        self.res = float(resolution)
        self.width = self.nx * self.res
        self.height = self.ny * self.res

    # -- coordinate helpers -------------------------------------------------
    @property
    def free_mask(self) -> np.ndarray:
        """Boolean [ny, nx] mask, True where a cell is free (drivable)."""
        return ~self.occ

    def world_to_cell(self, x: float, y: float) -> tuple[int, int]:
        return int(np.floor(x / self.res)), int(np.floor(y / self.res))  # (ix, iy)

    def cell_center(self, ix: int, iy: int) -> tuple[float, float]:
        return (ix + 0.5) * self.res, (iy + 0.5) * self.res

    def in_bounds(self, ix: int, iy: int) -> bool:
        return 0 <= ix < self.nx and 0 <= iy < self.ny

    def is_occupied(self, x: float, y: float) -> bool:
        """Occupancy at a world point; out-of-bounds counts as occupied."""
        ix, iy = self.world_to_cell(x, y)
        if not self.in_bounds(ix, iy):
            return True
        return bool(self.occ[iy, ix])

    # -- ray casting --------------------------------------------------------
    def ray_cast(self, x: float, y: float, theta: float,
                 max_range: float, step: float | None = None) -> float:
        """Distance from (x, y) along ``theta`` to the nearest obstacle.

        This is the expected measurement ``o_l`` of the paper (Sec. 3.2).

        Note: this is a sampling ray-caster. Rays that thread an obstacle
        *corner* exactly on a cell boundary are floating-point ambiguous (a
        handful of cells per map); that is harmless model noise, absorbed by the
        sensor model's variance. The localizer and simulator stay mutually
        consistent because the localizer always reads the precomputed cache.
        """
        if step is None:
            step = 0.25 * self.res
        c, s = np.cos(theta), np.sin(theta)
        d = 0.0
        while d <= max_range:
            if self.is_occupied(x + d * c, y + d * s):
                return d
            d += step
        return max_range


    #TODO MAYBE remove this 
    def precompute_expected_distances(self, n_theta: int, max_range: float,
                                      step: float | None = None) -> np.ndarray:
        """Pre-compute ``o_l`` for every cell and every absolute beam direction.

        Returns ``exp_dist[iy, ix, idir]`` = expected distance from the centre of
        cell (ix, iy) along absolute direction ``idir * 2*pi/n_theta``.

        Because the map is static this is computed **once** and shared by the
        sensor model, both filters, and the simulator's ground-truth scan. Beam
        bearings are aligned to this angular grid, so a scan from pose
        ``(ix, iy, ith)`` reads ``exp_dist[iy, ix, (ith + beam_offset) % n_theta]``
        -- a lookup instead of a fresh ray-cast (the only Sec. 3.4 optimisation
        we keep.
        """
        if step is None:
            step = 0.25 * self.res
        dtheta = 2.0 * np.pi / n_theta
        iy, ix = np.mgrid[0:self.ny, 0:self.nx]
        x0 = (ix + 0.5) * self.res                       # [ny, nx]
        y0 = (iy + 0.5) * self.res
        n_steps = int(np.ceil(max_range / step)) + 1

        exp_dist = np.full((self.ny, self.nx, n_theta), max_range, dtype=np.float32)
        for idir in range(n_theta):
            th = idir * dtheta
            c, s = np.cos(th), np.sin(th)
            hit = np.zeros((self.ny, self.nx), dtype=bool)
            dist = np.full((self.ny, self.nx), max_range, dtype=np.float32)
            for k in range(n_steps):
                d = k * step
                if d > max_range:
                    break
                jx = np.floor((x0 + d * c) / self.res).astype(int)
                jy = np.floor((y0 + d * s) / self.res).astype(int)
                inb = (jx >= 0) & (jx < self.nx) & (jy >= 0) & (jy < self.ny)
                occ_here = np.ones((self.ny, self.nx), dtype=bool)   # OOB = occupied
                occ_here[inb] = self.occ[np.clip(jy, 0, self.ny - 1)[inb],
                                         np.clip(jx, 0, self.nx - 1)[inb]]
                newly = occ_here & ~hit
                dist[newly] = d
                hit |= newly
                if hit.all():
                    break
            exp_dist[:, :, idir] = dist
        exp_dist[self.occ] = 0.0
        return exp_dist
