"""Example worlds (occupancy maps) the robot localises in.

Maps are defined in code (in metres) so they are reproducible. ``office_map`` is
deliberately **non-symmetric** so global localization converges to a single
pose; ``symmetric_corridor`` has repeated features so the belief correctly stays
multi-modal (a verification case from the plan).
"""

from __future__ import annotations

import numpy as np

from .occ_map import OccupancyMap


def _blank(width: float, height: float, res: float):
    nx = int(round(width / res))
    ny = int(round(height / res))
    return np.zeros((ny, nx), dtype=bool)


def _rect(occ: np.ndarray, res: float, x0: float, y0: float, x1: float, y1: float):
    """Fill the cells overlapping the world-rectangle [x0,x1] x [y0,y1] (metres)."""
    ny, nx = occ.shape
    ix0, ix1 = max(0, int(np.floor(x0 / res))), min(nx, int(np.ceil(x1 / res)))
    iy0, iy1 = max(0, int(np.floor(y0 / res))), min(ny, int(np.ceil(y1 / res)))
    occ[iy0:iy1, ix0:ix1] = True


def _outer_walls(occ, res, width, height, wall):
    _rect(occ, res, 0, 0, width, wall)
    _rect(occ, res, 0, height - wall, width, height)
    _rect(occ, res, 0, 0, wall, height)
    _rect(occ, res, width - wall, 0, width, height)


def office_map(res: float = 0.15, width: float = 10.0, height: float = 10.0,
               wall: float = 0.15) -> OccupancyMap:
    """A ~10x10 m office: outer walls, two rooms, a doorway, a pillar and an
    L-shaped feature. Asymmetric on purpose."""
    occ = _blank(width, height, res)
    _outer_walls(occ, res, width, height, wall)
    # horizontal divider at y=6 with a doorway gap (x in 4.0..5.5)
    _rect(occ, res, 0.0, 6.0, 4.0, 6.0 + wall)
    _rect(occ, res, 5.5, 6.0, width, 6.0 + wall)
    # vertical room divider, offset to break left/right symmetry
    _rect(occ, res, 4.0, wall, 4.0 + wall, 3.5)
    # free-standing pillar
    _rect(occ, res, 7.0, 2.0, 7.8, 2.8)
    # L-shaped feature, upper area
    _rect(occ, res, 1.5, 8.0, 3.0, 8.0 + wall)
    _rect(occ, res, 3.0 - wall, 7.0, 3.0, 8.0 + wall)
    return OccupancyMap(occ, res)


def symmetric_corridor(res: float = 0.15, width: float = 12.0, height: float = 4.0,
                       wall: float = 0.15) -> OccupancyMap:
    """A corridor with regularly spaced, identical niches -> ambiguous, so the
    belief stays multi-modal under global localization."""
    occ = _blank(width, height, res)
    _outer_walls(occ, res, width, height, wall)
    for cx in (2.0, 4.0, 6.0, 8.0, 10.0):
        _rect(occ, res, cx - 0.4, height - 0.8, cx + 0.4, height - wall)  # top niche
        _rect(occ, res, cx - 0.4, wall, cx + 0.4, 0.8)                    # bottom niche
    return OccupancyMap(occ, res)
