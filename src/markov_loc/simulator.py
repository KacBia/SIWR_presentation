"""A minimal 2D robot simulator: ground-truth motion, noisy odometry and noisy
range scans against the static map plus optional unmapped "people" (disks).

The localizer never sees the people or the true pose -- only the odometry and the
(corrupted) scans, exactly as a real robot would.
"""

from __future__ import annotations

import numpy as np


def wrap(a: float) -> float:
    return (a + np.pi) % (2 * np.pi) - np.pi


def _ray_disk(x, y, ang, cx, cy, r, max_range):
    """Distance along (x,y,ang) to a disk centred (cx,cy) radius r, else inf."""
    ux, uy = np.cos(ang), np.sin(ang)
    fx, fy = x - cx, y - cy
    b = fx * ux + fy * uy
    c = fx * fx + fy * fy - r * r
    disc = b * b - c
    if disc < 0:
        return np.inf
    sq = np.sqrt(disc)
    t = -b - sq
    if t < 0:
        t = -b + sq
    if t < 0 or t > max_range:
        return np.inf
    return t


class Robot:
    def __init__(self, occ_map, pose, n_theta, beam_offsets, max_range,
                 trans_noise=0.05, rot_noise=0.03, sensor_noise=0.05, rng=None):
        self.map = occ_map
        self.nth = n_theta
        self.dtheta = 2 * np.pi / n_theta
        self.beam_offsets = list(beam_offsets)
        self.max_range = max_range
        self.trans_noise = trans_noise        # actual executed translation noise
        self.rot_noise = rot_noise            # actual executed rotation noise
        self.sensor_noise = sensor_noise      # range measurement std [m]
        self.rng = rng or np.random.default_rng()
        self.true_pose = np.array(pose, dtype=float)   # (x, y, theta)

    def move(self, control):
        """Execute ``control = (dtrans, drot)`` with real-world slip; advance the
        true pose. Returns the odometry handed to the localizer (the commanded
        motion -- its discrepancy from the truth is covered by the motion model).
        """
        dtrans, drot = control
        a_t = dtrans + self.rng.normal(0, self.trans_noise * abs(dtrans) + 1e-3)
        a_r = drot + self.rng.normal(0, self.rot_noise * abs(drot)
                                     + 0.01 * abs(dtrans) + 1e-3)
        x, y, th = self.true_pose
        nx_, ny_ = x + a_t * np.cos(th), y + a_t * np.sin(th)
        if not self.map.is_occupied(nx_, ny_):       # don't drive into walls
            x, y = nx_, ny_
        self.true_pose = np.array([x, y, wrap(th + a_r)])
        return (dtrans, drot)

    def teleport(self, pose):
        """Kidnap the robot to a new pose."""
        self.true_pose = np.array(pose, dtype=float)

    def scan(self, people=None):
        """Noisy range scan from the true pose; ``people`` is a list of
        (cx, cy, r) disks that occlude beams (unmapped dynamic obstacles)."""
        x, y, th = self.true_pose
        ranges = np.empty(len(self.beam_offsets))
        for i, off in enumerate(self.beam_offsets):
            ang = th + off * self.dtheta
            d = self.map.ray_cast(x, y, ang, self.max_range)
            if people:
                for (cx, cy, r) in people:
                    d = min(d, _ray_disk(x, y, ang, cx, cy, r, self.max_range))
            d += self.rng.normal(0, self.sensor_noise)
            ranges[i] = np.clip(d, 0.0, self.max_range)
        return ranges


def crowd_around(pose, n=8, radius=1.2, person_r=0.25, rng=None):
    """Place ``n`` people in a ring around ``pose`` -- a dense crowd that corrupts
    a large fraction of the beams (for the centerpiece filter demo)."""
    rng = rng or np.random.default_rng()
    x, y, _ = pose
    people = []
    for k in range(n):
        a = 2 * np.pi * k / n + rng.normal(0, 0.15)
        rr = radius + rng.normal(0, 0.2)
        people.append((x + rr * np.cos(a), y + rr * np.sin(a), person_r))
    return people
