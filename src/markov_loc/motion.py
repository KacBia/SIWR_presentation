"""Motion (action) model -- the prediction step (Sec. 3.1, Eq. 21).

    Bel^-(l) = integral P(l | l', a) Bel(l') dl'

We use the "drive, then turn" odometry control ``a = (dtrans, drot)`` and realise
the convolution on the grid as **shift + blur**:

1. translate each orientation slice by ``dtrans`` *along its own heading*
   (each theta-slice shifts by a different ``(dtrans cos theta, dtrans sin theta)``);
2. blur spatially -- translational noise (variance grows with ``|dtrans|``);
3. roll the theta axis by ``drot`` (rotation moves mass *between* slices), with
   fractional interpolation so the mean rotation is not quantised to the grid;
4. blur circularly along theta -- rotational noise.

The coupling of per-slice translation directions with orientation uncertainty is
what produces the "banana-shaped" densities of Fig. 3.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

# Default motion-noise model. Standard deviations grow with the motion magnitude
# (mixture of zero-centred Gaussians, per the paper).
DEFAULT_NOISE = dict(
    a_trans=0.12,   # translational sigma per metre travelled        [m/m]
    a_rot=0.10,     # rotational sigma per radian turned             [rad/rad]
    a_rt=0.06,      # rotational sigma induced per metre travelled   [rad/m]
    min_xy=0.02,    # floor on translational sigma                   [m]
    min_th=0.02,    # floor on rotational sigma                      [rad]
)


def predict(bel: np.ndarray, control, res: float, n_theta: int, noise=None) -> np.ndarray:
    """Apply one motion update. ``control = (dtrans [m], drot [rad])``.

    Returns the (unnormalised) predicted belief; the localizer re-masks and
    normalises afterwards.
    """
    if noise is None:
        noise = DEFAULT_NOISE
    dtrans, drot = float(control[0]), float(control[1])
    dtheta = 2 * np.pi / n_theta

    # 1. translation: each theta-slice shifts along its own heading.
    out = np.empty_like(bel)
    for ith in range(n_theta):
        th = ith * dtheta
        dx, dy = dtrans * np.cos(th), dtrans * np.sin(th)
        # ndimage.shift takes (row=y, col=x) shifts, in cells.
        out[:, :, ith] = ndimage.shift(bel[:, :, ith], (dy / res, dx / res),
                                       order=1, mode="constant", cval=0.0)

    # 2. translational noise: spatial Gaussian blur (no blur across theta here).
    sig_xy = (noise["a_trans"] * abs(dtrans) + noise["min_xy"]) / res
    if sig_xy > 0:
        out = ndimage.gaussian_filter(out, sigma=(sig_xy, sig_xy, 0.0), mode="constant")

    # 3. rotation: roll theta axis by drot, interpolating the fractional bin.
    k = drot / dtheta
    k0 = int(np.floor(k))
    frac = k - k0
    out = (1 - frac) * np.roll(out, k0, axis=2) + frac * np.roll(out, k0 + 1, axis=2)

    # 4. rotational noise: circular Gaussian blur along theta.
    sig_th = (noise["a_rot"] * abs(drot) + noise["a_rt"] * abs(dtrans)
              + noise["min_th"]) / dtheta
    if sig_th > 0:
        out = ndimage.gaussian_filter1d(out, sigma=sig_th, axis=2, mode="wrap")

    np.clip(out, 0.0, None, out=out)
    return out
