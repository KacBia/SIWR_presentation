"""Matplotlib helpers: draw the world (map + robot + scan + people) and the
belief (orientation-marginal heatmap with a MAP-pose arrow)."""

from __future__ import annotations

import numpy as np

from .belief import xy_marginal


def _arrow(ax, pose, color, length=0.6, **kw):
    x, y, th = pose
    ax.annotate("", xy=(x + length * np.cos(th), y + length * np.sin(th)),
                xytext=(x, y),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=2.2), **kw)


def plot_world(ax, m, true_pose=None, scan=None, beam_offsets=None, n_theta=None,
               max_range=8.0, people=None, keep_mask=None, title=None):
    """Draw the static map, the (true) robot, its range scan and any people.

    Scan beams are coloured green (kept) / red (rejected) when ``keep_mask`` is
    given, else a single colour."""
    ax.clear()
    ax.imshow(m.occ, origin="lower", cmap="Greys", extent=[0, m.width, 0, m.height],
              vmin=0, vmax=1.4, interpolation="nearest")
    ax.set_xlim(0, m.width); ax.set_ylim(0, m.height)
    ax.set_aspect("equal"); ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    if title:
        ax.set_title(title)

    if people:
        for (cx, cy, r) in people:
            ax.add_patch(plt_circle(cx, cy, r, color="tab:orange", alpha=0.75, zorder=4))

    if scan is not None and true_pose is not None:
        x, y, th = true_pose
        dtheta = 2 * np.pi / n_theta
        for k, off in enumerate(beam_offsets):
            ang = th + off * dtheta
            d = scan[k]
            if keep_mask is None:
                col, a = "tab:cyan", 0.45
            else:
                col, a = ("limegreen", 0.7) if keep_mask[k] else ("red", 0.55)
            ax.plot([x, x + d * np.cos(ang)], [y, y + d * np.sin(ang)],
                    color=col, lw=0.8, alpha=a, zorder=3)

    if true_pose is not None:
        _arrow(ax, true_pose, "tab:blue")
        ax.plot(true_pose[0], true_pose[1], "o", color="tab:blue", ms=6, zorder=5)


def plot_belief(ax, m, bel, est_pose=None, true_pose=None, title=None, cmap="hot",
                gamma=0.45):
    """Heatmap of the orientation marginal Bel(x,y) with the MAP arrow (red) and
    the true pose (blue).

    A power-law display transform (``disp = (marg/max)**gamma``) is applied so a
    sharply peaked belief still "blooms" visibly and faint multi-modal mass shows
    -- otherwise a near-delta belief renders as an almost-black panel."""
    marg = bel if bel.ndim == 2 else xy_marginal(bel)
    mmax = float(marg.max())
    disp = (marg / mmax) ** gamma if mmax > 0 else marg
    ax.imshow(disp, origin="lower", cmap=cmap, vmin=0.0, vmax=1.0,
              extent=[0, m.width, 0, m.height], interpolation="nearest")
    # faint map overlay so structure is visible over the heatmap
    ax.contour(np.linspace(0, m.width, m.nx), np.linspace(0, m.height, m.ny),
               m.occ.astype(float), levels=[0.5], colors="deepskyblue",
               linewidths=0.6, alpha=0.6)
    ax.set_xlim(0, m.width); ax.set_ylim(0, m.height)
    ax.set_aspect("equal"); ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ax.set_title('')
    if title:
        ax.set_title(title)
    if est_pose is not None:
        _arrow(ax, est_pose, "red")
    if true_pose is not None:
        ax.plot(true_pose[0], true_pose[1], "o", color="deepskyblue",
                ms=7, mfc="none", mew=2, zorder=6)


def plt_circle(cx, cy, r, **kw):
    from matplotlib.patches import Circle
    return Circle((cx, cy), r, **kw)
