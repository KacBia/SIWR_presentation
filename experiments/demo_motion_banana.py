"""Motion model only: starting from a sharply peaked belief, repeatedly apply the
prediction step (no sensor updates) and watch the belief spread into the
characteristic "banana" shape of Fig. 3 -- orientation uncertainty couples with
translation so different headings drift in different directions.

Outputs results/motion_banana.png.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import _common as C
from markov_loc import office_map, MarkovLocalizer, belief


def main():
    m = office_map(0.15, width=10.0, height=10.0)
    exp = m.precompute_expected_distances(C.NTH, C.MAXR)
    loc = MarkovLocalizer(m, n_theta=C.NTH, max_range=C.MAXR, fov_deg=C.FOV,
                          sensor_params=C.SPARAMS, floor=0.0, exp_dist=exp)
    loc.set_gaussian((2.0, 5.0, 0.0), sigma_xy=0.12, sigma_th=0.05)

    snaps = [(0, loc.bel.copy())]
    for step in range(1, 9):
        loc.predict((0.5, 0.12))          # drive forward, turn slightly left
        snaps.append((step, loc.bel.copy()))

    show = [0, 3, 6, 8]
    fig, axes = plt.subplots(1, len(show), figsize=(4.0 * len(show), 4.2))
    for ax, s in zip(axes, show):
        bel = dict(snaps)[s]
        C.viz.plot_belief(ax, m, belief.xy_marginal(bel),
                          est_pose=belief.map_estimate(bel, m.res, C.NTH),
                          title="after %d motions" % s)
    fig.suptitle('Motion model: belief spreads into a "banana" (no sensor updates)',
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(C.out("motion_banana.png"), dpi=120)
    print("wrote", C.out("motion_banana.png"))


if __name__ == "__main__":
    main()
