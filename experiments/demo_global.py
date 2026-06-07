"""Global localization: the robot starts with a uniform belief (no idea where it
is) and localises itself as it drives, using only odometry and range scans.

Two scenes:
* office          -- a realistic, asymmetric map; the belief snaps to the true
                     pose almost immediately (a good sensor is very informative).
* symmetric corridor -- repeated, mirror-symmetric features keep the belief
                     genuinely *multi-modal* (several hypotheses at once) until
                     the robot drives to an end and breaks the symmetry. This is
                     the iconic advantage over unimodal (Kalman) filters (Fig. 2).

Outputs results/global_localization.gif, results/global_error.png,
        results/global_multimodal.gif and results/global_multimodal.png.
"""

import numpy as np

import _common as C
from markov_loc import office_map, symmetric_corridor


def main():
    # -- scene 1: office, fast convergence -------------------------------------
    m = office_map(0.15)
    rng = np.random.default_rng(3)
    tape, offs = C.record_tape(m, start_pose=(2.0, 2.2, 0.0), n_steps=18, rng=rng)
    exp = m.precompute_expected_distances(C.NTH, C.MAXR)
    frames = C.run_filter(m, tape, exp, filt=None)
    C.animate(m, tape, offs, [("belief  Bel(x, y)", frames)],
              C.out("global_localization.gif"),
              world_title="office  -  global localization")
    C.error_plot([("Markov localization", frames)], C.out("global_error.png"),
                 "Global localization from a uniform prior (office)")
    print("[office]   final err=%.2f m  H=%.2f" % (frames[-1]["err"], frames[-1]["entropy"]))

    # -- scene 2: symmetric corridor, sustained multi-modality -----------------
    mc = symmetric_corridor(0.15)
    rng = np.random.default_rng(0)
    # start OFF the symmetry axis (x=4); its mirror image (x=8) is a distinct
    # location, so the belief shows two well-separated hypotheses. Driving right
    # toward the asymmetric end eventually breaks the tie.
    tape_c, offs_c = C.record_tape(mc, start_pose=(4.0, 2.0, 0.0), n_steps=18, rng=rng)
    exp_c = mc.precompute_expected_distances(C.NTH, C.MAXR)
    frames_c = C.run_filter(mc, tape_c, exp_c, filt=None)
    C.animate(mc, tape_c, offs_c, [("belief  Bel(x, y)", frames_c)],
              C.out("global_multimodal.gif"),
              world_title="symmetric corridor  -  ambiguity stays multi-modal")
    # a still of an early multi-modal frame (the Fig.-2 picture)
    import matplotlib.pyplot as plt
    k = 0
    fig, (a, b) = plt.subplots(1, 2, figsize=(9.2, 3.6))
    C.viz.plot_world(a, mc, true_pose=tape_c[k]["true_pose"], scan=tape_c[k]["scan"],
                     beam_offsets=offs_c, n_theta=C.NTH, max_range=C.MAXR,
                     title="symmetric corridor (step %d)" % k)
    C.viz.plot_belief(b, mc, frames_c[k]["marg"], est_pose=frames_c[k]["est_pose"],
                      true_pose=tape_c[k]["true_pose"],
                      title="multi-modal belief\nH=%.2f" % frames_c[k]["entropy"])
    fig.suptitle("Multi-modal belief: several symmetric hypotheses at once")
    fig.tight_layout(); fig.savefig(C.out("global_multimodal.png"), dpi=120,); plt.close(fig)
    print("[corridor] step0 H=%.2f -> final H=%.2f" % (frames_c[0]["entropy"], frames_c[-1]["entropy"]))
    print("wrote", C.out("global_multimodal.png"))


if __name__ == "__main__":
    main()
