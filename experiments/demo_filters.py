"""CENTERPIECE: localization in a crowd.

The robot tracks its pose, then a dense crowd of people (unmapped, dynamic
obstacles) surrounds it and corrupts >50% of its range readings. We replay the
identical data through three localizers: no filter, the entropy filter and the
distance filter. Without filtering the belief diverges; both filters keep it
locked -- the paper's central claim.

Outputs results/filters_crowd.gif and results/filters_error.png.
"""

import numpy as np

import _common as C
from markov_loc import office_map, crowd_around

CROWD_FROM = 6


def main():
    m = office_map(0.15)
    rng = np.random.default_rng(7)

    def crowd(step, true_pose, rng):
        if step < CROWD_FROM:
            return None
        return crowd_around(true_pose, n=10, radius=1.1, person_r=0.28, rng=rng)

    tape, offs = C.record_tape(m, start_pose=(5.0, 4.2, 0.0), n_steps=24, rng=rng,
                               crowd_fn=crowd)
    exp = m.precompute_expected_distances(C.NTH, C.MAXR)
    crowd_corrupt = 100 * np.mean([t["corrupt"] for t in tape[CROWD_FROM:]])

    panels = [
        ("no filter",       C.run_filter(m, tape, exp, filt=None,       filt_from=CROWD_FROM)),
        ("entropy filter",  C.run_filter(m, tape, exp, filt="entropy",  filt_from=CROWD_FROM)),
        ("distance filter", C.run_filter(m, tape, exp, filt="distance", filt_from=CROWD_FROM)),
    ]
    # world shows the distance filter's beam decisions (green kept / red rejected)
    C.animate(m, tape, offs, panels, C.out("filters_crowd.gif"), fps=4,
              world_mask_from=2,
              world_title="crowd  (distance filter: red beams rejected)",
              suptitle="Markov localization in a crowd  (~%.0f%% of beams hit people)"
                       % crowd_corrupt)
    C.error_plot(panels, C.out("filters_error.png"),
                 "Localization error in a crowd  (filters engage at the dashed line)",
                 mark_step=CROWD_FROM, mark_label="crowd appears")

    for label, fr in panels:
        e = np.array([f["err"] for f in fr[CROWD_FROM:]])
        print("%-16s mean err in crowd = %.2f m  (max %.2f)" % (label, e.mean(), e.max()))


if __name__ == "__main__":
    main()
