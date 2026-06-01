"""Kidnapped-robot problem: after the robot has localised, it is teleported to a
new place without being told. Markov localization recovers because the belief
never collapses to exactly zero anywhere (the beam model's uniform/short terms
keep a probability floor).

We also compare filters: the *distance* filter still recovers, but the *entropy*
filter blocks recovery -- the contradicting readings from the new location
increase entropy, so an entropy filter (which only trusts belief-confirming
readings) rejects exactly the readings needed to escape the wrong pose. This is
noted in the paper (Sec. 3.3.1) and is a nice discussion point.

Outputs results/kidnapped.gif and results/kidnapped_error.png.
"""

import numpy as np

import _common as C
from markov_loc import office_map

KIDNAP_AT = 8
NEW_POSE = (2.2, 8.6, np.pi)


def main():
    m = office_map(0.15)
    rng = np.random.default_rng(11)
    tape, offs = C.record_tape(m, start_pose=(5.0, 4.5, 0.0), n_steps=28, rng=rng,
                               teleport_at=KIDNAP_AT, teleport_pose=NEW_POSE)
    exp = m.precompute_expected_distances(C.NTH, C.MAXR)

    panels = [
        ("no filter",       C.run_filter(m, tape, exp, filt=None,       filt_from=KIDNAP_AT)),
        ("distance filter", C.run_filter(m, tape, exp, filt="distance", filt_from=KIDNAP_AT)),
        ("entropy filter",  C.run_filter(m, tape, exp, filt="entropy",  filt_from=KIDNAP_AT)),
    ]
    C.animate(m, tape, offs, panels, C.out("kidnapped.gif"), fps=4,
              world_title="office  -  robot kidnapped at step %d" % KIDNAP_AT,
              suptitle="Kidnapped robot: recovery (entropy filter blocks it)")
    C.error_plot(panels, C.out("kidnapped_error.png"),
                 "Recovery after kidnapping", mark_step=KIDNAP_AT, mark_label="kidnap")

    for label, fr in panels:
        post = [f["err"] for f in fr[KIDNAP_AT:]]
        rec = next((j for j, e in enumerate(post) if e < 0.5), None)
        print("%-16s recovered in %s" % (label, f"{rec} steps" if rec is not None else "NEVER"))


if __name__ == "__main__":
    main()
