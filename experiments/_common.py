"""Shared harness for the demo scripts: locked parameters, a reusable
record/replay loop, an animator and an error-plot helper.

Design: the robot drives **once**, producing a *tape* of (odometry, scan,
true_pose, people). The tape is then replayed through one or more localizers
that differ only in which filter they use -- so every condition sees identical
data (the experimentally honest way to compare filters).
"""

from __future__ import annotations

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from markov_loc import MarkovLocalizer, Robot, viz, fan_offsets   # noqa: E402

# ---- locked configuration (validated for all three demos) -------------------
NTH = 36
MAXR = 8.0
FOV = 360.0
SIG = 0.45
SENSOR_NOISE = 0.15
FLOOR = 1e-4
SPARAMS = dict(sigma=SIG)
RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS, exist_ok=True)


def out(name: str) -> str:
    return os.path.join(RESULTS, name)


# ---- record a single ground-truth episode -----------------------------------
def record_tape(m, start_pose, n_steps, rng, crowd_fn=None,
                teleport_at=None, teleport_pose=None):
    offs = fan_offsets(NTH, FOV)
    bot = Robot(m, start_pose, NTH, offs, MAXR, sensor_noise=SENSOR_NOISE, rng=rng)
    tape = []
    for step in range(n_steps):
        if teleport_at is not None and step == teleport_at:
            bot.teleport(teleport_pose)
        people = crowd_fn(step, bot.true_pose, rng) if crowd_fn else None
        scan = bot.scan(people)
        clean = bot.scan(None)                       # robot's own (clean) nav sense
        true_pose = bot.true_pose.copy()             # pose at scan time
        front = clean[len(offs) // 2]
        turn = 0.45 if front < 1.1 else float(rng.normal(0, 0.06))
        dtrans = 0.0 if front < 0.7 else 0.3
        odo = bot.move((dtrans, turn if front < 1.1 else 0.0))
        corrupt = float(np.mean(np.abs(scan - clean) > 0.5))
        tape.append(dict(odom=odo, scan=scan, true_pose=true_pose,
                         people=people, corrupt=corrupt))
    return tape, offs


# ---- replay the tape through a localizer with a given filter -----------------
def run_filter(m, tape, exp_dist, filt=None, filt_from=0, init_pose=None,
               init_sigma_xy=0.3):
    loc = MarkovLocalizer(m, n_theta=NTH, max_range=MAXR, fov_deg=FOV,
                          sensor_params=SPARAMS, floor=FLOOR, exp_dist=exp_dist)
    if init_pose is not None:
        loc.set_gaussian(init_pose, sigma_xy=init_sigma_xy)
    frames = []
    for i, fr in enumerate(tape):
        keep = loc.correct(fr["scan"], filt=(filt if i >= filt_from else None))
        est = loc.map_estimate()
        err = float(np.hypot(est[0] - fr["true_pose"][0], est[1] - fr["true_pose"][1]))
        frames.append(dict(est_pose=est, marg=loc.xy_marginal().copy(),
                           err=err, entropy=loc.entropy(), keep_mask=keep))
        loc.predict(fr["odom"])
    return frames


# ---- visualisation -----------------------------------------------------------
def animate(m, tape, offs, panels, path, fps=1, world_title="environment",
            world_mask_from=None, suptitle=None):
    """``panels`` = list of (label, frames). One world panel + one belief panel
    per condition. ``world_mask_from`` indexes the panel whose beam keep-mask is
    drawn on the world (green=kept, red=rejected)."""
    n = len(panels)
    fig, axes = plt.subplots(1, 1 + n, figsize=(4.3 * (1 + n), 4.6))
    axes = np.atleast_1d(axes)
    axW, axBs = axes[0], axes[1:]

    # Fix the layout once, up front. Doing it per-frame makes the axes boxes
    # drift because the titles change width/content every step.
    if suptitle:
        fig.suptitle(suptitle, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94) if suptitle else None)
    # Freeze each axes' position so nothing re-flows during the animation.
    for ax in axes:
        ax.set_position(ax.get_position())

    def update(i):
        fr = tape[i]
        mask = panels[world_mask_from][1][i]["keep_mask"] if world_mask_from is not None else None
        axW.clear()
        viz.plot_world(axW, m, true_pose=fr["true_pose"], scan=fr["scan"],
                       beam_offsets=offs, n_theta=NTH, max_range=MAXR,
                       people=fr["people"], keep_mask=mask,
                       title=f"{world_title}\nstep {i}  ({100*fr['corrupt']:.0f}% beams on people)")
        for ax, (label, frames) in zip(axBs, panels):
            pf = frames[i]
            ax.clear()
            viz.plot_belief(ax, m, pf["marg"], est_pose=pf["est_pose"],
                            true_pose=fr["true_pose"],
                            title=f"{label}\nerr={pf['err']:.2f} m   H={pf['entropy']:.2f}")

    anim = FuncAnimation(fig, update, frames=len(tape), interval=1000 // fps)
    anim.save(path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print("wrote", path)
    return path


def error_plot(panels, path, title, mark_step=None, mark_label=""):
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    for label, frames in panels:
        ax.plot([f["err"] for f in frames], lw=2.2, label=label)
    if mark_step is not None:
        ax.axvline(mark_step, ls="--", color="gray")
        ax.text(mark_step + 0.2, ax.get_ylim()[1] * 0.92, mark_label, color="gray")
    ax.set_xlabel("step"); ax.set_ylabel("position error [m]")
    ax.set_title(title); ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)
    print("wrote", path)
    return path
