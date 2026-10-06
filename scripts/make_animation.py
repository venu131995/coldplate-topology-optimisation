"""Animation: the cold-plate channel network 'growing' during optimisation (8x hotspot case).

Left: design (blue = coolant channel, white = metal). Right: temperature field of that design.
Bottom: hotspot temperature vs iteration against the best straight-channel design.
Writes figures/optimisation_movie.mp4 and a lighter figures/optimisation_movie.gif.
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import imageio_ffmpeg
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import animation  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from coldplate import Grid, Params, density_filter, evaluate, optimise, project, solve, straight_channels  # noqa: E402

plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
FIG = ROOT / "figures"


def main() -> None:
    P = replace(Params(), hotspot=(0.55, 0.35, 0.12, 0.2, 8.0))
    res = optimise(P, 0.35, snapshot_every=2)
    H, g = density_filter(P.nx, P.ny, 2.5), Grid(P.nx, P.ny)
    frames = []
    for it, beta, q, x in res.snapshots:
        rho, _ = project(H @ x, beta)
        s = solve(rho, g, replace(P, q_kappa=q))
        frames.append((it, beta, rho, s.T))
    straight = min(evaluate(straight_channels(P, n, 0.35), P)["T_max"] for n in (5, 6, 7))
    hot = P.heat_source().reshape(P.ny, P.nx)
    hot_mask = hot > hot.min() * 1.5
    tmax_hist = np.array([max(h["Tmax"]) for h in res.history])
    vmax = np.percentile([f[3].max() for f in frames[len(frames) // 3:]], 90)

    fig = plt.figure(figsize=(10, 5.4), facecolor="white")
    gs = fig.add_gridspec(2, 2, height_ratios=[1.25, 0.75], hspace=0.35, wspace=0.08)
    ax_d, ax_t, ax_c = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])
    im_d = ax_d.imshow(frames[0][2].reshape(P.ny, P.nx), origin="lower", cmap="Blues", vmin=0, vmax=1)
    im_t = ax_t.imshow(frames[0][3].reshape(P.ny, P.nx), origin="lower", cmap="inferno", vmin=0, vmax=vmax)
    for ax, ttl in [(ax_d, "Channel layout (blue = coolant)"), (ax_t, "Temperature")]:
        ax.contour(hot_mask, levels=[0.5], colors="tab:red" if ax is ax_d else "w", linestyles="--", linewidths=0.9)
        ax.set_title(ttl, fontsize=10); ax.set_xticks([]); ax.set_yticks([])
    ax_d.annotate("coolant in →", (0.0, -0.09), xycoords="axes fraction", fontsize=8)
    ax_d.annotate("→ coolant out", (0.78, -0.09), xycoords="axes fraction", fontsize=8)
    ax_c.axhline(straight, color="0.4", ls="--", lw=1, label=f"Best straight channels: {straight:.3f}")
    line, = ax_c.plot([], [], color="tab:orange", lw=1.6, label="Optimised design (worst robust case)")
    ax_c.set_xlim(0, len(tmax_hist)); ax_c.set_ylim(0, max(tmax_hist[:5].max(), straight) * 1.1)
    ax_c.set_xlabel("Optimisation iteration"); ax_c.set_ylabel("Hotspot T_max"); ax_c.legend(fontsize=8, loc="upper right")
    title = fig.suptitle("", fontsize=11)

    def update(k):
        it, beta, rho, T = frames[k]
        im_d.set_data(rho.reshape(P.ny, P.nx)); im_t.set_data(T.reshape(P.ny, P.nx))
        line.set_data(np.arange(it + 1), tmax_hist[: it + 1])
        title.set_text(f"Topology optimisation of a chip cold plate (8x hotspot)   iteration {it}   sharpness beta = {beta}")
        return im_d, im_t, line, title

    hold = [len(frames) - 1] * 20                      # pause on the final design
    anim = animation.FuncAnimation(fig, update, frames=list(range(len(frames))) + hold, interval=60, blit=False)
    anim.save(FIG / "optimisation_movie.mp4", writer=animation.FFMpegWriter(fps=12, bitrate=2400), dpi=120)
    small = list(range(0, len(frames), 2)) + hold[:10]
    anim2 = animation.FuncAnimation(fig, update, frames=small, interval=110)
    anim2.save(FIG / "optimisation_movie.gif", writer=animation.PillowWriter(fps=9), dpi=70)
    print("frames", len(frames), "final Tmax", tmax_hist[-1], "straight", straight)


if __name__ == "__main__":
    main()
