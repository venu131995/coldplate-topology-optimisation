"""Fig. 7: best straight channels vs. the optimised 8x-hotspot design (rows: design / temperature),
from results/hotspot_8x_fields.npz."""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
L, H = 2.0, 1.0


def main():
    d = np.load(ROOT / "results" / "hotspot_8x_fields.npz")
    nx, ny = int(d["nx"]), int(d["ny"])
    q = d["q"].reshape(ny, nx)
    hot_mask = (q > q.min() * 1.5).astype(float)
    vmax = float(max(d["s_T"].max(), d["o_T"].max()))
    water = ListedColormap(["white", "#0b3c8c"])
    xc, yc = np.linspace(0, L, nx + 1), np.linspace(0, H, ny + 1)

    def hot(a):
        for c, lw, ls in (("w", 4.0, "solid"), ("#d7191c", 2.6, (0, (4, 2.5)))):
            a.contour(0.5 * (xc[1:] + xc[:-1]), 0.5 * (yc[1:] + yc[:-1]), hot_mask, levels=[0.5], colors=c,
                      linewidths=lw, linestyles=[ls])

    def peak(a, v):
        a.text(L - 0.03, H - 0.04, f"peak {v:.4f}", ha="right", va="top", fontsize=13, fontweight="bold",
               bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="k", lw=1.2))

    def field(a, vals, cmap, vmin, vmax):
        return a.imshow(np.asarray(vals).reshape(ny, nx), origin="lower", extent=[0, L, 0, H], cmap=cmap,
                        vmin=vmin, vmax=vmax, interpolation="nearest")

    fig = plt.figure(figsize=(11.0, 5.2))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 0.04], hspace=0.08, wspace=0.06)
    cols = [("s", f"Best straight channels (N = {int(d['n_ch'])})"), ("o", "Optimised channels")]
    for c, (k, name) in enumerate(cols):
        a = fig.add_subplot(gs[0, c])
        field(a, (d[f"{k}_rho"] > 0.5).astype(float), water, 0, 1); hot(a)
        a.set_title(name, fontsize=16, fontweight="bold")
        if c == 0: a.set_ylabel("Design", fontsize=14, fontweight="bold")
        a = fig.add_subplot(gs[1, c])
        im = field(a, d[f"{k}_T"], "jet", 0, vmax); hot(a); peak(a, float(d[f"{k}_Tmax"]))
        if c == 0: a.set_ylabel("Temperature", fontsize=14, fontweight="bold")
    for a in fig.axes:
        a.set_xticks([]); a.set_yticks([])
        for sp in a.spines.values(): sp.set_linewidth(1.5)
    fig.add_subplot(gs[0, 2]).axis("off")
    cb = fig.colorbar(im, cax=fig.add_subplot(gs[1, 2]))
    cb.set_label("temperature", fontsize=12, fontweight="bold"); cb.ax.tick_params(labelsize=10, width=1.5)
    for t in cb.ax.get_yticklabels(): t.set_fontweight("bold")
    fig.savefig(ROOT / "figures" / "fig7_hotspot_8x.png", dpi=200, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    main()
