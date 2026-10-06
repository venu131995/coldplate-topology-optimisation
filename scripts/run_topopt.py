"""Cold-plate topology optimisation study: main design, hotspot intensity, fluid-fraction sweep,
mesh refinement, adjoint verification and figures."""
from __future__ import annotations

import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import ndimage  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from coldplate import Grid, Params, blueprint, evaluate, gradient, optimise, solve, straight_channels  # noqa: E402

FIG, RES = ROOT / "figures", ROOT / "results"
VOL, RADIUS, MIN_FEATURE = 0.35, 2.5, 3
plt.rcParams.update({"figure.dpi": 130, "font.size": 9})


def min_feature_check(bp: np.ndarray, P: Params) -> dict:
    """Share of each phase removed by a morphological opening with a 3x3 element (small = no thin features)."""
    img = bp.reshape(P.ny, P.nx).astype(bool)
    st = np.ones((MIN_FEATURE, MIN_FEATURE), bool)
    fl = ndimage.binary_opening(img, st, border_value=1)
    so = ndimage.binary_opening(~img, st, border_value=1)
    return {"fluid_cells_lost_pct": round(100 * (img & ~fl).sum() / img.sum(), 2),
            "solid_cells_lost_pct": round(100 * (~img & ~so).sum() / (~img).sum(), 2)}


def straight_table(P: Params, vol: float) -> list[dict]:
    rows = []
    total = int(round(vol * P.ny))
    for n in range(3, 13):
        width = total / n
        fin = (P.ny - total) / n
        e = evaluate(straight_channels(P, n, vol), P)
        rows.append({"channels": n, "channel_width_cells": round(width, 2), "fin_width_cells": round(fin, 2),
                     "meets_min_feature": bool(min(width, fin) >= MIN_FEATURE), "T_max": e["T_max"], "T_mean": e["T_mean"],
                     "flow_rate": e["flow_rate"]})
    return rows


def run_case(P: Params, vol: float, radius: float = RADIUS, iters: int = 320) -> dict:
    t0 = time.time()
    res = optimise(P, vol, radius=radius, iters=iters)
    bp = blueprint(res, P, radius)
    e = evaluate(bp, P)
    table = straight_table(P, bp.mean())
    feasible = [r for r in table if r["meets_min_feature"]]
    best_feas = min(feasible, key=lambda r: r["T_max"])
    best_any = min(table, key=lambda r: r["T_max"])
    return {"res": res, "blueprint": bp, "eval": e, "straight": table, "runtime_s": round(time.time() - t0, 1),
            "summary": {"T_max": round(e["T_max"], 5), "T_mean": round(e["T_mean"], 5), "flow_rate": round(e["flow_rate"], 5),
                        "fluid_fraction": round(float(bp.mean()), 4), "energy_imbalance": e["energy_imbalance"],
                        "best_straight_same_min_feature": {"channels": best_feas["channels"], "T_max": round(best_feas["T_max"], 5)},
                        "reduction_vs_same_min_feature_pct": round(100 * (1 - e["T_max"] / best_feas["T_max"]), 1),
                        "best_straight_any_width": {"channels": best_any["channels"], "T_max": round(best_any["T_max"], 5),
                                                    "channel_width_cells": best_any["channel_width_cells"]},
                        "min_feature_check": min_feature_check(bp, P)}}


def adjoint_check() -> dict:
    P = Params(nx=16, ny=8)
    g = Grid(P.nx, P.ny)
    rho = np.random.default_rng(7).uniform(0.05, 0.95, g.n)
    J, dJ = gradient(solve(rho, g, P), g, P)
    fd = np.empty(g.n)
    h = 1e-6
    for i in range(g.n):
        rp, rm = rho.copy(), rho.copy()
        rp[i] += h; rm[i] -= h
        fd[i] = (gradient(solve(rp, g, P), g, P)[0] - gradient(solve(rm, g, P), g, P)[0]) / (2 * h)
    return {"dJ": dJ, "fd": fd, "max_rel_error": float(np.max(np.abs(dJ - fd)) / np.max(np.abs(fd)))}


def main() -> None:
    FIG.mkdir(exist_ok=True); RES.mkdir(exist_ok=True)
    t_start = time.time()
    P = Params()
    out = {"model": {k: (v if not isinstance(v, np.ndarray) else None) for k, v in P.__dict__.items() if k != "source"},
           "volume_fraction": VOL, "filter_radius_cells": RADIUS}

    chk = adjoint_check()
    out["adjoint_vs_finite_difference_max_rel_error"] = chk["max_rel_error"]

    main_case = run_case(P, VOL)
    out["main_hotspot_4x"] = main_case["summary"] | {"runtime_s": main_case["runtime_s"]}
    out["straight_channels_main"] = [{k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items()} for r in main_case["straight"]]

    P8 = replace(P, hotspot=(0.55, 0.35, 0.12, 0.2, 8.0))
    hot = run_case(P8, VOL)
    out["hotspot_8x"] = hot["summary"]

    sweep = []
    for v in (0.25, 0.45):
        c = run_case(P, v)
        sweep.append({"vol": v, **c["summary"]})
        if v == 0.25:
            low = c
        else:
            high = c
    sweep.insert(1, {"vol": VOL, **main_case["summary"]})
    out["fluid_fraction_sweep"] = sweep

    Pf = replace(P, nx=180, ny=90)
    fine = run_case(Pf, VOL, radius=RADIUS * 1.5)
    out["mesh_refinement_180x90"] = fine["summary"] | {"runtime_s": fine["runtime_s"]}
    out["runtime_total_s"] = round(time.time() - t_start, 1)
    (RES / "summary.json").write_text(json.dumps(out, indent=2, default=str))
    np.save(RES / "design_main_blueprint.npy", main_case["blueprint"])

    # ------------------------------------------------------------------ figures
    def show(ax, field, P_, title, cmap, vmax=None):
        im = ax.imshow(field.reshape(P_.ny, P_.nx), origin="lower", cmap=cmap, vmin=0, vmax=vmax, aspect="equal")
        ax.set_title(title, fontsize=9); ax.set_xticks([]); ax.set_yticks([])
        return im

    hot_mask = P.heat_source().reshape(P.ny, P.nx) > P.heat_source().min() * 1.5
    n7 = main_case["summary"]["best_straight_same_min_feature"]["channels"]
    st = straight_channels(P, n7, main_case["blueprint"].mean())
    e_st = evaluate(st, P)
    vmax = max(e_st["T_max"], main_case["eval"]["T_max"])
    fig, axs = plt.subplots(2, 2, figsize=(9, 5.2))
    show(axs[0, 0], st, P, f"Straight channels (N={n7}, same min. feature)", "Blues")
    show(axs[0, 1], main_case["blueprint"], P, "Topology-optimised (crisp blueprint)", "Blues")
    show(axs[1, 0], e_st["state"].T, P, f"Temperature, T_max = {e_st['T_max']:.4f}", "inferno", vmax)
    im = show(axs[1, 1], main_case["eval"]["state"].T, P, f"Temperature, T_max = {main_case['eval']['T_max']:.4f}", "inferno", vmax)
    for ax in axs[:, :].ravel():
        ax.contour(hot_mask, levels=[0.5], colors="w" if ax in axs[1] else "tab:red", linewidths=0.8, linestyles="--")
    fig.colorbar(im, ax=axs[1, :], shrink=0.8, label="Temperature (dimensionless)")
    fig.suptitle("Coolant enters left, exits right; dashed box = 4x chip hotspot", fontsize=9)
    fig.savefig(FIG / "fig1_design_vs_straight.png", bbox_inches="tight"); plt.close(fig)

    h = main_case["res"].history
    fig, ax = plt.subplots(figsize=(7, 3.2))
    J = np.array([r["J"] for r in h])
    for k, lab in enumerate(["eroded", "blueprint", "dilated"]):
        ax.plot(J[:, k], lw=1, label=lab)
    for it in range(50, len(h), 50):
        ax.axvline(it, color="0.8", lw=0.6)
    ax.set_xlabel("MMA iteration (grey lines: beta and permeability-penalty continuation)")
    ax.set_ylabel("p-norm temperature"); ax.legend(); ax.set_title("Robust optimisation history")
    fig.tight_layout(); fig.savefig(FIG / "fig2_convergence.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    ax.scatter(chk["fd"], chk["dJ"], s=10)
    lim = [min(chk["fd"].min(), chk["dJ"].min()), max(chk["fd"].max(), chk["dJ"].max())]
    ax.plot(lim, lim, "k--", lw=0.7)
    ax.set_xlabel("Central finite difference"); ax.set_ylabel("Discrete adjoint")
    ax.set_title(f"Gradient check: max rel. error {chk['max_rel_error']:.1e}")
    fig.tight_layout(); fig.savefig(FIG / "fig3_adjoint_check.png"); plt.close(fig)

    fig, axs = plt.subplots(1, 3, figsize=(10, 2.4))
    for ax, c, v in zip(axs, [low, main_case, high], [0.25, 0.35, 0.45]):
        show(ax, c["blueprint"], P, f"Fluid {int(v * 100)}%: T_max {c['eval']['T_max']:.4f}", "Blues")
    fig.tight_layout(); fig.savefig(FIG / "fig4_fluid_fraction_designs.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    vv = [s["vol"] for s in sweep]
    ax.plot(vv, [s["T_max"] for s in sweep], "o-", label="Topology-optimised")
    ax.plot(vv, [s["best_straight_same_min_feature"]["T_max"] for s in sweep], "s--", label="Best straight, same min. feature")
    ax.plot(vv, [s["best_straight_any_width"]["T_max"] for s in sweep], "^:", color="0.5", label="Best straight, any width (incl. sub-feature)")
    ax.set_xlabel("Fluid volume fraction"); ax.set_ylabel("Hotspot temperature T_max"); ax.legend(fontsize=7)
    ax.set_title("Trade-off: coolant volume vs hotspot temperature")
    fig.tight_layout(); fig.savefig(FIG / "fig5_fluid_fraction_tradeoff.png"); plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(9, 2.6))
    show(axs[0], main_case["blueprint"], P, f"120 x 60 (filter r = {RADIUS}): T_max {main_case['eval']['T_max']:.4f}", "Blues")
    show(axs[1], fine["blueprint"], Pf, f"180 x 90 (filter r = {RADIUS * 1.5}): T_max {fine['eval']['T_max']:.4f}", "Blues")
    fig.tight_layout(); fig.savefig(FIG / "fig6_mesh_refinement.png"); plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(9, 2.6))
    show(axs[0], hot["blueprint"], P8, f"8x hotspot design: T_max {hot['eval']['T_max']:.4f}", "Blues")
    show(axs[1], hot["eval"]["state"].T, P8, "Temperature", "inferno")
    hm8 = P8.heat_source().reshape(P8.ny, P8.nx) > P8.heat_source().min() * 1.5
    for ax in axs:
        ax.contour(hm8, levels=[0.5], colors="tab:red", linewidths=0.8, linestyles="--")
    fig.tight_layout(); fig.savefig(FIG / "fig7_hotspot_8x.png"); plt.close(fig)

    print(json.dumps({k: v for k, v in out.items() if k not in ("straight_channels_main", "model")}, indent=1, default=str))


if __name__ == "__main__":
    main()
