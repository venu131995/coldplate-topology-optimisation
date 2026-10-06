"""Optimisation loop and reference designs."""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from .model import Grid, Params, energy_balance, gradient, solve
from .optim import MMA, density_filter, project


@dataclass
class Result:
    x: np.ndarray                 # design variables
    rho: np.ndarray               # filtered + projected physical density
    history: list = field(default_factory=list)
    snapshots: list = field(default_factory=list)   # (iteration, beta, q_kappa, design x)


def physical(x, H, beta):
    xt = H @ x
    rho, drho = project(xt, beta)
    return rho, drho


def optimise(P: Params, vol_frac: float, radius: float = 2.5, iters: int = 320,
             beta_schedule: tuple = (1, 2, 4, 8, 16, 32), every: int = 50, x0: np.ndarray | None = None,
             robust: bool = True, eta_spread: float = 0.1, ks: float = 40.0,
             q_kappa_schedule: tuple = (8, 16, 32, 64, 128, 256), snapshot_every: int = 0,
             verbose: bool = False) -> Result:
    """Density-based topology optimisation with MMA.

    robust=True uses the eroded / blueprint / dilated formulation (Wang, Lazarov & Sigmund 2011):
    the objective is a smooth maximum (KS aggregate) of the three projected designs, so the optimiser
    cannot rely on features thinner than the filter length scale or on grey transition zones.
    The volume constraint applies to the blueprint (eta = 0.5).

    The permeability RAMP penalty is raised alongside beta. Without this, near-solid grey cells keep a
    small permeability and the optimiser learns to seep coolant through "porous metal" next to the
    hotspot -- a large, unphysical gain that disappears as soon as the design is made crisp.
    """
    g = Grid(P.nx, P.ny)
    H = density_filter(P.nx, P.ny, radius)
    x = np.full(g.n, vol_frac) if x0 is None else x0.copy()
    mma = MMA(g.n, move=0.2)
    etas = (0.5 + eta_spread, 0.5, 0.5 - eta_spread) if robust else (0.5,)
    hist, snaps, J0 = [], [], None
    for it in range(iters):
        stage = min(it // every, len(beta_schedule) - 1)
        beta = beta_schedule[stage]
        Pi = replace(P, q_kappa=q_kappa_schedule[min(stage, len(q_kappa_schedule) - 1)])
        xt = H @ x
        Js, dxs, Tmax = [], [], []
        for eta in etas:
            rho, drho = project(xt, beta, eta)
            s = solve(rho, g, Pi)
            J, dJ = gradient(s, g, Pi)
            Js.append(J); dxs.append(H.T @ (dJ * drho)); Tmax.append(float(s.T.max()))
            if eta == 0.5:
                rho_b, drho_b, flow_b = rho, drho, float(s.F_in.sum())
        Js = np.array(Js)
        J0 = J0 or float(Js.max())
        z = ks * Js / J0
        w = np.exp(z - z.max()); w /= w.sum()                       # KS weights = d(KS)/dJ_i
        J_ks = J0 / ks * (np.log(np.exp(z - z.max()).sum()) + z.max())
        dx = sum(wi * di for wi, di in zip(w, dxs)) / J0
        a = H.T @ (drho_b / g.n)
        b = vol_frac - rho_b.mean() + a @ x
        if snapshot_every and it % snapshot_every == 0:
            snaps.append((it, beta, Pi.q_kappa, x.copy()))
        x = mma.step(x, dx, a, b)
        hist.append({"it": it, "beta": beta, "J_ks": float(J_ks), "J": [float(j) for j in Js],
                     "vol": float(rho_b.mean()), "Tmax": Tmax, "flow": flow_b})
        if verbose and it % 20 == 0:
            print(f"it {it:3d}  beta {beta:2d}  J {np.round(Js, 4)}  vol {rho_b.mean():.3f}")
    rho, _ = project(H @ x, beta_schedule[-1], 0.5)
    return Result(x=x, rho=rho, history=hist, snapshots=snaps)


def blueprint(res: Result, P: Params, radius: float = 2.5) -> np.ndarray:
    """Crisp 0/1 design: threshold the filtered field at eta = 0.5 (the beta -> infinity limit)."""
    H = density_filter(P.nx, P.ny, radius)
    return ((H @ res.x) > 0.5).astype(float)


def evaluate(rho: np.ndarray, P: Params) -> dict:
    """Performance of a design (use crisp 0/1 densities for fair comparison)."""
    g = Grid(P.nx, P.ny)
    s = solve(rho, g, P)
    eb = energy_balance(s, g, P)
    return {"T_max": float(s.T.max()), "T_mean": float(s.T.mean()), "flow_rate": eb["flow_rate"],
            "pumping_power": eb["flow_rate"] * P.p_in, "fluid_fraction": float(rho.mean()),
            "energy_imbalance": eb["imbalance"], "state": s}


def straight_channels(P: Params, n_channels: int, vol_frac: float) -> np.ndarray:
    """N equally spaced straight channels from inlet to outlet with the given fluid fraction."""
    rho = np.zeros((P.ny, P.nx))
    total = int(round(vol_frac * P.ny))
    widths = [total // n_channels + (1 if c < total % n_channels else 0) for c in range(n_channels)]
    pitch = P.ny / n_channels
    for c, w in enumerate(widths):
        start = int(round((c + 0.5) * pitch - w / 2))
        rho[start:start + w, :] = 1.0
    return rho.ravel()
