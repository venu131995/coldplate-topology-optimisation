"""2.5-D cold-plate model: Darcy flow + advection-diffusion heat transfer, finite volumes, discrete adjoint.

Plan view of a cold plate on a structured nx-by-ny grid (unit cells). Coolant enters along the left
edge at pressure p_in and temperature 0 and leaves along the right edge at p = 0. Heat from the chip
below enters every cell as a volumetric source (with a hotspot). Each cell holds a design density
rho in [0, 1]: 1 = coolant channel, 0 = solid metal fin.

    flow:  div(kappa(rho) grad p) = 0,            u = -kappa grad p
    heat:  div(k(rho) grad T) - C u.grad T + Q = 0

RAMP interpolation makes intermediate densities unattractive: solid conducts heat well but blocks
flow; fluid carries heat away by advection but conducts poorly. That trade-off is what the
optimiser resolves. Discretisation: two-point flux with harmonic face averages, first-order upwind
advection (exactly conservative). The objective is a p-norm of temperature (a smooth surrogate for
the hotspot temperature); gradients come from the discrete adjoint of the coupled system.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


@dataclass(frozen=True)
class Params:
    nx: int = 120
    ny: int = 60
    kappa_min: float = 1e-4      # solid permeability relative to fluid
    q_kappa: float = 8.0         # RAMP penalty, flow
    k_solid: float = 1.0
    k_fluid: float = 0.01        # metal conducts ~100x better than coolant
    q_k: float = 8.0             # RAMP penalty, conduction
    C: float = 300.0             # advection strength (Peclet-type number)
    p_in: float = 1.0
    q_total: float = 1.0         # total heat load from the chip
    hotspot: tuple = (0.55, 0.35, 0.15, 0.30, 4.0)   # (x-centre, y-centre, half-width x, half-width y) as fractions, multiplier
    pnorm: float = 8.0
    source: np.ndarray = field(default=None, compare=False, repr=False)

    def heat_source(self) -> np.ndarray:
        if self.source is not None:
            return self.source
        x = (np.arange(self.nx) + 0.5) / self.nx
        y = (np.arange(self.ny) + 0.5) / self.ny
        X, Y = np.meshgrid(x, y)
        cx, cy, hx, hy, mult = self.hotspot
        q = np.ones_like(X)
        q[(abs(X - cx) <= hx) & (abs(Y - cy) <= hy)] = mult
        return (q / q.sum() * self.q_total).ravel()


def ramp(rho, q):
    return rho / (1 + q * (1 - rho))


def dramp(rho, q):
    return (1 + q) / (1 + q * (1 - rho)) ** 2


def kappa(rho, P: Params):
    return P.kappa_min + (1 - P.kappa_min) * ramp(rho, P.q_kappa)


def dkappa(rho, P: Params):
    return (1 - P.kappa_min) * dramp(rho, P.q_kappa)


def conductivity(rho, P: Params):
    """Solid-favouring interpolation: k = k_s at rho = 0, k_f at rho = 1."""
    s = 1 - rho
    return P.k_fluid + (P.k_solid - P.k_fluid) * ramp(s, P.q_k)


def dconductivity(rho, P: Params):
    return -(P.k_solid - P.k_fluid) * dramp(1 - rho, P.q_k)


class Grid:
    """Face connectivity for an nx-by-ny cell grid (cell id = j*nx + i)."""

    def __init__(self, nx: int, ny: int):
        self.nx, self.ny, self.n = nx, ny, nx * ny
        idx = np.arange(self.n).reshape(ny, nx)
        self.a = np.concatenate([idx[:, :-1].ravel(), idx[:-1, :].ravel()])     # "left/lower" cell
        self.b = np.concatenate([idx[:, 1:].ravel(), idx[1:, :].ravel()])       # "right/upper" cell
        self.inlet = idx[:, 0].copy()
        self.outlet = idx[:, -1].copy()


def harmonic(x, y):
    return 2 * x * y / (x + y)


def dharmonic_dx(x, y):
    return 2 * y**2 / (x + y) ** 2


@dataclass
class State:
    rho: np.ndarray
    p: np.ndarray
    T: np.ndarray
    F: np.ndarray        # interior face fluxes a -> b
    F_in: np.ndarray     # inlet face fluxes into the domain
    F_out: np.ndarray    # outlet face fluxes out of the domain
    kap: np.ndarray
    k: np.ndarray
    A_T: sp.csc_matrix
    A_p: sp.csc_matrix


def _assemble_pressure(g: Grid, kap, P: Params):
    tf = harmonic(kap[g.a], kap[g.b])
    rows = np.concatenate([g.a, g.b, g.a, g.b, g.inlet, g.outlet])
    cols = np.concatenate([g.a, g.b, g.b, g.a, g.inlet, g.outlet])
    vals = np.concatenate([tf, tf, -tf, -tf, 2 * kap[g.inlet], 2 * kap[g.outlet]])
    A = sp.csc_matrix((vals, (rows, cols)), shape=(g.n, g.n))
    b = np.zeros(g.n)
    b[g.inlet] = 2 * kap[g.inlet] * P.p_in
    return A, b, tf


def _assemble_heat(g: Grid, k, F, F_in, F_out, P: Params):
    gf = harmonic(k[g.a], k[g.b])
    Fp, Fm = np.maximum(F, 0), np.minimum(F, 0)
    out_in = np.maximum(-F_in, 0)                     # backflow through the inlet (normally zero)
    out_out = np.maximum(F_out, 0)
    # inlet: purely advective inflow at T = 0 (no conduction into the manifold); outlet: outflow BC
    rows = np.concatenate([g.a, g.b, g.a, g.b,                 # conduction
                           g.a, g.a, g.b, g.b,                 # advection
                           g.inlet, g.outlet])
    cols = np.concatenate([g.a, g.b, g.b, g.a,
                           g.a, g.b, g.a, g.b,
                           g.inlet, g.outlet])
    vals = np.concatenate([gf, gf, -gf, -gf,
                           P.C * Fp, P.C * Fm, -P.C * Fp, -P.C * Fm,
                           P.C * out_in, P.C * out_out])
    return sp.csc_matrix((vals, (rows, cols)), shape=(g.n, g.n)), gf


def solve(rho: np.ndarray, g: Grid, P: Params) -> State:
    kap, k = kappa(rho, P), conductivity(rho, P)
    A_p, b_p, tf = _assemble_pressure(g, kap, P)
    p = spla.spsolve(A_p, b_p)
    F = tf * (p[g.a] - p[g.b])
    F_in = 2 * kap[g.inlet] * (P.p_in - p[g.inlet])
    F_out = 2 * kap[g.outlet] * p[g.outlet]
    A_T, _ = _assemble_heat(g, k, F, F_in, F_out, P)
    T = spla.spsolve(A_T, P.heat_source())
    return State(rho, p, T, F, F_in, F_out, kap, k, A_T, A_p)


def objective(s: State, P: Params) -> tuple[float, np.ndarray]:
    """p-norm temperature J = (mean T^P)^(1/P) and dJ/dT."""
    n, pn = len(s.T), P.pnorm
    Tp = np.maximum(s.T, 0)
    m = np.mean(Tp**pn)
    J = m ** (1 / pn)
    dJ = (1 / pn) * m ** (1 / pn - 1) * pn * Tp ** (pn - 1) / n
    return float(J), dJ


def gradient(s: State, g: Grid, P: Params) -> tuple[float, np.ndarray]:
    """J and dJ/drho via the discrete adjoint of the coupled flow-heat system."""
    J, dJdT = objective(s, P)
    lam_T = spla.spsolve(s.A_T.T.tocsc(), dJdT)
    a, b, T, p, C = g.a, g.b, s.T, s.p, P.C
    tf = harmonic(s.kap[a], s.kap[b])
    T_up = np.where(s.F > 0, T[a], T[b])
    dlam = lam_T[a] - lam_T[b]

    # (dR_T/dp)^T lam_T  -- advection depends on p through the face fluxes
    rhs_p = np.zeros(g.n)
    w = dlam * C * T_up * tf
    np.add.at(rhs_p, a, w)
    np.add.at(rhs_p, b, -w)
    i_in, i_out = g.inlet, g.outlet
    back = (-s.F_in > 0)
    rhs_p[i_in] += lam_T[i_in] * C * T[i_in] * 2 * s.kap[i_in] * back
    rhs_p[i_out] += lam_T[i_out] * C * T[i_out] * 2 * s.kap[i_out] * (s.F_out > 0)
    lam_p = spla.spsolve(s.A_p.T.tocsc(), -rhs_p)

    dJ = np.zeros(g.n)
    # conduction conductances
    w_cond = dlam * (T[a] - T[b])
    dk = dconductivity(s.rho, P)
    np.add.at(dJ, a, -w_cond * dharmonic_dx(s.k[a], s.k[b]) * dk[a])
    np.add.at(dJ, b, -w_cond * dharmonic_dx(s.k[b], s.k[a]) * dk[b])
    # face fluxes through permeability (advection in R_T) and transmissibility (R_p)
    dkap = dkappa(s.rho, P)
    w_flux = dlam * C * T_up * (p[a] - p[b]) + (lam_p[a] - lam_p[b]) * (p[a] - p[b])
    np.add.at(dJ, a, -w_flux * dharmonic_dx(s.kap[a], s.kap[b]) * dkap[a])
    np.add.at(dJ, b, -w_flux * dharmonic_dx(s.kap[b], s.kap[a]) * dkap[b])
    # inlet / outlet boundary faces
    dJ[i_in] += -(lam_T[i_in] * C * T[i_in] * 2 * (p[i_in] - P.p_in) * back
                  + lam_p[i_in] * 2 * (p[i_in] - P.p_in)) * dkap[i_in]
    dJ[i_out] += -(lam_T[i_out] * C * T[i_out] * 2 * p[i_out] * (s.F_out > 0)
                   + lam_p[i_out] * 2 * p[i_out]) * dkap[i_out]
    return J, dJ


def energy_balance(s: State, g: Grid, P: Params) -> dict:
    """Heat in from the chip vs heat leaving by advection and inlet conduction."""
    q_in = float(P.heat_source().sum())
    adv_out = float(P.C * (np.maximum(s.F_out, 0) * s.T[g.outlet]).sum() + P.C * (np.maximum(-s.F_in, 0) * s.T[g.inlet]).sum())
    return {"heat_in": q_in, "advected_out": adv_out, "imbalance": q_in - adv_out,
            "flow_rate": float(s.F_in.sum()), "mixed_outlet_T": q_in / (P.C * float(s.F_out.sum()))}
