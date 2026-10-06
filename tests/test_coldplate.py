import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from coldplate import (MMA, Grid, Params, density_filter, energy_balance, gradient, project, solve,  # noqa: E402
                       straight_channels)


@pytest.fixture
def small():
    P = Params(nx=14, ny=8)
    g = Grid(P.nx, P.ny)
    rho = np.random.default_rng(0).uniform(0.05, 0.95, g.n)
    return P, g, rho


def test_adjoint_matches_finite_differences(small):
    P, g, rho = small
    J, dJ = gradient(solve(rho, g, P), g, P)
    h = 1e-6
    for i in [0, 13, 37, 60, 111]:
        rp, rm = rho.copy(), rho.copy()
        rp[i] += h; rm[i] -= h
        fd = (gradient(solve(rp, g, P), g, P)[0] - gradient(solve(rm, g, P), g, P)[0]) / (2 * h)
        assert dJ[i] == pytest.approx(fd, rel=1e-5, abs=1e-10)


def test_energy_and_mass_conservation(small):
    P, g, rho = small
    s = solve(rho, g, P)
    eb = energy_balance(s, g, P)
    assert abs(eb["imbalance"]) < 1e-12 * eb["heat_in"]
    assert s.F_in.sum() == pytest.approx(s.F_out.sum(), rel=1e-10)
    assert s.T.min() >= -1e-12                       # discrete maximum principle (upwind scheme)


def test_more_channels_cool_better_and_mixed_outlet_bound():
    P = Params(nx=60, ny=30)
    from coldplate import evaluate
    t = [evaluate(straight_channels(P, n, 0.4), P)["T_max"] for n in (2, 4, 6)]
    assert t[0] > t[1] > t[2]
    s = evaluate(straight_channels(P, 6, 0.4), P)["state"]
    eb = energy_balance(s, Grid(P.nx, P.ny), P)
    assert s.T.max() >= eb["mixed_outlet_T"]         # hottest point cannot be below the mixed outlet temperature


def test_straight_channel_fluid_fraction():
    P = Params(nx=20, ny=60)
    rho = straight_channels(P, 7, 0.35)
    assert rho.mean() == pytest.approx(21 / 60)


def test_filter_rows_sum_to_one_and_projection_derivative():
    H = density_filter(10, 6, 2.5)
    assert np.allclose(np.asarray(H.sum(axis=1)).ravel(), 1.0)
    x = np.linspace(0.01, 0.99, 50)
    y, dy = project(x, 8.0, 0.5)
    h = 1e-6
    fd = (project(x + h, 8.0, 0.5)[0] - project(x - h, 8.0, 0.5)[0]) / (2 * h)
    assert np.allclose(dy, fd, rtol=1e-6)
    assert project(np.array([0.0, 1.0]), 8.0, 0.5)[0] == pytest.approx([0.0, 1.0])


def test_mma_solves_constrained_convex_problem():
    # min sum c_i (x_i - t_i)^2  s.t. mean(x) <= 0.3 ; analytic KKT solution by bisection
    rng = np.random.default_rng(3)
    n = 40
    c, t = rng.uniform(0.5, 2, n), rng.uniform(0, 1, n)
    a, b = np.full(n, 1 / n), 0.3
    x, mma = np.full(n, 0.3), MMA(n, move=0.2)
    for _ in range(150):
        x = mma.step(x, 2 * c * (x - t), a, b)
    lo, hi = 0.0, 1e4
    for _ in range(100):
        lam = 0.5 * (lo + hi)
        xs = np.clip(t - lam / (2 * c * n), 0, 1)
        lo, hi = (lam, hi) if xs.mean() > b else (lo, lam)
    assert np.allclose(x, xs, atol=2e-3)


def test_local_and_global_conservation_mass_momentum_energy():
    from coldplate import conservation
    P = Params(nx=30, ny=16)
    g = Grid(P.nx, P.ny)
    for seed in (0, 1):
        rho = np.random.default_rng(seed).uniform(0, 1, g.n)
        c = conservation(solve(rho, g, P), g, P)
        assert abs(c["mass"]["global_imbalance_rel"]) < 1e-10 and c["mass"]["max_cell_imbalance_rel"] < 1e-10
        assert abs(c["momentum_power"]["closure_rel"]) < 1e-10      # pressure work = Darcy dissipation
        assert abs(c["energy"]["global_imbalance_rel"]) < 1e-10 and c["energy"]["max_cell_imbalance_rel"] < 1e-9
