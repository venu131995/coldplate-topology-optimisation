"""Density filter, Heaviside projection and the Method of Moving Asymptotes (one linear constraint)."""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def density_filter(nx: int, ny: int, radius: float) -> sp.csr_matrix:
    """Row-normalised linear 'hat' filter H, so rho_tilde = H x and dJ/dx = H^T dJ/drho_tilde."""
    r = int(np.ceil(radius))
    rows, cols, vals = [], [], []
    for j in range(ny):
        for i in range(nx):
            for dj in range(-r, r + 1):
                for di in range(-r, r + 1):
                    ii, jj = i + di, j + dj
                    if 0 <= ii < nx and 0 <= jj < ny:
                        w = radius - np.hypot(di, dj)
                        if w > 0:
                            rows.append(j * nx + i); cols.append(jj * nx + ii); vals.append(w)
    H = sp.csr_matrix((vals, (rows, cols)), shape=(nx * ny, nx * ny))
    return sp.diags(1 / np.asarray(H.sum(axis=1)).ravel()) @ H


def project(x: np.ndarray, beta: float, eta: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """Smoothed Heaviside projection (Wang, Lazarov & Sigmund 2011) and its derivative."""
    tb = np.tanh(beta * eta)
    den = tb + np.tanh(beta * (1 - eta))
    y = (tb + np.tanh(beta * (x - eta))) / den
    dy = beta * (1 - np.tanh(beta * (x - eta)) ** 2) / den
    return y, dy


class MMA:
    """Svanberg's MMA for  min f(x)  s.t.  a.x <= b (linear),  0 <= x <= 1.

    The objective is replaced at each iteration by a separable convex approximation built on moving
    asymptotes L < x < U; the linear constraint is kept exact. The subproblem is solved through its
    one-dimensional dual (bisection on the multiplier), with a vectorised inner root-find per variable.
    """

    def __init__(self, n: int, move: float = 0.2):
        self.move = move
        self.x1 = self.x2 = None
        self.low = self.upp = None
        self.it = 0

    def step(self, x: np.ndarray, df: np.ndarray, a: np.ndarray, b: float) -> np.ndarray:
        self.it += 1
        if self.it <= 2:
            self.low, self.upp = x - 0.5, x + 0.5
        else:
            sgn = (x - self.x1) * (self.x1 - self.x2)
            gam = np.where(sgn > 0, 1.2, np.where(sgn < 0, 0.7, 1.0))
            self.low = x - gam * (self.x1 - self.low)
            self.upp = x + gam * (self.upp - self.x1)
            self.low = np.clip(self.low, x - 10, x - 0.01)
            self.upp = np.clip(self.upp, x + 0.01, x + 10)
        lo = np.maximum.reduce([np.zeros_like(x), self.low + 0.1 * (x - self.low), x - self.move])
        hi = np.minimum.reduce([np.ones_like(x), self.upp - 0.1 * (self.upp - x), x + self.move])
        dfp, dfm = np.maximum(df, 0), np.maximum(-df, 0)
        reg = 1e-3 * np.abs(df) + 1e-6 / (self.upp - self.low)
        p = (self.upp - x) ** 2 * (1.001 * dfp + 0.001 * dfm + reg)
        q = (x - self.low) ** 2 * (0.001 * dfp + 1.001 * dfm + reg)

        def x_of(lam):
            # root of p/(U-x)^2 - q/(x-L)^2 + lam*a = 0, monotone increasing in x
            l, h = lo.copy(), hi.copy()
            for _ in range(60):
                m = 0.5 * (l + h)
                g = p / (self.upp - m) ** 2 - q / (m - self.low) ** 2 + lam * a
                pos = g > 0
                h = np.where(pos, m, h)
                l = np.where(pos, l, m)
            return 0.5 * (l + h)

        if a @ x_of(0.0) <= b:
            xn = x_of(0.0)
        else:
            l, h = 0.0, 1.0
            doublings = 0
            while a @ x_of(h) > b and doublings < 60:     # unreachable target within move limits:
                h *= 2                                     # take the most volume-reducing feasible step
                doublings += 1
            for _ in range(60):
                m = 0.5 * (l + h)
                if a @ x_of(m) > b:
                    l = m
                else:
                    h = m
            xn = x_of(h)
        self.x2, self.x1 = self.x1, x.copy()
        if self.x2 is None:
            self.x2 = x.copy()
        return xn
