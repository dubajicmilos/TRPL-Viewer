r"""hsapp.trpl_kinetics - differential lifetime and relative QFLS of a TRPL decay (numpy / scipy only, no Qt), 2026-11-17.

Definitions of github.com/barnlewis97/Perovskite-TRPL-Kinetics-Simulations (globalfit_functions.py: ``relative_QFLS`` and
``diff_lifetime`` with n = 2; MIT):

    dQFLS(t)    = QFLS(t) - QFLS(peak) = kT ln( PL(t) / PL(peak) )      (PL ~ n^2:  QFLS = Eg + kT ln(n^2 / n_ref^2))
    tau_diff(t) = ( -(1/2) d ln PL / dt )^-1

The derivative comes from a smooth model, not from the noisy points: a positive sum of exponentials on a fixed grid of lifetimes
(0.3 ns .. 20 x the time span, non-negative least squares), each component averaged over the REAL gate window of every point
(``Corrections/t_window_start_eff`` / ``_end_eff`` of a corrected or stitched file, else the nominal width).  A positive sum of
exponentials is log-convex, so tau_diff never decreases with time and noise cannot make it oscillate; a constant background (dark
error, light of the previous pulse) shows as a tau_diff that rises steeply at the end of the decay.  The fit is weighted by the local
scatter of the points around a first fit (two passes).  The point-wise derivative (``np.gradient`` of ln PL: the repo's own method on
simulated curves) is returned as well, for display as faint dots.

    diff_kinetics(t_ns, y, lo_ns=None, hi_ns=None, temperature_k=300) -> dict   (see its docstring)
"""
from __future__ import annotations

import math

import numpy as np
from scipy.optimize import lsq_linear, nnls

KB_EV_PER_K = 8.617333262e-5               # Boltzmann constant (eV / K); kT = 25.85 meV at 300 K
DEFAULT_T_K = 300.0                        # the repo's default temperature
N_TAU = 70                                 # lifetimes of the smooth model (as the 16 Nov script)
TAU_LO_NS = 0.3
MIN_POINTS = 5                             # decay points (after the peak, PL > 0) needed for a fit
REL_SIG_FIRST = 0.02                       # first pass: every point 2 % (relative)
REL_SIG_MIN, REL_SIG_MAX = 0.005, 0.5      # second pass: local scatter, clipped to these
SCATTER_HALF_WINDOW = 3                    # neighbours on each side for the local scatter
MARKER_TIMES_NS = (1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0)


def kt_ev(temperature_k=DEFAULT_T_K) -> float:
    """kT in eV; a temperature outside 1..2000 K (e.g. a missing / placeholder value) gives the 300 K value."""
    try:
        t = float(temperature_k)
    except (TypeError, ValueError):
        t = DEFAULT_T_K
    if not (math.isfinite(t) and 1.0 <= t <= 2000.0):
        t = DEFAULT_T_K
    return KB_EV_PER_K * t


def _window_average(s_lo, s_hi, taus):
    """(n, k) mean of exp(-s / tau_k) over the window [s_lo, s_hi] of every point (s = time after the reference; s_hi >= s_lo).
    exp(-s_lo/tau) (1 - exp(-w/tau)) / (w/tau), stable for w -> 0 (factor 1) and for windows that start before the reference
    (the exponent is clipped at +50, as trpl_schedule.DecayModel._gate_avg)."""
    s_lo = np.asarray(s_lo, float)[:, None]
    w = np.maximum(np.asarray(s_hi, float)[:, None] - s_lo, 0.0)
    x = w / taus[None, :]
    lead = np.exp(np.minimum(-s_lo / taus[None, :], 50.0))
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        frac = np.where(x > 1e-9, -np.expm1(-x) / np.where(x > 1e-9, x, 1.0), 1.0 - 0.5 * x)
    return lead * frac


def _nnls_fit(s_lo, s_hi, y, sig, taus):
    """Weighted non-negative least squares with unit-norm columns (the weighted design spans many decades).  Solver: bounded-variable
    least squares (scipy lsq_linear 'bvls'); scipy 1.14's ``nnls`` stops with "Maximum number of iterations" on steep decays (a
    bimolecular test decay over 5 decades), so it is only the last fallback after 'trf'."""
    A = _window_average(s_lo, s_hi, taus)
    Aw, b = A / sig[:, None], y / sig
    norm = np.sqrt(np.sum(Aw ** 2, axis=0))
    norm = np.where(norm > 0, norm, 1.0)
    An = Aw / norm[None, :]
    c = None
    for method in ('bvls', 'trf'):
        try:
            r = lsq_linear(An, b, bounds=(0.0, np.inf), method=method, tol=1e-12, max_iter=5000)
        except (ValueError, np.linalg.LinAlgError):
            continue
        if r.status >= 0 and np.all(np.isfinite(r.x)):
            c = np.maximum(r.x, 0.0)
            break
    if c is None:
        c, _ = nnls(An, b, maxiter=500 * len(taus))
    amp = c / norm
    return amp, A @ amp


def _local_scatter(r, half=SCATTER_HALF_WINDOW):
    """Robust SD (1.4826 x MAD) of the relative residuals r in a window of +-half neighbours around every point."""
    n = len(r)
    out = np.empty(n)
    for i in range(n):
        seg = r[max(0, i - half):min(n, i + half + 1)]
        out[i] = 1.4826 * float(np.median(np.abs(seg - np.median(seg))))
    return out


def diff_kinetics(t_ns, y, lo_ns=None, hi_ns=None, temperature_k=DEFAULT_T_K, n_out=300):
    """Smooth tau_diff and relative QFLS of one decay.

    ``t_ns``  time of every point (gate centre, any origin: the viewer's axis is the time after the PL onset)
    ``y``     PL of every point (any scale; only ratios are used)
    ``lo_ns`` / ``hi_ns``  the real gate window of every point as distances BEFORE / AFTER ``t_ns`` (default 0: instantaneous points)
    ``temperature_k``  for kT (300 K when missing or not physical)

    Only the points from the brightest one on with y > 0 are fitted (the rise is not a decay).  Returns a dict:
      ``t``, ``pl`` (smooth PL / PL(peak)), ``dqfls_meV``, ``tau_diff_ns``  -- n_out points, log-spaced from the peak to the last point
      ``raw_t``, ``raw_dqfls_meV``, ``raw_tau_ns``  -- the point-wise derivative (only finite, positive tau_diff)
      ``markers``  [(t, dqfls_meV, tau_diff_ns)] at 1, 10, 100 ns, ... after the axis origin, inside the fitted range
      ``fit_rms`` (relative rms of the fit), ``n_points``, ``kT_meV``, ``t_peak``, ``t_last``
    Raises ValueError when fewer than MIN_POINTS decay points are usable."""
    t = np.asarray(t_ns, float).ravel()
    yy = np.asarray(y, float).ravel()
    n = len(t)
    if len(yy) != n:
        raise ValueError('t and y differ in length (%d, %d)' % (n, len(yy)))
    lo = np.zeros(n) if lo_ns is None else np.asarray(lo_ns, float).ravel()
    hi = np.zeros(n) if hi_ns is None else np.asarray(hi_ns, float).ravel()
    if len(lo) != n or len(hi) != n:
        raise ValueError('gate windows differ in length from the points')
    lo = np.where(np.isfinite(lo), np.maximum(lo, 0.0), 0.0)
    hi = np.where(np.isfinite(hi), np.maximum(hi, 0.0), 0.0)
    fin = np.isfinite(t) & np.isfinite(yy)
    if not np.any(fin & (yy > 0)):
        raise ValueError('no positive PL value')
    order = np.argsort(np.where(fin, t, np.inf), kind='stable')
    t, yy, lo, hi, fin = t[order], yy[order], lo[order], hi[order], fin[order]
    ip = int(np.nanargmax(np.where(fin, yy, -np.inf)))
    use = fin & (yy > 0) & (np.arange(n) >= ip)
    if int(use.sum()) < MIN_POINTS:
        raise ValueError('only %d decay point(s) after the peak with PL > 0 (need %d)' % (int(use.sum()), MIN_POINTS))
    t_peak = float(t[ip])
    s, ys = t[use] - t_peak, yy[use]
    s_lo, s_hi = s - lo[use], s + hi[use]
    span = float(max(s_hi.max(), 1.0))
    taus = np.logspace(math.log10(TAU_LO_NS), math.log10(max(20.0 * span, 1e3)), N_TAU)
    scale = float(ys.max())
    yn = ys / scale
    amp, model = _nnls_fit(s_lo, s_hi, yn, REL_SIG_FIRST * yn, taus)          # pass 1: equal relative weights
    good = model > 0
    r = np.where(good, yn / np.where(good, model, 1.0) - 1.0, 0.0)
    sig_rel = np.clip(_local_scatter(r), REL_SIG_MIN, REL_SIG_MAX)
    amp2, model2 = _nnls_fit(s_lo, s_hi, yn, sig_rel * np.where(good, model, yn), taus)   # pass 2: local scatter
    if np.any(amp2 > 0):
        amp, model = amp2, model2
    if not np.any(amp > 0):
        raise ValueError('the smooth fit found no decay component')

    def smooth(sq):
        E = np.exp(-np.asarray(sq, float)[:, None] / taus[None, :])
        return E @ amp, -(E / taus[None, :]) @ amp

    p0 = float(smooth(np.array([0.0]))[0][0])                     # PL(peak) of the smooth model: dQFLS = 0 there
    s_last = float(s.max())
    s_first = float(min(max(np.min(s[s > 0]) if np.any(s > 0) else s_last, 1e-3), s_last))
    sq = np.unique(np.r_[0.0, np.logspace(math.log10(s_first), math.log10(max(s_last, s_first * 1.0001)), n_out - 1)])
    P, dP = smooth(sq)
    with np.errstate(divide='ignore', invalid='ignore'):
        tau = np.where(dP < 0, -2.0 * P / dP, np.inf)
    kt_mev = 1000.0 * kt_ev(temperature_k)
    ok = (P > 0) & np.isfinite(tau)
    out_t, out_p = sq[ok] + t_peak, P[ok] / p0
    out_q, out_tau = kt_mev * np.log(out_p), tau[ok]
    # point-wise derivative of the measured points (faint dots in the viewer)
    if len(s) >= 3:
        with np.errstate(divide='ignore', invalid='ignore'):
            g = np.gradient(np.log(ys), s)
            raw_tau = -2.0 / g
        keep = np.isfinite(raw_tau) & (raw_tau > 0)
    else:
        raw_tau, keep = np.full(len(s), np.nan), np.zeros(len(s), bool)
    raw_q = kt_mev * np.log(yn / p0)
    markers = []
    for tm in MARKER_TIMES_NS:
        if out_t.size and out_t[0] <= tm <= out_t[-1]:
            j = int(np.argmin(np.abs(out_t - tm)))
            markers.append((float(tm), float(out_q[j]), float(out_tau[j])))
    rms = float(np.sqrt(np.mean((model / yn - 1.0) ** 2)))
    return {'t': out_t, 'pl': out_p, 'dqfls_meV': out_q, 'tau_diff_ns': out_tau,
            'raw_t': s[keep] + t_peak, 'raw_dqfls_meV': raw_q[keep], 'raw_tau_ns': raw_tau[keep],
            'markers': markers, 'fit_rms': rms, 'n_points': int(len(s)), 'kT_meV': kt_mev, 't_peak': t_peak,
            't_last': float(s_last + t_peak)}


def value_at(res: dict, t_ns: float):
    """(dqfls_meV, tau_diff_ns) of a ``diff_kinetics`` result at the time nearest to ``t_ns`` (log scale), or None outside its range."""
    t = res['t']
    if not len(t) or not (t[0] <= t_ns <= t[-1]):
        return None
    tt = np.maximum(t - res['t_peak'], 1e-9)
    j = int(np.argmin(np.abs(np.log(tt) - math.log(max(t_ns - res['t_peak'], 1e-9)))))
    return float(res['dqfls_meV'][j]), float(res['tau_diff_ns'][j])
