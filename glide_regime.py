"""Glide-regime classifier: skip / capture / plunge.

A *diagnostic* classifier that reads the post-entry regime off an already-
integrated trajectory, following the entry-corridor criteria of Chapman
(NASA TR R-11 p. 209; TR R-55) and Vinh, Busemann & Culp Ch. 12:

  - SKIP     the vehicle lofts back out of the sensible atmosphere after a dip
             (Chapman's test: the density variable returns toward its entry
             value, i.e. the altitude re-ascends above the entry interface).
  - CAPTURE  the vehicle settles into a sustained quasi-equilibrium glide — a
             long run at small, near-constant flight-path angle — without
             re-exiting.
  - PLUNGE   the vehicle penetrates steeply to low altitude without a sustained
             glide, or its peak deceleration exceeds the limit (undershoot).

This is exact and speed-regime-agnostic: it does not approximate the dynamics,
it reads the regime off the real trajectory.  It is the regime (A) "capture"
predictor that the glide-damping (regime B, the ζ knob) presupposes.  See
GLIDE_CAPTURE_DESIGN.md for the full derivation and sources.

The companion *predictive* (closed-form) classifier — Chapman's periapsis
parameter F_p for near/super-circular entry — is a separate, future addition;
this diagnostic version needs no new physics and works for any entry speed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

G0 = 9.80665  # m/s^2

# Default re-entry interface (the "sensible atmosphere" edge used elsewhere in
# Thrusty for gating aerodynamic control).
ENTRY_INTERFACE_KM = 100.0
# Default deceleration limit (g) above which entry is an undershoot/plunge.
# Matches the worked corridor example in Chapman TR R-55 (10 g) and the RV
# pull-up cap default; override per vehicle via g_limit_g.
DECEL_LIMIT_G = 10.0
# Flight-path angle (deg) below which flight is considered "gliding" (shallow).
GLIDE_GAMMA_DEG = 3.0


@dataclass
class GlideRegime:
    """Verdict and supporting metrics for a post-entry trajectory."""
    verdict: str            # 'skip' | 'capture' | 'plunge' | 'no_entry'
    a_max_g: float          # peak deceleration during descent (g)
    n_reascents: int        # times altitude re-ascended above the entry interface
    above_frac: float       # fraction of post-apogee flight above the interface
    glide_frac: float       # fraction of in-atmosphere descent at shallow γ
    min_alt_km: float       # lowest altitude reached on the descent
    notes: str = ""

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return (f"{self.verdict.upper()}  (a_max={self.a_max_g:.1f}g, "
                f"re-ascents={self.n_reascents}, glide={self.glide_frac:.0%}, "
                f"above-interface={self.above_frac:.0%}, min_alt={self.min_alt_km:.0f} km)")


def classify_glide_regime(alt_km, speed_ms, t_s, *,
                          entry_alt_km: float = ENTRY_INTERFACE_KM,
                          g_limit_g: float = DECEL_LIMIT_G,
                          glide_gamma_deg: float = GLIDE_GAMMA_DEG,
                          g0: float = G0) -> GlideRegime:
    """Classify the post-entry regime of an integrated trajectory.

    Parameters
    ----------
    alt_km, speed_ms, t_s : array-like
        Altitude (km), inertial/relative speed (m/s) and time (s) samples of a
        full trajectory (launch through impact).  Only the post-apogee descent
        is analysed.
    entry_alt_km : float
        The sensible-atmosphere / re-entry interface altitude (km).
    g_limit_g : float
        Deceleration limit (g); peak descent deceleration above this is an
        undershoot → plunge.
    glide_gamma_deg : float
        |γ| below which a sample counts as gliding (shallow).

    Returns
    -------
    GlideRegime
    """
    alt = np.asarray(alt_km, dtype=float).ravel()
    spd = np.asarray(speed_ms, dtype=float).ravel()
    t = np.asarray(t_s, dtype=float).ravel()
    n = min(len(alt), len(spd), len(t))
    alt, spd, t = alt[:n], spd[:n], t[:n]
    if n < 3:
        return GlideRegime("no_entry", 0.0, 0, 0.0, 0.0,
                           float(alt.min()) if n else float("nan"),
                           "trajectory too short to classify")

    iap = int(np.argmax(alt))
    A, V, T = alt[iap:], spd[iap:], t[iap:]
    if len(A) < 3:
        return GlideRegime("no_entry", 0.0, 0, 0.0, 0.0, float(A.min()),
                           "no descent after apogee")

    min_alt = float(A.min())
    above = A > entry_alt_km
    above_frac = float(np.mean(above))

    # Never reached the sensible atmosphere within the integration window:
    # the vehicle is in vacuum/space — not a glide at all.
    if min_alt > entry_alt_km:
        return GlideRegime("no_entry", 0.0, 0, above_frac, 0.0, min_alt,
                           "descent never reached the entry interface")

    # Robust per-step rates (skip duplicate / non-increasing timestamps that
    # would make a naive np.gradient blow up).
    dt = np.diff(T)
    good = dt > 1e-6
    Vmid = 0.5 * (V[1:] + V[:-1])
    dV = np.diff(V)
    dh = np.diff(A) * 1000.0  # km → m
    decel_g = np.full(dt.shape, np.nan)
    decel_g[good] = -dV[good] / dt[good] / g0
    a_max_g = float(np.nanmax(decel_g)) if np.any(good) else 0.0

    # Flight-path angle per step (descent: γ < 0).
    gamma_deg = np.full(dt.shape, np.nan)
    vsafe = np.where(np.abs(Vmid) > 1.0, Vmid, np.nan)
    gamma_deg[good] = np.degrees(np.arcsin(np.clip(dh[good] / dt[good] / vsafe[good], -1.0, 1.0)))

    # Re-ascents above the entry interface after first dropping below it = skip.
    first_below = int(np.argmax(~above)) if (~above).any() else 0
    after = above[first_below:]
    n_reascents = int(np.sum((~after[:-1]) & (after[1:]))) if after.size > 1 else 0

    # Sustained shallow glide: fraction of in-atmosphere descent at small |γ|.
    in_atm = (A[1:] <= entry_alt_km) & good
    if np.any(in_atm):
        glide_frac = float(np.mean(np.abs(gamma_deg[in_atm]) < glide_gamma_deg))
    else:
        glide_frac = 0.0

    # --- Decision (order matters; see GLIDE_CAPTURE_DESIGN.md §5) -------------
    if a_max_g > g_limit_g:
        verdict, note = "plunge", f"peak deceleration {a_max_g:.1f}g exceeds {g_limit_g:.0f}g (undershoot)"
    elif n_reascents >= 1:
        verdict, note = "skip", f"re-ascended above the {entry_alt_km:.0f} km interface {n_reascents}× (lofts back out)"
    elif glide_frac > 0.5:
        verdict, note = "capture", f"sustained glide: {glide_frac:.0%} of in-atmosphere descent at |γ|<{glide_gamma_deg:.0f}°"
    else:
        verdict, note = "plunge", f"penetrated to {min_alt:.0f} km without a sustained glide and without skipping out"

    return GlideRegime(verdict, a_max_g, n_reascents, above_frac, glide_frac, min_alt, note)


def regime_from_result(result: dict, *,
                       g_limit_g: float | None = None,
                       rv=None, **kw) -> GlideRegime:
    """Convenience wrapper: classify directly from an ``integrate_trajectory``
    result dict (uses its ``alt``, ``speed`` and ``t`` arrays).

    If ``rv`` is given and ``g_limit_g`` is not, the RV's ``glider_pullup_g_max``
    is used as the deceleration (undershoot) limit.
    """
    if g_limit_g is None:
        g_limit_g = float(getattr(rv, "glider_pullup_g_max", DECEL_LIMIT_G)) if rv is not None else DECEL_LIMIT_G
    return classify_glide_regime(
        np.asarray(result["alt"]).ravel() / 1000.0,
        np.asarray(result["speed"]).ravel(),
        np.asarray(result["t"]).ravel(),
        g_limit_g=g_limit_g, **kw)
