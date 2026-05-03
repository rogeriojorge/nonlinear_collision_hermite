#!/usr/bin/env python3
"""Signed angular symmetry metrics for transverse Hermite/Landau diagnostics.

The functions here are intended for both distribution slices and signed
collision-update fields. They never clip negative values, which is essential
when diagnosing collision-operator residuals rather than positive distribution
functions.

For a polar slice ``g(r, alpha)``, with
``alpha = atan2(v_z, v_y)``, define

    gbar(r) = <g(r, alpha)>_alpha,
    delta(r, alpha) = g(r, alpha) - gbar(r).

The non-axisymmetric defect is measured with the velocity-space area weight
``r dr dalpha``; constants ``dr`` and ``dalpha`` cancel in the reported ratios:

    E_field  = sqrt( int r delta^2 dr dtheta / int r g^2 dr dtheta ),
    E_radial = sqrt( int r delta^2 dr dtheta / int r gbar^2 dr dtheta ).

For each harmonic ``m`` we project only the angular residual,

    a_m(r) = 2 <delta cos(m alpha)>_alpha,
    b_m(r) = 2 <delta sin(m alpha)>_alpha,
    delta_m = a_m(r) cos(m alpha) + b_m(r) sin(m alpha),

and report both ``||delta_m||/||g||`` and ``||delta_m||/||delta||`` with the
same ``r`` weighting. Thus ``E4_delta_fraction`` is the fraction of the total
non-axisymmetric power carried by the ``m=4`` angular harmonic.
"""

from __future__ import annotations

import argparse
from typing import Iterable

import numpy as np


def angular_symmetry_metrics_from_polar_plane(
    plane_rt,
    r,
    theta,
    rel_floor: float = 0.0,
    denominator: str = "field",
    m_list: Iterable[int] = (2, 4, 6, 8, 10, 12),
) -> dict:
    """Return signed, r-weighted angular symmetry diagnostics.

    Parameters
    ----------
    plane_rt:
        Values on a polar grid with shape (nr, nth).
    r, theta:
        One-dimensional radial and angular grids.
    rel_floor:
        If positive, radial rows with RMS below rel_floor*max(RMS) are excluded.
    denominator:
        Kept for API clarity. Both field and radial denominators are returned.
    m_list:
        Angular harmonics to project from delta = g - <g>_theta.
    """

    plane = np.asarray(plane_rt, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64)
    theta = np.asarray(theta, dtype=np.float64)
    if plane.ndim != 2:
        raise ValueError("plane_rt must have shape (nr, nth)")
    if plane.shape != (r.size, theta.size):
        raise ValueError(f"plane shape {plane.shape} does not match r/theta {(r.size, theta.size)}")
    if denominator not in {"field", "radial"}:
        raise ValueError("denominator must be 'field' or 'radial'")

    radial_mean = np.mean(plane, axis=1)
    delta = plane - radial_mean[:, None]
    radial_rms = np.sqrt(np.mean(plane * plane, axis=1))

    if rel_floor > 0.0:
        peak = float(np.max(radial_rms))
        mask_r = radial_rms >= float(rel_floor) * peak if peak > 0.0 and np.isfinite(peak) else np.zeros_like(r, dtype=bool)
    else:
        mask_r = np.ones_like(r, dtype=bool)
    if not np.any(mask_r):
        mask_r = np.ones_like(r, dtype=bool)

    rr = r[mask_r, None]
    plane_m = plane[mask_r, :]
    mean_m = radial_mean[mask_r]
    delta_m = delta[mask_r, :]

    denom_field = float(np.sum(rr * plane_m * plane_m)) + 1e-300
    denom_radial = float(np.sum(rr * mean_m[:, None] * mean_m[:, None])) + 1e-300
    denom_delta = float(np.sum(rr * delta_m * delta_m)) + 1e-300
    numer_delta = float(np.sum(rr * delta_m * delta_m))

    E_m_field = {}
    E_m_delta = {}
    for m in [int(x) for x in m_list]:
        cos_m = np.cos(m * theta)
        sin_m = np.sin(m * theta)
        a_m = 2.0 * np.mean(delta * cos_m[None, :], axis=1)
        b_m = 2.0 * np.mean(delta * sin_m[None, :], axis=1)
        proj = a_m[:, None] * cos_m[None, :] + b_m[:, None] * sin_m[None, :]
        proj_m = proj[mask_r, :]
        numer_m = float(np.sum(rr * proj_m * proj_m))
        E_m_field[m] = float(np.sqrt(numer_m / denom_field))
        E_m_delta[m] = float(np.sqrt(numer_m / denom_delta))

    dominant_m = max(E_m_delta, key=E_m_delta.get) if E_m_delta else None
    return {
        "E_ang_field": float(np.sqrt(numer_delta / denom_field)),
        "E_ang_radial": float(np.sqrt(numer_delta / denom_radial)),
        "E_m_field": E_m_field,
        "E_m_delta": E_m_delta,
        "dominant_m_by_delta": int(dominant_m) if dominant_m is not None else None,
        "E4_delta_fraction": float(E_m_delta.get(4, np.nan)),
        "radial_mean": radial_mean,
        "delta": delta,
        "mask_r": mask_r,
    }


def _selftest() -> None:
    r = np.linspace(0.0, 2.0, 96)
    theta = np.linspace(0.0, 2.0 * np.pi, 256, endpoint=False)
    eps = 1e-2
    for m in (2, 4):
        g = 1.0 + eps * np.cos(m * theta)[None, :]
        g = np.repeat(g, r.size, axis=0)
        out = angular_symmetry_metrics_from_polar_plane(g, r, theta, m_list=(2, 4, 6, 8), denominator="field")
        got = out["dominant_m_by_delta"]
        frac = out["E_m_delta"][m]
        print(f"[selftest] cos({m} alpha): dominant_m={got} E{m}_delta_fraction={frac:.12f}")
        if got != m or abs(frac - 1.0) > 5e-12:
            raise SystemExit(f"angular metric self-test failed for m={m}: got {got}, fraction={frac}")
    print("[selftest] angular signed metrics passed")


def main() -> None:
    ap = argparse.ArgumentParser(description="Self-test signed angular symmetry metrics")
    ap.add_argument("--selftest", action="store_true", help="Run analytic cos(m theta) checks.")
    args = ap.parse_args()
    if args.selftest:
        _selftest()
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
