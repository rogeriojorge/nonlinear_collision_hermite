#!/usr/bin/env python3
"""
Companion script to landau_hermite_jax_relative_entropy.py and
landau_hermite_jax_relative_entropy_2.py.

This script measures the loss of angular symmetry in the transverse
v_x = 0 slice. For each time t and Hermite truncation n_max, it
reconstructs

    g_t(v_y, v_z) = f(v_x = 0, v_y, v_z, t).

It then evaluates g_t directly on a polar grid (r, theta) in the
v_y-v_z plane and computes the angular mean at fixed radius,

    <g_t>(r) = (1 / 2 pi) integral g_t(r, theta) d theta.

Restricting to radii where <g_t>(r) >= eta max_r <g_t>(r), with
eta = 0.05 by default, the plotted quantity is

                     sum_mask [g_t(r,theta) - <g_t>(r)]^2
    E_ang(t) = sqrt( ------------------------------------ ).
                           sum_mask [<g_t>(r)]^2

E_ang(t) = 0 for an exactly circular v_y-v_z slice. Larger values
indicate stronger non-axisymmetric distortion.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List

import numpy as np

os.environ.setdefault("MPLBACKEND", "Agg")
HERE = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(HERE / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(HERE / ".cache"))

import matplotlib.pyplot as plt
from matplotlib.ticker import LogFormatterMathtext, LogLocator

from landau_hermite_jax_relative_entropy_2 import (
    angular_symmetry_error_tensor,
    compute_twostream_histories,
    configure_plot_style,
    prepare_polar_slice_grid,
)


def _parse_int_list(s: str) -> List[int]:
    return [int(x) for x in s.split(",") if x.strip() != ""]


def _angular_error_history(hist: np.ndarray, *, nmax: int, rmax: float, nr: int, nth: int) -> np.ndarray:
    polar_grid = prepare_polar_slice_grid(nmax=nmax, rmax=float(rmax), nr=int(nr), nth=int(nth))
    vals = np.empty(hist.shape[0], dtype=np.float64)
    for i, f in enumerate(hist):
        vals[i] = angular_symmetry_error_tensor(f, polar_grid=polar_grid, rel_floor=0.05)
    return vals


def _decade_ylim(curves: list[np.ndarray]) -> tuple[float, float]:
    vals = np.concatenate([np.ravel(np.asarray(c, dtype=np.float64)) for c in curves])
    vals = vals[np.isfinite(vals) & (vals > 0.0)]
    if vals.size == 0:
        return 1e-6, 1e0
    ymin = max(1e-3, 10.0 ** np.floor(np.log10(np.min(vals))))
    ymax = 10.0 ** np.ceil(np.log10(np.max(vals)))
    if ymax <= ymin:
        ymax = ymin * 10.0
    return float(ymin), float(ymax)


def main() -> None:
    ap = argparse.ArgumentParser(description="Angular-symmetry error sweep over Hermite truncation")
    ap.add_argument("--backend", choices=["jax", "numpy"], default="jax")
    ap.add_argument("--nmax_list", type=str, default="6,9,12,13,14,15")
    ap.add_argument("--Q", type=int, default=12)
    ap.add_argument("--maxK", type=int, default=256)
    ap.add_argument("--dt", type=float, default=0.15)
    ap.add_argument("--tmax", type=float, default=20.0)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--u", type=float, default=1.5)
    ap.add_argument("--grid_xlim", type=float, default=3.0)
    ap.add_argument("--polar_nr", type=int, default=64)
    ap.add_argument("--polar_nth", type=int, default=128)
    ap.add_argument("--linearized", choices=["on", "off"], default="on")
    ap.add_argument("--outprefix", type=str, default="Fig1_panel_3D_circularity_sweep")
    ap.add_argument("--dpi", type=int, default=220)
    args = ap.parse_args()

    configure_plot_style(int(args.dpi))

    nmax_list = _parse_int_list(str(args.nmax_list))
    colors = plt.cm.viridis(np.linspace(0.12, 0.92, len(nmax_list)))

    rows = 1 if str(args.linearized).lower() != "on" else 2
    fig, axes = plt.subplots(rows, 1, figsize=(6.6, 2.9 * rows), sharex=True, sharey=True, squeeze=False)
    ax_nl = axes[0, 0]
    ax_lin = axes[1, 0] if rows == 2 else None
    series_nl: list[tuple[int, np.ndarray, np.ndarray, np.ndarray, str, object]] = []
    series_lin: list[tuple[int, np.ndarray, np.ndarray, np.ndarray, str, object]] = []
    markers = ["o", "s", "^", "D", "v", "P", "X", "*"]
    linestyles = ["-", "--", "-.", ":", (0, (5, 1)), (0, (3, 1, 1, 1))]

    for idx, (color, nmax) in enumerate(zip(colors, nmax_list)):
        print(f"[run] nmax={nmax} ...", flush=True)
        tgrid, f_hist, f_hist_lin = compute_twostream_histories(
            backend=str(args.backend),
            nmax=int(nmax),
            Q=int(args.Q),
            maxK=int(args.maxK),
            dt=float(args.dt),
            tmax=float(args.tmax),
            steps=args.steps,
            u=float(args.u),
            linearized=str(args.linearized),
        )
        ang_nl = _angular_error_history(
            f_hist,
            nmax=int(nmax),
            rmax=float(args.grid_xlim),
            nr=int(args.polar_nr),
            nth=int(args.polar_nth),
        )
        series_nl.append((int(nmax), tgrid, np.maximum(ang_nl, 1e-18), color, markers[idx % len(markers)], linestyles[idx % len(linestyles)]))

        if ax_lin is not None and f_hist_lin is not None:
            ang_lin = _angular_error_history(
                f_hist_lin,
                nmax=int(nmax),
                rmax=float(args.grid_xlim),
                nr=int(args.polar_nr),
                nth=int(args.polar_nth),
            )
            series_lin.append((int(nmax), tgrid, np.maximum(ang_lin, 1e-18), color, markers[idx % len(markers)], linestyles[idx % len(linestyles)]))

    all_curves = [vals for _nmax, _t, vals, _color, _marker, _ls in series_nl] + [vals for _nmax, _t, vals, _color, _marker, _ls in series_lin]
    ymin, ymax = _decade_ylim(all_curves)

    for nmax, tgrid, vals, color, marker, ls in series_nl:
        ax_nl.plot(tgrid, vals, lw=2.2, color=color, ls=ls, marker=marker, ms=4.2, markevery=max(1, len(tgrid) // 14), label=fr"$n_{{\max}}={nmax}$")

    for nmax, tgrid, vals, color, marker, ls in series_lin:
        ax_lin.plot(tgrid, vals, lw=2.2, color=color, ls=ls, marker=marker, ms=4.2, markevery=max(1, len(tgrid) // 14), label=fr"$n_{{\max}}={nmax}$")

    major = LogLocator(base=10.0)
    minor = LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1)
    formatter = LogFormatterMathtext(base=10.0)

    ax_nl.set_ylabel("angular-symmetry error")
    ax_nl.set_yscale("log")
    ax_nl.set_ylim(ymin, ymax)
    ax_nl.yaxis.set_major_locator(major)
    ax_nl.yaxis.set_minor_locator(minor)
    ax_nl.yaxis.set_major_formatter(formatter)
    ax_nl.set_title("Nonlinear")
    ax_nl.grid(True, alpha=0.25, which="both")
    ax_nl.legend(frameon=True, fancybox=False, edgecolor="0.8", facecolor="white", framealpha=0.9, fontsize=9, loc="upper left")

    if ax_lin is not None:
        ax_lin.set_ylabel("angular-symmetry error")
        ax_lin.set_xlabel(r"$t$")
        ax_lin.set_yscale("log")
        ax_lin.set_ylim(ymin, ymax)
        ax_lin.yaxis.set_major_locator(major)
        ax_lin.yaxis.set_minor_locator(minor)
        ax_lin.yaxis.set_major_formatter(formatter)
        ax_lin.set_title("Linearized")
        ax_lin.grid(True, alpha=0.25, which="both")
    else:
        ax_nl.set_xlabel(r"$t$")

    fig.subplots_adjust(left=0.15, right=0.98, top=0.90, bottom=0.14, hspace=0.26)

    outprefix = str(args.outprefix)
    fig.savefig(f"{outprefix}.png", dpi=int(args.dpi), bbox_inches="tight")
    fig.savefig(f"{outprefix}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] wrote: {outprefix}.png and {outprefix}.pdf")


if __name__ == "__main__":
    main()
