#!/usr/bin/env python3
"""
Companion script to landau_hermite_jax_relative_entropy.py,
landau_hermite_jax_relative_entropy_2.py, and
landau_hermite_jax_relative_entropy_3.py.

This script sweeps numerical parameters and measures the same transverse
angular-symmetry error used in landau_hermite_jax_relative_entropy_3.py.

For each time t, the v_x = 0 slice is reconstructed,

    g_t(v_y, v_z) = f(v_x = 0, v_y, v_z, t),

evaluated directly on a polar grid (r, theta). Using the angular mean

    <g_t>(r) = (1 / 2 pi) integral g_t(r, theta) d theta,

and restricting to radii where <g_t>(r) >= eta max_r <g_t>(r), the script
reports

                     sum_mask [g_t(r,theta) - <g_t>(r)]^2
    E_ang(t) = sqrt( ------------------------------------ ).
                           sum_mask [<g_t>(r)]^2

E_ang(t) = 0 indicates perfect axial symmetry in the transverse plane.
"""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

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


@dataclass(frozen=True)
class Case:
    family: str
    label: str
    nmax: int
    Q: int
    maxK: int
    dt: float


def _parse_int_list(s: str) -> List[int]:
    return [int(x) for x in s.split(",") if x.strip() != ""]


def _parse_float_list(s: str) -> List[float]:
    return [float(x) for x in s.split(",") if x.strip() != ""]


def _parse_kernel_pairs(s: str) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    for item in s.split(","):
        item = item.strip()
        if not item:
            continue
        q, maxk = item.split(":")
        out.append((int(q), int(maxk)))
    return out


def _symmetry_error_history(hist: np.ndarray, *, nmax: int, rmax: float, nr: int, nth: int) -> np.ndarray:
    polar_grid = prepare_polar_slice_grid(nmax=nmax, rmax=float(rmax), nr=int(nr), nth=int(nth))
    vals = np.empty(hist.shape[0], dtype=np.float64)
    for i, f in enumerate(hist):
        vals[i] = angular_symmetry_error_tensor(f, polar_grid=polar_grid, rel_floor=0.05)
    return vals


def _decade_ylim(curves: Iterable[np.ndarray]) -> Tuple[float, float]:
    vals = np.concatenate([np.ravel(np.asarray(c, dtype=np.float64)) for c in curves])
    vals = vals[np.isfinite(vals) & (vals > 0.0)]
    if vals.size == 0:
        return 1e-6, 1e0
    ymin = max(1e-3, 10.0 ** np.floor(np.log10(np.min(vals))))
    ymax = 10.0 ** np.ceil(np.log10(np.max(vals)))
    if ymax <= ymin:
        ymax = ymin * 10.0
    return float(ymin), float(ymax)


def _cost_proxy(case: Case, tmax: float) -> float:
    steps = int(round(float(tmax) / float(case.dt)))
    p = case.nmax + 1
    return float(steps * case.Q * p**4)


def main() -> None:
    ap = argparse.ArgumentParser(description="Axial-symmetry parameter sweep")
    ap.add_argument("--backend", choices=["jax", "numpy"], default="jax")
    ap.add_argument("--nmax_list", type=str, default="6,9,12,13,14,15")
    ap.add_argument("--dt_list", type=str, default="0.15,0.10,0.075")
    ap.add_argument("--kernel_pairs", type=str, default="12:256,16:512,20:768")
    ap.add_argument("--tmax", type=float, default=15.0)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--u", type=float, default=1.5)
    ap.add_argument("--nu_LB", type=float, default=0.1)
    ap.add_argument("--grid_xlim", type=float, default=3.0)
    ap.add_argument("--polar_nr", type=int, default=64)
    ap.add_argument("--polar_nth", type=int, default=128)
    ap.add_argument("--linearized", choices=["on", "off"], default="on")
    ap.add_argument("--target_error", type=float, default=3e-2)
    ap.add_argument("--outprefix", type=str, default="Fig1_panel_3D_parameter_sweep")
    ap.add_argument("--dpi", type=int, default=220)
    args = ap.parse_args()

    configure_plot_style(int(args.dpi))

    nmax_list = _parse_int_list(str(args.nmax_list))
    dt_list = _parse_float_list(str(args.dt_list))
    kernel_pairs = _parse_kernel_pairs(str(args.kernel_pairs))
    ref_nmax = max(nmax_list)
    ref_dt = min(dt_list)
    base_dt = dt_list[0]
    base_Q, base_maxK = kernel_pairs[0]
    ref_Q, ref_maxK = kernel_pairs[-1]

    cases: List[Case] = []
    for nmax in nmax_list:
        cases.append(Case("vary nmax", fr"$n_{{\max}}={nmax}$", int(nmax), int(base_Q), int(base_maxK), float(base_dt)))
    for dt in dt_list:
        cases.append(Case("vary dt", fr"$\Delta t={dt:g}$", int(ref_nmax), int(ref_Q), int(ref_maxK), float(dt)))
    for Q, maxK in kernel_pairs:
        cases.append(Case("vary (Q,maxK)", fr"$Q={Q},\,K_{{\max}}={maxK}$", int(ref_nmax), int(Q), int(maxK), float(ref_dt)))

    family_order = ["vary nmax", "vary dt", "vary (Q,maxK)"]
    family_titles = {
        "vary nmax": fr"vary $n_{{\max}}$",
        "vary dt": fr"vary $\Delta t$",
        "vary (Q,maxK)": r"vary $(Q, K_{\max})$",
    }
    family_notes = {
        "vary nmax": fr"$\Delta t={base_dt:g},\ Q={base_Q},\ K_{{\max}}={base_maxK}$",
        "vary dt": fr"$n_{{\max}}={ref_nmax},\ Q={ref_Q},\ K_{{\max}}={ref_maxK}$",
        "vary (Q,maxK)": fr"$n_{{\max}}={ref_nmax},\ \Delta t={ref_dt:g}$",
    }

    cmap = {
        "vary nmax": plt.cm.viridis(np.linspace(0.12, 0.92, len(nmax_list))),
        "vary dt": plt.cm.magma(np.linspace(0.18, 0.88, len(dt_list))),
        "vary (Q,maxK)": plt.cm.cividis(np.linspace(0.18, 0.88, len(kernel_pairs))),
    }

    results = []
    curves_all: List[np.ndarray] = []
    color_index = {family: 0 for family in family_order}
    for case in cases:
        nu_LB_here = max(float(args.nu_LB), 1e-300)/(case.nmax-1)/(case.nmax-2)/(case.nmax-3)  # avoid zero collisionality (no LB damping) which can cause issues at low nmax
        print(f"[run] {case.family}: {case.label} ...", flush=True)
        tgrid, f_hist, f_hist_lin = compute_twostream_histories(
            backend=str(args.backend),
            nmax=case.nmax,
            Q=case.Q,
            maxK=case.maxK,
            dt=case.dt,
            tmax=float(args.tmax),
            steps=args.steps,
            u=float(args.u),
            linearized=str(args.linearized),
            nu_LB=nu_LB_here,
        )
        err_nl = np.maximum(
            _symmetry_error_history(
                f_hist,
                nmax=case.nmax,
                rmax=float(args.grid_xlim),
                nr=int(args.polar_nr),
                nth=int(args.polar_nth),
            ),
            1e-18,
        )
        err_lin = None
        if f_hist_lin is not None:
            err_lin = np.maximum(
                _symmetry_error_history(
                    f_hist_lin,
                    nmax=case.nmax,
                    rmax=float(args.grid_xlim),
                    nr=int(args.polar_nr),
                    nth=int(args.polar_nth),
                ),
                1e-18,
            )
            curves_all.append(err_lin)
        curves_all.append(err_nl)
        color = cmap[case.family][color_index[case.family]]
        color_index[case.family] += 1
        metrics = {
            "max_nl": float(np.max(err_nl)),
            "final_nl": float(err_nl[-1]),
            "max_lin": float(np.max(err_lin)) if err_lin is not None else float("nan"),
            "final_lin": float(err_lin[-1]) if err_lin is not None else float("nan"),
            "score": max(
                float(np.max(err_nl)),
                float(np.max(err_lin)) if err_lin is not None else 0.0,
            ),
            "cost": _cost_proxy(case, float(tgrid[-1])),
        }
        results.append((case, tgrid, err_nl, err_lin, color, metrics))
        print(
            f"      max_nl={metrics['max_nl']:.3e} final_nl={metrics['final_nl']:.3e} "
            f"max_lin={metrics['max_lin']:.3e} final_lin={metrics['final_lin']:.3e}",
            flush=True,
        )

    ymin, ymax = _decade_ylim(curves_all)
    rows = 1 if str(args.linearized).lower() != "on" else 2
    fig, axes = plt.subplots(rows, 3, figsize=(10.2, 2.9 * rows), sharex="col", sharey=True, squeeze=False)
    major = LogLocator(base=10.0)
    minor = LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1)
    formatter = LogFormatterMathtext(base=10.0)

    for col, family in enumerate(family_order):
        ax_nl = axes[0, col]
        fam_results = [item for item in results if item[0].family == family]
        for case, tgrid, err_nl, err_lin, color, _metrics in fam_results:
            ax_nl.plot(tgrid, err_nl, lw=2.1, color=color, label=case.label)
        ax_nl.set_title(family_titles[family])
        ax_nl.text(0.98, 0.03, family_notes[family], transform=ax_nl.transAxes, ha="right", va="bottom", fontsize=8)
        ax_nl.set_yscale("log")
        ax_nl.set_ylim(ymin, ymax)
        ax_nl.yaxis.set_major_locator(major)
        ax_nl.yaxis.set_minor_locator(minor)
        ax_nl.yaxis.set_major_formatter(formatter)
        ax_nl.grid(True, alpha=0.25, which="both")
        ax_nl.axhline(float(args.target_error), color="0.5", lw=1.0, ls=":", zorder=0)
        ax_nl.legend(frameon=True, fancybox=False, edgecolor="0.8", facecolor="white", framealpha=0.92, fontsize=8, loc="upper left")
        if col == 0:
            ax_nl.set_ylabel("angular-symmetry error")

        if rows == 2:
            ax_lin = axes[1, col]
            for case, tgrid, _err_nl, err_lin, color, _metrics in fam_results:
                if err_lin is not None:
                    ax_lin.plot(tgrid, err_lin, lw=2.1, color=color, label=case.label)
            ax_lin.set_yscale("log")
            ax_lin.set_ylim(ymin, ymax)
            ax_lin.yaxis.set_major_locator(major)
            ax_lin.yaxis.set_minor_locator(minor)
            ax_lin.yaxis.set_major_formatter(formatter)
            ax_lin.grid(True, alpha=0.25, which="both")
            ax_lin.axhline(float(args.target_error), color="0.5", lw=1.0, ls=":", zorder=0)
            ax_lin.set_xlabel(r"$t$")
            if col == 0:
                ax_lin.set_ylabel("angular-symmetry error")

    if rows == 2:
        axes[0, 0].text(-0.28, 0.5, "Nonlinear", rotation=90, transform=axes[0, 0].transAxes, va="center", ha="center", fontsize=10)
        axes[1, 0].text(-0.28, 0.5, "Linearized", rotation=90, transform=axes[1, 0].transAxes, va="center", ha="center", fontsize=10)
    else:
        axes[0, 0].set_xlabel(r"$t$")

    fig.subplots_adjust(left=0.10, right=0.99, top=0.90, bottom=0.14, wspace=0.18, hspace=0.28)

    outprefix = str(args.outprefix)
    fig.savefig(f"{outprefix}.png", dpi=int(args.dpi), bbox_inches="tight")
    fig.savefig(f"{outprefix}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] wrote: {outprefix}.png and {outprefix}.pdf")

    target = float(args.target_error)
    feasible_nl = [item for item in results if item[5]["max_nl"] <= target]
    if feasible_nl:
        feasible_nl = sorted(feasible_nl, key=lambda item: (item[5]["cost"], item[5]["max_nl"]))
        case, _tgrid, _err_nl, _err_lin, _color, metrics = feasible_nl[0]
        print(
            "[recommend nonlinear] smallest tested case meeting target "
            f"E_rad <= {target:.2e}: "
            f"nmax={case.nmax}, Q={case.Q}, maxK={case.maxK}, dt={case.dt:g} "
            f"(max_nl={metrics['max_nl']:.3e}, cost_proxy={metrics['cost']:.3e})"
        )
    else:
        best_nl = min(results, key=lambda item: (item[5]["max_nl"], item[5]["cost"]))
        case, _tgrid, _err_nl, _err_lin, _color, metrics = best_nl
        print(
            "[recommend nonlinear] no tested case met target; best tested case: "
            f"nmax={case.nmax}, Q={case.Q}, maxK={case.maxK}, dt={case.dt:g} "
            f"(max_nl={metrics['max_nl']:.3e}, cost_proxy={metrics['cost']:.3e})"
        )

    if str(args.linearized).lower() == "on":
        feasible_lin = [item for item in results if np.isfinite(item[5]["max_lin"]) and item[5]["max_lin"] <= target]
        if feasible_lin:
            feasible_lin = sorted(feasible_lin, key=lambda item: (item[5]["cost"], item[5]["max_lin"]))
            case, _tgrid, _err_nl, _err_lin, _color, metrics = feasible_lin[0]
            print(
                "[recommend linearized] smallest tested case meeting target "
                f"E_rad <= {target:.2e}: "
                f"nmax={case.nmax}, Q={case.Q}, maxK={case.maxK}, dt={case.dt:g} "
                f"(max_lin={metrics['max_lin']:.3e}, cost_proxy={metrics['cost']:.3e})"
            )
        else:
            best_lin = min(results, key=lambda item: (item[5]["max_lin"], item[5]["cost"]))
            case, _tgrid, _err_nl, _err_lin, _color, metrics = best_lin
            print(
                "[recommend linearized] no tested case met target; best tested case: "
                f"nmax={case.nmax}, Q={case.Q}, maxK={case.maxK}, dt={case.dt:g} "
                f"(max_lin={metrics['max_lin']:.3e}, cost_proxy={metrics['cost']:.3e})"
            )


if __name__ == "__main__":
    main()
