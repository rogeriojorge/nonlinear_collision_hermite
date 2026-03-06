#!/usr/bin/env python3
"""
Companion script to the Landau/Hermite diagnostics.

This figure is designed for publication use. It isolates the source of the
linearized symmetry breaking by showing:

1. the non-axisymmetric part of the matched Maxwellian background M,
2. the non-axisymmetric part of the nonlinear RHS Q(f0, f0),
3. the non-axisymmetric part of the linearized RHS
   L(h0) = Q(h0, M) + Q(M, h0),
4. how the corresponding angular-symmetry errors vary with n_max.

The top-row images display

    delta g(v_y, v_z) = g(v_y, v_z) - <g>_theta(r),

where g is the v_x = 0 slice and <g>_theta(r) is the angular average at fixed
radius r = sqrt(v_y^2 + v_z^2), evaluated on a polar grid and interpolated
back to the Cartesian plotting grid.

The scalar error used in the bottom panel is

                [ int (g - <g>_theta)^2 dA / int <g>_theta^2 dA ]^(1/2),

restricted to the positive-support region of the slice and evaluated
numerically by the helper in landau_hermite_jax_relative_entropy_2.py.
It vanishes for an exactly circular v_y-v_z slice.

The figure layout follows the standard solver-verification pattern used in the
kinetic numerics literature: field-level defect maps above, scalar defect norms
versus resolution below.
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
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import LogFormatterMathtext, LogLocator

from landau_hermite_jax_relative_entropy import (
    Species,
    build_ic_fig1_1sp_twostream,
    build_maxwellian_tensor_from_invariants,
    build_model_tables_np,
    invariants_from_tensor,
    make_linearized_rhs_1sp_numpy,
    prepare_entropy_grid,
    rhs_ab_np,
)
from landau_hermite_jax_relative_entropy_2 import (
    _psi0,
    _reconstruct_plane_vy_vz,
    _reconstruct_plane_vy_vz_polar,
    angular_symmetry_error_tensor,
    configure_plot_style,
    prepare_polar_slice_grid,
)


def _parse_int_list(s: str) -> List[int]:
    return [int(x) for x in s.split(",") if x.strip() != ""]


def _build_basis_maxwellian(nmax: int, sp: Species) -> np.ndarray:
    f = np.zeros((nmax + 1, nmax + 1, nmax + 1), dtype=np.float64)
    f[0, 0, 0] = 1.0 / (sp.vth**3)
    return f


def _cartesian_plane(f: np.ndarray, *, nmax: int, xlim: float, nx: int) -> tuple[np.ndarray, np.ndarray]:
    grid = prepare_entropy_grid(nmax=nmax, xlim=float(xlim), nx=int(nx))
    plane = _reconstruct_plane_vy_vz(f, grid.psi, _psi0(nmax + 1))
    return grid.x, plane


def _angular_residual_plane(
    f: np.ndarray,
    *,
    nmax: int,
    xlim: float,
    nx: int,
    polar_grid,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x, plane = _cartesian_plane(f, nmax=nmax, xlim=xlim, nx=nx)
    plane_polar = _reconstruct_plane_vy_vz_polar(f, polar_grid)
    radial_mean = np.mean(plane_polar, axis=1)
    Y, Z = np.meshgrid(x, x, indexing="ij")
    rho = np.sqrt(Y * Y + Z * Z)
    radial_interp = np.interp(rho.ravel(), polar_grid.r, radial_mean, left=radial_mean[0], right=radial_mean[-1]).reshape(rho.shape)
    residual = plane - radial_interp
    return x, plane, residual


def _collect_case(nmax: int, *, Q: int, maxK: int, u: float, xlim: float, nx: int, polar_nr: int, polar_nth: int):
    sp = Species(m=1.0, vth=1.0)
    T11 = build_model_tables_np(nmax=nmax, Q=Q, maxK=maxK, ma=sp.m, mb=sp.m, vtha=sp.vth, vthb=sp.vth, nu_ab=1.0)
    f0 = build_ic_fig1_1sp_twostream(nmax=nmax, sp=sp, u=u, enforce_nonneg=False).f
    inv0 = invariants_from_tensor(f0, sp)
    fM_match = build_maxwellian_tensor_from_invariants(nmax=nmax, sp=sp, inv=inv0, xlim=10.0, nx=2001)
    fM_basis = _build_basis_maxwellian(nmax=nmax, sp=sp)
    h_match = f0 - fM_match
    h_basis = f0 - fM_basis
    rhs_nl = rhs_ab_np(f0, f0, T11, use_tt=False, tt_tol=0.0, tt_rmax=1)
    L_match = make_linearized_rhs_1sp_numpy(T11, fM_match, use_tt=False, tt_tol=0.0, tt_rmax=1)
    L_basis = make_linearized_rhs_1sp_numpy(T11, fM_basis, use_tt=False, tt_tol=0.0, tt_rmax=1)
    rhs_lin_match = L_match(h_match)
    rhs_lin_basis = L_basis(h_basis)
    polar_grid = prepare_polar_slice_grid(nmax=nmax, rmax=float(xlim), nr=int(polar_nr), nth=int(polar_nth))
    metrics = {
        "M_match": angular_symmetry_error_tensor(fM_match, polar_grid=polar_grid),
        "rhs_nl": angular_symmetry_error_tensor(rhs_nl, polar_grid=polar_grid),
        "rhs_lin_match": angular_symmetry_error_tensor(rhs_lin_match, polar_grid=polar_grid),
        "rhs_lin_basis": angular_symmetry_error_tensor(rhs_lin_basis, polar_grid=polar_grid),
    }
    return {
        "sp": sp,
        "f0": f0,
        "fM_match": fM_match,
        "rhs_nl": rhs_nl,
        "rhs_lin_match": rhs_lin_match,
        "rhs_lin_basis": rhs_lin_basis,
        "metrics": metrics,
        "polar_grid": polar_grid,
    }


def _panel_image(ax, *, f: np.ndarray, nmax: int, xlim: float, nx: int, polar_grid, title: str, metric: float, cmap: str = "RdBu_r") -> None:
    x, plane, residual = _angular_residual_plane(f, nmax=nmax, xlim=xlim, nx=nx, polar_grid=polar_grid)
    plane_scale = max(np.max(np.abs(plane)), 1e-300)
    support_mask = np.abs(plane) >= 5.0e-3 * plane_scale
    residual_scale = max(np.max(np.abs(residual[support_mask])), 1e-300)
    residual_norm = np.full_like(residual, np.nan, dtype=np.float64)
    residual_norm[support_mask] = residual[support_mask] / residual_scale
    im = ax.imshow(
        residual_norm.T,
        origin="lower",
        extent=[float(x[0]), float(x[-1]), float(x[0]), float(x[-1])],
        cmap=cmap,
        vmin=-1.0,
        vmax=1.0,
        interpolation="nearest",
        aspect="equal",
    )
    ax.contour(
        x,
        x,
        np.where(np.isfinite(residual_norm), residual_norm, 0.0),
        levels=[-0.75, -0.4, 0.4, 0.75],
        colors=["#2166ac", "#67a9cf", "#ef8a62", "#b2182b"],
        linewidths=0.8,
    )
    ax.set_xlim(-xlim, xlim)
    ax.set_ylim(-xlim, xlim)
    ax.set_title(title)
    ax.set_xlabel(r"$v_y / v_{th}$")
    ax.set_ylabel(r"$v_z / v_{th}$")
    ax.text(
        0.03,
        0.08,
        fr"$E_{{\rm ang}}={metric:.2e}$",
        transform=ax.transAxes,
        va="bottom",
        ha="left",
        fontsize=9,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.82),
    )
    ax.tick_params(labelsize=8, pad=1)
    return im


def main() -> None:
    ap = argparse.ArgumentParser(description="Publication figure for symmetry breaking of the linearized operator")
    ap.add_argument("--nmax_list", type=str, default="6,9,12,14")
    ap.add_argument("--nmax_repr", type=int, default=12)
    ap.add_argument("--Q", type=int, default=12)
    ap.add_argument("--maxK", type=int, default=256)
    ap.add_argument("--u", type=float, default=1.5)
    ap.add_argument("--grid_xlim", type=float, default=3.0)
    ap.add_argument("--grid_nx", type=int, default=111)
    ap.add_argument("--polar_nr", type=int, default=64)
    ap.add_argument("--polar_nth", type=int, default=128)
    ap.add_argument("--outprefix", type=str, default="Fig1_panel_3D_linearized_symmetry_breaking")
    ap.add_argument("--dpi", type=int, default=220)
    args = ap.parse_args()

    configure_plot_style(int(args.dpi))

    nmax_list = _parse_int_list(str(args.nmax_list))
    repr_case = _collect_case(
        int(args.nmax_repr),
        Q=int(args.Q),
        maxK=int(args.maxK),
        u=float(args.u),
        xlim=float(args.grid_xlim),
        nx=int(args.grid_nx),
        polar_nr=int(args.polar_nr),
        polar_nth=int(args.polar_nth),
    )

    series = {
        "M_match": [],
        "rhs_nl": [],
        "rhs_lin_match": [],
        "rhs_lin_basis": [],
    }
    for nmax in nmax_list:
        out = _collect_case(
            int(nmax),
            Q=int(args.Q),
            maxK=int(args.maxK),
            u=float(args.u),
            xlim=float(args.grid_xlim),
            nx=int(args.grid_nx),
            polar_nr=int(args.polar_nr),
            polar_nth=int(args.polar_nth),
        )
        for key in series:
            series[key].append(out["metrics"][key])
        print(
            f"[nmax={nmax}] "
            f"E(M)={out['metrics']['M_match']:.3e} "
            f"E(Qff)={out['metrics']['rhs_nl']:.3e} "
            f"E(L_match h0)={out['metrics']['rhs_lin_match']:.3e} "
            f"E(L_basis h0)={out['metrics']['rhs_lin_basis']:.3e}",
            flush=True,
        )

    fig = plt.figure(figsize=(10.1, 6.25))
    gs = GridSpec(2, 3, figure=fig, height_ratios=[1.0, 0.92], hspace=0.28, wspace=0.22)

    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])
    ax3 = fig.add_subplot(gs[1, :])

    im0 = _panel_image(
        ax0,
        f=repr_case["fM_match"],
        nmax=int(args.nmax_repr),
        xlim=float(args.grid_xlim),
        nx=int(args.grid_nx),
        polar_grid=repr_case["polar_grid"],
        title=fr"Matched Background $M$",
        metric=repr_case["metrics"]["M_match"],
    )
    im1 = _panel_image(
        ax1,
        f=repr_case["rhs_nl"],
        nmax=int(args.nmax_repr),
        xlim=float(args.grid_xlim),
        nx=int(args.grid_nx),
        polar_grid=repr_case["polar_grid"],
        title=r"Nonlinear RHS $Q(f_0,f_0)$",
        metric=repr_case["metrics"]["rhs_nl"],
    )
    im2 = _panel_image(
        ax2,
        f=repr_case["rhs_lin_match"],
        nmax=int(args.nmax_repr),
        xlim=float(args.grid_xlim),
        nx=int(args.grid_nx),
        polar_grid=repr_case["polar_grid"],
        title=r"Linearized RHS $L(h_0;M)$",
        metric=repr_case["metrics"]["rhs_lin_match"],
    )

    for ax, label in zip([ax0, ax1, ax2, ax3], ["(a)", "(b)", "(c)", "(d)"]):
        ax.text(0.02, 0.98, label, transform=ax.transAxes, va="top", ha="left", fontsize=10, fontweight="bold")

    cbar = fig.colorbar(im2, ax=[ax0, ax1, ax2], fraction=0.024, pad=0.02)
    cbar.set_label(r"$\delta g / \max |\delta g|$ (per panel)")

    nmax_arr = np.asarray(nmax_list, dtype=np.int64)
    ax3.plot(nmax_arr, series["M_match"], color="#333333", marker="o", lw=2.0, label=r"matched $M$")
    ax3.plot(nmax_arr, series["rhs_nl"], color="#1f77b4", marker="s", lw=2.2, label=r"$Q(f_0,f_0)$")
    ax3.plot(nmax_arr, series["rhs_lin_match"], color="#d62728", marker="D", lw=2.2, label=r"$L(h_0)$ with matched $M$")
    ax3.plot(nmax_arr, series["rhs_lin_basis"], color="#2ca02c", marker="^", lw=2.0, ls="--", label=r"$L(h_0)$ with basis $M$")
    ax3.set_yscale("log")
    ax3.set_ylim(1e-3, 2e1)
    ax3.yaxis.set_major_locator(LogLocator(base=10.0))
    ax3.yaxis.set_major_formatter(LogFormatterMathtext(base=10.0))
    ax3.grid(True, which="major", alpha=0.24)
    ax3.grid(True, which="minor", alpha=0.10)
    ax3.set_xlabel(r"$n_{\max}$")
    ax3.set_ylabel(r"axisymmetry error $E_{\rm ang}$")
    ax3.set_xlim(float(nmax_arr[0]) - 0.4, float(nmax_arr[-1]) + 1.15)
    ax3.text(
        0.99,
        0.95,
        fr"$v_x=0$ slice, $n_{{\max}}^{{\rm map}}={int(args.nmax_repr)}$, $u={float(args.u):.1f},\ Q={int(args.Q)},\ K_{{\max}}={int(args.maxK)}$",
        transform=ax3.transAxes,
        ha="right",
        va="top",
        fontsize=9,
    )
    for key, color, label, dy in [
        ("M_match", "#333333", r"matched $M$", -0.08),
        ("rhs_nl", "#1f77b4", r"$Q(f_0,f_0)$", 0.10),
        ("rhs_lin_match", "#d62728", r"$L(h_0)$ with matched $M$", 0.02),
        ("rhs_lin_basis", "#2ca02c", r"$L(h_0)$ with basis $M$", -0.05),
    ]:
        ax3.text(
            float(nmax_arr[-1]) + 0.22,
            float(series[key][-1]) * (10.0 ** dy),
            label,
            color=color,
            va="center",
            ha="left",
            fontsize=9,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, boxstyle="round,pad=0.12"),
        )

    fig.savefig(f"{args.outprefix}.png", dpi=int(args.dpi), bbox_inches="tight")
    fig.savefig(f"{args.outprefix}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] wrote: {args.outprefix}.png and {args.outprefix}.pdf")


if __name__ == "__main__":
    main()
