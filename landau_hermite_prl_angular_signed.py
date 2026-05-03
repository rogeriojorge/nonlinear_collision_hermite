#!/usr/bin/env python3
"""Reviewer-proof signed angular diagnostic for the finite-Cartesian m=4 defect."""

from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLBACKEND", "Agg")
HERE = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(HERE / ".mplconfig"))

import matplotlib.pyplot as plt
import numpy as np

from landau_hermite_angular_metrics import angular_symmetry_metrics_from_polar_plane
from landau_hermite_jax_relative_entropy_1 import (
    Species,
    build_ic_fig1_1sp_twostream,
    build_maxwellian_tensor_from_invariants,
    build_model_tables_np,
    invariants_from_tensor,
    lb_collision_np,
    make_linearized_rhs_1sp_numpy,
    psi_1d,
    rhs_ab_np,
)
from landau_hermite_jax_relative_entropy_2 import (
    _reconstruct_plane_vy_vz_polar,
    prepare_polar_slice_grid,
)


def _parse_int_list(s: str) -> list[int]:
    return [int(x) for x in str(s).replace(";", ",").split(",") if x.strip()]


def _parse_float_list(s: str) -> list[float]:
    return [float(x) for x in str(s).replace(";", ",").split(",") if x.strip()]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def _basis_maxwellian(nmax: int, sp: Species) -> np.ndarray:
    p = nmax + 1
    f = np.zeros((p, p, p), dtype=np.float64)
    f[0, 0, 0] = 1.0 / (sp.vth**3)
    return f


def _case_fields(nmax: int, Q: int, maxK: int, u: float, nu_LB: float) -> dict[str, np.ndarray]:
    sp = Species(m=1.0, vth=1.0)
    T = build_model_tables_np(nmax=nmax, Q=Q, maxK=maxK, ma=sp.m, mb=sp.m, vtha=sp.vth, vthb=sp.vth, nu_ab=1.0)
    f0 = build_ic_fig1_1sp_twostream(nmax=nmax, sp=sp, u=u, enforce_nonneg=False).f
    inv0 = invariants_from_tensor(f0, sp)
    M_match = build_maxwellian_tensor_from_invariants(nmax=nmax, sp=sp, inv=inv0, xlim=10.0, nx=2001)
    M_basis = _basis_maxwellian(nmax, sp)
    h_match = f0 - M_match
    h_basis = f0 - M_basis
    rhs_nl = rhs_ab_np(f0, f0, T, use_tt=False, tt_tol=0.0, tt_rmax=1) + lb_collision_np(f0, float(nu_LB))
    L_match = make_linearized_rhs_1sp_numpy(T, M_match, use_tt=False, tt_tol=0.0, tt_rmax=1, nu_LB=float(nu_LB))
    L_basis = make_linearized_rhs_1sp_numpy(T, M_basis, use_tt=False, tt_tol=0.0, tt_rmax=1, nu_LB=float(nu_LB))
    return {
        "M_N": M_match,
        "Q(f0,f0)": rhs_nl,
        "L(h0;M_N)": L_match(h_match),
        "L(h0;M_basis)": L_basis(h_basis),
    }


def _metrics_for_field(f: np.ndarray, polar_grid, rel_floor: float, m_list: tuple[int, ...]) -> dict[str, Any]:
    plane_rt = _reconstruct_plane_vy_vz_polar(f, polar_grid)
    return angular_symmetry_metrics_from_polar_plane(
        plane_rt,
        polar_grid.r,
        polar_grid.theta,
        rel_floor=float(rel_floor),
        denominator="field",
        m_list=m_list,
    )


def _metric_row(
    *,
    section: str,
    nmax: int,
    field_name: str,
    met: dict[str, Any],
    m_list: tuple[int, ...],
    Q: int,
    maxK: int,
    u: float,
    nu_LB: float,
    polar_nr: int,
    polar_nalpha: int,
    rel_floor: float,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "section": section,
        "nmax": int(nmax),
        "p": int(nmax) + 1,
        "Q": int(Q),
        "maxK": int(maxK),
        "u": float(u),
        "nu_LB": float(nu_LB),
        "field": field_name,
        "polar_nr": int(polar_nr),
        "polar_nalpha": int(polar_nalpha),
        "rel_floor": float(rel_floor),
        "support_fraction_r": float(np.mean(met["mask_r"])),
        "E_ang_field": met["E_ang_field"],
        "E_ang_radial": met["E_ang_radial"],
        "E4_delta_fraction": met["E4_delta_fraction"],
        "dominant_m_by_delta": met["dominant_m_by_delta"],
    }
    for m in m_list:
        row[f"E{m}_field"] = met["E_m_field"][m]
        row[f"E{m}_delta"] = met["E_m_delta"][m]
    return row


def _cart_plane(f: np.ndarray, xlim: float, nx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    p = f.shape[0]
    x = np.linspace(-float(xlim), float(xlim), int(nx), dtype=np.float64)
    Y, Z = np.meshgrid(x, x, indexing="ij")
    psi0 = np.array([psi_1d(n, 0.0) for n in range(p)], dtype=np.float64)
    psi_y = np.stack([psi_1d(n, x) for n in range(p)], axis=0)
    psi_z = psi_y
    plane = np.einsum("nmp,n,my,pz->yz", f, psi0, psi_y, psi_z, optimize=True)
    return x, Y, Z, plane


def _cart_residual(f: np.ndarray, polar_grid, xlim: float, nx: int) -> tuple[np.ndarray, np.ndarray]:
    x, Y, Z, _plane = _cart_plane(f, xlim, nx)
    plane_rt = _reconstruct_plane_vy_vz_polar(f, polar_grid)
    radial_mean = np.mean(plane_rt, axis=1)
    rho = np.sqrt(Y * Y + Z * Z)
    radial_interp = np.interp(rho.ravel(), polar_grid.r, radial_mean, left=radial_mean[0], right=radial_mean[-1]).reshape(rho.shape)
    return x, _plane - radial_interp


def _plot(
    *,
    rows: list[dict[str, Any]],
    rep_fields: dict[str, np.ndarray],
    rep_metrics: dict[str, dict[str, Any]],
    rep_polar,
    args,
    m_list: tuple[int, ...],
) -> None:
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.labelsize": 8,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    colors = plt.get_cmap("tab10").colors
    fig = plt.figure(figsize=(7.05, 5.2), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.92])

    map_specs = [("Q(f0,f0)", "(a) nonlinear update"), ("L(h0;M_N)", "(b) linearized update")]
    residual_maps = {}
    for name, _label in map_specs:
        x, residual = _cart_residual(rep_fields[name], rep_polar, args.xlim, args.nx)
        residual_maps[name] = (x, residual, float(np.max(np.abs(residual))))
    common_scale = residual_maps["L(h0;M_N)"][2]
    if not (common_scale > 0.0 and np.isfinite(common_scale)):
        common_scale = max(v[2] for v in residual_maps.values())
    ims = []
    for j, (name, label) in enumerate(map_specs):
        ax = fig.add_subplot(gs[0, j])
        x, residual, panel_scale = residual_maps[name]
        shown = residual / common_scale if common_scale > 0.0 and np.isfinite(common_scale) else residual
        im = ax.imshow(
            shown.T,
            origin="lower",
            extent=[x[0], x[-1], x[0], x[-1]],
            cmap="RdBu_r",
            vmin=-1.0,
            vmax=1.0,
            interpolation="nearest",
        )
        ims.append(im)
        ax.set_aspect("equal")
        ax.set_xlabel(r"$v_y/v_{\mathrm{th}}$")
        ax.set_ylabel(r"$v_z/v_{\mathrm{th}}$")
        ax.text(0.02, 0.96, label, transform=ax.transAxes, va="top", ha="left", weight="bold", bbox=dict(facecolor="white", alpha=0.78, edgecolor="none"))
        ax.text(0.02, 0.08, rf"$\max|\delta g|={panel_scale:.2e}$", transform=ax.transAxes, va="bottom", ha="left", fontsize=7, bbox=dict(facecolor="white", alpha=0.78, edgecolor="none"))
    fig.colorbar(ims[-1], ax=[fig.axes[0], fig.axes[1]], fraction=0.045, pad=0.02, label=r"$\delta g/\max|\delta g|_{\rm lin}$")

    ax = fig.add_subplot(gs[1, 0])
    width = 0.38
    xloc = np.arange(len(m_list), dtype=float)
    for offset, name, color, label in [
        (-0.5 * width, "Q(f0,f0)", colors[1], r"nonlinear $Q$"),
        (0.5 * width, "L(h0;M_N)", colors[2], r"linearized $L_M$"),
    ]:
        vals = [rep_metrics[name]["E_m_delta"][m] for m in m_list]
        alpha = 0.48 if name == "Q(f0,f0)" else 0.86
        ax.bar(xloc + offset, vals, width=width, color=color, alpha=alpha, label=label)
    ax.set_xticks(xloc, [str(m) for m in m_list])
    ax.set_ylim(0.0, 1.08)
    ax.set_xlabel(r"angular harmonic $m$")
    ax.set_ylabel(r"$E_m/E_{\rm nonaxisym}$")
    ax.grid(True, axis="y", alpha=0.22)
    ax.text(0.02, 0.94, "(c)", transform=ax.transAxes, weight="bold")
    ax.legend(frameon=False, loc="upper right")
    ax.text(0.03, 0.06, "nonlinear amplitude is small; see (d)", transform=ax.transAxes, fontsize=6.8)

    ax = fig.add_subplot(gs[1, 1])
    scan = [r for r in rows if r["section"] == "nmax_scan"]
    styles = {
        "M_N": ("o-", colors[0], r"matched $M_N$"),
        "Q(f0,f0)": ("s-", colors[1], r"nonlinear $Q$"),
        "L(h0;M_N)": ("^-", colors[2], r"linearized $L_M$"),
    }
    for field_name, (style, color, label) in styles.items():
        vals = sorted([r for r in scan if r["field"] == field_name], key=lambda r: r["nmax"])
        ax.plot([r["nmax"] for r in vals], [r["E_ang_field"] for r in vals], style, color=color, label=label)
    ax.set_yscale("log")
    ax.set_xlabel(r"$n_{\max}$")
    ax.set_ylabel(r"$E_{\rm ang}$")
    ax.grid(True, alpha=0.22)
    ax.text(0.02, 0.94, "(d)", transform=ax.transAxes, weight="bold")
    ax.legend(frameon=False, loc="upper right")
    rep = rep_metrics["L(h0;M_N)"]
    ax.text(
        0.03,
        0.05,
        rf"$n_{{\max}}={args.nmax_repr}$: $E_4/E_{{\rm nonaxisym}}={rep['E_m_delta'][4]:.3f}$",
        transform=ax.transAxes,
        fontsize=7.5,
        bbox=dict(facecolor="white", alpha=0.82, edgecolor="0.85"),
    )

    fig.savefig(f"{args.outprefix}.pdf", bbox_inches="tight")
    fig.savefig(f"{args.outprefix}.png", dpi=int(args.dpi), bbox_inches="tight")


def main() -> None:
    ap = argparse.ArgumentParser(description="Signed angular PRL diagnostic for nonlinear vs linearized collision updates")
    ap.add_argument("--nmax_list", default="6,9,12,14")
    ap.add_argument("--nmax_repr", type=int, default=14)
    ap.add_argument("--Q", type=int, default=12)
    ap.add_argument("--maxK", type=int, default=256)
    ap.add_argument("--u", type=float, default=1.5)
    ap.add_argument("--nu_LB", type=float, default=0.0)
    ap.add_argument("--xlim", type=float, default=3.0)
    ap.add_argument("--nx", type=int, default=161)
    ap.add_argument("--polar_nr", type=int, default=96)
    ap.add_argument("--polar_nalpha", type=int, default=256)
    ap.add_argument("--rel_floor", type=float, default=0.05)
    ap.add_argument("--grid_check", default="129,257,513")
    ap.add_argument("--rel_floor_check", default="0,1e-4,1e-3,1e-2,5e-2")
    ap.add_argument("--filter_check_nu", default="0,0.02")
    ap.add_argument("--outprefix", default="Fig2_PRL_angular_signed_reviewproof")
    ap.add_argument("--csv", default="prl_angular_signed_reviewproof.csv")
    ap.add_argument("--json", default="prl_angular_signed_reviewproof.json")
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()

    nmax_list = _parse_int_list(args.nmax_list)
    m_list = (2, 4, 6, 8, 10, 12)
    rows: list[dict[str, Any]] = []
    by_key: dict[tuple[int, str], dict[str, Any]] = {}

    for nmax in nmax_list:
        print(f"[angular] nmax={nmax}", flush=True)
        polar_grid = prepare_polar_slice_grid(nmax=nmax, rmax=args.xlim, nr=args.polar_nr, nth=args.polar_nalpha)
        fields = _case_fields(nmax, args.Q, args.maxK, args.u, args.nu_LB)
        for field_name, arr in fields.items():
            met = _metrics_for_field(arr, polar_grid, args.rel_floor, m_list)
            row = _metric_row(
                section="nmax_scan",
                nmax=nmax,
                field_name=field_name,
                met=met,
                m_list=m_list,
                Q=args.Q,
                maxK=args.maxK,
                u=args.u,
                nu_LB=args.nu_LB,
                polar_nr=args.polar_nr,
                polar_nalpha=args.polar_nalpha,
                rel_floor=args.rel_floor,
            )
            rows.append(row)
            by_key[(nmax, field_name)] = row
            print(
                f"[angular] {field_name:14s} nmax={nmax:2d} "
                f"E_ang={row['E_ang_field']:.3e} E2={row['E2_delta']:.3f} "
                f"E4={row['E4_delta']:.3f} E6={row['E6_delta']:.3f} E8={row['E8_delta']:.3f} "
                f"dominant_m={row['dominant_m_by_delta']}",
                flush=True,
            )

    rep_polar = prepare_polar_slice_grid(nmax=args.nmax_repr, rmax=args.xlim, nr=args.polar_nr, nth=args.polar_nalpha)
    rep_fields = _case_fields(args.nmax_repr, args.Q, args.maxK, args.u, args.nu_LB)
    rep_metrics = {name: _metrics_for_field(arr, rep_polar, args.rel_floor, m_list) for name, arr in rep_fields.items()}

    robustness_rows: list[dict[str, Any]] = []
    # Polar-grid convergence for the representative fields.
    for nalpha in _parse_int_list(args.grid_check):
        nr = max(33, int(round(nalpha / 2)))
        grid = prepare_polar_slice_grid(nmax=args.nmax_repr, rmax=args.xlim, nr=nr, nth=nalpha)
        for field_name, arr in rep_fields.items():
            met = _metrics_for_field(arr, grid, args.rel_floor, m_list)
            robustness_rows.append(
                _metric_row(
                    section="grid_check",
                    nmax=args.nmax_repr,
                    field_name=field_name,
                    met=met,
                    m_list=m_list,
                    Q=args.Q,
                    maxK=args.maxK,
                    u=args.u,
                    nu_LB=args.nu_LB,
                    polar_nr=nr,
                    polar_nalpha=nalpha,
                    rel_floor=args.rel_floor,
                )
            )
    # Radial-support sensitivity; support is based on radial RMS, never positivity.
    for rel_floor in _parse_float_list(args.rel_floor_check):
        for field_name, arr in rep_fields.items():
            met = _metrics_for_field(arr, rep_polar, rel_floor, m_list)
            robustness_rows.append(
                _metric_row(
                    section="rel_floor_check",
                    nmax=args.nmax_repr,
                    field_name=field_name,
                    met=met,
                    m_list=m_list,
                    Q=args.Q,
                    maxK=args.maxK,
                    u=args.u,
                    nu_LB=args.nu_LB,
                    polar_nr=args.polar_nr,
                    polar_nalpha=args.polar_nalpha,
                    rel_floor=rel_floor,
                )
            )
    # Filter sensitivity is a separate robustness check, not the default instantaneous diagnostic.
    for nu in _parse_float_list(args.filter_check_nu):
        fields = _case_fields(args.nmax_repr, args.Q, args.maxK, args.u, nu)
        for field_name, arr in fields.items():
            met = _metrics_for_field(arr, rep_polar, args.rel_floor, m_list)
            robustness_rows.append(
                _metric_row(
                    section="filter_check",
                    nmax=args.nmax_repr,
                    field_name=field_name,
                    met=met,
                    m_list=m_list,
                    Q=args.Q,
                    maxK=args.maxK,
                    u=args.u,
                    nu_LB=nu,
                    polar_nr=args.polar_nr,
                    polar_nalpha=args.polar_nalpha,
                    rel_floor=args.rel_floor,
                )
            )

    csv_path = Path(args.csv)
    robust_csv = csv_path.with_name(csv_path.stem + "_robustness.csv")
    _write_csv(csv_path, rows)
    _write_csv(robust_csv, robustness_rows)
    meta = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "args": vars(args),
        "angle_name": "alpha = atan2(v_z, v_y)",
        "metric": "signed r dr d alpha weighted L2; no clipping; support mask from radial RMS only",
        "main_csv": str(csv_path),
        "robustness_csv": str(robust_csv),
    }
    Path(args.json).write_text(json.dumps(meta, indent=2))

    _plot(rows=rows, rep_fields=rep_fields, rep_metrics=rep_metrics, rep_polar=rep_polar, args=args, m_list=m_list)

    rep_row = by_key.get((args.nmax_repr, "L(h0;M_N)"))
    if rep_row is None:
        rep_row = _metric_row(
            section="representative",
            nmax=args.nmax_repr,
            field_name="L(h0;M_N)",
            met=rep_metrics["L(h0;M_N)"],
            m_list=m_list,
            Q=args.Q,
            maxK=args.maxK,
            u=args.u,
            nu_LB=args.nu_LB,
            polar_nr=args.polar_nr,
            polar_nalpha=args.polar_nalpha,
            rel_floor=args.rel_floor,
        )
    print(
        "[fact] representative linearized collision update: "
        f"E_ang={rep_row['E_ang_field']:.3e}, E2/E_nonaxisym={rep_row['E2_delta']:.3f}, "
        f"E4/E_nonaxisym={rep_row['E4_delta']:.3f}, E6/E_nonaxisym={rep_row['E6_delta']:.3f}, "
        f"E8/E_nonaxisym={rep_row['E8_delta']:.3f}, dominant_m={rep_row['dominant_m_by_delta']}, "
        f"grid=({args.polar_nr},{args.polar_nalpha}), rel_floor={args.rel_floor}, nu_LB={args.nu_LB}"
    )
    print(f"[ok] wrote {csv_path}, {robust_csv}, {args.json}, {args.outprefix}.pdf, {args.outprefix}.png")


if __name__ == "__main__":
    main()
