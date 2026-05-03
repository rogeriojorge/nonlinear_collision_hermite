#!/usr/bin/env python3
"""Reviewer-proof PRL benchmark for the Hermite Landau tensor barrier.

The script measures the array-storage barrier associated with a dense
coefficient tensor C_{k,k_a,k_b} of size p^9, the actual one-center/SOE
working arrays used by the implementation, wall-clock timings, explicit
low-order dense validation, dense stored-tensor contraction timing while
feasible, and Q-convergence of the collision evaluation.
"""

from __future__ import annotations

import os

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")
os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
import csv
import dataclasses
import json
import math
import platform
import resource
import sys
import time
import tracemalloc
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np

from landau_hermite_jax_relative_entropy_1 import (
    Species,
    ModelTablesNP,
    build_S_np,
    build_ic_fig1_1sp_twostream,
    build_model_tables_np,
    invariants_from_tensor,
    rhs_ab_np,
    rhs_ab_with_S_np,
)


def _parse_int_list(s: str) -> list[int]:
    return [int(x) for x in str(s).replace(";", ",").split(",") if x.strip()]


def _array_bytes_dataclass(obj) -> int:
    total = 0
    for f in dataclasses.fields(obj):
        val = getattr(obj, f.name)
        if isinstance(val, np.ndarray):
            total += int(val.nbytes)
    return total


def _rss_bytes() -> int | None:
    try:
        import psutil  # type: ignore

        return int(psutil.Process().memory_info().rss)
    except Exception:
        return None


def _ru_maxrss_bytes() -> int:
    val = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # macOS reports bytes; Linux reports KiB. This repository is mostly run on
    # macOS, but keep the conversion robust enough for CI/Linux.
    if sys.platform.startswith("darwin"):
        return val
    return val * 1024


def _time_samples(fn: Callable[[], Any], repeats: int, warmup: int) -> tuple[dict[str, float], Any]:
    for _ in range(max(0, int(warmup))):
        fn()
    samples: list[float] = []
    last = None
    for _ in range(max(1, int(repeats))):
        t0 = time.perf_counter()
        last = fn()
        samples.append(time.perf_counter() - t0)
    arr = np.asarray(samples, dtype=np.float64)
    med = float(np.median(arr))
    stats = {
        "median": med,
        "mad": float(np.median(np.abs(arr - med))),
        "q25": float(np.quantile(arr, 0.25)),
        "q75": float(np.quantile(arr, 0.75)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }
    return stats, last


def _peak_measure(fn: Callable[[], Any]) -> dict[str, float | int | None]:
    rss0 = _rss_bytes()
    ru0 = _ru_maxrss_bytes()
    tracemalloc.start()
    t0 = time.perf_counter()
    try:
        fn()
    finally:
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    rss1 = _rss_bytes()
    ru1 = _ru_maxrss_bytes()
    return {
        "wall_time_s": float(time.perf_counter() - t0),
        "tracemalloc_peak_bytes": int(peak),
        "rss_delta_bytes": None if rss0 is None or rss1 is None else int(rss1 - rss0),
        "ru_maxrss_delta_bytes": int(max(0, ru1 - ru0)),
    }


def _soe_error(T: ModelTablesNP, nmax: int) -> tuple[float, float, int]:
    max_A = int(3 * (2 * int(nmax) + 1))
    A = np.arange(max_A + 1, dtype=np.float64)
    approx = np.sum(T.w_nodes[:, None] * (T.s_nodes[:, None] ** (2.0 * A[None, :])), axis=0)
    exact = 1.0 / (2.0 * A + 1.0)
    err_abs = np.max(np.abs(approx - exact))
    err_rel = np.max(np.abs(approx - exact) / np.maximum(np.abs(exact), 1e-300))
    return float(err_abs), float(err_rel), max_A


def _format_bytes(num: float) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    x = float(num)
    for unit in units:
        if abs(x) < 1024.0 or unit == units[-1]:
            return f"{x:.3g} {unit}"
        x /= 1024.0
    return f"{x:.3g} TiB"


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


def _build_dense_tensor(nmax: int, Q: int, maxK: int) -> tuple[ModelTablesNP, np.ndarray, float]:
    T = build_model_tables_np(nmax=nmax, Q=Q, maxK=maxK, ma=1.0, mb=1.0, vtha=1.0, vthb=1.0, nu_ab=1.0)
    p = int(nmax) + 1
    N = p**3
    C = np.zeros((N, N, N), dtype=np.float64)
    basis = np.eye(N, dtype=np.float64)
    t0 = time.perf_counter()
    for ia in range(N):
        ea = basis[ia].reshape(p, p, p)
        for ib in range(N):
            eb = basis[ib].reshape(p, p, p)
            C[:, ia, ib] = rhs_ab_np(ea, eb, T, use_tt=False, tt_tol=0.0, tt_rmax=1).reshape(-1)
    dense_build_time = time.perf_counter() - t0
    return T, C, float(dense_build_time)


def _dense_tensor_validate(nmax: int, Q: int, maxK: int, u: float, T: ModelTablesNP | None = None, C: np.ndarray | None = None, dense_build_time: float | None = None) -> dict[str, Any]:
    sp = Species(m=1.0, vth=1.0)
    if T is None or C is None:
        T, C, dense_build_time = _build_dense_tensor(nmax, Q, maxK)
    assert dense_build_time is not None
    p = int(nmax) + 1
    N = p**3
    fa = build_ic_fig1_1sp_twostream(nmax=nmax, sp=sp, u=u, enforce_nonneg=False).f
    fb = build_ic_fig1_1sp_twostream(nmax=nmax, sp=sp, u=0.8 * u, enforce_nonneg=False).f
    dense_rhs = np.einsum("kab,a,b->k", C, fa.reshape(-1), fb.reshape(-1), optimize=True).reshape(p, p, p)
    ref_rhs = rhs_ab_np(fa, fb, T, use_tt=False, tt_tol=0.0, tt_rmax=1)
    rel = float(np.linalg.norm(dense_rhs - ref_rhs) / (np.linalg.norm(ref_rhs) + 1e-300))
    row: dict[str, Any] = {
        "nmax": int(nmax),
        "p": int(p),
        "N": int(N),
        "dense_validate_rel_error": rel,
        "dense_validate_build_time_s": float(dense_build_time),
        "dense_validate_bytes": int(C.nbytes),
    }
    total = float(C.size)
    for tol in (1e-14, 1e-12, 1e-10):
        nnz = int(np.count_nonzero(np.abs(C) > tol))
        tag = f"{tol:.0e}".replace("-", "m")
        row[f"nnz_gt_{tag}"] = nnz
        row[f"nnz_frac_gt_{tag}"] = float(nnz / total)
    return row


def _dense_apply_timing_rows(
    *,
    max_nmax: int,
    Q: int,
    maxK: int,
    u: float,
    repeats: int,
    warmup: int,
    time_budget_s: float,
    memory_budget_gib: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    sp = Species(m=1.0, vth=1.0)
    rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    memory_budget_bytes = float(memory_budget_gib) * 1024.0**3
    stopped = False
    for nmax in range(1, int(max_nmax) + 1):
        p = nmax + 1
        N = p**3
        dense_bytes = 8 * (p**9)
        base: dict[str, Any] = {
            "nmax": int(nmax),
            "p": int(p),
            "N": int(N),
            "dense_tensor_bytes": int(dense_bytes),
            "dense_apply_time_budget_s": float(time_budget_s),
            "dense_apply_memory_budget_gib": float(memory_budget_gib),
        }
        if stopped:
            rows.append({**base, "status": "skipped_after_budget_stop", "dense_apply_time_s": math.nan})
            continue
        if dense_bytes > memory_budget_bytes:
            rows.append({**base, "status": "skipped_memory_budget", "dense_apply_time_s": math.nan})
            stopped = True
            continue
        print(f"[dense-apply] building dense tensor nmax={nmax} p={p} ({_format_bytes(dense_bytes)})", flush=True)
        try:
            T, C, build_time = _build_dense_tensor(nmax, Q, maxK)
            vrow = _dense_tensor_validate(nmax, Q, maxK, u, T=T, C=C, dense_build_time=build_time)
            validation_rows.append(vrow)
            f = build_ic_fig1_1sp_twostream(nmax=nmax, sp=sp, u=u, enforce_nonneg=False).f.reshape(-1)
            stats, dense_flat = _time_samples(
                lambda C=C, f=f: np.einsum("kab,a,b->k", C, f, f, optimize=True),
                repeats=max(1, int(repeats)),
                warmup=max(0, int(warmup)),
            )
            ref = rhs_ab_np(f.reshape(p, p, p), f.reshape(p, p, p), T, use_tt=False, tt_tol=0.0, tt_rmax=1).reshape(-1)
            abs_err = float(np.linalg.norm(dense_flat - ref))
            ref_norm = float(np.linalg.norm(ref))
            rel = float(abs_err / ref_norm) if ref_norm > 1e-14 else math.nan
            row = {
                **base,
                "status": "ok" if np.isfinite(rel) else "ok_near_zero_reference",
                "dense_build_time_s": float(build_time),
                "dense_apply_time_s": stats["median"],
                "dense_apply_time_mad_s": stats["mad"],
                "dense_apply_abs_error": abs_err,
                "dense_apply_ref_norm": ref_norm,
                "dense_apply_rel_error": rel,
            }
            rows.append(row)
            print(f"[dense-apply] nmax={nmax} apply={stats['median']:.3e}s rel_error={rel:.3e} status={row['status']}", flush=True)
            if stats["median"] > float(time_budget_s):
                stopped = True
        except Exception as exc:
            rows.append({**base, "status": f"failed: {exc}", "dense_apply_time_s": math.nan})
            stopped = True
    return rows, validation_rows


def _run_q_sweep(nmax: int, Q_values: list[int], maxK: int, u: float, repeats: int) -> list[dict[str, Any]]:
    sp = Species(m=1.0, vth=1.0)
    fa = build_ic_fig1_1sp_twostream(nmax=nmax, sp=sp, u=u, enforce_nonneg=False).f
    fb = build_ic_fig1_1sp_twostream(nmax=nmax, sp=sp, u=u, enforce_nonneg=False).f
    results: list[dict[str, Any]] = []
    rhs_by_Q: dict[int, np.ndarray] = {}
    max_Q = max(Q_values)
    for Q in Q_values:
        print(f"[q-sweep] nmax={nmax} Q={Q}", flush=True)
        build_stats, T = _time_samples(
            lambda Q=Q: build_model_tables_np(nmax=nmax, Q=Q, maxK=maxK, ma=1.0, mb=1.0, vtha=1.0, vthb=1.0, nu_ab=1.0),
            repeats=1,
            warmup=0,
        )
        rhs_stats, rhs = _time_samples(
            lambda T=T: rhs_ab_np(fa, fb, T, use_tt=False, tt_tol=0.0, tt_rmax=1),
            repeats=max(1, int(repeats)),
            warmup=0,
        )
        rhs_by_Q[int(Q)] = rhs
        soe_abs, soe_rel, max_A = _soe_error(T, nmax)
        results.append(
            {
                "nmax": int(nmax),
                "p": int(nmax) + 1,
                "Q": int(Q),
                "Q_ref": int(max_Q),
                "table_build_time_s": build_stats["median"],
                "rhs_total_time_s": rhs_stats["median"],
                "soe_max_A": int(max_A),
                "soe_err_abs": float(soe_abs),
                "soe_err_rel": float(soe_rel),
            }
        )
    rhs_ref = rhs_by_Q[max_Q]
    ref_norm = float(np.linalg.norm(rhs_ref)) + 1e-300
    for row in results:
        Q = int(row["Q"])
        row["rhs_rel_error_vs_Qref"] = float(np.linalg.norm(rhs_by_Q[Q] - rhs_ref) / ref_norm)
    return results


def _plot(rows: list[dict[str, Any]], dense_rows: list[dict[str, Any]], dense_apply_rows: list[dict[str, Any]], q_rows: list[dict[str, Any]], outprefix: str, dpi: int) -> None:
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
    p = np.array([r["p"] for r in rows], dtype=float)
    gib = 1024.0**3
    fig, axs = plt.subplots(1, 3, figsize=(7.1, 2.35), constrained_layout=True)

    ax = axs[0]
    ax.plot(p, np.array([r["dense_tensor_bytes"] for r in rows]) / gib, "o-", color=colors[0])
    ax.plot(p, np.array([r["model_tables_bytes"] for r in rows]) / gib, "s-", color=colors[1])
    ax.plot(p, np.array([r["working_set_bytes"] for r in rows]) / gib, "^-", color=colors[2])
    ax.set_yscale("log")
    ax.set_xlabel(r"$p=n_{\max}+1$")
    ax.set_ylabel("array storage (GiB)")
    ax.grid(True, alpha=0.22)
    ax.text(0.02, 0.94, "(a)", transform=ax.transAxes, weight="bold")
    ax.text(6.0, 0.075, r"dense $8p^9$", color=colors[0], fontsize=6.8)
    ax.text(5.55, 7.4e-4, "basis/Coulomb tables", color=colors[1], fontsize=6.5)
    pick = sorted(rows, key=lambda r: int(r["p"]))[-1]
    ax.annotate(
        f"dense tensor: {_format_bytes(pick['dense_tensor_bytes'])}",
        xy=(pick["p"], pick["dense_tensor_bytes"] / gib),
        xytext=(10.2, 84.0),
        textcoords="data",
        arrowprops=dict(arrowstyle="->", lw=0.8, color="0.25"),
        fontsize=6.7,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.82, pad=1.5),
    )
    ax.annotate(
        f"working set: {_format_bytes(pick['working_set_bytes'])}",
        xy=(pick["p"], pick["working_set_bytes"] / gib),
        xytext=(9.2, 0.030),
        textcoords="data",
        arrowprops=dict(arrowstyle="->", lw=0.7, color="0.35"),
        fontsize=6.7,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.82, pad=1.5),
    )

    ax = axs[1]
    timing_specs = [
        ("table_build_time_s", "precompute tables", "o", colors[0]),
        ("S_build_time_s", "prepare contraction", "s", colors[1]),
        ("rhs_apply_time_s", r"collision update $\dot f=Q(f,f)$", "d", colors[3]),
    ]
    for key, label, marker, color in timing_specs:
        y = np.array([r[key] for r in rows], dtype=float)
        yerr = np.array([r.get(key.replace("_s", "_mad_s"), 0.0) for r in rows], dtype=float)
        ax.errorbar(p, y, yerr=yerr, marker=marker, color=color, lw=1.3, capsize=2)
    dense_ok = [r for r in dense_apply_rows if str(r.get("status", "")).startswith("ok") and np.isfinite(float(r.get("dense_apply_time_s", math.nan)))]
    if dense_ok:
        ax.plot(
            [r["p"] for r in dense_ok],
            [r["dense_apply_time_s"] for r in dense_ok],
            "x--",
            color="0.25",
            lw=1.2,
        )
        ax.text(2.25, 4.4e-5, "dense tensor\ncontraction", color="0.25", fontsize=6.4)
    ax.set_yscale("log")
    ax.set_xlabel(r"$p=n_{\max}+1$")
    ax.set_ylabel("median wall time (s)")
    ax.grid(True, alpha=0.22)
    ax.text(0.02, 0.94, "(b)", transform=ax.transAxes, weight="bold")
    ax.text(8.65, 1.25e-2, "precompute tables", color=colors[0], fontsize=6.4)
    ax.text(8.75, 6.0e-2, "prepare contraction", color=colors[1], fontsize=6.4)
    ax.text(9.35, 1.85, r"collision update $\dot f=Q(f,f)$", color=colors[3], fontsize=6.4)

    ax = axs[2]
    if q_rows:
        q = np.array([r["Q"] for r in q_rows], dtype=float)
        rhs_err = np.array([r["rhs_rel_error_vs_Qref"] for r in q_rows], dtype=float)
        ax.plot(q, rhs_err, "o-", color=colors[0])
    ax.set_yscale("log")
    ax.set_xlabel(r"SOE quadrature points $Q$")
    ax.set_ylabel(r"relative error in $\dot f=Q(f,f)$")
    ax.grid(True, alpha=0.22)
    ax.text(0.02, 0.94, "(c)", transform=ax.transAxes, weight="bold")
    ax.text(0.98, 0.92, r"$Q_{\rm ref}=48$", transform=ax.transAxes, ha="right", va="top", fontsize=6.8)

    fig.savefig(f"{outprefix}.pdf", bbox_inches="tight")
    fig.savefig(f"{outprefix}.png", dpi=int(dpi), bbox_inches="tight")


def main() -> None:
    ap = argparse.ArgumentParser(description="Reviewer-proof PRL benchmark for one-center/SOE Hermite Landau evaluation")
    ap.add_argument("--nmax_list", default="4,6,8,10,12,14,16")
    ap.add_argument("--Q", type=int, default=12)
    ap.add_argument("--maxK", type=int, default=256)
    ap.add_argument("--u", type=float, default=1.5)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--tt_tol", type=float, default=1e-10)
    ap.add_argument("--tt_rmax", type=int, default=48)
    ap.add_argument("--dense_validate_nmax", type=int, default=3)
    ap.add_argument("--dense_apply_time_budget", type=float, default=30.0)
    ap.add_argument("--dense_apply_memory_budget_gib", type=float, default=8.0)
    ap.add_argument("--dense_apply_max_nmax", type=int, default=4)
    ap.add_argument("--q_sweep_nmax", type=int, default=12)
    ap.add_argument("--q_sweep", default="6,8,10,12,16,24,32,48")
    ap.add_argument("--q_sweep_repeats", type=int, default=1)
    ap.add_argument("--outprefix", default="Fig1_PRL_tensor_barrier")
    ap.add_argument("--csv", default="prl_benchmark_reviewproof.csv")
    ap.add_argument("--json", default="prl_benchmark_reviewproof.json")
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()

    rows: list[dict[str, Any]] = []
    sp = Species(m=1.0, vth=1.0)
    for nmax in _parse_int_list(args.nmax_list):
        p = nmax + 1
        N = p**3
        print(f"[bench] nmax={nmax} p={p} N={N}", flush=True)
        table_stats, T = _time_samples(
            lambda nmax=nmax: build_model_tables_np(nmax=nmax, Q=args.Q, maxK=args.maxK, ma=1.0, mb=1.0, vtha=1.0, vthb=1.0, nu_ab=1.0),
            repeats=max(1, args.repeats),
            warmup=0,
        )
        fa = build_ic_fig1_1sp_twostream(nmax=nmax, sp=sp, u=float(args.u), enforce_nonneg=False).f
        fb = build_ic_fig1_1sp_twostream(nmax=nmax, sp=sp, u=float(args.u), enforce_nonneg=False).f
        S_stats, S_pair = _time_samples(lambda: build_S_np(fb, T), repeats=args.repeats, warmup=args.warmup)
        S1, S2 = S_pair
        apply_stats, rhs_ref = _time_samples(
            lambda: rhs_ab_with_S_np(fa, S1, S2, T, use_tt=False, tt_tol=0.0, tt_rmax=1),
            repeats=args.repeats,
            warmup=args.warmup,
        )
        total_stats, _ = _time_samples(
            lambda: rhs_ab_np(fa, fb, T, use_tt=False, tt_tol=0.0, tt_rmax=1),
            repeats=args.repeats,
            warmup=args.warmup,
        )
        tt_stats = {"median": math.nan, "mad": math.nan}
        tt_err = math.nan
        tt_valid = False
        try:
            tt_stats, rhs_tt = _time_samples(
                lambda: rhs_ab_with_S_np(fa, S1, S2, T, use_tt=True, tt_tol=args.tt_tol, tt_rmax=args.tt_rmax),
                repeats=max(1, min(args.repeats, 3)),
                warmup=min(args.warmup, 1),
            )
            tt_err = float(np.linalg.norm(rhs_tt - rhs_ref) / (np.linalg.norm(rhs_ref) + 1e-300))
            tt_valid = bool(np.isfinite(tt_err) and tt_err < 1e-6)
        except Exception as exc:
            print(f"[bench] TT skipped for nmax={nmax}: {exc}", flush=True)

        peak_table = _peak_measure(lambda nmax=nmax: build_model_tables_np(nmax=nmax, Q=args.Q, maxK=args.maxK, ma=1.0, mb=1.0, vtha=1.0, vthb=1.0, nu_ab=1.0))
        peak_S = _peak_measure(lambda: build_S_np(fb, T))
        peak_apply = _peak_measure(lambda: rhs_ab_with_S_np(fa, S1, S2, T, use_tt=False, tt_tol=0.0, tt_rmax=1))

        inv_rate = invariants_from_tensor(rhs_ref, sp)
        soe_abs, soe_rel, max_A = _soe_error(T, nmax)
        model_bytes = _array_bytes_dataclass(T)
        working_bytes = model_bytes + int(S1.nbytes) + int(S2.nbytes)
        row: dict[str, Any] = {
            "nmax": int(nmax),
            "p": int(p),
            "N": int(N),
            "dense_tensor_entries": int(p**9),
            "dense_tensor_bytes": int(8 * (p**9)),
            "model_tables_bytes": int(model_bytes),
            "S1_bytes": int(S1.nbytes),
            "S2_bytes": int(S2.nbytes),
            "working_set_bytes": int(working_bytes),
            "table_build_time_s": table_stats["median"],
            "table_build_time_mad_s": table_stats["mad"],
            "S_build_time_s": S_stats["median"],
            "S_build_time_mad_s": S_stats["mad"],
            "rhs_apply_time_s": apply_stats["median"],
            "rhs_apply_time_mad_s": apply_stats["mad"],
            "rhs_total_time_s": total_stats["median"],
            "rhs_total_time_mad_s": total_stats["mad"],
            "rhs_tt_time_s": tt_stats["median"],
            "rhs_tt_time_mad_s": tt_stats.get("mad", math.nan),
            "tt_rel_error": float(tt_err),
            "tt_valid": bool(tt_valid),
            "mass_rate_abs": float(abs(rhs_ref[0, 0, 0])),
            "momentum_rate_norm": float(np.linalg.norm(inv_rate[1:4])),
            "energy_rate_abs": float(abs(inv_rate[4])),
            "soe_max_A": int(max_A),
            "soe_err_abs": float(soe_abs),
            "soe_err_rel": float(soe_rel),
            "table_tracemalloc_peak_bytes": peak_table["tracemalloc_peak_bytes"],
            "S_tracemalloc_peak_bytes": peak_S["tracemalloc_peak_bytes"],
            "rhs_apply_tracemalloc_peak_bytes": peak_apply["tracemalloc_peak_bytes"],
            "table_rss_delta_bytes": peak_table["rss_delta_bytes"],
            "S_rss_delta_bytes": peak_S["rss_delta_bytes"],
            "rhs_apply_rss_delta_bytes": peak_apply["rss_delta_bytes"],
            "table_ru_maxrss_delta_bytes": peak_table["ru_maxrss_delta_bytes"],
            "S_ru_maxrss_delta_bytes": peak_S["ru_maxrss_delta_bytes"],
            "rhs_apply_ru_maxrss_delta_bytes": peak_apply["ru_maxrss_delta_bytes"],
        }
        rows.append(row)
        print(
            f"[bench] p={p}: dense={_format_bytes(row['dense_tensor_bytes'])} "
            f"working={_format_bytes(working_bytes)} total_rhs={row['rhs_total_time_s']:.3e}s "
            f"SOE_rel={soe_rel:.3e} TT_err={tt_err:.3e}",
            flush=True,
        )

    dense_apply_rows, dense_rows = _dense_apply_timing_rows(
        max_nmax=max(int(args.dense_validate_nmax), int(args.dense_apply_max_nmax)),
        Q=args.Q,
        maxK=args.maxK,
        u=args.u,
        repeats=max(1, min(args.repeats, 3)),
        warmup=min(args.warmup, 1),
        time_budget_s=float(args.dense_apply_time_budget),
        memory_budget_gib=float(args.dense_apply_memory_budget_gib),
    )
    dense_rows = [r for r in dense_rows if int(r["nmax"]) <= int(args.dense_validate_nmax)]
    if dense_rows:
        for drow in dense_rows:
            print(
                f"[dense] nmax={drow['nmax']} rel_error={drow['dense_validate_rel_error']:.3e} "
                f"nnz(1e-12)={drow['nnz_frac_gt_1em12']:.3e}",
                flush=True,
            )

    q_values = _parse_int_list(args.q_sweep)
    q_rows = _run_q_sweep(args.q_sweep_nmax, q_values, args.maxK, args.u, args.q_sweep_repeats) if q_values else []

    csv_path = Path(args.csv)
    dense_csv = csv_path.with_name(csv_path.stem + "_dense_validation.csv")
    dense_apply_csv = csv_path.with_name(csv_path.stem + "_dense_apply.csv")
    q_csv = csv_path.with_name(csv_path.stem + "_q_sweep.csv")
    _write_csv(csv_path, rows)
    _write_csv(dense_csv, dense_rows)
    _write_csv(dense_apply_csv, dense_apply_rows)
    _write_csv(q_csv, q_rows)

    meta: dict[str, Any] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python": sys.version,
        "numpy": np.__version__,
        "argv": sys.argv,
        "args": vars(args),
        "thread_env": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")},
        "plotted_memory_quantity": "array nbytes: dense 8*p^9, ModelTablesNP numpy arrays, plus S1/S2 temporary arrays for one collision evaluation",
        "extra_csv": {"dense_validation": str(dense_csv), "dense_apply": str(dense_apply_csv), "q_sweep": str(q_csv)},
        "dense_apply": {
            "time_budget_s": float(args.dense_apply_time_budget),
            "memory_budget_gib": float(args.dense_apply_memory_budget_gib),
            "max_nmax": int(args.dense_apply_max_nmax),
        },
    }
    try:
        import jax  # type: ignore

        meta["jax"] = jax.__version__
    except Exception as exc:
        meta["jax"] = f"not importable: {exc}"
    Path(args.json).write_text(json.dumps(meta, indent=2))
    _plot(rows, dense_rows, dense_apply_rows, q_rows, args.outprefix, args.dpi)

    if rows:
        pick = sorted(rows, key=lambda r: int(r["p"]))[-1]
        comp = pick["dense_tensor_bytes"] / max(float(pick["working_set_bytes"]), 1.0)
        print(
            f"[fact] At p={pick['p']}, dense p^9 storage would require {_format_bytes(pick['dense_tensor_bytes'])}, "
            f"while one-center/SOE tables plus one collision-evaluation working set use {_format_bytes(pick['working_set_bytes'])} "
            f"(factor {comp:.3e})."
        )
        if dense_rows:
            best_dense = dense_rows[-1]
            print(
                f"[fact] Explicit dense assembly through nmax={best_dense['nmax']} agrees with the tensorized collision evaluation to "
                f"{best_dense['dense_validate_rel_error']:.3e}."
            )
        if q_rows:
            default = next((r for r in q_rows if int(r["Q"]) == int(args.Q)), q_rows[-1])
            print(
                f"[fact] Collision-evaluation convergence at nmax={args.q_sweep_nmax}: Q={default['Q']} differs from "
                f"Q_ref={default['Q_ref']} by {default['rhs_rel_error_vs_Qref']:.3e}."
            )
        dense_ok = [r for r in dense_apply_rows if str(r.get("status", "")).startswith("ok")]
        if dense_ok:
            last_dense = sorted(dense_ok, key=lambda r: int(r["p"]))[-1]
            print(
                f"[fact] Dense stored-tensor contraction timing reached p={last_dense['p']} "
                f"(nmax={last_dense['nmax']}) with median {last_dense['dense_apply_time_s']:.3e}s."
            )
        print(f"[ok] wrote {csv_path}, {dense_csv}, {dense_apply_csv}, {q_csv}, {args.json}, {args.outprefix}.pdf, {args.outprefix}.png")


if __name__ == "__main__":
    main()
