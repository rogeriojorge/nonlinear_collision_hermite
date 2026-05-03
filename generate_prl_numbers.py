#!/usr/bin/env python3
"""Generate TeX macros and a Markdown summary from PRL evidence CSV/JSON files."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _f(row: dict[str, str], key: str, default: float = math.nan) -> float:
    try:
        val = row.get(key, "")
        return float(val) if val not in ("", None) else default
    except Exception:
        return default


def _i(row: dict[str, str], key: str, default: int = 0) -> int:
    try:
        val = row.get(key, "")
        return int(float(val)) if val not in ("", None) else default
    except Exception:
        return default


def _format_sci_tex(x: float, sig: int = 3) -> str:
    if not math.isfinite(x):
        return r"\mathrm{nan}"
    if x == 0.0:
        return "0"
    exp = int(math.floor(math.log10(abs(x))))
    mant = x / (10.0**exp)
    decimals = max(0, sig - 1)
    return rf"{mant:.{decimals}f}\times10^{{{exp}}}"


def _format_float(x: float, sig: int = 3) -> str:
    if not math.isfinite(x):
        return "nan"
    return f"{x:.{sig}g}"


def _format_bytes_tex(num: float) -> str:
    units = [(1024.0**4, "TiB"), (1024.0**3, "GiB"), (1024.0**2, "MiB"), (1024.0, "KiB"), (1.0, "B")]
    for scale, unit in units:
        if abs(num) >= scale or unit == "B":
            val = num / scale
            if val >= 100:
                s = f"{val:.0f}"
            elif val >= 10:
                s = f"{val:.0f}" if abs(val - round(val)) < 0.05 else f"{val:.1f}"
            else:
                s = f"{val:.3g}"
            return rf"{s}\,\mathrm{{{unit}}}"
    return rf"{num:.3g}\,\mathrm{{B}}"


def _format_bytes_md(num: float) -> str:
    tex = _format_bytes_tex(num)
    return tex.replace(r"\,", " ").replace(r"\mathrm{", "").replace("}", "")


def _pick_benchmark(rows: list[dict[str, str]], bench_p: int | None = None) -> dict[str, str]:
    if not rows:
        raise SystemExit("No benchmark rows found")
    if bench_p is not None:
        for row in rows:
            if _i(row, "p") == int(bench_p):
                return row
        raise SystemExit(f"Requested --bench_p={bench_p}, but that p is not in the benchmark CSV")
    return sorted(rows, key=lambda r: _i(r, "p"))[-1]


def _pick_dense(dense_rows: list[dict[str, str]], meta: dict[str, Any]) -> tuple[int, float]:
    if dense_rows:
        row = sorted(dense_rows, key=lambda r: _i(r, "nmax"))[-1]
        return _i(row, "nmax"), _f(row, "dense_validate_rel_error")
    dense = meta.get("dense_validation", {}) if isinstance(meta, dict) else {}
    if "dense_validate_nmax" in dense:
        return int(dense.get("dense_validate_nmax", 0)), float(dense.get("dense_validate_rel_error", math.nan))
    return 0, math.nan


def _pick_angular(rows: list[dict[str, str]]) -> dict[str, str]:
    if not rows:
        raise SystemExit("No angular rows found")
    candidates = [r for r in rows if r.get("field") == "L(h0;M_N)"]
    return sorted(candidates or rows, key=lambda r: _i(r, "nmax"))[-1]


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate PRL TeX macros and Markdown summary")
    ap.add_argument("--benchmark_csv", default="prl_benchmark_reviewproof.csv")
    ap.add_argument("--benchmark_json", default="prl_benchmark_reviewproof.json")
    ap.add_argument("--dense_csv", default="prl_benchmark_reviewproof_dense_validation.csv")
    ap.add_argument("--dense_apply_csv", default="prl_benchmark_reviewproof_dense_apply.csv")
    ap.add_argument("--q_csv", default="prl_benchmark_reviewproof_q_sweep.csv")
    ap.add_argument("--angular_csv", default="prl_angular_signed_reviewproof.csv")
    ap.add_argument("--angular_json", default="prl_angular_signed_reviewproof.json")
    ap.add_argument("--bench_p", type=int, default=None, help="Optional p value for headline benchmark macros; default is largest measured p.")
    ap.add_argument("--tex", default="prl_numbers.tex")
    ap.add_argument("--summary", default="prl_results_summary.md")
    args = ap.parse_args()

    bench_rows = _read_csv(Path(args.benchmark_csv)) or _read_csv(Path("prl_benchmark_results.csv"))
    bench_meta = _read_json(Path(args.benchmark_json)) or _read_json(Path("prl_benchmark_meta.json"))
    dense_rows = _read_csv(Path(args.dense_csv))
    dense_apply_rows = _read_csv(Path(args.dense_apply_csv))
    q_rows = _read_csv(Path(args.q_csv))
    angular_rows = _read_csv(Path(args.angular_csv)) or _read_csv(Path("prl_angular_signed_metrics.csv"))
    angular_meta = _read_json(Path(args.angular_json))

    brow = _pick_benchmark(bench_rows, args.bench_p)
    dense_nmax, dense_err = _pick_dense(dense_rows, bench_meta)
    arow = _pick_angular(angular_rows)

    dense_bytes = _f(brow, "dense_tensor_bytes")
    working_bytes = _f(brow, "working_set_bytes")
    table_bytes = _f(brow, "model_tables_bytes")
    compression = dense_bytes / max(working_bytes, 1.0)
    collision_eval_time = _f(brow, "rhs_apply_time_s")
    if not math.isfinite(collision_eval_time):
        collision_eval_time = _f(brow, "rhs_total_time_s")
    dense_apply_budget = float(bench_meta.get("dense_apply", {}).get("time_budget_s", bench_meta.get("args", {}).get("dense_apply_time_budget", math.nan))) if bench_meta else math.nan
    max_soe = max((_f(r, "soe_err_rel") for r in bench_rows), default=math.nan)
    q_default = int(float(bench_meta.get("args", {}).get("Q", 12))) if bench_meta else 12
    q_ref = max((_i(r, "Q") for r in q_rows), default=0)
    q_default_row = next((r for r in q_rows if _i(r, "Q") == q_default), None)
    rhs_q_default_err = _f(q_default_row, "rhs_rel_error_vs_Qref") if q_default_row else math.nan
    rhs_q_max_err = max((_f(r, "rhs_rel_error_vs_Qref") for r in q_rows), default=math.nan)
    q_sweep_nmax = _i(q_rows[0], "nmax") if q_rows else 0

    nu_lb = _f(arow, "nu_LB", 0.0)
    efour = _f(arow, "E4_delta")
    eang = _f(arow, "E_ang_field")
    dom = _i(arow, "dominant_m_by_delta")

    macros = [
        "% Auto-generated by generate_prl_numbers.py. Do not edit by hand.",
        rf"\newcommand{{\HermiteCoeffFormula}}{{N=p^3}}",
        rf"\newcommand{{\DenseStorageFormula}}{{8p^9}}",
        rf"\newcommand{{\BenchP}}{{{_i(brow, 'p')}}}",
        rf"\newcommand{{\BenchNmax}}{{{_i(brow, 'nmax')}}}",
        rf"\newcommand{{\BenchNcoeff}}{{{_i(brow, 'N')}}}",
        rf"\newcommand{{\DenseTensorMemory}}{{{_format_bytes_tex(dense_bytes)}}}",
        rf"\newcommand{{\MeasuredTableMemory}}{{{_format_bytes_tex(table_bytes)}}}",
        rf"\newcommand{{\MeasuredWorkingSetMemory}}{{{_format_bytes_tex(working_bytes)}}}",
        rf"\newcommand{{\WorkingSetMemory}}{{{_format_bytes_tex(working_bytes)}}}",
        rf"\newcommand{{\CompressionFactor}}{{{_format_sci_tex(compression)}}}",
        rf"\newcommand{{\MedianRhsTimeAtBenchP}}{{{_format_float(collision_eval_time, 3)}\,\mathrm{{s}}}}",
        rf"\newcommand{{\CollisionEvalTimeAtBenchP}}{{{_format_float(collision_eval_time, 3)}\,\mathrm{{s}}}}",
        rf"\newcommand{{\DenseApplyTimeBudget}}{{{_format_float(dense_apply_budget, 3)}\,\mathrm{{s}}}}",
        rf"\newcommand{{\DenseValidationNmax}}{{{dense_nmax}}}",
        rf"\newcommand{{\DenseValidationError}}{{{_format_sci_tex(dense_err)}}}",
        rf"\newcommand{{\MaxSOEQuadratureError}}{{{_format_sci_tex(max_soe)}}}",
        rf"\newcommand{{\RhsQConvergenceNmax}}{{{q_sweep_nmax}}}",
        rf"\newcommand{{\RhsQDefault}}{{{q_default}}}",
        rf"\newcommand{{\RhsQReference}}{{{q_ref}}}",
        rf"\newcommand{{\RhsQDefaultError}}{{{_format_sci_tex(rhs_q_default_err)}}}",
        rf"\newcommand{{\MaxRhsQConvergenceError}}{{{_format_sci_tex(rhs_q_max_err)}}}",
        rf"\newcommand{{\CollisionEvalQDefault}}{{{q_default}}}",
        rf"\newcommand{{\CollisionEvalQReference}}{{{q_ref}}}",
        rf"\newcommand{{\CollisionEvalQError}}{{{_format_sci_tex(rhs_q_default_err)}}}",
        rf"\newcommand{{\AngularNmax}}{{{_i(arow, 'nmax')}}}",
        rf"\newcommand{{\AngularNuLB}}{{{_format_float(nu_lb, 3)}}}",
        rf"\newcommand{{\AngularEang}}{{{_format_sci_tex(eang)}}}",
        rf"\newcommand{{\EfourFraction}}{{{_format_float(efour, 3)}}}",
        rf"\newcommand{{\AngularDominantM}}{{{dom}}}",
        "",
    ]
    Path(args.tex).write_text("\n".join(macros))

    dense_md = _format_bytes_md(dense_bytes)
    working_md = _format_bytes_md(working_bytes)
    table_md = _format_bytes_md(table_bytes)
    summary = f"""# PRL Results Summary

Generated from CSV/JSON evidence files; numbers are not hand-entered in the manuscript.

## Storage reduction

At `p={_i(brow, 'p')}` (`nmax={_i(brow, 'nmax')}`, `N={_i(brow, 'N')}`), direct dense storage of the bilinear coefficient tensor would require `{dense_md}` (`8 p^9` doubles). The measured one-center/SOE table storage is `{table_md}`, and the measured tables plus temporary arrays for one collision evaluation are `{working_md}`. The array-storage reduction factor relative to the dense tensor is `{compression:.3e}`.

## Timing

At `p={_i(brow, 'p')}`, the measured median tensorized collision-update time is `{collision_eval_time:.3e} s` using the single-thread environment recorded in `{args.benchmark_json}`. This is a measured tensorized timing, not a claim of dense high-`p` runtime speedup.

## Explicit dense-assembly validation

For low order, explicit dense tensor assembly and contraction are compared directly with the tensorized collision evaluation. At `nmax={dense_nmax}`, the relative difference is `{dense_err:.3e}`. This validates that explicit dense assembly and the tensorized contraction produce the same bilinear map; it is not by itself an independent proof of the analytic Coulomb/Talmi formula.

## SOE quadrature and collision-evaluation convergence

The largest stored SOE quadrature relative error over the benchmark rows is `{max_soe:.3e}`. In the Q sweep at `nmax={q_sweep_nmax}`, `Q={q_default}` differs from `Q_ref={q_ref}` by `{rhs_q_default_err:.3e}` in collision-update relative norm; the largest collision-update difference over the sweep is `{rhs_q_max_err:.3e}`.

## Angular D4 projection

For the representative signed diagnostic at `nmax={_i(arow, 'nmax')}` and `nu_LB={nu_lb:g}`, the linearized collision update has `E_ang={eang:.3e}`, dominant measured harmonic `m={dom}`, and `E4/E_nonaxisym={efour:.3f}`. The metric is signed, unclipped, and weighted with `r dr d alpha` in the `v_y,v_z` plane.
"""
    Path(args.summary).write_text(summary)
    print(f"[ok] wrote {args.tex} and {args.summary}")
    print(summary)


if __name__ == "__main__":
    main()
