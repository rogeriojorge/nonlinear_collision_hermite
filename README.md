# nonlinear_collision_hermite
Repository for code used in nonlinear collision operator study using Hermite polynomials.

## PRL evidence figures

The PRL support scripts are designed to run from the repository root and write
publication-ready PDFs/PNGs plus machine-readable CSV/JSON data.

```bash
python landau_hermite_prl_benchmark.py \
  --nmax_list 4,6,8,10,12,14 \
  --Q 24 \
  --maxK 256 \
  --dense_apply_time_budget 30.0 \
  --dense_apply_memory_budget_gib 8.0 \
  --dense_apply_max_nmax 4 \
  --outprefix Fig1_PRL_tensor_barrier \
  --csv prl_benchmark_reviewproof.csv \
  --json prl_benchmark_reviewproof.json
```

Outputs:
- `Fig1_PRL_tensor_barrier_onecol.pdf` and `Fig1_PRL_tensor_barrier_onecol.png`
- `Fig1_PRL_tensor_barrier.pdf` and `Fig1_PRL_tensor_barrier.png`
- `prl_benchmark_reviewproof.csv`
- `prl_benchmark_reviewproof.json`
- `prl_benchmark_reviewproof_dense_validation.csv`
- `prl_benchmark_reviewproof_dense_apply.csv`
- `prl_benchmark_reviewproof_q_sweep.csv`

This benchmark compares the theoretical dense `p^9 = N^3` collision tensor
storage against the actual one-center/SOE table and collision-evaluation working-set storage, measures
table construction and collision-update timings with single-thread settings, validates the
SOE quadrature and collision-evaluation convergence with a `Q` sweep, performs
explicit dense tensor assembly only at low order, and times stored dense-tensor
contractions only while they remain within the requested time/memory budgets.

```bash
python landau_hermite_prl_angular_signed.py \
  --nmax_list 6,9,12,14 \
  --nmax_repr 14 \
  --Q 12 \
  --maxK 256 \
  --nu_LB 0.0 \
  --outprefix Fig2_PRL_angular_signed_reviewproof \
  --csv prl_angular_signed_reviewproof.csv \
  --json prl_angular_signed_reviewproof.json
```

Outputs:
- `Fig2_PRL_angular_signed_reviewproof.pdf` and `Fig2_PRL_angular_signed_reviewproof.png`
- `prl_angular_signed_reviewproof.csv`
- `prl_angular_signed_reviewproof.json`
- `prl_angular_signed_reviewproof_robustness.csv`

This figure uses signed, `r dr d alpha`-weighted angular diagnostics on the
`v_x=0` slice with `alpha = atan2(v_z, v_y)`. It does not clip signed fields. It
reports the non-axisymmetric defect of the matched projected Maxwellian, the
nonlinear collision update, and the linearized collision update, plus the angular-harmonic spectrum
(`m=2,4,6,8,10,12`). The robustness CSV records polar-grid, support-threshold,
and small-filter sensitivity checks.

```bash
python generate_prl_numbers.py
```

Outputs:
- `prl_numbers.tex`
- `prl_results_summary.md`

These files are generated from the CSV/JSON evidence and keep manuscript
numbers, captions, and summaries synchronized.

```bash
python landau_hermite_angular_metrics.py --selftest
```

This checks that the angular metric correctly identifies analytic `cos(2 theta)`
and `cos(4 theta)` perturbations.

The helper runner executes the same sequence:

```bash
./run_prl_figures.sh
```

Dense `p^9` tensor construction is intentionally restricted to low-order
validation. It is not part of the production algorithm and should not be enabled
for large `nmax`.
