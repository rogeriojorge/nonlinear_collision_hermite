# nonlinear_collision_hermite
Repository for code used in nonlinear collision operator study using Hermite polynomials.

## Optional dense contraction with P/M precontraction

The NumPy pair RHS and JAX builder accept `contraction="rearranged"`:

```python
rhs = build_jax_functions(tables, contraction="rearranged")
df = rhs(fa, fb)

df_numpy = rhs_ab_np(fa, fb, tables, use_tt=False, tt_tol=0., tt_rmax=0,
                     contraction="rearranged")
```

This evaluates the same bilinear SOE operator by contracting each
one-dimensional projection `P` with its kernel factor `M` before multiplying
across dimensions. It avoids the severe floating-point cancellation possible
when a rounded dense `S` is multiplied by high-order triple-Hermite factors.
It uses full dense coefficient arrays and does not support the optional TT
path. Existing calls retain `contraction="canonical"`.

For repeated NumPy evaluations, compute `W = rearranged_pair_tables_np(tables)`
once and pass it to `rhs_ab_rearranged_np(fa, fb, tables, W)`; the JAX builder
already caches these tables in its returned function. This rearrangement does
not impose positivity, filtering, or invariant projection, and is not a general
floating-point accuracy guarantee.

Run the focused low-order equivalence, dtype, mode-selection and high-order
polynomial-reference checks with:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu \
  python -m unittest -v test_rearranged_contraction
```

The high-order test uses the input direction `fa[16,16,16]=1` and an analytic
Maxwellian `fb`, with the masses, scales and pair coefficient specified in the
test. Its reference evaluates the existing one-dimensional P/M formulas as
polynomials in the SOE integration variable on `[0,1]`: multiply the three axis
polynomials, apply the outer creation ladder and pair prefactor, and integrate
each monomial `c_k s^k` as `c_k/(k+1)`. The resulting corner coefficient is
`-12.12547373505867054708649179599768` at 120 decimal digits. The regression
tolerance `1e-6` allows the finite `Q=24` quadrature and float64 table errors;
it is not a claim of 120-digit accuracy for the float64 implementation.

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
