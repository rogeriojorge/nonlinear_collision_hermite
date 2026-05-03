#!/usr/bin/env bash
set -euo pipefail

# Reviewer-proof PRL evidence bundle. Expected runtime is machine dependent;
# on a recent laptop this usually takes a few minutes because it includes
# repeated NumPy collision-update timings, low-order dense assembly, and a Q sweep.
python landau_hermite_prl_benchmark.py \
  --nmax_list 4,6,8,10,12,14 \
  --Q 24 \
  --maxK 256 \
  --repeats 5 \
  --warmup 1 \
  --dense_validate_nmax 3 \
  --dense_apply_time_budget 30.0 \
  --dense_apply_memory_budget_gib 8.0 \
  --dense_apply_max_nmax 4 \
  --q_sweep_nmax 12 \
  --q_sweep 6,8,10,12,16,24,32,48 \
  --outprefix Fig1_PRL_tensor_barrier \
  --csv prl_benchmark_reviewproof.csv \
  --json prl_benchmark_reviewproof.json

# Signed, unclipped, r dr d alpha-weighted angular diagnostic. The default
# nu_LB=0 keeps this an instantaneous Landau collision-update diagnostic; nonzero filters
# are only reported in the robustness CSV.
python landau_hermite_prl_angular_signed.py \
  --nmax_list 6,9,12,14 \
  --nmax_repr 14 \
  --Q 12 \
  --maxK 256 \
  --nu_LB 0.0 \
  --outprefix Fig2_PRL_angular_signed_reviewproof \
  --csv prl_angular_signed_reviewproof.csv \
  --json prl_angular_signed_reviewproof.json

python landau_hermite_angular_metrics.py --selftest
python generate_prl_numbers.py
