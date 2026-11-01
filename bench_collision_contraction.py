"""Stage-split timing of one collision evaluation: MPO einsum (baseline) vs Gauss-Hermite nodes.

Single thread, A/B/A/B alternation on the same two-stream state; refuses a dirty tree.
    python bench_collision_contraction.py --n 5,7,9,11,13,15,17,19,21 --Q 24 --reps 5 --out bench.json
    python bench_collision_contraction.py --species 1,2,3,4 --n 9,13,17 --Q 24 --reps 5 --out species.json

The --species mode separates table precomputation, JIT compilation and repeated updates for all
N_s^2 ordered species pairs, each with its own tables (masses 4^s, equal temperature).
"""
import argparse
import json
import os
import subprocess

os.environ.update(
    OMP_NUM_THREADS="1",
    OPENBLAS_NUM_THREADS="1",
    JAX_PLATFORMS="cpu",
    XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1",
)
import time

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

import landau_hermite_jax_relative_entropy_1 as L


def baseline_build_S(T):
    """`build_S_jax` of `build_jax_functions`, jitted alone."""
    M, wq = jnp.asarray(T.M_buildS), jnp.asarray(T.wq_prefI)
    p, pk = T.p, T.p_kp

    def kron(Mx, My, Mz, f):
        t = jnp.tensordot(Mx, f, axes=(1, 0))
        t = jnp.tensordot(My, t, axes=(1, 1)).transpose(1, 0, 2)
        return jnp.tensordot(Mz, t, axes=(1, 2)).transpose(1, 2, 0)

    def build(fb):
        def for_q(Mq):
            F = Mq.reshape((36, 3, pk, p))
            return jax.vmap(kron, (0, 0, 0, None))(F[:, 0], F[:, 1], F[:, 2], fb).reshape((3, 3, 4, pk, pk, pk))

        s = jnp.sum(jax.vmap(for_q)(M) * wq[:, None, None, None, None, None, None], axis=0)
        return s[:, :, 0] + s[:, :, 1], s[:, :, 2] + s[:, :, 3]

    return jax.jit(build)


def timed(fn, *args):
    t0 = time.perf_counter()
    jax.block_until_ready(fn(*args))
    return time.perf_counter() - t0


def compile_stats(fn, *args):
    t0 = time.perf_counter()
    exe = fn.lower(*args).compile()
    mem = exe.memory_analysis()
    return time.perf_counter() - t0, int(mem.temp_size_in_bytes) if mem is not None else -1


def species_row(n, Q, Ns, reps, sha):
    """Precomputation, compile and per-update wall-clock for all ordered pairs of Ns species."""
    masses = [4.0**s for s in range(Ns)]
    vth = [(2.0 / m) ** 0.5 for m in masses]
    sp = L.Species(m=1.0, vth=1.0)
    fs = [jnp.asarray(L.build_ic_fig1_1sp_twostream(n - 1, sp, 1.5).f)] * Ns
    t0 = time.perf_counter()
    tables = {
        (a, b): L.build_model_tables_np(nmax=n - 1, Q=Q, maxK=256, ma=masses[a], mb=masses[b], vtha=vth[a], vthb=vth[b], nu_ab=1.0)
        for a in range(Ns)
        for b in range(Ns)
    }
    row = dict(n=n, Q=Q, species=Ns, commit=sha, t_tables=time.perf_counter() - t0)
    updates = {}
    for name, builder in (("mpo", L.build_jax_functions), ("gauss", L.build_jax_functions_gauss)):
        t0 = time.perf_counter()
        rhs = {pair: builder(T) for pair, T in tables.items()}
        row[f"t_setup_{name}"] = time.perf_counter() - t0
        update = jax.jit(lambda fs, rhs=rhs: [sum(rhs[a, b](fs[a], fs[b]) for b in range(Ns)) for a in range(Ns)])
        row[f"compile_{name}"], row[f"temp_{name}"] = compile_stats(update, fs)
        jax.block_until_ready(update(fs))
        updates[name] = update
    times = {k: [] for k in updates}
    for _ in range(reps):  # A/B alternation
        for k, fn in updates.items():
            times[k].append(timed(fn, fs))
    row.update({f"t_update_{k}": min(v) for k, v in times.items()})
    row.update({f"tmed_update_{k}": sorted(v)[len(v) // 2] for k, v in times.items()})
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", default="7,9,11,13,15,17,19,21")
    ap.add_argument("--Q", default="24")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--out", default="")
    ap.add_argument("--species", default="", help="comma-separated species counts; selects the all-pairs table")
    args = ap.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    git = lambda *a: subprocess.check_output(["git", *a], cwd=here, text=True).strip()
    if git("status", "--porcelain", "--untracked-files=no"):
        raise SystemExit("tracked files modified: commit before benchmarking")
    rows, sha = [], git("rev-parse", "HEAD")
    print("commit", sha, "jax", jax.__version__, flush=True)
    if args.species:
        for Ns in map(int, args.species.split(",")):
            for n in map(int, args.n.split(",")):
                rows.append(species_row(n, int(args.Q.split(",")[0]), Ns, args.reps, sha))
                print(json.dumps(rows[-1]), flush=True)
                if args.out:
                    json.dump(rows, open(args.out, "w"), indent=1)
        return
    sp = L.Species(m=1.0, vth=1.0)
    for n in map(int, args.n.split(",")):
        for Q in map(int, args.Q.split(",")):
            t0 = time.perf_counter()
            T = L.build_model_tables_np(nmax=n - 1, Q=Q, maxK=256, ma=1.0, mb=1.0, vtha=1.0, vthb=1.0, nu_ab=1.0)
            t_tables = time.perf_counter() - t0
            f = jnp.asarray(L.build_ic_fig1_1sp_twostream(n - 1, sp, 1.5).f)
            mpo, mpo_S = L.build_jax_functions(T), baseline_build_S(T)
            gauss, (gauss_S, gauss_C) = L.build_jax_functions_gauss(T), L.build_jax_functions_gauss(T, split_stages=True)
            S = gauss_S(f)
            row = dict(n=n, Q=Q, commit=sha, t_tables=t_tables)
            row["compile_mpo"], row["temp_mpo"] = compile_stats(mpo, f, f)
            row["compile_gauss"], row["temp_gauss"] = compile_stats(gauss, f, f)
            times = {k: [] for k in ("mpo", "mpo_S", "gauss", "gauss_S", "gauss_C")}
            calls = dict(mpo=(mpo, f, f), mpo_S=(mpo_S, f), gauss=(gauss, f, f), gauss_S=(gauss_S, f), gauss_C=(gauss_C, f, S))
            for fn, *a in calls.values():
                jax.block_until_ready(fn(*a))
            for _ in range(args.reps):  # A/B alternation
                for k, (fn, *a) in calls.items():
                    times[k].append(timed(fn, *a))
            row.update({f"t_{k}": min(v) for k, v in times.items()})
            row.update({f"tmed_{k}": sorted(v)[len(v) // 2] for k, v in times.items()})
            rows.append(row)
            print(json.dumps(row), flush=True)
            if args.out:
                json.dump(rows, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
