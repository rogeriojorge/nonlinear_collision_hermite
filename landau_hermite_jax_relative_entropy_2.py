#!/usr/bin/env python3
"""
Companion script to landau_hermite_jax_relative_entropy.py

Creates a publication-ready 3D velocity-space visualization of the evolving
1-species distribution using nested isosurfaces at fixed f-levels. The default
setup mirrors the Fig1 twostream case and renders three time snapshots.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np

# Matplotlib cache/config to a local writable dir.
os.environ.setdefault("MPLBACKEND", "Agg")
HERE = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(HERE / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(HERE / ".cache"))

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from landau_hermite_jax_relative_entropy import (
    Species,
    _maybe_import_jax,
    build_ic_fig1_1sp_twostream,
    build_integrators_jax,
    build_jax_functions,
    build_maxwellian_tensor_from_invariants,
    build_model_tables_np,
    integrate_1sp_numpy,
    invariants_from_tensor,
    make_linearized_rhs_1sp_jax,
    make_linearized_rhs_1sp_numpy,
    prepare_entropy_grid,
    rhs_ab_np,
)

try:
    from skimage.measure import marching_cubes

    _HAVE_SKIMAGE = True
except Exception:
    _HAVE_SKIMAGE = False


# ----------------------------
# Utilities
# ----------------------------


def _parse_float_list(s: str) -> List[float]:
    return [float(x) for x in s.split(",") if x.strip() != ""]


def _nearest_indices(tgrid: np.ndarray, targets: Iterable[float]) -> List[int]:
    idxs = []
    for t in targets:
        idxs.append(int(np.argmin(np.abs(tgrid - float(t)))))
    return idxs


def _reconstruct_fxyz(f: np.ndarray, psi: np.ndarray) -> np.ndarray:
    return np.einsum("nmp,nx,my,pz->xyz", f, psi, psi, psi, optimize=True)


def _add_isosurface_skimage(ax, fxyz: np.ndarray, x0: float, dx: float, level: float, color: str, alpha: float) -> None:
    verts, faces, _normals, _values = marching_cubes(fxyz, level=level, spacing=(dx, dx, dx))
    verts = verts + np.array([x0, x0, x0])
    mesh = Poly3DCollection(verts[faces], alpha=alpha, facecolor=color, edgecolor="none")
    mesh.set_facecolor(color)
    ax.add_collection3d(mesh)


def _add_isosurface_scatter(ax, fxyz: np.ndarray, x: np.ndarray, level: float, color: str, alpha: float, tol: float = 0.06, max_points: int = 25000) -> None:
    # Fallback: plot points near the isosurface.
    if level <= 0:
        return
    rel = np.abs(fxyz - level) / max(level, 1e-300)
    mask = rel < tol
    if not np.any(mask):
        return
    coords = np.column_stack(np.where(mask))
    if coords.shape[0] > max_points:
        rng = np.random.default_rng(0)
        keep = rng.choice(coords.shape[0], size=max_points, replace=False)
        coords = coords[keep]
    xs = x[coords[:, 0]]
    ys = x[coords[:, 1]]
    zs = x[coords[:, 2]]
    ax.scatter(xs, ys, zs, s=2.0, c=color, alpha=alpha, depthshade=False)


def _style_axes(ax, xlim: float, label: str) -> None:
    ax.set_xlim(-xlim, xlim)
    ax.set_ylim(-xlim, xlim)
    ax.set_zlim(-xlim, xlim)
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlabel(r"$v_x / v_{th}$", labelpad=6)
    ax.set_ylabel(r"$v_y / v_{th}$", labelpad=6)
    ax.set_zlabel(r"$v_z / v_{th}$", labelpad=4)
    ax.set_title(label, pad=2)
    ax.grid(False)
    # Clean, publication-friendly 3D panes.
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
        axis.pane.set_edgecolor((0.85, 0.85, 0.85, 0.8))
    ax.tick_params(axis="both", which="major", pad=2, labelsize=9)
    # Keep z-label aligned inward for clean layout.
    try:
        ax.zaxis.set_rotate_label(False)
        ax.zaxis.label.set_rotation(90)
        ax.zaxis.set_label_coords(-0.08, 0.5)
    except Exception:
        pass


# ----------------------------
# Main
# ----------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description="3D velocity-space isosurface panel for Fig1 twostream evolution")
    ap.add_argument("--backend", choices=["jax", "numpy"], default="jax")
    ap.add_argument("--nmax", type=int, default=6)
    ap.add_argument("--Q", type=int, default=12)
    ap.add_argument("--maxK", type=int, default=256)
    ap.add_argument("--dt", type=float, default=0.15)
    ap.add_argument("--tmax", type=float, default=15.0)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--u", type=float, default=1.5)
    ap.add_argument("--grid_xlim", type=float, default=3.0)
    ap.add_argument("--grid_nx", type=int, default=55)
    ap.add_argument("--levels", type=str, default="0.6,0.3,0.1", help="Fractions of f_max(t=0) for isosurfaces")
    ap.add_argument("--snapshots", type=str, default="0.0,0.5,1.0", help="Fractions of tmax (or absolute times if >1)")
    ap.add_argument("--outprefix", type=str, default="Fig1_panel_3D")
    ap.add_argument("--dpi", type=int, default=220)
    ap.add_argument("--view", type=str, default="25,35", help="Camera elev,azim in degrees")
    ap.add_argument("--linearized", choices=["on", "off"], default="on")
    args = ap.parse_args()

    # Styling for publication quality.
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "mathtext.fontset": "dejavuserif",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "figure.dpi": float(args.dpi),
        }
    )

    backend = str(args.backend)
    nmax = int(args.nmax)
    p = nmax + 1

    if args.steps is None:
        steps = int(round(float(args.tmax) / float(args.dt)))
    else:
        steps = int(args.steps)
    tmax = steps * float(args.dt)
    tgrid = np.linspace(0.0, tmax, steps + 1)

    # Species and model tables.
    sp1 = Species(m=1.0, vth=1.0)
    T11 = build_model_tables_np(nmax=nmax, Q=int(args.Q), maxK=int(args.maxK), ma=sp1.m, mb=sp1.m, vtha=sp1.vth, vthb=sp1.vth, nu_ab=1.0)

    # Initial condition (twostream).
    ic = build_ic_fig1_1sp_twostream(nmax=nmax, sp=sp1, u=float(args.u), enforce_nonneg=False)
    f0 = ic.f

    # Integrate in time (nonlinear).
    if backend == "jax":
        jax, jnp = _maybe_import_jax("jax")
        assert jax is not None and jnp is not None
        rhs11 = build_jax_functions(T11)
        integrate_1sp_j, _ = build_integrators_jax(lambda f: rhs11(f, f), lambda a, b: (a, b), "ssprk3")
        integrate_1sp_j = jax.jit(integrate_1sp_j, static_argnums=(2,))
        f_hist = integrate_1sp_j(jnp.asarray(f0), float(args.dt), steps)
        jax.block_until_ready(f_hist)
        f_hist = np.array(f_hist)
    else:
        def rhs1_np(f):
            return rhs_ab_np(f, f, T11, use_tt=False, tt_tol=0.0, tt_rmax=1)
        f_hist = integrate_1sp_numpy(rhs1_np, f0, float(args.dt), steps, "ssprk3")

    # Optional linearized evolution about the equilibrium Maxwellian with the same invariants.
    f_hist_lin = None
    if str(args.linearized).lower() == "on":
        inv0 = invariants_from_tensor(f0, sp1)
        fM = build_maxwellian_tensor_from_invariants(nmax=nmax, sp=sp1, inv=inv0, xlim=10.0, nx=2001)
        if backend == "jax":
            jax, jnp = _maybe_import_jax("jax")
            assert jax is not None and jnp is not None
            L_apply = make_linearized_rhs_1sp_jax(rhs11, jnp.asarray(fM))
            integrate_1sp_lin, _ = build_integrators_jax(lambda f: L_apply(f), lambda a, b: (a, b), "ssprk3")
            integrate_1sp_lin = jax.jit(integrate_1sp_lin, static_argnums=(2,))
            df0 = jnp.asarray(f0 - fM)
            df_hist = integrate_1sp_lin(df0, float(args.dt), steps)
            jax.block_until_ready(df_hist)
            f_hist_lin = np.array(df_hist) + fM[None, ...]
        else:
            L_apply_np = make_linearized_rhs_1sp_numpy(T11, fM, use_tt=False, tt_tol=0.0, tt_rmax=1)
            df_hist = integrate_1sp_numpy(L_apply_np, f0 - fM, float(args.dt), steps, "ssprk3")
            f_hist_lin = df_hist + fM[None, ...]

    # Entropy grid reused to reconstruct f(x,y,z).
    grid = prepare_entropy_grid(nmax=nmax, xlim=float(args.grid_xlim), nx=int(args.grid_nx))
    x = grid.x
    dx = grid.dx

    # Snapshot times.
    snap_raw = _parse_float_list(str(args.snapshots))
    if all(t <= 1.0 for t in snap_raw):
        snap_times = [float(t) * tmax for t in snap_raw]
    else:
        snap_times = [float(t) for t in snap_raw]
    idxs = _nearest_indices(tgrid, snap_times)

    # Use levels as fractions of f_max(t) at each snapshot (robust to decay).
    level_fracs = _parse_float_list(str(args.levels))

    # Colors and alphas (outer surfaces lighter).
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    alphas = [0.6, 0.35, 0.15, 0.05]

    ncols = len(idxs)
    rows = 2 if (f_hist_lin is not None) else 1
    fig = plt.figure(figsize=(3.5 * ncols, 3.6 * rows))

    view = _parse_float_list(str(args.view))
    if len(view) != 2:
        view = [25.0, 35.0]
    elev, azim = float(view[0]), float(view[1])
    histories: List[Tuple[str, np.ndarray]] = [("Nonlinear", f_hist)]
    if f_hist_lin is not None:
        histories.append(("Linearized", f_hist_lin))

    patches = [Patch(facecolor=colors[k % len(colors)], edgecolor="none", alpha=alphas[k % len(alphas)], label=fr"$f/f_{{\max}}(t)={level_fracs[k]:.2f}$") for k in range(len(level_fracs))]

    top = 0.90
    bottom = 0.07
    hspace = 0.22
    row_centers = [top - (i + 0.5) * (top - bottom) / rows for i in range(rows)]

    for row, (row_label, hist) in enumerate(histories):
        for col, idx in enumerate(idxs):
            ax = fig.add_subplot(rows, ncols, row * ncols + col + 1, projection="3d")
            fxyz = _reconstruct_fxyz(hist[idx], grid.psi)
            fmax = float(np.max(fxyz))
            if (not np.isfinite(fmax)) or fmax <= 0.0:
                continue
            levels = [fmax * frac for frac in level_fracs]
            for k, level in enumerate(levels):
                color = colors[k % len(colors)]
                alpha = alphas[k % len(alphas)]
                if (level <= np.min(fxyz)) or (level >= np.max(fxyz)) or (not np.isfinite(level)):
                    continue
                if _HAVE_SKIMAGE:
                    _add_isosurface_skimage(ax, fxyz, x0=float(x[0]), dx=dx, level=level, color=color, alpha=alpha)
                else:
                    _add_isosurface_scatter(ax, fxyz, x=x, level=level, color=color, alpha=alpha)

            ax.view_init(elev=elev, azim=azim)
            _style_axes(ax, xlim=float(args.grid_xlim), label=fr"$t = {tgrid[idx]:.2f}$")

        # Row label on the left margin.
        fig.text(0.008, row_centers[row], row_label, rotation=90, va="center", ha="center", fontsize=10)

    fig.subplots_adjust(left=0.035, right=0.995, top=top, bottom=bottom, wspace=0.08, hspace=hspace)
    fig.legend(handles=patches, loc="upper center", bbox_to_anchor=(0.5, 0.965), ncol=len(level_fracs), frameon=False)

    outprefix = str(args.outprefix)
    fig.savefig(f"{outprefix}.png", dpi=int(args.dpi), bbox_inches="tight")
    fig.savefig(f"{outprefix}.pdf", bbox_inches="tight")
    print(f"[ok] wrote: {outprefix}.png and {outprefix}.pdf")

    if not _HAVE_SKIMAGE:
        print("[note] skimage not available; used scatter fallback for isosurfaces.")


if __name__ == "__main__":
    main()
