"""Independent direct solve of the homogeneous Landau equation (Fourier-spectral, after Pareschi, Russo & Toscani 2000), two-stream test.

C(a, b) = c d_i[(U_ij * b) d_j a - a (d_j U_ij * b)], with U_ij truncated at |u| < R. Shares nothing with the Hermite code.
Options: the kernel cutoff R (default R = L; R = 0.906 L is also used), --correct for the least-squares conservation
correction of the collision output (Zhang & Gamba 2017, Eq. 36-37), --linearized for dh/dt = C(h, M) + C(M, h) about the true Maxwellian.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import jax
import jax.numpy as jnp
import numpy as np

from landau_hermite_angular_metrics import angular_symmetry_metrics_from_polar_plane as metric

jax.config.update("jax_enable_x64", True)


class Solver:
    def __init__(s, N, L, R=None, correct=False):
        s.N, s.L, s.dv = N, L, 2 * L / N
        s.v = -L + s.dv * np.arange(N)
        s.V = np.meshgrid(s.v, s.v, s.v, indexing="ij")
        R = L if R is None else R
        k1, kz = np.pi / L * np.fft.fftfreq(N, 1 / N), np.pi / L * np.fft.rfftfreq(N, 1 / N)
        k = np.meshgrid(k1, k1, kz, indexing="ij")
        K = np.sqrt(sum(a**2 for a in k))
        K[0, 0, 0] = 1.0
        x = K * R
        Upar, Uperp = 8 * np.pi / K**2 * (1 - np.sin(x) / x), 4 * np.pi / K**2 * (np.sin(x) / x - np.cos(x))
        Upar[0, 0, 0] = Uperp[0, 0, 0] = 4 * np.pi * R**2 / 3
        U = {(i, j): jnp.asarray(Uperp * (i == j) + (Upar - Uperp) * k[i] * k[j] / K**2) for i in range(3) for j in range(i, 3)}
        mask = np.ones_like(K)
        mask[N // 2, :, :] = mask[:, N // 2, :] = mask[:, :, -1] = 0
        s.mask, s.kj, s.Up, s.U = jnp.asarray(mask), [jnp.asarray(a) for a in k], jnp.asarray(Upar), U
        s.kmax = np.pi / s.dv
        s.w = s.dv**3
        phi = np.stack([np.ones((N, N, N)), *s.V, sum(a**2 for a in s.V)])
        s.A = jnp.asarray(phi * s.w)
        G = jnp.asarray(np.linalg.inv(np.tensordot(phi, phi, axes=([1, 2, 3], [1, 2, 3])) * s.w**2))
        project = lambda q: q - jnp.tensordot(G @ jnp.tensordot(s.A, q, axes=3), s.A, axes=1)
        s.project = project if correct else lambda q: q

    def ifft(s, A):
        return jnp.fft.irfftn(A, s=(s.N,) * 3)

    def C(s, a, b, c=0.5):
        A, B = jnp.fft.rfftn(a) * s.mask, jnp.fft.rfftn(b) * s.mask
        grad = [s.ifft(1j * kj * A) for kj in s.kj]
        D = {ij: s.ifft(u * B) for ij, u in s.U.items()}
        div = 0
        for i in range(3):
            flux = sum(D[min(i, j), max(i, j)] * grad[j] for j in range(3)) - s.ifft(1j * s.kj[i] * s.Up * B) * a
            div = div + 1j * s.kj[i] * jnp.fft.rfftn(flux)
        return s.project(c * s.ifft(div * s.mask))

    def stable_dt(s, f, cfl=1.2, c=0.5):
        F = jnp.fft.rfftn(jnp.asarray(f))
        return cfl / (c * 3 * max(float(jnp.abs(s.ifft(s.U[i, i] * F)).max()) for i in range(3)) * s.kmax**2)

    def two_stream(s, u0=1.5):
        M = lambda x: np.exp(-(x**2)) / np.pi**0.5
        return 0.5 * (M(s.V[0] - u0) + M(s.V[0] + u0)) * M(s.V[1]) * M(s.V[2])

    def maxwellian(s, tau):
        return np.exp(-sum(a**2 for a in s.V) / tau) / (np.pi * tau) ** 1.5

    def e_ang(s, f):
        """Angular defect of the v_x = 0 slice by exact trigonometric interpolation, with the repository's metric."""
        r, th = np.linspace(0, 4.5, 91), np.linspace(0, 2 * np.pi, 128, endpoint=False)
        k = np.pi / s.L * np.fft.fftfreq(s.N, 1 / s.N)
        k[s.N // 2] = 0
        F = np.fft.fft2(np.asarray(f)[s.N // 2]) / s.N**2
        F[s.N // 2, :] = F[:, s.N // 2] = 0
        Ey, Ez = np.exp(1j * np.outer(np.outer(r, np.cos(th)).ravel() + s.L, k)), np.exp(1j * np.outer(np.outer(r, np.sin(th)).ravel() + s.L, k))
        plane = np.einsum("ab,pa,pb->p", F, Ey, Ez, optimize=True).real.reshape(r.size, th.size)
        return metric(plane, r, th, rel_floor=0.05, m_list=(2, 4, 6, 8, 10, 12))["E_ang_field"]


def crossing(t, y, level):
    if not (np.asarray(y) <= level).any():
        return float("nan")
    i = int(np.argmax(np.asarray(y) <= level))
    return t[i - 1] + (t[i] - t[i - 1]) * (y[i - 1] - level) / (y[i - 1] - y[i])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--N", type=int, default=48)
    ap.add_argument("--L", type=float, default=8.0)
    ap.add_argument("--R", type=float, default=None)
    ap.add_argument("--correct", action="store_true")
    ap.add_argument("--linearized", action="store_true")
    ap.add_argument("--tmax", type=float, default=15.0)
    ap.add_argument("--save", type=float, default=0.15, help="sampling interval; times are linear interpolations between samples")
    a = ap.parse_args()
    s = Solver(a.N, a.L, a.R, a.correct)
    f0 = s.two_stream()
    r2 = sum(v**2 for v in s.V)
    tau = float(2 / 3 * (f0 * r2).sum() * s.w)  # 2.5 for u0 = 1.5
    M = s.maxwellian(tau)
    Mj = jnp.asarray(M)
    R = (lambda h: s.C(h, Mj) + s.C(Mj, h)) if a.linearized else (lambda f: s.C(f, f))
    y, base = (jnp.asarray(f0 - M), M) if a.linearized else (jnp.asarray(f0), 0.0)
    sub = int(np.ceil(a.save / min(s.stable_dt(f0), s.stable_dt(M)) * (2 if a.linearized else 1)))
    dt = a.save / sub

    def step(y, _):
        y1 = y + dt * R(y)
        y2 = 0.75 * y + 0.25 * (y1 + dt * R(y1))
        return y / 3 + 2 / 3 * (y2 + dt * R(y2)), None

    chunk = jax.jit(lambda y: jax.lax.scan(step, y, None, length=sub)[0])
    Rj = jax.jit(R)
    rows = []
    for i in range(int(round(a.tmax / a.save)) + 1):
        f = np.asarray(y) + base
        Tx, Tz = (f * s.V[0] ** 2).sum() * s.w, (f * s.V[2] ** 2).sum() * s.w
        W = (f * r2).sum() * s.w
        m = f > 1e-12 * f.max()
        lnf = np.log(f[m])
        # D is the Kullback-Leibler divergence from the Maxwellian with the same n, P, W; S the entropy.
        rows.append((i * a.save, (Tz - Tx) / (W / 3), (f[m] * (lnf - np.log(M[m]))).sum() * s.w, -(f[m] * lnf).sum() * s.w, -(np.asarray(Rj(y))[m] * lnf).sum() * s.w, W))
        if i < int(round(a.tmax / a.save)):
            y = chunk(y)
    t, A, D, S, dS, W = np.array(rows).T
    print(f"N={a.N} L={a.L} dt={dt:.5f} t(A/A0=0.3)={crossing(t, A / A[0], 0.3):.4f} t(D/D0=0.1)={crossing(t, D / D[0], 0.1):.4f}")
    print(f"t={t[-1]:g}: S={S[-1]:.5f} dS/dt={dS[-1]:.3e} E_ang={s.e_ang(f):.2e} max energy drift={np.abs(W / W[0] - 1).max():.1e}")


if __name__ == "__main__":
    main()
