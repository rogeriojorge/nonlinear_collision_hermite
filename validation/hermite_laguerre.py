"""Hermite-Laguerre check at zero perpendicular wavenumber.

With mu = y^2 + z^2, L_j(mu) = sum_a W[j, a] Psi_2a(y) Psi_2(j-a)(z), W[j, a] = (-1)^j C(j, a) sqrt((2a)! (2j-2a)!) / (2^j j!), an isometry.
(1) The tensor operator through W against a direct 5-D Gauss quadrature of the weak form in the Hermite-Laguerre representation, which
uses no ladder product, bracket, Boys function, sum of exponentials or Cartesian coefficient.
(2) Evolution of the (P+1)(J+1) axisymmetric unknowns against the masked Cartesian run.
"""
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import roots_laguerre

import landau_hermite_jax_relative_entropy_1 as H

jax.config.update("jax_enable_x64", True)


def W(j):
    f = math.factorial
    return np.array([(-1) ** j * math.comb(j, a) * math.sqrt(f(2 * a) * f(2 * j - 2 * a)) / (2**j * f(j)) for a in range(j + 1)])


def cart_to_hl(F, P, J):
    return jnp.stack([jnp.einsum("Pa,a->P", F[: P + 1, 2 * jnp.arange(j + 1), 2 * (j - jnp.arange(j + 1))], W(j)) for j in range(J + 1)], 1)


def hl_to_cart(l, p):
    F = jnp.zeros((p, p, p))
    for j in range(l.shape[1]):
        a = jnp.arange(j + 1)
        F = F.at[: l.shape[0], 2 * a, 2 * (j - a)].add(jnp.outer(l[:, j], W(j)))
    return F


def hermite(P, x):
    out = [jnp.ones_like(x), jnp.sqrt(2.0) * x]
    for n in range(1, P):
        out.append(jnp.sqrt(2.0 / (n + 1)) * x * out[n] - jnp.sqrt(n / (n + 1.0)) * out[n - 1])
    H_ = jnp.stack(out[: P + 1])
    r = jnp.sqrt(2.0 * jnp.arange(1, P + 1)).reshape((-1,) + (1,) * x.ndim)
    return H_, jnp.concatenate([jnp.zeros_like(x)[None], r * H_[:-1]])


def laguerre(J, mu):
    out = [jnp.ones_like(mu), 1 - mu]
    for j in range(1, J):
        out.append(((2 * j + 1 - mu) * out[j] - j * out[j - 1]) / (j + 1))
    L = jnp.stack(out[: J + 1])
    return L, jnp.concatenate([jnp.zeros_like(mu)[None], -jnp.cumsum(L[:-1], axis=0)])


def field(l, v, P, J):
    """g and d_j g - 2 v_j g, for f = g exp(-v^2) / pi^(3/2)."""
    x, y, z = v[..., 0], v[..., 1], v[..., 2]
    Hx, dH = hermite(P, x)
    L, dL = laguerre(J, y * y + z * z)
    g = jnp.einsum("Pj,P...,j...->...", l, Hx, L)
    gx, gm = jnp.einsum("Pj,P...,j...->...", l, dH, L), jnp.einsum("Pj,P...,j...->...", l, Hx, dL)
    return g, jnp.stack([gx - 2 * x * g, 2 * y * (gm - g), 2 * z * (gm - g)], -1)


def direct(la, lb, va=1.0, vb=1.0, ratio=1.0, NU=24, Nt=30, Na=40):
    """Weak form: C_{P,j} = -(1/2) int dv dv' d_i[Psi_P L_j](v) U_ij (f_b' d_j f_a - ratio f_a d'_j f_b'), nodes in the (u, U) frame."""
    P, J = la.shape[0] - 1, la.shape[1] - 1
    c, s = va / math.hypot(va, vb), vb / math.hypot(va, vb)
    x, w = np.polynomial.hermite.hermgauss(NU)
    U = jnp.asarray(np.stack(np.meshgrid(x, x, x, indexing="ij"), -1).reshape(-1, 3))
    wU = jnp.asarray(np.einsum("i,j,k->ijk", w, w, w).ravel())
    t, wt = roots_laguerre(Nt)
    ca, wa = np.polynomial.legendre.leggauss(Na)

    @jax.jit
    def one(n, r):
        u = r * n
        v, vp = s * U + c * u, c * U - s * u
        ga, Da = field(la, v, P, J)
        gb, Db = field(lb, vp, P, J)
        B = gb[..., None] * Da - ratio * ga[..., None] * Db
        F = B - (B @ n)[..., None] * n
        y, z = v[..., 1], v[..., 2]
        Hx, dH = hermite(P, v[..., 0])
        L, dL = laguerre(J, y * y + z * z)
        return jnp.einsum("Pq,jq,q->Pj", dH, L, wU * F[..., 0]) + jnp.einsum("Pq,jq,q->Pj", Hx, dL, wU * 2 * (y * F[..., 1] + z * F[..., 2]))

    tot = 0.0
    for ti, wi in zip(t, wt):
        for ai, wj in zip(ca, wa):
            tot = tot + wi * wj * np.pi * np.asarray(one(jnp.array([ai, math.sqrt(1 - ai * ai), 0.0]), math.sqrt(ti)))
    return -0.5 * c / np.pi**3 * tot


def bi_maxwellian(P, J, tpar, tperp):
    l = np.zeros((P + 1, J + 1))
    for k in range(P // 2 + 1):
        l[2 * k] = math.sqrt(math.factorial(2 * k)) * (tpar - 1) ** k / (2**k * math.factorial(k))
    return l * (1 - tperp) ** np.arange(J + 1)[None, :]


def two_stream(P, J, u, tperp=1.0):
    c = [(math.sqrt(2) * u) ** n / math.sqrt(math.factorial(n)) if n % 2 == 0 else 0.0 for n in range(P + 1)]
    return np.outer(c, (1 - tperp) ** np.arange(J + 1))


def operator(p, Q=48, ma=1.0, mb=1.0, va=1.0, vb=1.0):
    return H.build_jax_functions_gauss(H.build_model_tables_np(nmax=p - 1, Q=Q, maxK=256, ma=ma, mb=mb, vtha=va, vthb=vb, nu_ab=1.0))


def main():
    P, J, p = 10, 5, 11
    cases = [
        ("bi-Maxwellian 1.4, 0.7", bi_maxwellian(P, J, 1.4, 0.7), bi_maxwellian(P, J, 1.4, 0.7), {}),
        ("two-stream 1.5", two_stream(P, J, 1.5), two_stream(P, J, 1.5), {}),
        ("two-stream 1.5, Tperp 0.7", two_stream(P, J, 1.5, 0.7), two_stream(P, J, 1.5, 0.7), {}),
        ("cross, vb/va 0.7, ma/mb 0.5", bi_maxwellian(P, J, 1.3, 0.8), two_stream(P, J, 1.0), dict(ma=1.0, mb=2.0, va=1.0, vb=0.7)),
    ]
    for name, la, lb, kw in cases:
        la, lb = jnp.asarray(la), jnp.asarray(lb)
        tensor = cart_to_hl(operator(p, **kw)(hl_to_cart(la, p), hl_to_cart(lb, p)), P, J)
        va, vb = kw.get("va", 1.0), kw.get("vb", 1.0)
        ref = direct(la, lb, va, vb, kw.get("ma", 1.0) / kw.get("mb", 1.0) * va / vb)
        print(f"{name}: tensor through W vs direct, relative difference {np.linalg.norm(tensor - ref) / np.linalg.norm(ref):.1e}")

    p, P, J, dt, steps = 13, 12, 6, 0.15, 67
    rhs = operator(p, Q=24)
    i, j, k = np.indices((p, p, p))
    mask = jnp.asarray((j + k <= P) * 1.0)
    f0 = jnp.asarray(H.build_ic_fig1_1sp_twostream(P, H.Species(1.0, 1.0), 1.5).f)

    def ssp(R, y):
        y1 = y + dt * R(y)
        y2 = 0.75 * y + 0.25 * (y1 + dt * R(y1))
        return y / 3 + 2 / 3 * (y2 + dt * R(y2))

    def R_hl(l):
        F = hl_to_cart(l, p)
        return cart_to_hl(rhs(F, F), P, J)

    step_hl, step_c = jax.jit(lambda l: ssp(R_hl, l)), jax.jit(lambda F: ssp(lambda g: mask * rhs(g, g), F))
    l, F = cart_to_hl(f0, P, J), f0
    for _ in range(steps):
        l, F = step_hl(l), step_c(F)
    ref = cart_to_hl(F, P, J)
    print(f"{(P + 1) * (J + 1)} unknowns, t={steps * dt:.2f}: vs masked Cartesian run, relative difference {np.linalg.norm(ref - l) / np.linalg.norm(ref):.1e}")


if __name__ == "__main__":
    main()
