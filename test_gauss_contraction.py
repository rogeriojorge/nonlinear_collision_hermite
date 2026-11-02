"""Gauss-Hermite collision contraction against the MPO path, the dense tensor, and the invariants."""
import unittest

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

from landau_hermite_jax_relative_entropy_1 import (
    Species,
    build_ic_fig1_1sp_twostream,
    build_jax_functions,
    build_jax_functions_gauss,
    build_model_tables_np,
    invariants_from_tensor,
    rhs_ab_np,
)


def tables(nmax, Q=24, ma=1.0, mb=1.0, vtha=1.0, vthb=1.0, nu=1.0):
    return build_model_tables_np(nmax=nmax, Q=Q, maxK=256, ma=ma, mb=mb, vtha=vtha, vthb=vthb, nu_ab=nu)


class GaussContractionTest(unittest.TestCase):
    def test_matches_mpo_and_numpy_on_random_tensors(self):
        rng = np.random.default_rng(216)
        sa, sb = np.sqrt(2.0), np.sqrt(2.0 / 11.0)
        for nmax in (3, 6):
            fa, fb = rng.normal(size=(2, nmax + 1, nmax + 1, nmax + 1))
            for T in (tables(nmax), tables(nmax, ma=1.0, mb=11.0, vtha=sa, vthb=sb)):
                ref = np.asarray(build_jax_functions(T)(fa, fb))
                # Random high-order tensors expose the table round-off of the MPO path (about 1e-10 at nmax = 6).
                atol = (1e-12 if nmax == 3 else 1e-9) * np.abs(ref).max()
                np.testing.assert_allclose(np.asarray(build_jax_functions_gauss(T)(fa, fb)), ref, rtol=0, atol=atol)
                np.testing.assert_allclose(rhs_ab_np(fa, fb, T, use_tt=False, tt_tol=0.0, tt_rmax=0), ref, rtol=0, atol=atol)

    def test_matches_dense_tensor(self):
        T, p = tables(2), 3
        rhs = build_jax_functions_gauss(T)
        basis = np.eye(p**3).reshape(-1, p, p, p)
        C = np.array([[np.asarray(rhs(a, b)).ravel() for b in basis] for a in basis])
        rng = np.random.default_rng(1)
        fa, fb = rng.normal(size=(2, p, p, p))
        dense = np.einsum("abk,a,b->k", C, fa.ravel(), fb.ravel()).reshape(p, p, p)
        ref = rhs_ab_np(fa, fb, T, use_tt=False, tt_tol=0.0, tt_rmax=0)
        np.testing.assert_allclose(dense, ref, rtol=0, atol=1e-12 * np.abs(ref).max())

    def test_invariants_at_roundoff(self):
        sp = Species(m=1.0, vth=1.0)
        f = build_ic_fig1_1sp_twostream(12, sp, 1.5).f
        self_rate = np.asarray(build_jax_functions_gauss(tables(12))(f, f))
        np.testing.assert_allclose(invariants_from_tensor(self_rate, sp), 0.0, atol=1e-12)
        self.assertGreater(np.linalg.norm(self_rate), 0.1)

        sa, sb = Species(m=1.0, vth=np.sqrt(2.0)), Species(m=11.0, vth=np.sqrt(2.0 / 11.0))
        common = dict(ma=1.0, mb=11.0, vtha=sa.vth, vthb=sb.vth, nu=1.0)
        back = dict(ma=11.0, mb=1.0, vtha=sb.vth, vthb=sa.vth, nu=11.0)  # equal physical pair coefficient
        rng = np.random.default_rng(3)
        fa, fb = rng.normal(size=(2, 5, 5, 5))
        ia = invariants_from_tensor(np.asarray(build_jax_functions_gauss(tables(4, **common))(fa, fb)), sa)
        ib = invariants_from_tensor(np.asarray(build_jax_functions_gauss(tables(4, **back))(fb, fa)), sb)
        np.testing.assert_allclose([ia[0], ib[0]], 0.0, atol=1e-12)
        np.testing.assert_allclose(ia[1:] + ib[1:], 0.0, atol=1e-11)


if __name__ == "__main__":
    unittest.main()
