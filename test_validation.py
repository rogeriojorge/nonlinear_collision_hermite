"""Short checks of the validation scripts."""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "validation"))
import jax.numpy as jnp
import numpy as np

from fourier_landau import Solver
from hermite_laguerre import W, bi_maxwellian, cart_to_hl, direct, hl_to_cart, operator
from particle_landau import PL, moments


class ValidationTest(unittest.TestCase):
    def test_fourier_correction_conserves_and_maxwellian_is_fixed(self):
        s = Solver(24, 8.0, correct=True)
        f = jnp.asarray(s.two_stream())
        np.testing.assert_allclose(np.tensordot(np.asarray(s.A), np.asarray(s.C(f, f)), axes=3), 0.0, atol=1e-13)
        M = jnp.asarray(s.maxwellian(2.5))
        self.assertLess(float(jnp.abs(s.C(M, M)).max()), 1e-3 * float(jnp.abs(s.C(f, f)).max()))

    def test_map_is_isometric_and_agrees_with_direct_quadrature(self):
        for j in range(6):
            self.assertAlmostEqual(float(W(j) @ W(j)), 1.0, delta=1e-14)
        P, J, p = 4, 2, 5
        l = jnp.asarray(bi_maxwellian(P, J, 1.4, 0.7))
        np.testing.assert_allclose(cart_to_hl(hl_to_cart(l, p), P, J), l, atol=1e-14)
        F = hl_to_cart(l, p)
        ref = direct(l, l, NU=12, Nt=14, Na=16)
        np.testing.assert_allclose(cart_to_hl(operator(p, Q=16)(F, F), P, J), ref, atol=1e-11)

    def test_particle_pairs_conserve_and_relax(self):
        for scheme in ("ta", "nanbu"):
            for pairing in ("perm", "shift"):
                p = PL(N=20000, dt=1/16, scheme=scheme, pairing=pairing, threads=1)
                v0 = p.init(seed=1)
                m0, m1 = moments(p.to_numpy(v0)), moments(p.to_numpy(p.advance(v0, 122)))  # t = 7.625
                np.testing.assert_allclose(m1["P"], m0["P"], atol=1e-13)
                self.assertAlmostEqual(m1["W"], m0["W"], delta=1e-13 * m0["W"])
                self.assertAlmostEqual(m1["A"] / m0["A"], 0.3, delta=0.06)  # grid solvers reach 0.3 at t = 7.60


if __name__ == "__main__":
    unittest.main()
