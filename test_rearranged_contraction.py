"""Checks for the optional full dense Landau contraction."""

import unittest

import numpy as np
from scipy.special import gammaln

from landau_hermite_jax_relative_entropy_1 import (
    build_jax_functions,
    build_model_tables_np,
    rearranged_pair_tables_np,
    rhs_ab_np,
    rhs_ab_rearranged_np,
)


class RearrangedContractionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.table = build_model_tables_np(
            nmax=3, Q=8, maxK=32, ma=1.0, mb=11.0,
            vtha=np.sqrt(2.0), vthb=np.sqrt(2.0 / 11.0), nu_ab=1.0,
        )
        rng = np.random.default_rng(216)
        cls.fa = rng.normal(size=(4, 4, 4))
        cls.fb = rng.normal(size=(4, 4, 4))

    def test_numpy_matches_canonical_and_cached_tables(self):
        t = self.table
        canonical = rhs_ab_np(self.fa, self.fb, t, use_tt=False, tt_tol=0.0, tt_rmax=0)
        rearranged = rhs_ab_np(self.fa, self.fb, t, use_tt=False, tt_tol=0.0,
                               tt_rmax=0, contraction="rearranged")
        cached = rhs_ab_rearranged_np(self.fa, self.fb, t, rearranged_pair_tables_np(t))
        np.testing.assert_allclose(rearranged, canonical, rtol=2e-10, atol=2e-10)
        np.testing.assert_array_equal(rearranged, cached)

    def test_jax_matches_numpy(self):
        import jax

        jax.config.update("jax_platform_name", "cpu")
        jax.config.update("jax_enable_x64", True)
        t = self.table
        result = np.asarray(build_jax_functions(t, contraction="rearranged")(self.fa, self.fb))
        expected = rhs_ab_np(self.fa, self.fb, t, use_tt=False, tt_tol=0.0,
                             tt_rmax=0, contraction="rearranged")
        np.testing.assert_allclose(result, expected, rtol=2e-10, atol=2e-10)
        result32 = np.asarray(build_jax_functions(t, contraction="rearranged")(
            self.fa.astype(np.float32), self.fb.astype(np.float32)))
        expected32 = rhs_ab_rearranged_np(self.fa.astype(np.float32), self.fb.astype(np.float32), t)
        np.testing.assert_allclose(result32, expected32, rtol=2e-10, atol=2e-10)

    def test_high_order_corner_against_polynomial_antiderivative(self):
        import jax

        jax.config.update("jax_enable_x64", True)
        sa, sb = np.sqrt(2.), np.sqrt(2./11.)
        table = build_model_tables_np(nmax=16, Q=24, maxK=80,
            ma=1., mb=11., vtha=sa, vthb=sb, nu_ab=50*(sb/sa)**3)
        direction = np.zeros((17, 17, 17))
        direction[-1, -1, -1] = 1.
        # Analytic AW coefficients of a Maxwellian with variance 1.2/11.
        k = np.arange(9)
        c = np.zeros(17)
        c[::2] = np.exp(.5*gammaln(2*k+1)-k*np.log(2)-gammaln(k+1)) * .2**k
        background = .2/sb**3 * np.einsum("i,j,k->ijk", c, c, c)
        # Independent 120-digit polynomial antiderivative of the P/M formulas
        # for the (16,16,16) output; it does not use either dense contraction.
        expected = -12.12547373505867054708649179599768
        result = np.asarray(build_jax_functions(table, contraction="rearranged")(
            direction, background))[-1, -1, -1]
        self.assertAlmostEqual(result, expected, delta=1e-6)

    def test_opt_in_and_invalid_modes(self):
        t = self.table
        original = rhs_ab_np(self.fa, self.fb, t, use_tt=False, tt_tol=0.0, tt_rmax=0)
        explicit = rhs_ab_np(self.fa, self.fb, t, use_tt=False, tt_tol=0.0,
                             tt_rmax=0, contraction="canonical")
        np.testing.assert_array_equal(original, explicit)
        with self.assertRaisesRegex(ValueError, "unknown contraction"):
            rhs_ab_np(self.fa, self.fb, t, use_tt=False, tt_tol=0.0,
                      tt_rmax=0, contraction="invalid")
        with self.assertRaisesRegex(ValueError, "use_tt=False"):
            rhs_ab_np(self.fa, self.fb, t, use_tt=True, tt_tol=0.0,
                      tt_rmax=0, contraction="rearranged")
        with self.assertRaisesRegex(ValueError, "unknown contraction"):
            build_jax_functions(t, contraction="invalid")


if __name__ == "__main__":
    unittest.main()
