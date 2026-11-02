"""Binary Monte-Carlo Coulomb collisions in JAX for the homogeneous one-species Landau equation
     df/dt = c d_i Int U_ij(v-v') [f' d_j f - f d'_j f'] dv',  U_ij = (u^2 delta_ij - u_i u_j)/u^3,  c = 1/2, Int f = n = 1,
in the units of the Hermite code (M(v) = exp(-v^2)/sqrt(pi), times in 1/nu^ab). Matching the weak forms gives, for each pair,
     <Delta v_i Delta v_j>/dt = 2c U_ij n   and   <Delta v_i>/dt = -4c u_i n/u^3,
that is <theta^2> = 16 c n dt/u^3, hence
     Takizuka & Abe (1977): tan(theta/2) ~ Normal(0, 4 c n dt/u^3)
     Nanbu (1997):          s = 8 c n dt/u^3, coth A - 1/A = exp(-s), cos(theta) = 1 + ln(U + (1-U) exp(-2A))/A.
Each particle meets one partner per step, and the relative velocity is rotated, so each pair conserves momentum and energy to round-off.

Usage:
    from particle_landau import setup_threads; setup_threads(1)      # before any other JAX work; thread count = host devices
    from particle_landau import PL
    p = PL(N=10**6, dt=1/16, scheme="ta", pairing="perm", threads=1); v = p.init(seed=0); v = p.advance(v, 16)
Pairing: "perm" is a full random permutation in each shard (the textbook method, sort based in JAX, N log N);
"shift" rotates the array and pairs the two halves at a random offset, which is O(N) and cache friendly but not a published scheme; use "perm" when in doubt.
With threads > 1 each thread collides within its own shard, then all shards exchange equal pieces, so one step mixes the population globally."""
import os


def setup_threads(threads):
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ["XLA_FLAGS"] = (f"--xla_force_host_platform_device_count={threads} --xla_cpu_multi_thread_eigen=false "
                               "intra_op_parallelism_threads=1")
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = "1"


import numpy as np


def _jax():
    import jax
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_default_prng_impl", "rbg")
    return jax


def nanbu_cos(s, U, jnp):
    """cos(theta) with <cos(theta)> = exp(-s): Nanbu (1997) inverse Langevin, two Newton steps from a rational first guess."""
    x = jnp.exp(-jnp.clip(s, 1e-4, 12.0))
    A = x*(3-x*x)/(1-x*x)
    for _ in range(2):
        L = jnp.where(A < 0.02, A/3-A**3/45, 1/jnp.tanh(A)-1/A)
        dL = jnp.where(A < 0.02, 1/3-A*A/15, 1/(A*A)-1/jnp.sinh(jnp.minimum(A, 300.))**2)
        A = A-(L-x)/dL
    c = 1+jnp.log(U+(1-U)*jnp.exp(-2*A))/A
    c = jnp.where(s < 1e-4, 1+s*jnp.log(U), c)
    c = jnp.where(s > 12.0, 2*U-1, c)
    return jnp.clip(c, -1.0, 1.0)


class PL:
    def __init__(s, N, dt, scheme="ta", pairing="perm", threads=1, c=0.5, n=1.0):
        jax = s.jax = _jax()
        import jax.numpy as jnp
        s.jnp = jnp
        s.T = threads
        s.m = (N//(2*threads*threads))*2*threads   # particles per shard: even and divisible by the number of shards
        s.N = s.m*threads
        s.dt, s.scheme, s.pairing, s.c, s.n = dt, scheme, pairing, c, n
        s._seg = {}

    def collide(s, a, b, key):
        jax, jnp = s.jax, s.jnp
        k1, k2 = jax.random.split(key)
        u = a-b
        u2 = jnp.sum(u*u, -1)
        U = jnp.sqrt(u2)
        Uinv3 = 1/jnp.maximum(U*u2, 1e-300)
        if s.scheme == "ta":
            d = jax.random.normal(k1, U.shape)*jnp.sqrt(4*s.c*s.n*s.dt*Uinv3)
            d2 = d*d
            den = 1/(1+d2)
            sn, omc = 2*d*den, 2*d2*den
        elif s.scheme == "nanbu":
            cs = nanbu_cos(8*s.c*s.n*s.dt*Uinv3, jax.random.uniform(k1, U.shape), jnp)
            sn, omc = jnp.sqrt(jnp.maximum(1-cs*cs, 0.)), 1-cs
        else:
            raise ValueError(s.scheme)
        ph = 2*jnp.pi*jax.random.uniform(k2, U.shape)
        cp, sp = jnp.cos(ph), jnp.sin(ph)
        ux, uy, uz = u[..., 0], u[..., 1], u[..., 2]
        Up = jnp.sqrt(ux*ux+uy*uy)
        ok = Up > 0
        ex = jnp.where(ok, ux/jnp.where(ok, Up, 1.), 1.)
        ey = jnp.where(ok, uy/jnp.where(ok, Up, 1.), 0.)
        du = jnp.stack([ex*uz*sn*cp-ey*U*sn*sp-ux*omc, ey*uz*sn*cp+ex*U*sn*sp-uy*omc, -Up*sn*cp-uz*omc], -1)   # Takizuka & Abe Eq. (4a-c) form
        return a+.5*du, b-.5*du

    def _step(s, v, key):
        jax, jnp = s.jax, s.jnp
        k1, k2, k3 = jax.random.split(key, 3)
        h = s.m//2
        if s.pairing == "perm":
            v = jax.random.permutation(k1, v, axis=0)
            a, b = s.collide(v[:h], v[h:], k2)
        else:
            r = jax.random.randint(k1, (2,), 0, s.m)
            v = jnp.roll(v, r[0], axis=0)
            a, b = s.collide(v[:h], jnp.roll(v[h:], r[1] % h, axis=0), k2)
        v = jnp.concatenate([a, b], 0)
        if s.T > 1:   # all-to-all: each shard gives one piece to every other shard
            v = jax.lax.all_to_all(v.reshape(s.T, s.m//s.T, 3), "i", 0, 0).reshape(s.m, 3)
        return v

    def _segment(s, nsteps):
        if nsteps in s._seg:
            return s._seg[nsteps]
        jax = s.jax

        def seg(v, key):
            def body(c, k):
                return s._step(c, k), None
            v, _ = jax.lax.scan(body, v, jax.random.split(key, nsteps))
            return v
        s._seg[nsteps] = jax.pmap(seg, axis_name="i") if s.T > 1 else jax.jit(seg)
        return s._seg[nsteps]

    def init(s, seed=0, u0=1.5, sampler=None):
        """two-stream by default; sampler(rng, n) -> (n,3) array otherwise. Plain Monte Carlo sampling, no quiet start."""
        jax, jnp = s.jax, s.jnp
        rng = np.random.default_rng(seed)
        s.key = jax.random.PRNGKey(seed+12345)
        v = (sampler or (lambda r, n: two_stream(r, n, u0)))(rng, s.N)
        if s.T > 1:
            return v.reshape(s.T, s.m, 3)   # pmap distributes the leading axis over the host devices
        return jnp.asarray(v)

    def advance(s, v, nsteps):
        s.key, k = s.jax.random.split(s.key)
        if s.T > 1:
            k = s.jax.random.split(k, s.T)
        return s._segment(nsteps)(v, k)

    def to_numpy(s, v):
        return np.asarray(v).reshape(s.N, 3)


def two_stream(rng, n, u0=1.5):
    v = rng.standard_normal((n, 3))/np.sqrt(2)
    v[:, 0] += u0*(2*rng.integers(0, 2, n)-1)
    return v


def moments(v):
    """v: (N,3) numpy. Exact second moments and the moment-based quantities shared with the grid solvers."""
    P = v.mean(0)
    v2 = v*v
    T = v2.mean(0)-P*P
    x4 = float((v2[:, 0]**2).mean())
    r2 = v2[:, 1]+v2[:, 2]
    m4 = float((v2[:, 1]**2-6*v2[:, 1]*v2[:, 2]+v2[:, 2]**2).mean())
    return dict(P=P.tolist(), Px=float(v[:, 0].sum()), Tx=float(T[0]), Ty=float(T[1]), Tz=float(T[2]), W=float(v2.sum(1).mean()),
                A=float((T[2]-T[0])/(T.sum()/3)), x4=x4, kx=x4/float(v2[:, 0].mean())**2, m4=m4/float((r2*r2).mean()))
