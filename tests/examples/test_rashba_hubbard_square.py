import os
import sys
from collections import namedtuple
import numpy as np
import jax
from jax import numpy as jnp

__here__ = os.path.abspath(__file__)
__module_path__ = os.path.dirname(os.path.dirname(os.path.dirname(__here__)))
sys.path.insert(0, __module_path__)

from admf import mf_optimize, expectation, get_fe

dimensions = namedtuple("dimensions", ["nx", "ny"])
basis = namedtuple("basis", ["x", "y", "spin"])


def generate_basis(dimensions):
    for x in range(dimensions.nx):
        for y in range(dimensions.ny):
            for spin in [0, 1]:
                yield basis(x, y, spin)


loc = {}
rloc = {}
uloc = {}
d = dimensions(2, 3)
j = 0
for i, site in enumerate(generate_basis(d)):
    loc[site] = i
    rloc[i] = site
    if site.spin == 0:
        uloc[site] = j
        j += 1

hsize = len(loc)


def nn(b, nx, ny):
    yield basis(b.x, (b.y + 1) % ny, b.spin)  # down
    yield basis(b.x, (b.y - 1) % ny, b.spin)  # up
    yield basis((b.x - 1) % nx, b.y, b.spin)  # left
    yield basis((b.x + 1) % nx, b.y, b.spin)  # right


def spin_flip(b):
    return basis(b.x, b.y, 1 if b.spin == 0 else 0)


def rashba(b, nx, ny, lmbd=1):
    spin = 1 if b.spin == 0 else 0
    r = basis(b.x, (b.y + 1) % ny, spin)
    yield r, 1.0j * lmbd
    r = basis(b.x, (b.y - 1) % ny, spin)
    d = np.array([0, -1, 0])
    yield r, -1.0j * lmbd
    r = basis((b.x - 1) % nx, b.y, spin)
    yield r, 1.0 * lmbd * (-1) ** b.spin
    r = basis((b.x + 1) % nx, b.y, spin)
    yield r, -1.0 * lmbd * (-1) ** b.spin


K, RS = [jnp.zeros([hsize, hsize], dtype=jnp.complex64) for _ in range(2)]

for site in loc:
    for nsite in nn(site, d.nx, d.ny):
        K = K.at[loc[site], loc[nsite]].add(1.0)
    for nsite, rash in rashba(site, d.nx, d.ny):
        RS = RS.at[loc[site], loc[nsite]].add(rash)

const = namedtuple("const", ["t", "lbd", "u", "beta"])
var = namedtuple("var", ["mu", "xm", "ym", "zm"])


def hansatz(const, var):
    # kinetic
    hm = const.t * K + const.lbd * RS
    for site in loc:
        nsite = spin_flip(site)
        if site.spin == 0:  # up:
            hm = hm.at[loc[site], loc[site]].add(var.zm[uloc[site]])
            hm = hm.at[loc[site], loc[nsite]].add(
                var.xm[uloc[site]] - 1.0j * var.ym[uloc[site]]
            )

        else:
            hm = hm.at[loc[site], loc[site]].add(-var.zm[uloc[nsite]])
            hm = hm.at[loc[site], loc[nsite]].add(
                var.xm[uloc[nsite]] + 1.0j * var.ym[uloc[nsite]]
            )
    hm += (var.mu) * jnp.eye(hsize)
    return hm


def h(const, var):
    return const.t * K + const.lbd * RS - 0.5 * const.u * jnp.eye(hsize)


def hint(const, var, e, v):
    energy = 0
    for site in uloc:  # interaction part by wick expansion
        nsite = spin_flip(site)
        cross = expectation(loc[site], loc[nsite], const.beta, e, v)
        energy += const.u * (
            expectation(loc[site], loc[site], const.beta, e, v)
            * expectation(loc[nsite], loc[nsite], const.beta, e, v)
            - jnp.conj(cross) * cross
        )
    return energy


def generate_random_matrix(n, shape, seed=42):
    assert n >= 1
    key = jax.random.PRNGKey(seed)
    yield jax.random.normal(key, shape)
    for _ in range(n - 1):
        key, subkey = jax.random.split(key)
        yield jax.random.normal(key, shape)


t1, t2, t3 = generate_random_matrix(3, [int(len(loc) / 2)])

init_params = var(0.0, t1, t2, t3)

result = namedtuple("result", ["const", "var", "energy", "s"])
so = namedtuple("so", ["o", "x", "y", "z"])


def get_S(site, beta, e, v):
    usite = loc[basis(site.x, site.y, 0)]
    dsite = loc[basis(site.x, site.y, 1)]
    uu = expectation(usite, usite, beta, e, v)
    ud = expectation(usite, dsite, beta, e, v)
    du = expectation(dsite, usite, beta, e, v)
    dd = expectation(dsite, dsite, beta, e, v)
    return so(uu + dd, ud + du, 1j * (du - ud), uu - dd)


f, _ = get_fe(hansatz, h, hint)


def vf(const, var):
    print(f(const, var))


def test_square_rashba_hubbard():
    const_params = const(1.0, 1.0, 5.0, 5.0)
    num_iter = 50
    var_params = mf_optimize(
        hansatz,
        h,
        hint,
        const_params,
        init_params,
        num_iter,
        verbose_sep=20,
        verbose_func=vf,
        step_size=0.001,
    )

    e, v = jnp.linalg.eigh(hansatz(const_params, var_params))
    r = {}
    for site in uloc:
        r[site] = get_S(site, const_params.beta, e, v)
    assert f(const_params, var_params) < -21.2

    """
    # 空间磁矩分布可视化
    from matplotlib import pyplot as plt
    x = []
    y = []
    xy = []
    sx = []
    sy = []
    sxy= []
    sz = []
    
    for k, v in r.items():
        x.append(k.x)
        y.append(k.y)
        xy.append([k.x, k.y])
        sx.append(np.real(v.x))
        sy.append(np.real(v.y))
        sxy.append([v.x, v.y])
        sz.append(np.real(v.z))
    plt.quiver(x, y, sx, sy, sz)
    """
