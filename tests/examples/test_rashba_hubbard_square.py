import os
import sys
from collections import namedtuple
from jax import numpy as jnp

__here__ = os.path.abspath(__file__)
__module_path__ = os.path.dirname(os.path.dirname(os.path.dirname(__here__)))
sys.path.insert(0, __module_path__)

from admf import mf_optimize, expectation, get_fe, utils


dimensions = namedtuple("dimensions", ["nx", "ny"])
basis = namedtuple("basis", ["x", "y", "spin"])


def generate_basis(dimensions):
    for x in range(dimensions.nx):
        for y in range(dimensions.ny):
            for spin in [0, 1]:
                yield basis(x, y, spin)


d = dimensions(2, 3)
loc, rloc = utils.loc_index(generate_basis(d))
hsize = len(loc)
uloc, _ = utils.loc_index(generate_basis(d), lambda b: b.spin == 0)


def nn(b, nx, ny):
    yield basis(b.x, (b.y + 1) % ny, b.spin)  # down
    yield basis(b.x, (b.y - 1) % ny, b.spin)  # up
    yield basis((b.x - 1) % nx, b.y, b.spin)  # left
    yield basis((b.x + 1) % nx, b.y, b.spin)  # right


def rashba(b, nx, ny, lmbd=1):
    spin = 1 if b.spin == 0 else 0
    r = basis(b.x, (b.y + 1) % ny, spin)
    yield r, 1.0j * lmbd
    r = basis(b.x, (b.y - 1) % ny, spin)
    yield r, -1.0j * lmbd
    r = basis((b.x - 1) % nx, b.y, spin)
    yield r, 1.0 * lmbd * (-1) ** b.spin
    r = basis((b.x + 1) % nx, b.y, spin)
    yield r, -1.0 * lmbd * (-1) ** b.spin


K, RS = utils.generate_jnp_zeros(2, [hsize, hsize])
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
        nsite = utils.spin_flip(site)
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
    hm += var.mu * jnp.eye(hsize)
    return hm


def h(const, var):
    return const.t * K + const.lbd * RS - 0.5 * const.u * jnp.eye(hsize)


hint = utils.hubbard_int(loc)

t1, t2, t3 = utils.generate_jnp_random_normal(3, [int(len(loc) / 2)])

init_params = var(0.0, t1, t2, t3)

result = namedtuple("result", ["const", "var", "energy", "s"])

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
        r[site] = utils.measure_S(loc, site, const_params.beta, e, v)
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
        sx.append(np.real(v.x))
        sy.append(np.real(v.y))
        sz.append(np.real(v.z))
    plt.quiver(x, y, sx, sy, sz)
    """
