import os
import sys
from collections import namedtuple
import numpy as np
from jax import numpy as jnp

__here__ = os.path.abspath(__file__)
__module_path__ = os.path.dirname(os.path.dirname(os.path.dirname(__here__)))
sys.path.insert(0, __module_path__)

from admf import mf_optimize, expectation, get_fe, utils


basis = namedtuple("basis", ["x", "y", "sub", "spin"])


def generate_lattice(nx, ny):
    i = 0
    j = 0
    while i < nx and j < ny:
        yield basis(i, j, 0, 0)
        yield basis(i, j, 0, 1)
        yield basis(i, j, 1, 0)
        yield basis(i, j, 1, 1)
        i += 1
        if i >= nx:
            i = 0
            j += 1


def nn(t, nx, ny):
    if t.sub == 0:  # A sublattice
        yield basis((t.x - 1) % nx, (t.y - 1) % ny, 1, t.spin)
        yield basis(t.x, (t.y - 1) % ny, 1, t.spin)
        yield basis(t.x, t.y, 1, t.spin)
    elif t.sub == 1:  # B sublattice
        yield basis(t.x, (t.y + 1) % ny, 0, t.spin)
        yield basis((t.x + 1) % nx, (t.y + 1) % ny, 0, t.spin)
        yield basis(t.x, t.y, 0, t.spin)


def real_position(t):
    return (
        np.sqrt(3) * t.x - np.sqrt(3) / 2.0 * t.y,
        -3 / 2 * t.y if t.sub == 0 else -3 / 2 * t.y - 1,
    )


def rashba(t, nx, ny, lmbd=1):
    if t.spin == 0:
        sigma = np.array([1, -1j, 0])
    else:
        sigma = np.array([1, 1j, 0])
    for site in nn(t, nx, ny):
        dsite = utils.spin_flip(site)
        dx, dy = real_position(dsite)
        ux, uy = real_position(dsite)
        d = np.array([dx - ux, dy - uy, 0.0])
        yield dsite, 1j * lmbd * np.cross(sigma, d)[2]


nx = 3
ny = 3
loc, _ = utils.loc_index(generate_lattice(nx, ny))
uloc, _ = utils.loc_index(generate_lattice(nx, ny), lambda t: t.spin == 0)
hsize = len(loc)
K, RS = utils.generate_np_zeros(2, [hsize, hsize])
for site in loc:
    for hopsite in nn(site, nx, ny):
        K[loc[site], loc[hopsite]] = 1
    for rashbasite, lam in rashba(site, nx, ny):
        RS[loc[site], loc[rashbasite]] = lam


def hansatz(const, var):
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


const = namedtuple("const", ["t", "lbd", "u", "beta"])
var = namedtuple("var", ["mu", "xm", "ym", "zm"])
t1, t2, t3 = utils.generate_jnp_random_normal(3, [int(hsize / 2)])

init_params = var(0.0, t1, t2, t3)
const_params = const(1.0, 1.0, 6.0, 5.0)


def test_honeycomb_rashba_hubbard():
    var_params = mf_optimize(hansatz, h, hint, const_params, init_params, 200, 50)
    f, _ = get_fe(hansatz, h, hint)
    assert f(const_params, var_params) < -58.0
