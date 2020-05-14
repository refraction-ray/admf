import os
import sys
from collections import namedtuple
import numpy as np
from jax import numpy as jnp

__here__ = os.path.abspath(__file__)
__module_path__ = os.path.dirname(os.path.dirname(os.path.dirname(__here__)))
sys.path.insert(0, __module_path__)

from admf import mf_optimize, expectation, get_fe, utils


def generate_lattice(nx, ny):
    i = 0
    j = 0
    while i < nx and j < ny:
        yield (i, j, 0, 0)
        yield (i, j, 0, 1)
        yield (i, j, 1, 0)
        yield (i, j, 1, 1)  # x, y, sublattice, spin
        i += 1
        if i >= nx:
            i = 0
            j += 1


def nn(t, nx, ny):
    if t[2] == 0:  # A sublattice
        yield ((t[0] - 1) % nx, (t[1] - 1) % ny, 1, t[3])
        yield (t[0], (t[1] - 1) % ny, 1, t[3])
        yield (t[0], t[1], 1, t[3])
    elif t[2] == 1:  # B sublattice
        yield (t[0], (t[1] + 1) % ny, 0, t[3])
        yield ((t[0] + 1) % nx, (t[1] + 1) % ny, 0, t[3])
        yield (t[0], t[1], 0, t[3])


def rashba(t, nx, ny, lmbd=1):
    spin = 1 if t[3] == 0 else 0
    if t[3] == 0:
        sigma = np.array([1, -1j, 0])
    else:
        sigma = np.array([1, 1j, 0])
    if t[2] == 0:  # A sublattice
        r = ((t[0] - 1) % nx, (t[1] - 1) % ny, 1, spin)
        d = np.array([-np.sqrt(3) / 2, -1 / 2, 0])
        yield r, 1j * lmbd * np.cross(sigma, d)[2]
        r = (t[0], (t[1] - 1) % ny, 1, spin)
        d = np.array([np.sqrt(3) / 2, -1 / 2, 0])
        yield r, 1j * lmbd * np.cross(sigma, d)[2]
        r = (t[0], t[1], 1, spin)
        d = np.array([0, 1, 0])
        yield r, 1j * lmbd * np.cross(sigma, d)[2]
    elif t[2] == 1:  # B sublattice
        r = (t[0], (t[1] + 1) % ny, 0, spin)
        d = np.array([-np.sqrt(3) / 2, 1 / 2, 0])
        yield r, 1j * lmbd * np.cross(sigma, d)[2]
        r = ((t[0] + 1) % nx, (t[1] + 1) % ny, 0, spin)
        d = np.array([np.sqrt(3) / 2, 1 / 2, 0])
        yield r, 1j * lmbd * np.cross(sigma, d)[2]
        r = (t[0], t[1], 0, spin)
        d = np.array([0, -1, 0])
        yield r, 1j * lmbd * np.cross(sigma, d)[2]


def flip_spin(t):
    nt = list(t)
    nt[3] = 1 if t[3] == 0 else 0
    nt = tuple(nt)
    return nt


nx = 3
ny = 3
loc = {}
for i, site in enumerate(generate_lattice(nx, ny)):
    loc[site] = i  # index dict
uloc = [i for i in loc if i[3] == 0]


(
    kinetic,
    rashbaterm,
    uuaterm,
    ddaterm,
    uubterm,
    ddbterm,
    udaterm,
    duaterm,
    udbterm,
    dubterm,
) = utils.generate_np_zeros(10, [len(loc), len(loc)])


for site in loc:
    for hopsite in nn(site, nx, ny):
        kinetic[loc[site], loc[hopsite]] = 1
    for rashbasite, lam in rashba(site, nx, ny):
        rashbaterm[loc[site], loc[rashbasite]] = lam

    if site[3] == 0 and site[2] == 0:
        uuaterm[loc[site], loc[site]] = 1
        nsite = flip_spin(site)
        udaterm[loc[site], loc[nsite]] = 1
    elif site[3] == 0 and site[2] == 1:
        uubterm[loc[site], loc[site]] = 1
        nsite = flip_spin(site)
        udbterm[loc[site], loc[nsite]] = 1
    elif site[3] == 1 and site[2] == 0:
        ddaterm[loc[site], loc[site]] = 1
        nsite = flip_spin(site)
        duaterm[loc[site], loc[nsite]] = 1
    else:
        ddbterm[loc[site], loc[site]] = 1
        nsite = flip_spin(site)
        dubterm[loc[site], loc[nsite]] = 1


def hansatz(const, var):
    return (
        const.t * kinetic
        + const.lbd * rashbaterm
        + const.u / 2 * var.mua * uuaterm
        + const.u / 2 * (1 - var.mua) * ddaterm
        + const.u / 2 * (1 - var.mub) * ddbterm
        + const.u / 2 * var.mub * uubterm
        + var.mu * (uuaterm + uubterm + ddaterm + ddbterm)
        - const.u / 2 * (var.deltaa + 1.0j * var.deltaai) * udaterm
        - const.u / 2 * (var.deltaa - 1.0j * var.deltaai) * duaterm
        - const.u / 2 * (var.deltab + 1.0j * var.deltabi) * udbterm
        - const.u / 2 * (var.deltab - 1.0j * var.deltabi) * dubterm
    )


def h(const, var):
    return (
        (-0.5 * const.u) * (uuaterm + uubterm + ddaterm + ddbterm)
        + const.t * kinetic
        + const.lbd * rashbaterm
    )


def hint(const, var, e, v):
    energy = 0
    for site in uloc:  # interaction part by wick expansion
        nsite = flip_spin(site)
        cross = expectation(loc[site], loc[nsite], const.beta, e, v)
        energy += const.u * (
            expectation(loc[site], loc[site], const.beta, e, v)
            * expectation(loc[nsite], loc[nsite], const.beta, e, v)
            - jnp.conj(cross) * cross
        )
    return energy


const = namedtuple("const", ["t", "lbd", "u", "beta"])
var = namedtuple("var", ["mu", "mua", "mub", "deltaa", "deltab", "deltaai", "deltabi"])
const_params = const(1.0, 1.0, 6.0, 5.0)
init_params = var(0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0)


def test_honeycomb_rashba_hubbard():
    var_params = mf_optimize(hansatz, h, hint, const_params, init_params, 200, 50)
    f, _ = get_fe(hansatz, h, hint)
    assert f(const_params, var_params) < -58.0
