from collections import namedtuple
import jax
import numpy as np
from jax import numpy as jnp

from .framework import expectation


def loc_index(l, cond=None):
    """

    :param l: Iterable. generator is ok.
    :param cond: Optional[Callable[[NamedTuple], bool]].
            element in l only be recorded into loc dict when True
    :return: Dict[NamedTuple, int], Dict[int, NamedTuple]. Maps between basis and integer index.
    """
    loc = {}
    rloc = {}
    i = 0
    for j in l:
        if cond is None or cond(j):
            loc[j] = i
            rloc[i] = j
            i += 1
    return loc, rloc


def generate_jnp_random_normal(n, shape, seed=42):
    assert n >= 1
    key = jax.random.PRNGKey(seed)
    yield jax.random.normal(key, shape)
    for _ in range(n - 1):
        key, subkey = jax.random.split(key)
        yield jax.random.normal(key, shape)


def generate_np_zeros(n, shape, dtype=np.complex64):
    return [np.zeros(shape, dtype=dtype) for _ in range(n)]


def generate_jnp_zeros(n, shape, dtype=jnp.complex64):
    return [jnp.zeros(shape, dtype=dtype) for _ in range(n)]


def spin_flip(b, attr="spin", flip_func=None):
    l = []
    for f in b._fields:
        if f != attr:
            l.append(getattr(b, f))
        else:
            if flip_func is None:
                flip_func = lambda s: 1 if s == 0 else 0
            l.append(flip_func(getattr(b, f)))
    return type(b)(*l)


mo = namedtuple("mo", ["o", "x", "y", "z"])


def measure_S(loc, site, beta, e, v):
    usite = loc[site]
    dsite = loc[spin_flip(site)]
    uu = expectation(usite, usite, beta, e, v)
    ud = expectation(usite, dsite, beta, e, v)
    du = expectation(dsite, usite, beta, e, v)
    dd = expectation(dsite, dsite, beta, e, v)
    return mo(uu + dd, ud + du, 1j * (du - ud), uu - dd)


def hubbard_int(loc, uloc=None, spin_flip_func=None, u="u"):
    """
    directly return Callable used as hint in mf_optimize with Hubbard type interaction

    :param loc:
    :param uloc:
    :param spin_flip_func:
    :param u:
    :return:
    """
    if spin_flip_func is None:
        spin_flip_func = spin_flip
    if uloc is None:
        uloc = [k for k in loc if k.spin == 0]

    def hint(const, var, e, v):
        energy = 0
        for site in uloc:  # interaction part by wick expansion
            nsite = spin_flip_func(site)
            cross = expectation(loc[site], loc[nsite], const.beta, e, v)
            energy += (
                expectation(loc[site], loc[site], const.beta, e, v)
                * expectation(loc[nsite], loc[nsite], const.beta, e, v)
                - jnp.conj(cross) * cross
            )
        if u:
            energy *= getattr(const, u)
        return energy

    return hint
