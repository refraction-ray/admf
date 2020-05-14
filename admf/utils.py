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


def hubbard_int(loc, uloc, spin_flip, u="u"):
    def hint(const, var, e, v):
        energy = 0
        for site in uloc:  # interaction part by wick expansion
            nsite = spin_flip(site)
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
