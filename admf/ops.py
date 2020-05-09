"""
redefine gradient for some operations with numerical stability consideration
"""

from jax import numpy as jnp
from jax import custom_jvp, custom_vjp


@custom_jvp
def fermion_weight(x):
    return 1 / (1 + jnp.exp(x))


@fermion_weight.defjvp
def fermion_weight_jvp(primals, tangents):
    (x,) = primals
    (x_dot,) = tangents
    primal_out = fermion_weight(x)
    tangent_out = jnp.where(
        jnp.abs(x) <= 80.0, -1.0 / (2 + jnp.exp(x) + jnp.exp(-x)), 0.0
    )
    return primal_out, tangent_out * x_dot


@custom_jvp
def log1exp(x):
    return jnp.where(x <= 80.0, jnp.log(1 + jnp.exp(x)), x)


@log1exp.defjvp
def log1exp_jvp(primals, tangents):
    (x,) = primals
    (x_dot,) = tangents
    primal_out = log1exp(x)
    tangent_out = jnp.where(x >= -80.0, 1 / (1 + jnp.exp(-x)), 0)
    return primal_out, tangent_out * x_dot


@custom_vjp
def eigh(A):
    return jnp.linalg.eigh(A)


def _safe_reciprocal(x, epsilon=1e-20):
    return x / (x * x + epsilon)


def jaxeigh_fwd(A):
    e, v = eigh(A)
    return (e, v), (A, e, v)


def jaxeigh_bwd(r, tangents):
    a, e, v = r
    de, dv = tangents
    eye_n = jnp.eye(a.shape[-1], dtype=a.dtype)
    f = _safe_reciprocal(e[..., jnp.newaxis, :] - e[..., jnp.newaxis] + eye_n) - eye_n
    middle = jnp.diag(de) + jnp.multiply(f, (v.T @ dv))
    grad_a = jnp.conj(v) @ middle @ v.T
    return (grad_a,)


eigh.defvjp(jaxeigh_fwd, jaxeigh_bwd)
