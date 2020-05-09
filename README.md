# ADMF

This is the implementation of traditional mean field approach in quantum many-body physics but equipped with AD engine (powered by jax for now).

So called mean field approach is just an optimization problem to minimize the free energy with the parameterized ensemble ansatz (determined by non-interacting version of parameterized Hamiltonian in conventional approach).

For the correct perpective on mean field theory and its relation to variational inference, see [this post](https://jaan.io/how-does-physics-connect-machine-learning/).

Not a very fancy or big project, but it just works. This can be used in daily research with complicated models beyond toy Ising model. One can utilize codebase here to do conventional mean field stuies in a more elegant and swift way, and that's all.

The project has some customized operators so that exponential explosion and degeneracy problem in AD of ``eigh`` can be avoided.

Refer to ``tests/examples`` for some integrated examples of mean field theory on real world quantum models.

**Disclaimer**: This project is not extensively tested and benchmarked, so I have no guarantee that there is no bug or loophole in the program or the algorithm. The numerical results may be wrong.