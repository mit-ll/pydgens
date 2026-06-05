# Copyright 2026 MIT Lincoln Laboratory
# SPDX-License-Identifier: MIT

# Analysis script for jax compilation and runtime
import jax
import jax.numpy as jnp
import time

def runtime_analysis_1(T,n,m):
    # A = jnp.eye(n)[None, :, :].repeat(T, axis=0)
    A = jnp.ones((T, n, n))
    B = jnp.ones((T, n, m))
    x = jnp.ones((T, n))
    u = jnp.ones((T, m))

    def linearize_one(A_, B_, x_, u_):
        return B_.T @ A_ @ x_ + u_

    @jax.jit
    def run():
        return jax.vmap(linearize_one)(A, B, x, u)

    t0 = time.time()
    result = run().block_until_ready()
    t1 = time.time()
    print("TTFE (JAX):", t1 - t0)
    print(jnp.round(result, 6))

    extimes = []
    for i in range(100):
        t0 = time.time()
        run().block_until_ready()
        t1 = time.time()
        extimes.append(t1-t0)
    
    avg_extime = jnp.mean(jnp.asarray(extimes))
    max_extime = jnp.max(jnp.asarray(extimes))
    print("Steady state average execution time (JAX):", avg_extime)
    print("Steady state maximum execution time (JAX):", max_extime)

if __name__ == "__main__":
    runtime_analysis_1(100, 32, 16)