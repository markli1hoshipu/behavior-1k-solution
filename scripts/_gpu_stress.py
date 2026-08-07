"""Can these GPUs be driven above 0% SM at all? Pure compute, then a collective."""
import time
import jax, jax.numpy as jnp

n = jax.local_device_count()
print(f"devices={n}  backend={jax.default_backend()}", flush=True)

# 1) per-device matmul: should peg SM with no communication involved
N = 8192
x = jnp.ones((n, N, N), dtype=jnp.bfloat16)
f = jax.pmap(lambda a: a @ a)
y = f(x); y.block_until_ready()
t0 = time.perf_counter(); iters = 0
while time.perf_counter() - t0 < 30:
    y = f(y); iters += 1
y.block_until_ready()
dt = time.perf_counter() - t0
flops = 2 * N**3 * n * iters
print(f"[matmul] {iters} iters in {dt:.1f}s = {flops/dt/1e12:.1f} TFLOP/s across {n} GPUs", flush=True)

# 2) all-reduce: exercises NCCL, which is where the hang happened
g = jax.pmap(lambda a: jax.lax.psum(a, "i"), axis_name="i")
z = jnp.ones((n, 1024, 1024), dtype=jnp.float32)
w = g(z); w.block_until_ready()
t0 = time.perf_counter(); iters = 0
while time.perf_counter() - t0 < 20:
    w = g(w); iters += 1
    if iters % 50 == 0: w.block_until_ready()
w.block_until_ready()
dt = time.perf_counter() - t0
print(f"[allreduce] {iters} iters in {dt:.1f}s = {iters/dt:.0f}/s  sum ok={float(w[0,0,0])>0}", flush=True)
print("STRESS DONE", flush=True)
