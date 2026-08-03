# 🦆 QuACK: A Quirky Assortment of CuTe Kernels 🦆

Kernels are written in the [CuTe-DSL](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/dsl_introduction.html).

## Installation

``` bash
# For CUDA 12.9:
pip install quack-kernels

# For CUDA 13.x:
pip install 'quack-kernels[cu13]' --extra-index-url https://download.pytorch.org/whl/cu130

# Do not use uv for CUDA 13.x installs yet: it can race/install
# nvidia-cutlass-dsl[cu13] in the wrong order (NVIDIA/cutlass#3259):
# https://github.com/NVIDIA/cutlass/issues/3259

# Optional: install NVIDIA matmul heuristics for better untuned GEMM configs
pip install 'quack-kernels[heuristics]'

# Optional: JAX bindings (pulls in jax and jax-tvm-ffi)
pip install 'quack-kernels[jax]'
```

## Requirements

- H100, B200/B300, or RTX 50 GPU
- CUDA toolkit 12.9+
- Python 3.12

There is also an opt-in ROCm RMSNorm backend built on FlyDSL, for MI355X
(gfx950). See [ROCm / FlyDSL](#rocm--flydsl-) below; it has its own extra and a
separate import path.

## Kernels 🐥

- 🦆 RMSNorm forward + backward
- 🦆 Softmax forward + backward
- 🦆 Cross entropy forward + backward
- 🦆 Layernorm forward + backward
- 🦆 Hopper gemm + epilogue
- 🦆 Blackwell gemm + epilogue
- 🦆 Blackwell GeForce gemm + epilogue

## Usage

```
from quack import rmsnorm, softmax, cross_entropy
```

JAX bindings are also available for some kernels (see [docs/jax.md](docs/jax.md)):

```
from quack.softmax_jax import softmax
```

## ROCm / FlyDSL 🐥

RMSNorm forward and backward also run on AMD MI355X (gfx950) through a separate
backend built on [FlyDSL](https://github.com/ROCm/FlyDSL). It is opt-in and does
not change the CUDA path.

```
pip install 'quack-kernels[flydsl]'
```

The CuTe kernels need `cutlass`, which is CUDA-only, so `from quack import
rmsnorm` does not work on a ROCm host. Import the backend explicitly:

```
from quack.rmsnorm_flydsl import rmsnorm
```

`rmsnorm()` takes the same arguments as `quack.rmsnorm`.

**Use it in eager code.** Under `torch.compile` the kernel is an opaque custom
op, so Inductor cannot fuse RMSNorm into neighbouring kernels the way it fuses
plain PyTorch. Measured on a transformer block with two RMSNorms around two
matmuls, this backend is 1.35x faster than eager PyTorch and 1.57x slower than
compiled PyTorch.

What this backend does not do yet:

- rows wider than 8192, or any row length that is not a multiple of 8 (4 for
  fp32) — use `torch.nn.functional.rms_norm` for those
- gfx950 only; other architectures are rejected rather than assumed to work
- layernorm, and the lower-level `rmsnorm_fwd` / `rmsnorm_bwd` entry points

Forward and backward autotuning are available through
`quack.rmsnorm_flydsl.rmsnorm_autotuned` and search only when
`FLYDSL_AUTOTUNE=1` is set. Design notes and measured results are in
[AI/flydsl_rmsnorm_notes.md](AI/flydsl_rmsnorm_notes.md).

## Documentations

- [JAX interface](docs/jax.md) — optional `jax` + `jax-tvm-ffi` bindings, see `quack/softmax_jax.py` for an example.

[2025-07-10] We have a comprehensive
[blogpost](media/2025-07-10-membound-sol.md) on how to get memory-bound kernels
to speed-of-light, right in the comfort of Python thanks to the [CuTe-DSL](https://docs.nvidia.com/cutlass/media/docs/pythonDSL/cute_dsl_general/dsl_introduction.html).

## Performance

<div align="center">
<figure>
  <img
  src="media/bf16_kernel_benchmarks_single_row.svg"
  >
</figure>
</div>

See our [blogpost](media/2025-07-10-membound-sol.md) for the details.

## Development

To set up the development environment:

```bash
pip install -e '.[dev]'
pre-commit install

# For CUDA 13.x:
pip install 'quack-kernels[dev,cu13]' --extra-index-url https://download.pytorch.org/whl/cu130

# Do not use uv for CUDA 13.x installs yet; use pip instead.
# See https://github.com/NVIDIA/cutlass/issues/3259
```
