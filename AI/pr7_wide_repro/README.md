# PR7 wide-row RMSNorm reproduction

This record checks the `8192 x 262144`, BF16 activation / FP32 weight headline
from PR7 on two MI355X nodes. It compares the original PR7 commit (`4413997`)
with the proposed PR5 integration and separates kernel behavior from GPU-node
health.

## Method

`benchmarks/repro_pr7_wide.py`:

- verifies each provider against an FP32 reference before timing;
- pins the imported Quack and FlyDSL paths and records their versions;
- warms both providers for three seconds so MI355X clocks reach steady state;
- alternates provider order each round;
- times batches with one event pair rather than one pair per launch;
- rotates two 4 GiB inputs; and
- measures 512 MiB bandwidth probes before and after as a contention canary.

Both nodes used Python 3.10.12, torch
`2.9.1+rocm7.2.0.git7e1940d4`, HIP `7.2.26015-fc0010cf6a`, FlyDSL
`0.3.0.dev765`, and Triton `3.6.0+git42270451`.

## Result

| node | operation | FlyDSL | torch.compile | torch / FlyDSL | BW canary |
| --- | --- | ---: | ---: | ---: | ---: |
| `mia1-p02-g23` | forward | 2.448 ms | 2.395 ms | 0.978x | 0.997 |
| `smci355-ccs-aus-n08-09` | forward | 2.577 ms | 2.427 ms | 0.942x | 0.996 |
| `mia1-p02-g23` | backward | 3.929 ms | 5.180 ms | 1.318x | 0.999 |
| `smci355-ccs-aus-n08-09` | backward | 4.160 ms | 5.408 ms | 1.300x | 0.994 |

The original PR7 commit and the PR5 integration are indistinguishable on the
remote node: forward measured 2.575/2.431 ms at `4413997` and 2.574/2.429 ms
after integration.

The nodes are healthy. Their opening/closing best bandwidth probes were
6.81/6.79 TB/s and 6.54/6.51 TB/s respectively, with less than 0.7% drift.
The remote node had no KFD processes on any of its eight GPUs.

The backward speedup reproduces. The claimed forward speedup does not:
torch.compile is 2.2% to 5.8% faster under the controlled protocol. The old
benchmark has no steady-state warmup, provider-order control, provenance, or
contention canary, so its 2.526/2.916 ms forward pair should not be retained as
a merge claim.

## Reproduce

From the repository root with one idle GPU visible:

```bash
export HIP_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0
export ARCH=gfx950 FLYDSL_GPU_ARCH=gfx950
export PYTHONPATH="$PWD"

python benchmarks/repro_pr7_wide.py \
  --operation fwd \
  --output AI/pr7_wide_repro/local_fwd.json

python benchmarks/repro_pr7_wide.py \
  --operation bwd \
  --calls-per-sample 2 \
  --output AI/pr7_wide_repro/local_bwd.json
```

Only quote a run when `status` is `passed`, both correctness gates printed
`PASS`, and `contention_canary.quiet` is true.
