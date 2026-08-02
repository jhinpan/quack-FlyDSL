"""Does the FlyDSL forward actually stay correct above MAX_N?

The notes assert it runs to N=262144 at bf16 accuracy indistinguishable from
shapes under the cap, but no committed sidecar carries those numbers. Probe
only; nothing is written back and nothing is committed from here.
"""

import json
import sys

import torch

sys.path.insert(0, "/root/quack-FlyDSL-review")

from quack import rmsnorm_flydsl as fd  # noqa: E402
from quack.flydsl import rmsnorm_config  # noqa: E402


def ref(x, w, eps):
    x32 = x.float()
    return (x32 * torch.rsqrt(x32.square().mean(-1, keepdim=True) + eps) * w.float()).to(x.dtype)


def main():
    dev = torch.device("cuda")
    dtype = torch.bfloat16
    eps = 1e-6
    shipped = rmsnorm_config.MAX_N
    assert fd.MAX_N == shipped, "the two MAX_N bindings already disagree"

    out = {
        "what": "FlyDSL fwd accuracy with MAX_N lifted, vs the same kernel under the cap",
        "device": torch.cuda.get_device_name(0),
        "dtype": "bfloat16",
        "shipped_max_n": shipped,
        "rows": [],
    }

    rmsnorm_config.MAX_N = 1 << 20
    fd.MAX_N = 1 << 20
    try:
        for n in (4096, 8192, 16384, 32768, 65536, 131072, 262144):
            m = max(1, (1 << 24) // n)
            torch.manual_seed(0)
            x = torch.randn(m, n, device=dev, dtype=dtype)
            w = torch.randn(n, device=dev, dtype=dtype)
            try:
                got = fd.rmsnorm(x, w, eps=eps)
                exp = ref(x, w, eps)
                d = (got.float() - exp.float()).abs()
                rel = (d / exp.float().abs().clamp_min(1e-6)).mean().item()
                amax = d.max().item()
                ok = torch.isfinite(got).all().item()
                out["rows"].append(
                    {
                        "N": n,
                        "m": m,
                        "over_cap": n > shipped,
                        "status": "ok" if ok else "nonfinite",
                        "mean_rel_err": rel,
                        "max_abs_err": amax,
                    }
                )
            except Exception as e:  # noqa: BLE001
                out["rows"].append(
                    {
                        "N": n,
                        "m": m,
                        "over_cap": n > shipped,
                        "status": "raised",
                        "error": f"{type(e).__name__}: {e}"[:300],
                    }
                )
            del x, w
            torch.cuda.empty_cache()
    finally:
        rmsnorm_config.MAX_N = shipped
        fd.MAX_N = shipped

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
